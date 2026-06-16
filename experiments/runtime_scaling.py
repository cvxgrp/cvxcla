"""Runtime-scaling experiment for the CLA paper.

Measures the wall-clock time to trace the entire long-only efficient frontier as
a function of the number of assets n, for n in {20, 40, 80, 160, 320, 640}. To
make the dense and factor (Woodbury) backends solve the *identical* matrix -- so
they trace the same frontier and the comparison is a pure per-solve cost
difference -- the covariance is a ground-truth K-factor model
``Sigma = diag(d) + U diag(delta) U^T`` with a fixed number of factors K. The
dense backend receives ``Sigma`` materialised; the factor backend receives the
structured ``FactorCovariance``.

For reference we also time PyPortfolioOpt's ``CLA`` (the Bailey & Lopez de Prado
critical-line implementation) on the same dense problem, when it is installed.
It is a pure-Python reference implementation and is timed once per size (the
cvxcla backends are timed REPEATS times) because it is orders of magnitude
slower at the larger sizes.

Prints a table and, when matplotlib is available, writes paper/scaling.pdf
(log-log runtime vs n for each implementation).

Usage:
    uv run python experiments/runtime_scaling.py
"""

from __future__ import annotations

import time

import numpy as np

from cvxcla import CLA, FactorCovariance

SIZES = [20, 40, 80, 160, 320, 640]
N_FACTORS = 10
SEED = 7
REPEATS = 3


def make_problem(rng: np.random.Generator, n: int, k: int) -> tuple[np.ndarray, FactorCovariance, dict]:
    """Return (dense Sigma, FactorCovariance, problem dict) for a size-n factor model.

    The two covariance representations are mathematically identical, so both
    backends trace the same frontier; only the per-solve linear algebra differs.
    """
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    factor = FactorCovariance(d=d, u=u, delta=delta)
    dense = np.diag(d) + (u * delta) @ u.T
    problem = {
        "mean": rng.uniform(0.0, 1.0, n),  # dispersed expected returns
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }
    return dense, factor, problem


def median_trace(covariance: object, problem: dict) -> tuple[int, float]:
    """Return (turning points, median trace seconds) over REPEATS repetitions."""
    cla = CLA(covariance=covariance, **problem)
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        cla = CLA(covariance=covariance, **problem)
        times.append(time.perf_counter() - start)
    return len(cla), float(np.median(times))


def pypfopt_trace(dense: np.ndarray, mean: np.ndarray) -> tuple[int, float] | None:
    """Time PyPortfolioOpt's CLA (Bailey & Lopez de Prado) once, if installed.

    Returns (turning points, seconds) or None when PyPortfolioOpt is absent.
    """
    try:
        from pypfopt.cla import CLA as PYPFOPT_CLA
    except ImportError:
        return None
    start = time.perf_counter()
    ppo = PYPFOPT_CLA(mean, dense, weight_bounds=(0, 1))
    ppo._solve()  # compute the full critical line (all turning points)
    elapsed = time.perf_counter() - start
    return len(ppo.w), elapsed


def main() -> None:
    """Run the scaling sweep, print a table, and write the figure."""
    ns, dense_times, factor_times, ppo_times, points = [], [], [], [], []
    for n in SIZES:
        rng = np.random.default_rng(SEED)
        dense, factor, problem = make_problem(rng, n, N_FACTORS)
        n_pts, t_dense = median_trace(dense, problem)
        n_pts_f, t_factor = median_trace(factor, problem)
        if n_pts != n_pts_f:
            msg = f"backends disagree at n={n}: {n_pts} vs {n_pts_f}"
            raise RuntimeError(msg)
        ppo = pypfopt_trace(dense, problem["mean"])
        ns.append(n)
        points.append(n_pts)
        dense_times.append(t_dense)
        factor_times.append(t_factor)
        ppo_times.append(ppo[1] if ppo else None)
        ppo_str = f"  pypfopt={ppo[1] * 1e3:9.1f} ms (pts={ppo[0]})" if ppo else "  pypfopt=n/a"
        print(
            f"n={n:4d}  points={n_pts:4d}  dense={t_dense * 1e3:8.1f} ms  "
            f"factor={t_factor * 1e3:8.1f} ms  speedup={t_dense / t_factor:5.2f}x{ppo_str}"
        )

    # Empirical log-log slope (runtime ~ n^p) over the largest few sizes.
    def slope(times: list[float]) -> float:
        x = np.log(np.array(ns[-4:], dtype=float))
        y = np.log(np.array(times[-4:]))
        return float(np.polyfit(x, y, 1)[0])

    print(f"\ndense  empirical exponent p (time ~ n^p): {slope(dense_times):.2f}")
    print(f"factor empirical exponent p (time ~ n^p): {slope(factor_times):.2f}")
    if all(t is not None for t in ppo_times):
        print(f"pypfopt empirical exponent p (time ~ n^p): {slope([t for t in ppo_times if t]):.2f}")

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping paper/scaling.pdf")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    have_ppo = any(t is not None for t in ppo_times)
    if have_ppo:
        pn = [n for n, t in zip(ns, ppo_times, strict=True) if t is not None]
        pt = [t for t in ppo_times if t is not None]
        ax.loglog(pn, pt, "-^", ms=4, color="#7f7f7f", label="PyPortfolioOpt CLA (Bailey--Lopez de Prado)")
    ax.loglog(ns, dense_times, "-o", ms=4, color="#c00000", label="cvxcla, dense backend")
    ax.loglog(ns, factor_times, "-s", ms=4, color="#1f4e79", label=f"cvxcla, factor backend (Woodbury, K={N_FACTORS})")
    ax.set_xlabel("Number of assets $n$")
    ax.set_ylabel("Frontier trace time [s]")
    ax.set_title("CLA runtime vs problem size", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = "paper/scaling.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

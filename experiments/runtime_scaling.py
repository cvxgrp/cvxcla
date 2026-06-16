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
critical-line implementation) on the same dense problem, when it is installed,
and an incremental-inverse baseline (``InverseCLA`` from ``inverse_cla.py``) that
shares cvxcla's vectorised event logic but maintains ``Sigma_FF^{-1}`` through
rank-1 updates instead of re-solving each step. The baseline-vs-cvxcla-dense
comparison cleanly isolates the linear-algebra strategy (maintained inverse vs
fresh block-eliminated solve), since only that differs between them. (PyPortfolioOpt
differs from both algorithmically --- it recomputes a full inverse per candidate
each step --- so it is reported for context, not as a clean vectorisation control.)
All implementations are timed with the same protocol --- the median of REPEATS
repetitions --- so the comparison is apples-to-apples.

Prints a table and, when matplotlib is available, writes docs/paper/scaling.pdf
(log-log runtime vs n for each implementation).

Usage:
    uv run --with pyportfolioopt python experiments/runtime_scaling.py
"""

from __future__ import annotations

import time

import numpy as np

from cvxcla import CLA, FactorCovariance

try:  # the explicit-inverse baseline lives alongside this script
    from inverse_cla import InverseCLA
except ImportError:  # pragma: no cover - allows import from other CWDs
    InverseCLA = None

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


def median_trace_inverse(dense: np.ndarray, problem: dict) -> tuple[int, float] | None:
    """Time the vectorised explicit-inverse baseline over REPEATS repetitions.

    Returns (turning points, median seconds) or None when the baseline module
    is unavailable. Mirrors ``median_trace`` so the protocol is identical.
    """
    if InverseCLA is None:
        return None
    kw = {
        "mean": problem["mean"],
        "covariance": dense,
        "lower_bounds": problem["lower_bounds"],
        "upper_bounds": problem["upper_bounds"],
    }
    inv = InverseCLA(**kw)
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        inv = InverseCLA(**kw)
        times.append(time.perf_counter() - start)
    return len(inv), float(np.median(times))


def pypfopt_trace(dense: np.ndarray, mean: np.ndarray) -> tuple[int, float] | None:
    """Time PyPortfolioOpt's CLA (Bailey & Lopez de Prado) if installed.

    Timed over REPEATS repetitions with the same protocol as the other
    implementations (median of REPEATS), so the comparison is apples-to-apples.
    Returns (turning points, median seconds) or None when PyPortfolioOpt is
    absent.
    """
    try:
        from pypfopt.cla import CLA as PYPFOPT_CLA
    except ImportError:
        return None
    times = []
    n_pts = 0
    for _ in range(REPEATS):
        start = time.perf_counter()
        ppo = PYPFOPT_CLA(mean, dense, weight_bounds=(0, 1))
        ppo._solve()  # compute the full critical line (all turning points)
        times.append(time.perf_counter() - start)
        n_pts = len(ppo.w)
    return n_pts, float(np.median(times))


def main() -> None:
    """Run the scaling sweep, print a table, and write the figure."""
    ns, dense_times, factor_times, ppo_times, inv_times, points = [], [], [], [], [], []
    for n in SIZES:
        rng = np.random.default_rng(SEED)
        dense, factor, problem = make_problem(rng, n, N_FACTORS)
        n_pts, t_dense = median_trace(dense, problem)
        n_pts_f, t_factor = median_trace(factor, problem)
        if n_pts != n_pts_f:
            msg = f"backends disagree at n={n}: {n_pts} vs {n_pts_f}"
            raise RuntimeError(msg)
        inv = median_trace_inverse(dense, problem)
        if inv and inv[0] != n_pts:
            msg = f"inverse baseline disagrees at n={n}: {inv[0]} vs {n_pts}"
            raise RuntimeError(msg)
        ppo = pypfopt_trace(dense, problem["mean"])
        ns.append(n)
        points.append(n_pts)
        dense_times.append(t_dense)
        factor_times.append(t_factor)
        ppo_times.append(ppo[1] if ppo else None)
        inv_times.append(inv[1] if inv else None)
        inv_str = f"  inverse={inv[1] * 1e3:9.1f} ms" if inv else "  inverse=n/a"
        ppo_str = f"  pypfopt={ppo[1] * 1e3:9.1f} ms (pts={ppo[0]})" if ppo else "  pypfopt=n/a"
        print(
            f"n={n:4d}  points={n_pts:4d}  dense={t_dense * 1e3:8.1f} ms  "
            f"factor={t_factor * 1e3:8.1f} ms  speedup={t_dense / t_factor:5.2f}x{inv_str}{ppo_str}"
        )

    # The clean comparison is the incremental-inverse baseline vs the dense
    # backend: same vectorised event logic, differing only in the linear-algebra
    # strategy (maintained inverse vs fresh block-eliminated solve). PyPortfolioOpt
    # differs algorithmically (per-candidate full inverse), so its ratio is an
    # overall figure, not a clean control.
    if inv_times[-1] is not None:
        n_last = ns[-1]
        strategy = dense_times[-1] / inv_times[-1]  # solve cost / incremental-inverse cost
        line = (
            f"\nat n={n_last}: incremental inverse is {strategy:.2f}x the speed of the "
            f"dense fresh-solve backend; factor/dense = {dense_times[-1] / factor_times[-1]:.1f}x"
        )
        if ppo_times[-1] is not None:
            line += f"; dense/pypfopt (overall) = {ppo_times[-1] / dense_times[-1]:.0f}x"
        print(line)

    # Empirical log-log slope (runtime ~ n^p) over the largest few sizes.
    def slope(times: list[float]) -> float:
        x = np.log(np.array(ns[-4:], dtype=float))
        y = np.log(np.array(times[-4:]))
        return float(np.polyfit(x, y, 1)[0])

    print(f"\ndense   empirical exponent p (time ~ n^p): {slope(dense_times):.2f}")
    print(f"factor  empirical exponent p (time ~ n^p): {slope(factor_times):.2f}")
    if all(t is not None for t in inv_times):
        print(f"inverse empirical exponent p (time ~ n^p): {slope([t for t in inv_times if t]):.2f}")
    if all(t is not None for t in ppo_times):
        print(f"pypfopt empirical exponent p (time ~ n^p): {slope([t for t in ppo_times if t]):.2f}")

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter, ScalarFormatter
    except ImportError:
        print("matplotlib not available - skipping docs/paper/scaling.pdf")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    have_ppo = any(t is not None for t in ppo_times)
    if have_ppo:
        pn = [n for n, t in zip(ns, ppo_times, strict=True) if t is not None]
        pt = [t for t in ppo_times if t is not None]
        ax.loglog(pn, pt, "-^", ms=4, color="#7f7f7f", label="PyPortfolioOpt CLA (Bailey--Lopez de Prado)")
    if any(t is not None for t in inv_times):
        vn = [n for n, t in zip(ns, inv_times, strict=True) if t is not None]
        vt = [t for t in inv_times if t is not None]
        ax.loglog(vn, vt, "-D", ms=4, color="#2ca02c", label="vectorised explicit-inverse baseline")
    ax.loglog(ns, dense_times, "-o", ms=4, color="#c00000", label="cvxcla, dense backend")
    ax.loglog(ns, factor_times, "-s", ms=4, color="#1f4e79", label=f"cvxcla, factor backend (Woodbury, K={N_FACTORS})")
    ax.set_xlabel("Number of assets $n$")
    ax.set_ylabel("Frontier trace time [s]")
    ax.set_title("CLA runtime vs problem size", fontsize=9)

    # Label the x-axis at the actual problem sizes as plain integers, not the
    # default powers of ten (which never coincide with 20, 40, ..., 640 and
    # leave cluttered minor-tick labels on a log axis).
    ax.set_xticks(ns)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(ns[0] * 0.85, ns[-1] * 1.18)
    ax.tick_params(axis="x", labelsize=8)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = "docs/paper/scaling.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

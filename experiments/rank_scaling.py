"""Factor-rank scaling experiment for the CLA paper.

Fixes the universe at n = 320 assets and varies the number of factors
K in {20, 40, 80, 160, 320} of a ground-truth factor covariance
``Sigma = diag(d) + U diag(delta) U^T``. For each K the full long-only frontier
is traced once with the dense backend and once with the structured
``FactorCovariance`` (Woodbury) backend. Both solve the identical Sigma, so they
return the same frontier; only the per-solve cost differs.

The point: the dense backend's cost is independent of K (it factorises the same
n x n free block regardless of rank), while the Woodbury solve costs
``O(n_F K^2 + K^3)`` per step and grows with K -- so the factor backend's
advantage shrinks as K -> n and disappears once the low-rank part is no longer
low rank.

Prints a table and, when matplotlib is available, writes docs/paper/rank_scaling.pdf
(runtime vs K at fixed n).

Usage:
    uv run python experiments/rank_scaling.py
"""

from __future__ import annotations

import time

import numpy as np

from cvxcla import CLA, FactorCovariance

N_ASSETS = 320
RANKS = [20, 40, 80, 160, 320]
SEED = 11
REPEATS = 3


def make_problem(rng: np.random.Generator, n: int, k: int) -> tuple[np.ndarray, FactorCovariance, dict]:
    """Return (dense Sigma, FactorCovariance, problem dict) for an n-asset, K-factor model."""
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


def main() -> None:
    """Run the rank sweep, print a table, and write the figure."""
    ks, dense_times, factor_times, points = [], [], [], []
    for k in RANKS:
        rng = np.random.default_rng(SEED)
        dense, factor, problem = make_problem(rng, N_ASSETS, k)
        n_pts, t_dense = median_trace(dense, problem)
        n_pts_f, t_factor = median_trace(factor, problem)
        if n_pts != n_pts_f:
            msg = f"backends disagree at K={k}: {n_pts} vs {n_pts_f}"
            raise RuntimeError(msg)
        ks.append(k)
        points.append(n_pts)
        dense_times.append(t_dense)
        factor_times.append(t_factor)
        print(
            f"K={k:4d}  points={n_pts:4d}  dense={t_dense * 1e3:8.1f} ms  "
            f"factor={t_factor * 1e3:8.1f} ms  speedup={t_dense / t_factor:5.2f}x"
        )

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter, ScalarFormatter
    except ImportError:
        print("matplotlib not available - skipping docs/paper/rank_scaling.pdf")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.loglog(ks, dense_times, "-o", ms=4, color="#c00000", label="cvxcla, dense backend")
    ax.loglog(ks, factor_times, "-s", ms=4, color="#1f4e79", label="cvxcla, factor backend (Woodbury)")
    ax.set_xlabel(f"Number of factors $K$ (fixed $n={N_ASSETS}$)")
    ax.set_ylabel("Frontier trace time [s]")
    ax.set_title(f"CLA runtime vs factor rank ($n={N_ASSETS}$)", fontsize=9)
    ax.set_xticks(ks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(ks[0] * 0.85, ks[-1] * 1.18)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = "docs/paper/rank_scaling.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

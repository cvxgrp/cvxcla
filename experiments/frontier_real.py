"""Empirical CLA experiment on real S&P 500 returns.

Loads the daily-return matrix produced by ``experiments/fetch_sp500.py`` and:

1. traces the entire long-only efficient frontier of the full universe from the
   full-history sample covariance (the dense backend), and times a rank-K
   diagonal-plus-low-rank (Woodbury) approximation of the same covariance;
2. demonstrates the degeneracy limitation: on a short estimation window
   (T < N, a rank-deficient sample covariance) the event logic fails.

Writes the real-data frontier figure to ``docs/paper/real_frontier.pdf``.

Usage:
    uv run python experiments/fetch_sp500.py    # once, to download the data
    uv run python experiments/frontier_real.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from cvxcla import CLA, FactorCovariance

DATA = Path(__file__).parent / "data" / "sp500_pct_returns.parquet"
SHORT_WINDOW = 120  # trading days < N -> rank-deficient sample covariance
N_FACTORS = 20
REPEATS = 5


def problem(n: int) -> dict:
    """Long-only, fully-invested box-constrained problem of size n."""
    return {
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }


def median_trace(mean: np.ndarray, covariance: object, n: int) -> tuple[int, float]:
    """Return (turning points, median trace seconds) over REPEATS repetitions."""
    cla = CLA(mean=mean, covariance=covariance, **problem(n))
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        cla = CLA(mean=mean, covariance=covariance, **problem(n))
        times.append(time.perf_counter() - start)
    return cla, float(np.median(times))


def factor_estimate(cov: np.ndarray, k: int) -> FactorCovariance:
    """Diagonal-plus-low-rank estimate from the top-k eigenpairs of ``cov``."""
    evals, evecs = np.linalg.eigh(cov)
    top = np.argsort(evals)[::-1][:k]
    u = evecs[:, top]
    delta = np.clip(evals[top], 1e-12, None)
    d = np.clip(np.diag(cov) - np.einsum("ij,j,ij->i", u, delta, u), 1e-10, None)
    return FactorCovariance(d=d, u=u, delta=delta)


def main() -> None:
    """Run the empirical experiment, print statistics, and write the figure."""
    returns = pd.read_parquet(DATA)
    r = returns.to_numpy()
    t_days, n = r.shape
    mean = r.mean(axis=0)
    sample_cov = np.cov(r, rowvar=False)

    span = f"{returns.index[0].date()} -> {returns.index[-1].date()}"
    print(f"universe                : {n} assets x {t_days} days ({span})")
    print(f"covariance condition no.: {np.linalg.cond(sample_cov):,.0f}")

    # 1. Full-history frontier (dense) + Woodbury approximation.
    cla, dense_time = median_trace(mean, sample_cov, n)
    factor_cla, factor_time = median_trace(mean, factor_estimate(sample_cov, N_FACTORS), n)
    print(f"dense frontier          : {len(cla)} turning points in {dense_time * 1e3:.1f} ms (median)")
    print(f"factor (K={N_FACTORS}) approx    : {len(factor_cla)} turning points in {factor_time * 1e3:.1f} ms (median)")

    frontier = cla.frontier
    ann = 252.0  # annualise daily moments for readable axes
    ret = frontier.returns * ann
    vol = frontier.volatility * np.sqrt(ann)
    sharpe, _ = frontier.max_sharpe
    print(f"annualised return range : [{ret.min():.3f}, {ret.max():.3f}]")
    print(f"annualised vol range    : [{vol.min():.3f}, {vol.max():.3f}]")
    print(f"max Sharpe (annualised) : {sharpe * np.sqrt(ann):.3f}")

    # 2. Degeneracy demonstration on a short (rank-deficient) estimation window.
    short = r[-SHORT_WINDOW:]
    short_cov = np.cov(short, rowvar=False)
    rank = int(np.linalg.matrix_rank(short_cov))
    try:
        CLA(mean=short.mean(axis=0), covariance=short_cov, **problem(n))
        print(f"short window W={SHORT_WINDOW}     : traced (unexpected)")
    except Exception as exc:  # noqa: BLE001 - we are demonstrating the failure
        print(f"short window W={SHORT_WINDOW}     : rank(S)={rank} < N={n} -> {type(exc).__name__}: {exc}")

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping docs/paper/real_frontier.pdf")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(vol, ret, "-o", ms=2.5, lw=1.0, color="#1f4e79", label="Efficient frontier")
    ax.scatter(vol[[0, -1]], ret[[0, -1]], color="#c00000", zorder=5, s=18)
    ax.annotate("max return", (vol[0], ret[0]), textcoords="offset points", xytext=(-6, 4), ha="right", fontsize=8)
    ax.annotate("min variance", (vol[-1], ret[-1]), textcoords="offset points", xytext=(8, -2), fontsize=8)
    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised expected return")
    ax.set_title(f"S&P 500 efficient frontier: {n} assets, {t_days} days ({len(cla)} turning points)", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = "docs/paper/real_frontier.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Empirical CLA experiment on real S&P 500 returns.

Loads the frozen daily-return matrix committed at
``experiments/data/sp500_pct_returns.parquet`` and:

1. traces the entire long-only efficient frontier of the full universe from the
   full-history sample covariance (the dense backend), and times rank-K
   diagonal-plus-low-rank (Woodbury) approximations of the same covariance at
   several values of K to demonstrate the real-data payoff of the structured
   backend;
2. demonstrates the degeneracy limitation: on a short estimation window
   (T < N, a rank-deficient sample covariance) the event logic fails.

Writes the real-data frontier figure to ``docs/paper/real_frontier.pdf`` and
a timing-comparison bar chart to ``docs/paper/real_timing.pdf``.

The return matrix is a frozen snapshot committed to the repository, so this
experiment reproduces deterministically and offline. ``experiments/fetch_sp500.py``
is only needed to refresh that snapshot from live sources, not to run this script.

Usage:
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
# Factor ranks to sweep: chosen to span from a very low-rank approximation
# (K=5, ~1% of 494 assets) through a moderate-rank one (K=40, ~8%), covering
# the regime typical risk models inhabit (K << n).
FACTOR_RANKS = [5, 10, 20, 40]
REPEATS = 5


def problem(n: int) -> dict:
    """Long-only, fully-invested box-constrained problem of size n."""
    return {
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }


def median_trace(mean: np.ndarray, covariance: object, n: int) -> tuple[object, float]:
    """Return (last CLA, median trace seconds) over REPEATS repetitions."""
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
    """Run the empirical experiment, print statistics, and write the figures."""
    returns = pd.read_parquet(DATA)
    r = returns.to_numpy()
    t_days, n = r.shape
    mean = r.mean(axis=0)
    sample_cov = np.cov(r, rowvar=False)

    span = f"{returns.index[0].date()} -> {returns.index[-1].date()}"
    print(f"universe                : {n} assets x {t_days} days ({span})")
    print(f"covariance condition no.: {np.linalg.cond(sample_cov):,.0f}")

    # 1. Full-history frontier (dense backend).
    cla, dense_time = median_trace(mean, sample_cov, n)
    print(f"dense frontier          : {len(cla)} turning points in {dense_time * 1e3:.1f} ms (median)")

    # 2. Factor backends at several ranks K: real-data payoff demonstration.
    print(f"\n{'K':>4}  {'points':>6}  {'factor [ms]':>11}  {'dense [ms]':>10}  {'speedup':>7}")
    print("-" * 48)
    factor_rows: list[tuple[int, int, float, float]] = []
    for k in FACTOR_RANKS:
        fc = factor_estimate(sample_cov, k)
        factor_cla, factor_time = median_trace(mean, fc, n)
        speedup = dense_time / factor_time
        factor_rows.append((k, len(factor_cla), factor_time, speedup))
        print(f"{k:>4}  {len(factor_cla):>6}  {factor_time * 1e3:>11.1f}  {dense_time * 1e3:>10.1f}  {speedup:>6.1f}x")

    # 3. Frontier geometry (dense frontier, for the paper's frontier figure).
    frontier = cla.frontier
    ann = 252.0  # annualise daily moments for readable axes
    ret = frontier.returns * ann
    vol = frontier.volatility * np.sqrt(ann)
    sharpe, _ = frontier.max_sharpe
    print(f"\nannualised return range : [{ret.min():.3f}, {ret.max():.3f}]")
    print(f"annualised vol range    : [{vol.min():.3f}, {vol.max():.3f}]")
    print(f"max Sharpe (annualised) : {sharpe * np.sqrt(ann):.3f}")

    # 4. Degeneracy demonstration on a short (rank-deficient) estimation window.
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
        print("matplotlib not available - skipping docs/paper/real_frontier.pdf and real_timing.pdf")
        return

    # Figure A: the efficient frontier (unchanged from the original).
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
    out_frontier = "docs/paper/real_frontier.pdf"
    fig.savefig(out_frontier)
    print(f"wrote {out_frontier}")
    plt.close(fig)

    # Figure B: timing comparison — dense vs factor at each K (bar chart).
    ks = [row[0] for row in factor_rows]
    factor_ms = [row[2] * 1e3 for row in factor_rows]
    x = np.arange(len(ks))
    width = 0.35

    fig2, ax2 = plt.subplots(figsize=(5.0, 3.4))
    ax2.bar(
        x - width / 2,
        [dense_time * 1e3] * len(ks),
        width,
        label="Dense (sample cov.)",
        color="#c00000",
        alpha=0.85,
    )
    ax2.bar(x + width / 2, factor_ms, width, label="Factor (Woodbury)", color="#1f4e79", alpha=0.85)

    # Annotate speedup above each pair.
    for i, (_, _, _ft, su) in enumerate(factor_rows):
        ax2.text(i, dense_time * 1e3 + 0.5, f"{su:.1f}x", ha="center", va="bottom", fontsize=7.5, color="#555555")

    ax2.set_xlabel("Factor rank $K$")
    ax2.set_ylabel("Frontier trace time [ms]")
    ax2.set_title(f"Dense vs factor backend on S\\&P 500 ($N={n}$ assets)", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    out_timing = "docs/paper/real_timing.pdf"
    fig2.savefig(out_timing)
    print(f"wrote {out_timing}")
    plt.close(fig2)


if __name__ == "__main__":
    main()

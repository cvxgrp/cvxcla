"""Numerical experiment for the CLA paper: 20 assets, 50 days.

Simulates T = 50 daily return observations for N = 20 assets from a
K = 5 factor model and traces the entire long-only, fully-invested efficient
frontier with the Critical Line Algorithm. The small size keeps the frontier's
piecewise-linear corner structure clearly visible in the figure.

We adopt the factor risk model itself as the covariance,
``Sigma = diag(d) + U diag(delta) U^T`` (the standard practice -- a structured
risk model rather than the noisy sample covariance), and estimate expected
returns by the sample mean over the 50 days. The frontier is then traced twice
with two mathematically identical representations of the same Sigma: the dense
matrix and the structured ``FactorCovariance`` (Woodbury) backend. Because the
matrix is identical, both backends trace the *same* frontier; only the per-solve
linear algebra differs.

Prints the summary statistics quoted in docs/paper/cla.tex and, when matplotlib is
available, writes the frontier figure to docs/paper/frontier.pdf.

Usage:
    uv run python experiments/frontier_20x50.py
"""

from __future__ import annotations

import time

import numpy as np

from cvxcla import CLA, FactorCovariance

N_ASSETS = 20
N_DAYS = 50
N_FACTORS = 5
SEED = 42
REPEATS = 5


def build_model(rng: np.random.Generator) -> tuple[FactorCovariance, np.ndarray]:
    """Return the ground-truth factor covariance and the per-asset expected returns."""
    u = rng.standard_normal((N_ASSETS, N_FACTORS)) / np.sqrt(N_ASSETS)
    delta = rng.uniform(0.5, 2.0, N_FACTORS) * N_ASSETS
    d = rng.uniform(0.5, 2.0, N_ASSETS)
    expected = rng.uniform(0.0, 1.0, N_ASSETS)  # dispersed expected returns
    return FactorCovariance(d=d, u=u, delta=delta), expected


def simulate_returns(rng: np.random.Generator, factor: FactorCovariance, expected: np.ndarray) -> np.ndarray:
    """Simulate (N_DAYS, N_ASSETS) returns whose population covariance is ``factor``."""
    factor_returns = rng.standard_normal((N_DAYS, N_FACTORS)) * np.sqrt(factor.delta)
    idiosyncratic = rng.standard_normal((N_DAYS, N_ASSETS)) * np.sqrt(factor.d)
    return expected + factor_returns @ factor.u.T + idiosyncratic


def trace(problem: dict, covariance: object) -> tuple[object, float]:
    """Trace the frontier ``REPEATS`` times and return (last CLA, median seconds)."""
    times = []
    cla = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        cla = CLA(covariance=covariance, **problem)
        times.append(time.perf_counter() - start)
    return cla, float(np.median(times))


def main() -> None:
    """Run the experiment, print statistics, and write the frontier figure."""
    rng = np.random.default_rng(SEED)
    factor, expected = build_model(rng)
    returns = simulate_returns(rng, factor, expected)

    mean = returns.mean(axis=0)
    dense_cov = np.diag(factor.d) + (factor.u * factor.delta) @ factor.u.T

    problem = {
        "mean": mean,
        "lower_bounds": np.zeros(N_ASSETS),
        "upper_bounds": np.ones(N_ASSETS),
        "a": np.ones((1, N_ASSETS)),
        "b": np.ones(1),
    }

    cla, dense_time = trace(problem, dense_cov)
    factor_cla, factor_time = trace(problem, factor)
    if len(cla) != len(factor_cla):
        msg = f"backends disagree: {len(cla)} vs {len(factor_cla)}"
        raise RuntimeError(msg)

    frontier = cla.frontier
    returns_f = frontier.returns
    vol_f = frontier.volatility
    max_sharpe, _ = frontier.max_sharpe
    cond = float(np.linalg.cond(dense_cov))

    print(f"problem                 : {N_ASSETS} assets x {N_DAYS} days, {N_FACTORS}-factor model")
    print(f"covariance condition no.: {cond:,.1f}")
    print(f"turning points          : {len(cla)}  (dense and factor agree)")
    print(f"dense trace  (median)   : {dense_time * 1e3:.1f} ms  ({dense_time / len(cla) * 1e3:.2f} ms/point)")
    print(f"factor trace (median)   : {factor_time * 1e3:.1f} ms  ({factor_time / len(cla) * 1e3:.2f} ms/point)")
    print(f"factor speedup          : {dense_time / factor_time:.2f}x")
    print(f"expected-return range   : [{returns_f.min():.4f}, {returns_f.max():.4f}]")
    print(f"volatility range        : [{vol_f.min():.4f}, {vol_f.max():.4f}]")
    print(f"max Sharpe (model units): {max_sharpe:.4f}")

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping docs/paper/frontier.pdf")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(vol_f, returns_f, "-o", ms=2.5, lw=1.0, color="#1f4e79", label="Efficient frontier")
    ax.scatter(vol_f[[0, -1]], returns_f[[0, -1]], color="#c00000", zorder=5, s=18)
    ax.annotate(
        "max return", (vol_f[0], returns_f[0]), textcoords="offset points", xytext=(-6, 4), ha="right", fontsize=8
    )
    ax.annotate("min variance", (vol_f[-1], returns_f[-1]), textcoords="offset points", xytext=(8, -2), fontsize=8)
    ax.set_xlabel("Volatility (model units)")
    ax.set_ylabel("Expected return (model units)")
    ax.set_title(f"Efficient frontier: {N_ASSETS} assets, {N_DAYS} days ({len(cla)} turning points)", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = "docs/paper/frontier.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

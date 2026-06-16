"""Numerical experiment for the CLA paper: 200 assets, 2000 days.

Simulates T = 2000 daily return observations for N = 200 assets from a
K = 15 factor model, estimates the sample mean vector and sample covariance
matrix, and traces the entire long-only, fully-invested efficient frontier
with the Critical Line Algorithm. The same frontier is traced twice: once with
the dense covariance backend and once with a diagonal-plus-low-rank factor
estimate of the sample covariance (the Woodbury backend). Prints the summary
statistics quoted in paper/cla.tex and, when matplotlib is available, writes the
frontier figure to paper/frontier.pdf.

Usage:
    uv run python experiments/frontier_200x2000.py
"""

from __future__ import annotations

import time

import numpy as np

from cvxcla import CLA, FactorCovariance

N_ASSETS = 200
N_DAYS = 2000
N_FACTORS = 15
SEED = 42
REPEATS = 5


def simulate_returns(rng: np.random.Generator) -> np.ndarray:
    """Return a (N_DAYS, N_ASSETS) matrix of synthetic daily returns.

    A K-factor data-generating process: each asset loads on N_FACTORS common
    factors plus a well-conditioned idiosyncratic shock, with a dispersed
    cross-section of expected returns. N_DAYS > N_ASSETS, so the sample
    covariance is full rank.
    """
    loadings = rng.standard_normal((N_ASSETS, N_FACTORS)) / np.sqrt(N_ASSETS)
    factor_var = rng.uniform(0.5, 2.0, N_FACTORS) * N_ASSETS
    factor_returns = rng.standard_normal((N_DAYS, N_FACTORS)) * np.sqrt(factor_var)
    idio_var = rng.uniform(0.5, 2.0, N_ASSETS)
    idiosyncratic = rng.standard_normal((N_DAYS, N_ASSETS)) * np.sqrt(idio_var)
    expected = rng.uniform(0.0, 1.0, N_ASSETS)  # dispersed expected returns
    return expected + factor_returns @ loadings.T + idiosyncratic


def factor_estimate(sample_cov: np.ndarray, k: int) -> FactorCovariance:
    """Diagonal-plus-low-rank estimate of ``sample_cov`` from its top-k eigenpairs.

    Keeps the leading ``k`` eigenpairs as the low-rank part and matches the
    full diagonal of the sample covariance via the idiosyncratic residual, so
    ``diag(estimate) == diag(sample_cov)`` exactly.
    """
    evals, evecs = np.linalg.eigh(sample_cov)
    top = np.argsort(evals)[::-1][:k]
    u = evecs[:, top]
    delta = evals[top]
    residual = np.diag(sample_cov) - np.einsum("ij,j,ij->i", u, delta, u)
    d = np.clip(residual, 1e-6, None)
    return FactorCovariance(d=d, u=u, delta=delta)


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
    returns = simulate_returns(rng)

    mean = returns.mean(axis=0)
    sample_cov = np.cov(returns, rowvar=False)

    problem = {
        "mean": mean,
        "lower_bounds": np.zeros(N_ASSETS),
        "upper_bounds": np.ones(N_ASSETS),
        "a": np.ones((1, N_ASSETS)),
        "b": np.ones(1),
    }

    cla, dense_time = trace(problem, sample_cov)
    factor_cla, factor_time = trace(problem, factor_estimate(sample_cov, N_FACTORS))

    frontier = cla.frontier
    returns_f = frontier.returns
    vol_f = frontier.volatility
    max_sharpe, _ = frontier.max_sharpe
    cond = float(np.linalg.cond(sample_cov))

    print(f"problem                 : {N_ASSETS} assets x {N_DAYS} days, {N_FACTORS}-factor model")
    print(f"covariance condition no.: {cond:,.1f}")
    print(f"turning points (dense)  : {len(cla)}")
    print(f"turning points (factor) : {len(factor_cla)}")
    print(f"dense trace  (median)   : {dense_time * 1e3:.1f} ms  ({dense_time / len(cla) * 1e3:.2f} ms/point)")
    print(f"factor trace (median)   : {factor_time * 1e3:.1f} ms  ({factor_time / len(factor_cla) * 1e3:.2f} ms/point)")
    print(f"factor speedup          : {dense_time / factor_time:.2f}x")
    print(f"expected-return range   : [{returns_f.min():.4f}, {returns_f.max():.4f}]")
    print(f"volatility range        : [{vol_f.min():.4f}, {vol_f.max():.4f}]")
    print(f"max Sharpe (model units): {max_sharpe:.4f}")

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping paper/frontier.pdf")
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
    out = "paper/frontier.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

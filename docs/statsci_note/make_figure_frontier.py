"""Figure 2 for the Statistical Science note: the 20-asset efficient frontier.

The portfolio face of the identity, as a concrete Critical Line Algorithm trace.
We simulate T = 50 daily returns for N = 20 assets from a K = 5 factor model, adopt
the factor covariance ``Sigma = diag(d) + U diag(delta) U^T`` as the risk model and the
sample mean as expected returns, and trace the long-only, fully invested frontier
(budget ``1' w = 1``, ``0 <= w <= 1``). The frontier is returned exactly as a finite
sequence of turning points; we plot them in expected-return / volatility space.

Self-contained and seeded so the figure reproduces. Change ``SEED`` to vary it (e.g. to
distinguish this illustration from the companion software paper's frontier figure).

Usage:
    uv run --with matplotlib --with scipy python docs/statsci_note/make_figure_frontier.py
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from cvxcla import CLA, FactorCovariance

N_ASSETS = 20
N_DAYS = 50
N_FACTORS = 5
SEED = 42


def main() -> None:
    """Build the factor model, trace the long-only frontier, and write the figure."""
    rng = np.random.default_rng(SEED)
    u = rng.standard_normal((N_ASSETS, N_FACTORS)) / np.sqrt(N_ASSETS)
    delta = rng.uniform(0.5, 2.0, N_FACTORS) * N_ASSETS
    d = rng.uniform(0.5, 2.0, N_ASSETS)
    expected = rng.uniform(0.0, 1.0, N_ASSETS)  # dispersed expected returns
    factor = FactorCovariance(d=d, u=u, delta=delta)

    # Simulate returns whose population covariance is the factor model, then estimate
    # the expected returns by the sample mean (as in the note).
    factor_returns = rng.standard_normal((N_DAYS, N_FACTORS)) * np.sqrt(delta)
    idiosyncratic = rng.standard_normal((N_DAYS, N_ASSETS)) * np.sqrt(d)
    returns = expected + factor_returns @ u.T + idiosyncratic
    mean = returns.mean(axis=0)

    cla = CLA(
        mean=mean,
        covariance=factor,
        lower_bounds=np.zeros(N_ASSETS),
        upper_bounds=np.ones(N_ASSETS),
        a=np.ones((1, N_ASSETS)),
        b=np.ones(1),
    )
    frontier = cla.frontier
    vol, ret = frontier.volatility, frontier.returns
    print(f"{len(cla)} turning points; return range [{ret.min():.3f}, {ret.max():.3f}]")

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(vol, ret, "-o", ms=2.5, lw=1.0, color="#1f4e79")
    ax.scatter(vol[[0, -1]], ret[[0, -1]], color="#c00000", zorder=5, s=18)
    ax.annotate("max return", (vol[0], ret[0]), textcoords="offset points", xytext=(-6, 4), ha="right", fontsize=8)
    ax.annotate("min variance", (vol[-1], ret[-1]), textcoords="offset points", xytext=(8, -2), fontsize=8)
    ax.set_xlabel("Volatility (model units)")
    ax.set_ylabel("Expected return (model units)")
    ax.set_title(f"Efficient frontier: {N_ASSETS} assets, {N_DAYS} days ({len(cla)} turning points)", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = "docs/statsci_note/frontier.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

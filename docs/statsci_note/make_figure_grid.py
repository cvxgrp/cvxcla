"""Second figure: an exact constrained path versus a grid that misses its corners.

Traces a non-negative LASSO path (a constrained instance: standard LASSO does not
sign-constrain) exactly with cvxcla, then overlays what a practitioner gridding the
penalty would report: solutions sampled at a coarse set of lambda values and joined
by straight lines. The grid points lie on the true path, but linear interpolation
between them cuts the corners that the exact homotopy resolves.

Usage:
    uv run --with matplotlib --with scipy python docs/statsci_note/make_figure_grid.py
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

try:
    from cvxcla.lasso import Lasso
except ImportError as exc:  # pragma: no cover
    msg = "run from the repo root so cvxcla is importable"
    raise SystemExit(msg) from exc


def beta_at(lams_path: np.ndarray, betas_path: np.ndarray, lam: float) -> np.ndarray:
    """Exact coefficient vector at penalty ``lam`` by interpolation on the path."""
    return np.array([np.interp(lam, lams_path[::-1], betas_path[::-1, j]) for j in range(betas_path.shape[1])])


def main() -> None:
    """Trace an exact non-negative LASSO path and contrast it with a coarse grid."""
    rng = np.random.default_rng(5)
    m, n = 70, 6
    x = rng.standard_normal((m, n))
    x = x - x.mean(axis=0, keepdims=True)
    beta_true = np.zeros(n)
    beta_true[[0, 1, 3]] = [2.2, 1.4, 0.8]
    y = x @ beta_true + 0.5 * rng.standard_normal(m)
    y = y - y.mean()

    lasso = Lasso(x=x, y=y, nonneg=True)
    lams = np.array([bp.lam for bp in lasso.path])
    betas = np.array([bp.beta for bp in lasso.path])

    grid = np.linspace(lams.min(), lams.max(), 6)
    grid_betas = np.array([beta_at(lams, betas, g) for g in grid])

    show = [j for j in range(n) if np.abs(betas[:, j]).max() > 1e-6][:3]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for lam in lams:
        ax.axvline(lam, color="0.9", lw=0.5, zorder=0)
    for k, j in enumerate(show):
        col = colors[k % len(colors)]
        ax.plot(lams, betas[:, j], "-", lw=1.6, color=col, label=rf"$\beta_{{{j + 1}}}$ exact")
        ax.plot(lams, betas[:, j], "o", ms=3, color=col)
        ax.plot(grid, grid_betas[:, j], "--", lw=1.2, color=col, alpha=0.9)
        ax.plot(grid, grid_betas[:, j], "s", ms=4, mfc="white", mec=col)
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.set_xlabel(r"penalty $\lambda$")
    ax.set_ylabel(r"coefficient $\beta_j(\lambda)$")
    ax.set_title("Exact constrained path (solid) vs. a 6-point grid (dashed)", fontsize=10)
    ax.invert_xaxis()
    ax.margins(x=0.02)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    out = "docs/statsci_note/path_grid_misses.pdf"
    fig.savefig(out)
    print(f"wrote {out} ({len(lams)} exact breakpoints, {len(grid)} grid points)")


if __name__ == "__main__":
    main()

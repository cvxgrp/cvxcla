"""Generate the 'one path, two readings' figure for the Statistical Science note.

Traces a small LASSO path with cvxcla and plots the piecewise-linear coefficient
paths beta_j(lambda). The same breakpoints carry two vocabularies: a LASSO 'enter'
(a correlation reaching lambda) is a CLA blocked-multiplier sign change; a LASSO
'leave' (a coefficient reaching zero) is a CLA free weight reaching a bound. The
curve is the shared homotopy of the perspective; the figure is real output, not a
schematic.

Usage:
    uv run --with matplotlib --with scipy python docs/statsci_note/make_figure.py
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


def main() -> None:
    """Trace a small LASSO path and write the dual-reading figure."""
    rng = np.random.default_rng(2)
    m, n = 60, 8
    x = rng.standard_normal((m, n))
    x = x - x.mean(axis=0, keepdims=True)
    beta_true = np.zeros(n)
    beta_true[[0, 2, 5]] = [2.0, -1.6, 1.1]
    y = x @ beta_true + 0.5 * rng.standard_normal(m)
    y = y - y.mean()

    lasso = Lasso(x=x, y=y)
    lams = np.array([bp.lam for bp in lasso.path])
    betas = np.array([bp.beta for bp in lasso.path])  # (k, n)

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for lam in lams:
        ax.axvline(lam, color="0.88", lw=0.5, zorder=0)
    for j in range(n):
        ax.plot(lams, betas[:, j], "-o", ms=2.5, lw=1.4)
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.set_xlabel(r"parameter $\lambda$  (LASSO penalty $\;\equiv\;$ CLA risk aversion)")
    ax.set_ylabel(r"coordinate $\beta_j(\lambda)\;=\;w_j(\lambda)$")
    ax.set_title("One piecewise-linear path, two readings", fontsize=10)
    ax.invert_xaxis()  # the homotopy runs as lambda decreases
    ax.margins(x=0.02)
    ax.text(
        0.015,
        0.03,
        f"{len(lams)} breakpoints; each is a LASSO enter/leave $=$ a CLA bound/multiplier event",
        transform=ax.transAxes,
        fontsize=7.5,
        color="0.35",
    )
    fig.tight_layout()
    out = "docs/statsci_note/path_both_sides.pdf"
    fig.savefig(out)
    print(f"wrote {out} ({len(lams)} breakpoints)")


if __name__ == "__main__":
    main()

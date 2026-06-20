"""The identity figure: the LASSO path and the mean-variance QP are one curve.

Demonstrates Theorem 1 by tracing the *same* data two ways and overlaying them in
coefficient space, plotted against the shared parameter ||beta||_1 (the l1 budget on
the regression side, the gross-exposure cap on the portfolio side):

  * the LASSO regularisation path, traced by cvxcla's homotopy (solid lines);
  * the gross-exposure-constrained mean-variance solutions, solved *independently*
    as QPs with Sigma = X^T X and mu = X^T y (open markers): for each cap c,
        minimize  1/2 ||y - X beta||^2   s.t.  ||beta||_1 <= c.

The markers land on the homotopy path, which is the content of the identity: the two
formulations are the same curve. The script prints the max discrepancy as a check
that this is a genuine demonstration, not a relabelling.

Usage:
    uv run --with cvxpy --with matplotlib --with scipy python docs/statsci_note/make_figure_identity.py
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

try:
    import cvxpy as cp

    from cvxcla.lasso import Lasso
except ImportError as exc:  # pragma: no cover
    msg = "run from the repo root with cvxpy available"
    raise SystemExit(msg) from exc


def lasso_beta_at(l1_path: np.ndarray, betas_path: np.ndarray, c: float) -> np.ndarray:
    """Coefficient vector on the LASSO path at l1-norm ``c`` (linear interpolation)."""
    order = np.argsort(l1_path)
    return np.array([np.interp(c, l1_path[order], betas_path[order, j]) for j in range(betas_path.shape[1])])


def main() -> None:
    """Trace the LASSO path and the constrained QP on one problem and overlay them."""
    rng = np.random.default_rng(1)
    m, n = 80, 6
    x = rng.standard_normal((m, n))
    x = x - x.mean(axis=0, keepdims=True)
    beta_true = np.array([2.0, -1.5, 0.0, 1.1, 0.0, 0.0])
    y = x @ beta_true + 0.5 * rng.standard_normal(m)
    y = y - y.mean()

    # LASSO homotopy (cvxcla): the regression / penalty formulation.
    lasso = Lasso(x=x, y=y)
    betas_path = np.array([bp.beta for bp in lasso.path])
    l1_path = np.abs(betas_path).sum(axis=1)
    l1_max = float(l1_path.max())

    # Gross-exposure-constrained mean-variance QP (CVXPY): the portfolio formulation,
    # solved independently at a grid of caps c. Sigma = X^T X, mu = X^T y are implicit
    # in the least-squares objective 1/2||y - X beta||^2.
    caps = np.linspace(0.05 * l1_max, 0.97 * l1_max, 9)
    qp_betas, qp_l1 = [], []
    for c in caps:
        beta = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(y - x @ beta)), [cp.norm1(beta) <= c])
        prob.solve(solver=cp.CLARABEL)
        qp_betas.append(beta.value)
        qp_l1.append(float(np.abs(beta.value).sum()))
    qp_betas = np.array(qp_betas)
    qp_l1 = np.array(qp_l1)

    # Verification: the QP solutions must lie on the LASSO path (Theorem 1).
    discrepancy = max(
        float(np.max(np.abs(qp_betas[i] - lasso_beta_at(l1_path, betas_path, qp_l1[i])))) for i in range(len(caps))
    )
    print(f"max |QP beta - LASSO-path beta| at matched ||beta||_1: {discrepancy:.1e}")
    print(f"LASSO breakpoints: {len(l1_path)}; QP caps: {len(caps)}")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    show = [j for j in range(n) if np.abs(betas_path[:, j]).max() > 1e-6]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    order = np.argsort(l1_path)
    for k, j in enumerate(show):
        col = colors[k % len(colors)]
        ax.plot(l1_path[order], betas_path[order, j], "-", lw=1.6, color=col, label=rf"$\beta_{{{j + 1}}}$")
        ax.plot(qp_l1, qp_betas[:, j], "o", ms=6, mfc="none", mec=col, mew=1.3)
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.set_xlabel(r"$\|\beta\|_1$  ($\ell_1$ budget $\;\equiv\;$ gross-exposure cap)")
    ax.set_ylabel(r"coefficient $=$ weight")
    ax.set_title("LASSO homotopy (lines) vs. mean-variance QP (markers): one curve", fontsize=10)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    out = "docs/statsci_note/path_identity.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

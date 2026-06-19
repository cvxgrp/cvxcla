"""LASSO-path experiment: the CLA engine traces a LASSO, checked against scikit-learn.

The Critical Line Algorithm and the LARS/LASSO homotopy are two instances of the
same parametric active-set scheme (see the paper, sec. on homotopy methods). This
script makes that concrete and *verifies* it: the same ``cvxcla.pathtracer.trace``
loop that walks the efficient frontier is driven by ``cvxcla.Lasso`` to trace the
entire LASSO regularisation path, and the resulting path is compared, breakpoint by
breakpoint, against scikit-learn's ``lars_path`` (the reference LARS/LASSO
implementation).

Conventions. ``cvxcla.Lasso`` traces the penalty form
``1/2 ||y - X beta||^2 + lam ||beta||_1`` with ``lam`` in ``[0, ||X^T y||_inf]``.
scikit-learn's ``lars_path(method="lasso")`` reports its breakpoints as
``alphas`` in the averaged form ``1/(2 m) ||y - X beta||^2 + alpha ||beta||_1`` for
``m`` samples, so the two parameters are related by ``lam = m * alpha``. We map
scikit-learn's breakpoints into cvxcla's ``lam`` and compare coefficient vectors.

Prints a table and, when matplotlib is available, writes docs/paper/lasso_path.pdf
(the coefficient paths from both implementations overlaid).

Usage:
    uv run --with scikit-learn --with matplotlib python experiments/lasso_path.py
"""

from __future__ import annotations

import numpy as np

from cvxcla import Lasso

N_SAMPLES = 60
N_FEATURES = 12
N_TRUE_NONZERO = 4
NOISE = 0.10
SEED = 0


def make_problem(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return a standardised design matrix ``X`` and centred response ``y``.

    Columns of ``X`` are standardised (zero mean, unit norm direction) and ``y`` is
    centred, the conventional LARS setup with no intercept, so cvxcla and
    scikit-learn solve the identical problem.
    """
    x = rng.standard_normal((N_SAMPLES, N_FEATURES))
    x = x - x.mean(axis=0, keepdims=True)
    x = x / np.linalg.norm(x, axis=0, keepdims=True)
    beta_true = np.zeros(N_FEATURES)
    beta_true[:N_TRUE_NONZERO] = rng.uniform(1.0, 3.0, N_TRUE_NONZERO) * rng.choice([-1.0, 1.0], N_TRUE_NONZERO)
    y = x @ beta_true + NOISE * rng.standard_normal(N_SAMPLES)
    y = y - y.mean()
    return x, y


def main() -> None:
    """Trace the LASSO path with cvxcla, compare to scikit-learn, and plot it."""
    rng = np.random.default_rng(SEED)
    x, y = make_problem(rng)

    lasso = Lasso(x=x, y=y)
    cvxcla_breaks = sorted(lasso.path, key=lambda bp: bp.lam, reverse=True)
    print(f"cvxcla LASSO path: {len(cvxcla_breaks)} breakpoints, lam_max={lasso.lam_max:.4f}")

    try:
        from sklearn.linear_model import lars_path
    except ImportError:
        print("scikit-learn not installed; skipping the cross-check and the figure.")
        print("  install it with:  uv run --with scikit-learn --with matplotlib python experiments/lasso_path.py")
        return

    # scikit-learn's reference LARS/LASSO path. Its alphas are in the averaged
    # (1/2m) loss convention, so lam = m * alpha maps them into cvxcla's penalty.
    alphas, _active, coefs = lars_path(x, y, method="lasso")
    lambdas = alphas * N_SAMPLES
    print(f"sklearn lars_path: {len(alphas)} breakpoints, lam_max={lambdas[0]:.4f}")

    # Compare coefficient vectors at every scikit-learn breakpoint, reading cvxcla's
    # piecewise-linear path at the matching lam. Agreement to solver precision
    # certifies that the same homotopy is being traced.
    max_abs = 0.0
    print("\n   lam      ||beta||_1   max|beta_cvxcla - beta_sklearn|")
    for i, lam in enumerate(lambdas):
        beta_sklearn = coefs[:, i]
        beta_cvxcla = lasso.solution(float(lam))
        diff = float(np.max(np.abs(beta_cvxcla - beta_sklearn)))
        max_abs = max(max_abs, diff)
        print(f"  {lam:8.4f}  {np.abs(beta_sklearn).sum():9.4f}   {diff:.2e}")

    print(f"\nmax coefficient discrepancy over the whole path: {max_abs:.2e}")
    print(f"paths agree to solver precision: {max_abs < 1e-8}")

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping docs/paper/lasso_path.pdf")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    cl = np.array([bp.lam for bp in cvxcla_breaks])
    cb = np.array([bp.beta for bp in cvxcla_breaks])
    for j in range(N_FEATURES):
        ax.plot(cl, cb[:, j], "-", color="#1f4e79", lw=1.2, alpha=0.9)
    # scikit-learn breakpoints as markers on top, to show coincidence.
    for j in range(N_FEATURES):
        ax.plot(lambdas, coefs[j, :], "o", ms=3, color="#c00000", alpha=0.7)
    ax.plot([], [], "-", color="#1f4e79", label="cvxcla Lasso (path)")
    ax.plot([], [], "o", ms=3, color="#c00000", label="scikit-learn lars_path")
    ax.set_xlabel(r"Penalty $\lambda$")
    ax.set_ylabel(r"Coefficients $\beta_j(\lambda)$")
    ax.set_title("LASSO regularisation path: cvxcla vs scikit-learn", fontsize=9)
    ax.invert_xaxis()  # path is traced from lam_max down to 0
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = "docs/paper/lasso_path.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

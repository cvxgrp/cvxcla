"""Validate the constrained LASSO path against a per-lambda general QP solver.

``cvxcla.Lasso`` traces the LASSO regularisation path through the same parametric
active-set engine as the Critical Line Algorithm, and -- like the CLA -- it now
admits general linear inequality constraints ``G beta <= h`` (with ``h > 0``, so the
path starts from the feasible ``beta = 0``). This script checks that claim the same
way ``validate_constraints.py`` checks the constrained frontier: it traces a
constrained path and re-solves every segment midpoint as a general QP.

For a deterministic standardised regression with a group-exposure cap ``G beta <= h``
we

  (a) trace the constrained path with ``cvxcla.Lasso``, and
  (b) at the midpoint ``lambda`` of every finite segment read the path coefficients
      and re-solve

          min  1/2 ||y - X beta||^2 + lambda ||beta||_1   s.t.  G beta <= h

      from scratch with CVXPY / Clarabel (tight tolerances),

and report the maximum coefficient discrepancy and the worst constraint violation.
Agreement to solver tolerance, with the cap active along the path, confirms the
constrained homotopy traces the exact constrained LASSO, not an approximation.

Usage:
    uv run --with cvxpy python experiments/validate_lasso_constraints.py
"""

from __future__ import annotations

from itertools import pairwise

import cvxpy as cp
import numpy as np

from cvxcla import Lasso

N_OBS = 60
N_FEATURES = 12
N_GROUPS = 3
SEED = 5


def make_regression(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, G, h): a standardised regression with binding group caps.

    ``G`` sums the coefficients within each of ``N_GROUPS`` contiguous groups, and
    ``h`` caps each group sum below its unconstrained least-squares level so the cap
    genuinely binds along the path. ``h > 0`` keeps ``beta = 0`` feasible.
    """
    x = rng.standard_normal((N_OBS, N_FEATURES))
    x = (x - x.mean(0)) / x.std(0)  # standardise columns
    beta_true = rng.standard_normal(N_FEATURES)
    y = x @ beta_true + 0.1 * rng.standard_normal(N_OBS)
    y = y - y.mean()

    group = np.arange(N_FEATURES) % N_GROUPS
    g = np.array([(group == j).astype(float) for j in range(N_GROUPS)])
    beta_ols = np.linalg.lstsq(x, y, rcond=None)[0]
    # cap each group sum at a positive fraction of its (absolute) OLS level
    h = np.maximum(np.abs(g @ beta_ols) * 0.4, 0.1)
    return x, y, g, h


def qp_solution(x: np.ndarray, y: np.ndarray, lam: float, g: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Solve the constrained LASSO at penalty ``lam`` with CVXPY / Clarabel."""
    n = x.shape[1]
    beta = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(y - x @ beta) + lam * cp.norm1(beta))
    cp.Problem(objective, [g @ beta <= h]).solve(
        solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12
    )
    return np.asarray(beta.value, dtype=float)


def main() -> None:
    """Trace the constrained LASSO and check every segment midpoint against a QP."""
    rng = np.random.default_rng(SEED)
    x, y, g, h = make_regression(rng)
    lasso = Lasso(x=x, y=y, g=g, h=h)

    points = sorted(lasso.path, key=lambda bp: bp.lam)
    errors = []
    violations = []
    active_mid = 0
    for lo, hi in pairwise(points):
        if hi.lam <= lasso.tol:
            continue
        lam = 0.5 * (lo.lam + hi.lam)
        beta_path = lasso.solution(lam)
        beta_qp = qp_solution(x, y, lam, g, h)
        errors.append(float(np.max(np.abs(beta_path - beta_qp))))
        violations.append(float(np.max(g @ beta_path - h)))
        if np.any(g @ beta_path - h > -1e-7):
            active_mid += 1

    err = np.array(errors)
    print(f"regression              : {N_OBS} x {N_FEATURES}, {N_GROUPS} group caps G beta <= h")
    print(f"breakpoints             : {len(lasso.path)}")
    print(f"segment midpoints tested: {err.size}")
    print(f"midpoints with cap active: {active_mid} / {err.size}")
    print(f"max |beta_path - beta_QP|: {err.max():.2e}")
    print(f"max constraint violation : {max(violations):.2e}")
    verdict = "EXACT (matches constrained QP to solver tolerance)" if err.max() < 1e-5 else "MISMATCH"
    print(f"verdict                 : {verdict}")


if __name__ == "__main__":
    main()

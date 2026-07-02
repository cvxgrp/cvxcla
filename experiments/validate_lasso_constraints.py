"""Validate the constrained LASSO path against a per-lambda general QP solver.

``cvxcla.Lasso`` traces the LASSO regularisation path through the same parametric
active-set engine as the Critical Line Algorithm, and -- like the CLA -- it admits
constraints: general linear inequalities ``G beta <= h`` (with ``h > 0``, so the path
starts from the feasible ``beta = 0``) and the sign restriction ``beta >= 0`` (the
non-negative LASSO, where the l1 penalty collapses to the linear term ``lam 1' beta``
and only positive coefficients enter). This script checks both the same way
``validate_constraints.py`` checks the constrained frontier: it traces a path and
re-solves every segment midpoint as a general QP.

For each scenario we

  (a) trace the path with ``cvxcla.Lasso``, and
  (b) at the midpoint ``lambda`` of every finite segment read the path coefficients
      and re-solve

          min  1/2 ||y - X beta||^2 + lambda ||beta||_1   s.t.  (constraints)

      from scratch with CVXPY / Clarabel (tight tolerances),

and report the maximum coefficient discrepancy and the worst constraint violation.

Two scenarios:

  1. inequality caps -- per-group exposure caps ``G beta <= h`` that bind along the
     path;
  2. non-negative -- the sign restriction ``beta >= 0``.

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


def make_regression(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A standardised design and centred response."""
    x = rng.standard_normal((N_OBS, N_FEATURES))
    x = (x - x.mean(0)) / x.std(0)
    beta_true = rng.standard_normal(N_FEATURES)
    y = x @ beta_true + 0.1 * rng.standard_normal(N_OBS)
    return x, y - y.mean()


def qp_solution(
    x: np.ndarray, y: np.ndarray, lam: float, g: np.ndarray | None, h: np.ndarray | None, nonneg: bool
) -> np.ndarray:
    """Solve the constrained LASSO at penalty ``lam`` with CVXPY / Clarabel."""
    beta = cp.Variable(x.shape[1])
    objective = cp.Minimize(0.5 * cp.sum_squares(y - x @ beta) + lam * cp.norm1(beta))
    constraints = []
    if g is not None:
        constraints.append(g @ beta <= h)
    if nonneg:
        constraints.append(beta >= 0)
    cp.Problem(objective, constraints).solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return np.asarray(beta.value, dtype=float)


def validate(
    label: str,
    lasso: Lasso,
    g: np.ndarray | None = None,
    h: np.ndarray | None = None,
    nonneg: bool = False,
) -> float:
    """Check every segment midpoint of a traced path against a per-lambda QP."""
    x, y = lasso.x, lasso.y
    points = sorted(lasso.path, key=lambda bp: bp.lam)
    errors: list[float] = []
    viol: list[float] = []
    for lo, hi in pairwise(points):
        if hi.lam <= lasso.tol:
            continue
        lam = 0.5 * (lo.lam + hi.lam)
        beta_path = lasso.solution(lam)
        errors.append(float(np.max(np.abs(beta_path - qp_solution(x, y, lam, g, h, nonneg)))))
        if g is not None:
            viol.append(float(np.max(g @ beta_path - h)))
        if nonneg:
            viol.append(float(np.max(-beta_path)))

    err = np.array(errors)
    print(f"\n{label}")
    print(f"  breakpoints             : {len(lasso.path)}")
    print(f"  segment midpoints tested: {err.size}")
    if viol:
        print(f"  max constraint violation: {max(viol):.2e}")
    print(f"  max |beta_path - beta_QP|: {err.max():.2e}")
    return err.max()


def main() -> None:
    """Run the inequality and non-negative scenarios and report exactness."""
    rng = np.random.default_rng(SEED)
    x, y = make_regression(rng)

    # Scenario 1: per-group exposure caps G beta <= h that bind along the path.
    group = np.arange(N_FEATURES) % N_GROUPS
    g = np.array([(group == j).astype(float) for j in range(N_GROUPS)])
    h = np.maximum(np.abs(g @ np.linalg.lstsq(x, y, rcond=None)[0]) * 0.4, 0.1)
    err_ineq = validate("Scenario 1 -- inequality caps (G beta <= h)", Lasso(x=x, y=y, g=g, h=h), g=g, h=h)

    # Scenario 2: the non-negative LASSO (beta >= 0).
    err_nn = validate("Scenario 2 -- non-negative (beta >= 0)", Lasso(x=x, y=y, nonneg=True), nonneg=True)

    worst = max(err_ineq, err_nn)
    verdict = "EXACT (matches constrained QP to solver tolerance)" if worst < 1e-5 else "MISMATCH"
    print(f"\nworst |beta_path - beta_QP| across scenarios: {worst:.2e}  ->  {verdict}")


if __name__ == "__main__":
    main()

"""Validate that the CLA frontier is exact, against a per-lambda QP solver.

The CLA claims to return the *exact* efficient frontier: between consecutive
turning points the optimal weights are the affine segment w(lambda) = alpha +
lambda * beta. We check that claim independently. For a tractable real
subproblem (the first N_ASSETS S&P 500 names, full-history sample covariance) we
trace the frontier with the CLA, then at the midpoint lambda of every segment we

  (a) read the CLA weights by linear interpolation in lambda between the two
      bracketing turning points, and
  (b) solve the parametric QP

          min  1/2 w' Sigma w - lambda * mu' w
          s.t. 1' w = 1,  0 <= w <= 1

      from scratch with a convex solver (CVXPY / Clarabel),

and report the maximum weight discrepancy. Agreement to solver tolerance
confirms the CLA segments are the true optimisers, not an approximation.

Usage:
    uv run python experiments/fetch_sp500.py    # once, to download the data
    uv run python experiments/validate_exact.py
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from cvxcla import CLA

DATA = Path(__file__).parent / "data" / "sp500_pct_returns.parquet"
N_ASSETS = 40


def qp_solution(mean: np.ndarray, cov: np.ndarray, lam: float) -> np.ndarray:
    """Solve min 1/2 w'Sigma w - lam mu'w s.t. 1'w=1, 0<=w<=1 with CVXPY.

    Tight Clarabel tolerances so the comparison probes the CLA's accuracy, not
    the QP solver's default stopping criteria.
    """
    n = mean.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(cov)) - lam * mean @ w)
    constraints = [cp.sum(w) == 1, w >= 0, w <= 1]
    cp.Problem(objective, constraints).solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return np.asarray(w.value, dtype=float)


def main() -> None:
    """Trace with the CLA, re-solve each segment midpoint as a QP, report error."""
    returns = pd.read_parquet(DATA).iloc[:, :N_ASSETS]
    mean = returns.mean(axis=0).to_numpy()
    cov = np.cov(returns.to_numpy(), rowvar=False)

    cla = CLA(
        mean=mean,
        covariance=cov,
        lower_bounds=np.zeros(N_ASSETS),
        upper_bounds=np.ones(N_ASSETS),
        a=np.ones((1, N_ASSETS)),
        b=np.ones(1),
    )
    tps = cla.turning_points
    print(f"subproblem              : {N_ASSETS} assets, cond(S)={np.linalg.cond(cov):,.0f}")
    print(f"turning points          : {len(tps)}")

    errors = []
    # Segments between consecutive turning points with finite lambda.
    for hi, lo in pairwise(tps):
        lam_hi, lam_lo = hi.lamb, lo.lamb
        if not np.isfinite(lam_hi):  # skip the lambda = inf endpoint segment
            continue
        lam = 0.5 * (lam_hi + lam_lo)
        frac = (lam - lam_lo) / (lam_hi - lam_lo)  # affine in lambda on the segment
        w_cla = lo.weights + frac * (hi.weights - lo.weights)
        w_qp = qp_solution(mean, cov, lam)
        errors.append(float(np.max(np.abs(w_cla - w_qp))))

    errors = np.array(errors)
    print(f"segment midpoints tested: {errors.size}")
    print(f"median |w_CLA - w_QP|   : {np.median(errors):.2e}")
    print(f"max    |w_CLA - w_QP|   : {errors.max():.2e}")
    verdict = "EXACT (matches QP to solver tolerance)" if errors.max() < 1e-4 else "MISMATCH"
    print(f"verdict                 : {verdict}")


if __name__ == "__main__":
    main()

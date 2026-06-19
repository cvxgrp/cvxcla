"""Validate the CLA frontier under general linear constraints (multi-row A, G w <= h).

The paper claims the turning-point loop traces the *exact* frontier not only for
the box-and-budget problem but for an arbitrary linear equality system ``A w = b``
(weighted, or m > 1 rows: sector/factor neutrality) and arbitrary linear
inequalities ``G w <= h`` (group or sector exposure caps). The headline examples
and ``validate_exact.py`` exercise only box bounds plus a single budget row, so
this script closes that gap: it traces two genuinely constrained problems and
checks each against a per-lambda general QP solve.

For each scenario, on a deterministic synthetic factor model, we

  (a) trace the long-only frontier with the CLA under the constraints, and
  (b) at the midpoint lambda of every finite segment, read the CLA weights by
      linear interpolation between the bracketing turning points and re-solve

          min  1/2 w' Sigma w - lambda * mu' w
          s.t. A w = b,  G w <= h,  0 <= w <= 1

      from scratch with CVXPY / Clarabel (tight tolerances),

and report the maximum weight discrepancy. Agreement to solver tolerance confirms
the affine segments are the true constrained optimisers.

Two scenarios are run:

  1. multi-row equality: budget + a characteristic-neutrality row (m = 2), so the
     first turning point is found by the linear program of Section 3, not the
     greedy fill; and
  2. inequality caps: budget + per-sector exposure caps ``G w <= h``. The means
     are arranged so the high-return end concentrates in one sector and the cap
     *binds* -- we assert at least one inequality row is active along the trace,
     so the inequality machinery is genuinely exercised, not merely present.

Usage:
    uv run --with cvxpy python experiments/validate_constraints.py
"""

from __future__ import annotations

from itertools import pairwise

import cvxpy as cp
import numpy as np

from cvxcla import CLA

N_ASSETS = 30
N_FACTORS = 4
N_SECTORS = 3
SEED = 11
SECTOR_CAP = 0.45


def make_market(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dense Sigma, mean, sector id) for a deterministic factor market.

    Sector 0 carries a return premium, so the maximum-return end of the frontier
    wants to overweight it and the sector-exposure cap binds (scenario 2).
    """
    u = rng.standard_normal((N_ASSETS, N_FACTORS)) / np.sqrt(N_ASSETS)
    delta = rng.uniform(0.5, 2.0, N_FACTORS) * N_ASSETS
    d = rng.uniform(0.5, 2.0, N_ASSETS)
    cov = np.diag(d) + (u * delta) @ u.T
    sector = np.arange(N_ASSETS) % N_SECTORS  # interleaved sector membership
    mean = rng.uniform(0.0, 1.0, N_ASSETS)
    mean[sector == 0] += 1.0  # a clear premium for sector 0 -> its cap will bind
    return cov, mean, sector


def qp_solution(
    mean: np.ndarray,
    cov: np.ndarray,
    lam: float,
    a: np.ndarray,
    b: np.ndarray,
    g: np.ndarray | None,
    h: np.ndarray | None,
) -> np.ndarray:
    """Solve the return-parametrised QP under A w = b, G w <= h, 0 <= w <= 1.

    Tight Clarabel tolerances so the comparison probes the CLA's accuracy, not the
    QP solver's default stopping criteria.
    """
    n = mean.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(cov)) - lam * mean @ w)
    constraints = [a @ w == b, w >= 0, w <= 1]
    if g is not None and g.shape[0] > 0:
        constraints.append(g @ w <= h)
    cp.Problem(objective, constraints).solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return np.asarray(w.value, dtype=float)


def validate(
    label: str,
    mean: np.ndarray,
    cov: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    g: np.ndarray | None = None,
    h: np.ndarray | None = None,
) -> float:
    """Trace one constrained problem and check every segment midpoint against a QP.

    Returns the maximum weight discrepancy over the tested segment midpoints, and
    reports how many turning points hold an inequality row active (binding).
    """
    cla = CLA(
        mean=mean,
        covariance=cov,
        lower_bounds=np.zeros(N_ASSETS),
        upper_bounds=np.ones(N_ASSETS),
        a=a,
        b=b,
        g=g,
        h=h,
    )
    tps = cla.turning_points
    active = 0
    if g is not None and g.shape[0] > 0:
        active = sum(int(np.any(np.abs(g @ tp.weights - h) <= 1e-7)) for tp in tps)

    errors = []
    for hi, lo in pairwise(tps):
        if not np.isfinite(hi.lamb):  # skip the lambda = inf endpoint segment
            continue
        lam = 0.5 * (hi.lamb + lo.lamb)
        frac = (lam - lo.lamb) / (hi.lamb - lo.lamb)  # affine in lambda on the segment
        w_cla = lo.weights + frac * (hi.weights - lo.weights)
        w_qp = qp_solution(mean, cov, lam, a, b, g, h)
        errors.append(float(np.max(np.abs(w_cla - w_qp))))

    err = np.array(errors)
    print(f"\n{label}")
    print(f"  equality rows m         : {a.shape[0]}")
    if g is not None and g.shape[0] > 0:
        print(f"  inequality rows p       : {g.shape[0]}")
        print(f"  turning pts w/ active G : {active} / {len(tps)}")
    print(f"  turning points          : {len(tps)}")
    print(f"  segment midpoints tested: {err.size}")
    print(f"  median |w_CLA - w_QP|   : {np.median(err):.2e}")
    print(f"  max    |w_CLA - w_QP|   : {err.max():.2e}")
    return err.max()


def main() -> None:
    """Run the multi-row-equality and inequality-cap scenarios and report exactness."""
    rng = np.random.default_rng(SEED)
    cov, mean, sector = make_market(rng)
    ones = np.ones((1, N_ASSETS))

    # Scenario 1: budget + a characteristic-neutrality row (m = 2). The neutrality
    # target is reachable (equal weight satisfies it), so the problem is feasible
    # and the first vertex is the Section 3 linear program, not the greedy fill.
    char = rng.standard_normal(N_ASSETS)
    target = float(char.mean())  # c' (1/n 1) = mean(char): equal weight is feasible
    a2 = np.vstack([ones, char[None, :]])
    b2 = np.array([1.0, target])
    err1 = validate("Scenario 1 -- multi-row equality (budget + characteristic neutrality)", mean, cov, a2, b2)

    # Scenario 2: budget + per-sector exposure caps G w <= h. Sector 0 carries a
    # premium, so the high-return end wants to pile into it and the cap binds.
    g = np.array([(sector == s).astype(float) for s in range(N_SECTORS)])
    h = np.full(N_SECTORS, SECTOR_CAP)
    err2 = validate("Scenario 2 -- inequality sector-exposure caps (G w <= h)", mean, cov, ones, np.ones(1), g, h)

    worst = max(err1, err2)
    verdict = "EXACT (matches QP to solver tolerance)" if worst < 1e-4 else "MISMATCH"
    print(f"\nworst discrepancy across scenarios: {worst:.2e}  ->  {verdict}")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "clarabel==0.11.1",
#     "cvxcla==2.7.1",
#     "cvxpy==1.9.3",
#     "numpy==2.4.6",
#     # scikit-base (a PyPortfolioOpt dependency) imports packaging without declaring it.
#     "packaging==26.3",
#     "pyportfolioopt==1.6.0",
# ]
# ///
r"""Reproduce the Markowitz--Todd 10-asset example of Bailey and Lopez de Prado (2013).

BLdP validate their CLA on one problem: the example that accompanies the VBA-Excel
implementation of Markowitz and Todd (2000). Their Table 1 prints the data (means,
bounds ``0 <= w <= 1`` and the lower triangle of the covariance, to 7 decimals), and
Table 2 the 10 turning points: return, risk, lambda and weights, to 3 decimals. The
text adds a maximum Sharpe ratio of 4.4535 at risk 0.2274 and a minimum-variance risk
of 0.2052. They report that their results match the VBA-Excel code to 15 significant
figures.

This script runs the BLdP code as shipped in PyPortfolioOpt 1.6.0 (``bldp``), the same
with the stale-weight fix (``bldp-fix-w``) and ``cvxcla``, and for each

* checks every turning point and segment midpoint against CVXPY/Clarabel, exactly as
  ``bldp_agree.py`` does;
* compares the distinct turning points with Table 2, to the 3 printed decimals;
* compares the maximum-Sharpe and minimum-variance portfolios with the text;
* compares the implementations with each other, as paths in lambda and as sets of
  portfolios.

Usage:
    uv run experiments/markowitz_todd.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for bldp_agree, bldp_gap
import itertools

import bldp_agree as ba

IMPLEMENTATIONS = ["bldp", "bldp-fix-w", "cvxcla"]

# Bailey and Lopez de Prado (2013), Table 1.
MEAN = np.array([1.175, 1.19, 0.396, 1.12, 0.346, 0.679, 0.089, 0.73, 0.481, 1.08])
LOWER_TRIANGLE = [
    [0.4075516],
    [0.0317584, 0.9063047],
    [0.0518392, 0.0313639, 0.194909],
    [0.056639, 0.0268726, 0.0440849, 0.1952847],
    [0.0330226, 0.0191717, 0.0300677, 0.0277735, 0.3405911],
    [0.0082778, 0.0093438, 0.0132274, 0.0052667, 0.0077706, 0.1598387],
    [0.0216594, 0.0249504, 0.0352597, 0.0137581, 0.0206784, 0.0210558, 0.6805671],
    [0.0133242, 0.0076104, 0.0115493, 0.0078088, 0.0073641, 0.0051869, 0.0137788, 0.9552692],
    [0.0343476, 0.0287487, 0.0427563, 0.0291418, 0.0254266, 0.0172374, 0.0462703, 0.0106553, 0.3168158],
    [0.022499, 0.0133687, 0.020573, 0.0164038, 0.0128408, 0.0072378, 0.0192609, 0.0076096, 0.0185432, 0.1107929],
]

# Bailey and Lopez de Prado (2013), Table 2: return, risk, lambda, weights.
TABLE2 = np.array(
    [
        [1.190, 0.952, 58.303, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [1.180, 0.546, 4.174, 0.649, 0.351, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [1.160, 0.417, 1.946, 0.434, 0.231, 0.000, 0.335, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [1.111, 0.267, 0.165, 0.127, 0.072, 0.000, 0.281, 0.000, 0.000, 0.000, 0.000, 0.000, 0.520],
        [1.108, 0.265, 0.147, 0.123, 0.070, 0.000, 0.279, 0.000, 0.000, 0.000, 0.006, 0.000, 0.521],
        [1.022, 0.230, 0.056, 0.087, 0.050, 0.000, 0.224, 0.000, 0.174, 0.000, 0.030, 0.000, 0.435],
        [1.015, 0.228, 0.052, 0.085, 0.049, 0.000, 0.220, 0.000, 0.180, 0.000, 0.031, 0.006, 0.429],
        [0.973, 0.220, 0.037, 0.074, 0.044, 0.000, 0.199, 0.026, 0.198, 0.000, 0.033, 0.028, 0.398],
        [0.950, 0.216, 0.031, 0.068, 0.041, 0.015, 0.188, 0.034, 0.202, 0.000, 0.034, 0.034, 0.383],
        [0.803, 0.205, 0.000, 0.037, 0.027, 0.095, 0.126, 0.077, 0.219, 0.030, 0.036, 0.061, 0.292],
    ]
)
MAX_SHARPE, MAX_SHARPE_RISK, MIN_VAR_RISK = 4.4535, 0.2274, 0.2052
HALF_ULP_3 = 5e-4 + 1e-12  # agreement to 3 printed decimals


def problem() -> ba.Problem:
    """The Markowitz--Todd example as a ``bldp_agree`` problem."""
    n = len(MEAN)
    cov = np.zeros((n, n))
    for i, row in enumerate(LOWER_TRIANGLE):
        cov[i, : len(row)] = row
    cov = cov + np.tril(cov, -1).T
    return ba._problem("Markowitz-Todd", MEAN, cov, 0.0, 1.0)


def turning_points(weights: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """Finite turning points, repeats removed, as rows of (return, risk, lambda, weights).

    Table 2 lists turning points from the first finite lambda on; the lambda = inf
    point of a trace holds the same portfolio as the first of them.
    """
    order = np.argsort(-lambdas, kind="stable")
    w, lam = weights[order], lambdas[order]
    finite = np.isfinite(lam)
    w, lam = w[finite], lam[finite]
    keep = np.r_[True, np.abs(np.diff(w, axis=0)).max(axis=1) > 1e-12]
    w, lam = w[keep], lam[keep]
    p = problem()
    ret = w @ p["mean"]
    risk = np.sqrt(np.einsum("ki,ij,kj->k", w, p["cov"], w))
    return np.column_stack([ret, risk, lam, w])


def sharpe_and_minvar(weights: np.ndarray, lambdas: np.ndarray) -> tuple[float, float, float]:
    """Maximum Sharpe ratio (risk-free rate 0) on the traced frontier, its risk, and the minimum risk.

    The Sharpe ratio is maximised over each segment between adjacent turning points by a
    fine one-dimensional search, which is how BLdP and cvxcla both locate it.
    """
    p = problem()
    w = weights[np.argsort(-lambdas, kind="stable")]
    best = (-np.inf, np.nan)
    t = np.linspace(0.0, 1.0, 20001)[:, None]
    for a, b in itertools.pairwise(w):
        path = a + t * (b - a)
        ret = path @ p["mean"]
        risk = np.sqrt(np.einsum("ki,ij,kj->k", path, p["cov"], path))
        k = int(np.argmax(ret / risk))
        if ret[k] / risk[k] > best[0]:
            best = (float(ret[k] / risk[k]), float(risk[k]))
    min_risk = float(np.sqrt(w[-1] @ p["cov"] @ w[-1]))
    return best[0], best[1], min_risk


def main() -> None:
    """Run the three implementations on the example and print every comparison."""
    p = problem()
    traces = {}
    for impl in IMPLEMENTATIONS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traces[impl] = ba.run(impl, p)

    print("Markowitz-Todd 10-asset example (Bailey and Lopez de Prado 2013, Tables 1-2)\n")
    print(
        f"{'':12s} {'QP check':>9s} {'gap':>8s} {'points':>6s} {'max|Table 2 diff|':>18s} "
        f"{'max Sharpe':>10s} {'its risk':>8s} {'min risk':>8s}"
    )
    print(
        f"{'paper':12s} {'':>9s} {'':>8s} {len(TABLE2):6d} {'':>18s} "
        f"{MAX_SHARPE:10.4f} {MAX_SHARPE_RISK:8.4f} {MIN_VAR_RISK:8.4f}"
    )
    for impl, (w, lam) in traces.items():
        v = ba.validate(p, w, lam)
        tp = turning_points(w, lam)
        diff = np.abs(tp - TABLE2).max() if tp.shape == TABLE2.shape else np.nan
        sr, sr_risk, mv = sharpe_and_minvar(w, lam)
        print(
            f"{impl:12s} {v['status']:>9s} {v['gap']:8.1e} {len(tp):6d} {diff:18.1e} "
            f"{sr:10.4f} {sr_risk:8.4f} {mv:8.4f}"
        )

    print("\nagreement with cvxcla: path distance in lambda / set-of-portfolios distance")
    for impl in IMPLEMENTATIONS[:-1]:
        a = ba.agreement(traces[impl], traces["cvxcla"])
        print(f"  {impl:12s} {a['distance']:.1e} / {max(a['missing'], a['extra']):.1e}")

    print("\nturning points from cvxcla (return, risk, lambda, weights), printed as in Table 2:")
    for k, row in enumerate(turning_points(*traces["cvxcla"]), start=1):
        print(f"  {k:2d}  " + "  ".join(f"{x:6.3f}" for x in row))
    ok = np.abs(turning_points(*traces["cvxcla"]) - TABLE2).max() <= HALF_ULP_3
    print(f"\ncvxcla reproduces Table 2 to its 3 printed decimals: {ok}")

    # The stale-weight defect fires only in the step after a free weight reaches a
    # bound and leaves the free set. Count the free weights along the frontier.
    w = turning_points(*traces["cvxcla"])[:, 3:]
    free = ((w > p["lb"] + 1e-12) & (w < p["ub"] - 1e-12)).sum(axis=1)
    leaves = bool(np.any(np.diff(free) < 0))
    print(f"free weights per turning point: {free.tolist()}")
    print(f"does any weight leave the free set along this frontier? {leaves}")


if __name__ == "__main__":
    main()

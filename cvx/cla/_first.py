from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cvxpy as cp

from cvx.cla.types import MATRIX, BOOLEAN_VECTOR


@dataclass(frozen=True)
class Next:
    free: BOOLEAN_VECTOR
    weights: MATRIX
    lamb: float = np.inf
    mean: float = -np.inf

def init_algo(mean: MATRIX, lower_bounds: MATRIX, upper_bounds: MATRIX) -> Next:
    """The key insight behind Markowitz’s CLA is to find first the
    turning point associated with the highest expected return, and then
    compute the sequence of turning points, each with a lower expected
    return than the previous.That first turning point consists in the
    smallest subset of assets with highest return such that the sum of
    their upper boundaries equals or exceeds one.

    We sort the expected returns in descending order.
    This gives us a sequence for searching for the
    first free asset. All weights are initially set to their lower bounds,
    and following the sequence from the previous step, we move those
    weights from the lower to the upper bound until the sum of weights
    exceeds one. If possible the last iterated weight is then reduced
    to comply with the constraint that the sum of weights equals one.
    This last weight is the first free asset,
    and the resulting vector of weights the first turning point.
    """

    if np.any(lower_bounds > upper_bounds):
        raise ValueError("Lower bounds must be less than or equal to upper bounds")

    # Initialize weights to lower bounds
    weights = np.copy(lower_bounds)
    free = np.full_like(mean, False, dtype=np.bool_)

    # Move weights from lower to upper bound
    # until sum of weights hits or exceeds 1
    for index in np.argsort(mean)[::-1]:
        weights[index] = upper_bounds[index]
        if np.sum(weights) >= 1:
            weights[index] -= np.sum(weights) - 1
            free[index] = True
            break

    if not np.any(free):
        # We have not reached the sum of weights of 1...
        raise ValueError("Could not construct a fully invested portfolio")

    # Return first turning point, the point with the highest expected return.
    return Next(free=free, weights=weights, mean=float(mean.T @ weights))


def init_algo_lp(
    mean: MATRIX,
    lower_bounds: MATRIX,
    upper_bounds: MATRIX,
    A_eq: MATRIX | None = None,
    b_eq: MATRIX | None = None,
    A_ub: MATRIX | None = None,
    b_ub: MATRIX | None = None,
) -> Next:
    if A_eq is None:
        A_eq = np.atleast_2d(np.ones_like(mean))

    if b_eq is None:
        b_eq = np.array([1.0])

    if A_ub is None:
        A_ub = np.atleast_2d(np.zeros_like(mean))

    if b_ub is None:
        b_ub = np.array([0.0])

    w = cp.Variable(mean.shape[0], "weights")

    objective = cp.Maximize(mean.T @ w)
    constraints = [
        A_eq @ w == b_eq,
        A_ub @ w <= b_ub,
        lower_bounds <= w,
        w <= upper_bounds,
    ]

    cp.Problem(objective, constraints).solve()

    w = w.value

    # compute the distance from the closest bound
    distance = np.min(
        np.array([np.abs(w - lower_bounds), np.abs(upper_bounds - w)]), axis=0
    )

    # which element has the largest distance to any bound?
    # Even if all assets are at their bounds,
    # we get a (somewhat random) free asset.
    index = np.argmax(distance)

    free = np.full_like(mean, False, dtype=np.bool_)
    free[index] = True

    return Next(free=free, weights=w, mean=mean.T @ w)

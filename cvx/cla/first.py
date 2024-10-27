#    Copyright 2023 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from __future__ import annotations

import cvxpy as cp
import numpy as np

from .types import MATRIX, TurningPoint


#
def init_algo(mean: MATRIX, lower_bounds: MATRIX, upper_bounds: MATRIX) -> TurningPoint:
    """The key insight behind Markowitz’s CLA is to find first the
    turning point associated with the highest expected return, and then
    compute the sequence of turning points, each with a lower expected
    return than the previous. That first turning point consists in the
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
    for index in np.argsort(-mean):
        weights[index] += np.min(
            [upper_bounds[index] - lower_bounds[index], 1.0 - np.sum(weights)]
        )
        if np.sum(weights) >= 1:
            free[index] = True
            break

    # free = _free(weights, lower_bounds, upper_bounds)

    if not np.any(free):
        #    # We have not reached the sum of weights of 1...
        raise ValueError("Could not construct a fully invested portfolio")

    # Return first turning point, the point with the highest expected return.
    return TurningPoint(free=free, weights=weights)


def init_algo_lp(
    mean: MATRIX,
    lower_bounds: MATRIX,
    upper_bounds: MATRIX,
    A_eq: MATRIX | None = None,
    b_eq: MATRIX | None = None,
    solver=cp.CLARABEL,
    **kwargs
    # A_ub: MATRIX | None = None,
    # b_ub: MATRIX | None = None,
) -> TurningPoint:
    if A_eq is None:
        A_eq = np.atleast_2d(np.ones_like(mean))

    if b_eq is None:
        b_eq = np.array([1.0])

    # if A_ub is None:
    #    A_ub = np.atleast_2d(np.zeros_like(mean))

    # if b_ub is None:
    #    b_ub = np.array([0.0])

    w = cp.Variable(mean.shape[0], "weights")

    objective = cp.Maximize(mean.T @ w)
    constraints = [
        A_eq @ w == b_eq,
        # A_ub @ w <= b_ub,
        lower_bounds <= w,
        w <= upper_bounds,
        cp.sum(w) == 1.0,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, **kwargs)
    # check status of problem is optimal
    if problem.status != cp.OPTIMAL:
        raise ValueError("Could not construct a fully invested portfolio")

    # assert problem.status == cp.OPTIMAL
    # print(problem.status)
    # print(status)

    w = w.value

    # compute the distance from the closest bound
    # distance = np.min(
    #    np.array([np.abs(w - lower_bounds), np.abs(upper_bounds - w)]), axis=0
    # )

    # which element has the largest distance to any bound?
    # Even if all assets are at their bounds,
    # we get a (somewhat random) free asset.
    # index = np.argmax(distance)

    # free = np.full_like(mean, False, dtype=np.bool_)
    # free[index] = True

    free = _free(w, lower_bounds, upper_bounds)

    return TurningPoint(free=free, weights=w)


def _free(w, lower_bounds, upper_bounds):
    distance = np.min(
        np.array([np.abs(w - lower_bounds), np.abs(upper_bounds - w)]), axis=0
    )

    index = np.argmax(distance)

    free = np.full_like(w, False, dtype=np.bool_)
    free[index] = True
    return free

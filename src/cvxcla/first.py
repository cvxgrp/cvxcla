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
"""First turning point computation for the Critical Line Algorithm.

This module provides functions to compute the first turning point on the efficient frontier,
which is the portfolio with the highest expected return that satisfies the constraints.
Two implementations are provided: a direct algorithm and a linear programming approach.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import TurningPoint


#
def init_algo(
    mean: NDArray[np.float64], lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
) -> TurningPoint:
    """Compute the first turning point.

    The key insight behind Markowitz’s CLA is to find first the
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
        weights[index] += np.min([upper_bounds[index] - lower_bounds[index], 1.0 - np.sum(weights)])
        if np.sum(weights) >= 1:
            free[index] = True
            break

    # free = _free(weights, lower_bounds, upper_bounds)

    if not np.any(free):
        #    # We have not reached the sum of weights of 1...
        raise ValueError("Could not construct a fully invested portfolio")

    # Return first turning point, the point with the highest expected return.
    return TurningPoint(free=free, weights=weights)


def _free(
    w: NDArray[np.float64], lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Determine which asset should be free in the turning point.

    This helper function identifies the asset that should be marked as free
    in the turning point. It selects the asset that is furthest from its bounds,
    which helps ensure numerical stability in the algorithm.

    Args:
        w: Vector of portfolio weights.
        lower_bounds: Vector of lower bounds for asset weights.
        upper_bounds: Vector of upper bounds for asset weights.

    Returns:
        A boolean vector indicating which asset is free (True) and which are blocked (False).

    """
    # Calculate the distance from each weight to its nearest bound
    distance = np.min(np.array([np.abs(w - lower_bounds), np.abs(upper_bounds - w)]), axis=0)

    # Find the index of the asset furthest from its bounds
    index = np.argmax(distance)

    # Create a boolean vector with only that asset marked as free
    free = np.full_like(w, False, dtype=np.bool_)
    free[index] = True
    return free

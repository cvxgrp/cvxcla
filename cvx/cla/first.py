from __future__ import annotations

import numpy as np

from cvx.cla.types import MATRIX, Next


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

    # we don't care about the means only about their order.
    # In descending order they are
    order = np.argsort(mean)[::-1]

    # 2) Initialize weights to lower bounds
    weights = np.copy(lower_bounds)
    free = np.full_like(mean, False, dtype=np.bool_)

    # 3) Move weights from lower to upper bound until sum of weights exceeds 1
    for index, bound in zip(order, upper_bounds[order]):
        weights[index] = bound
        if np.sum(weights) >= 1:
            weights[index] -= np.sum(weights) - 1
            free[index] = True
            break

    # 4) If we get here, all weights are at their upper bound
    return Next(free=free, weights=weights, lamb=None)

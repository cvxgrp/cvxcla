"""First turning point computation for the Critical Line Algorithm.

This module provides functions to compute the first turning point on the efficient frontier,
which is the portfolio with the highest expected return that satisfies the constraints.
Two implementations are provided: a direct algorithm and a linear programming approach.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog  # type: ignore[import-untyped]

from .types import TurningPoint


#
def init_algo(
    mean: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    total: float = 1.0,
) -> TurningPoint:
    """Compute the first turning point for a single all-ones budget constraint.

    The key insight behind Markowitz's CLA is to find first the
    turning point associated with the highest expected return, and then
    compute the sequence of turning points, each with a lower expected
    return than the previous. That first turning point consists in the
    smallest subset of assets with highest return such that the sum of
    their upper boundaries equals or exceeds the budget ``total``.

    We sort the expected returns in descending order.
    This gives us a sequence for searching for the
    first free asset. All weights are initially set to their lower bounds,
    and following the sequence from the previous step, we move those
    weights from the lower to the upper bound until the sum of weights
    reaches ``total``. The last iterated weight is then reduced
    to comply with the constraint that the sum of weights equals ``total``.
    This last weight is the first free asset,
    and the resulting vector of weights the first turning point.

    Args:
        mean: Vector of expected returns.
        lower_bounds: Lower box bounds.
        upper_bounds: Upper box bounds.
        total: Target sum of weights (the right-hand side ``b`` of the all-ones
            budget constraint ``sum(w) = total``; ``1`` for fully-invested,
            ``0`` for dollar-neutral, ``> 1`` for a leveraged total).
    """
    if np.any(lower_bounds > upper_bounds):
        msg = "Lower bounds must be less than or equal to upper bounds"
        raise ValueError(msg)

    # Initialize weights to lower bounds
    weights = np.copy(lower_bounds).astype(np.float64)
    free = np.full_like(mean, False, dtype=np.bool_)

    # Move weights from lower to upper bound until the sum reaches ``total``. The
    # check needs a tolerance: the increment ``total - sum(weights)`` can bring the
    # sum to ``total`` only up to floating-point error, and without the slack the
    # loop would move on and mark the NEXT asset (sitting on its bound) as free
    # while the genuinely interior asset stays blocked.
    for index in np.argsort(-mean):
        weights[index] += np.min([upper_bounds[index] - lower_bounds[index], total - np.sum(weights)])
        if np.sum(weights) >= total - 1e-12:
            free[index] = True
            break

    if not np.any(free):
        #    # We have not reached the target sum of weights...
        msg = "Could not construct a fully invested portfolio"
        raise ValueError(msg)

    # Return first turning point, the point with the highest expected return.
    return TurningPoint(free=free, weights=weights)


def first_vertex_lp(
    mean: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    tol: float,
) -> TurningPoint:
    """Compute the first turning point for a general equality system ``A w = b``.

    The maximum-return vertex of the feasible polytope
    ``{w : A w = b, lower <= w <= upper}`` is a linear program,
    ``maximize mean @ w``. The greedy fill of :func:`init_algo` only solves the
    single all-ones budget; for a general (weighted, or multi-row) ``A`` we solve
    the LP directly with HiGHS (via :func:`scipy.optimize.linprog`), which returns
    a vertex, and read the free set off the solution (the assets strictly inside
    their box bounds).

    Args:
        mean: Vector of expected returns.
        lower_bounds: Lower box bounds.
        upper_bounds: Upper box bounds.
        a: Equality-constraint matrix (``m x n``).
        b: Equality-constraint right-hand side (length ``m``).
        tol: Tolerance for classifying an asset as free (strictly interior).

    Returns:
        The maximum-return vertex as a :class:`TurningPoint`.

    Raises:
        ValueError: If the linear program is infeasible or unbounded (the
            constraints admit no maximum-return vertex).
    """
    # maximize mean @ w  ==  minimize -mean @ w, subject to A w = b and the box.
    result = linprog(
        c=-np.asarray(mean, dtype=np.float64),
        A_eq=np.asarray(a, dtype=np.float64),
        b_eq=np.asarray(b, dtype=np.float64),
        bounds=list(zip(lower_bounds, upper_bounds, strict=True)),
        method="highs",
    )
    if not result.success:
        msg = f"Could not find a maximum-return vertex (linear program: {result.message})"
        raise ValueError(msg)

    weights = np.asarray(result.x, dtype=np.float64)
    free = (weights > lower_bounds + tol) & (weights < upper_bounds - tol)
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

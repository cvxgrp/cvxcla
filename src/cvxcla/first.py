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
        # No asset ended up interior: the bounds cannot sum to the target.
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
    g: NDArray[np.float64] | None = None,
    h: NDArray[np.float64] | None = None,
) -> TurningPoint:
    """Compute the first turning point for a general ``A w = b``, ``G w <= h`` system.

    The maximum-return vertex of the feasible polytope
    ``{w : A w = b, G w <= h, lower <= w <= upper}`` is a linear program,
    ``maximize mean @ w``. The greedy fill of :func:`init_algo` only solves the
    single all-ones budget with no inequality rows; for a general (weighted, or
    multi-row) ``A`` or any ``G`` we solve the LP directly with HiGHS (via
    :func:`scipy.optimize.linprog`), which returns a vertex. The free set is read
    off the solution (assets strictly inside their box bounds) and the initial
    active inequality set off the tight rows (``g_i w`` at ``h_i`` to tolerance).

    Args:
        mean: Vector of expected returns.
        lower_bounds: Lower box bounds.
        upper_bounds: Upper box bounds.
        a: Equality-constraint matrix (``m x n``).
        b: Equality-constraint right-hand side (length ``m``).
        tol: Tolerance for classifying an asset as free (strictly interior) and a
            row as active (tight).
        g: Inequality-constraint matrix (``p x n``); ``None`` means no rows.
        h: Inequality-constraint right-hand side (length ``p``).

    Returns:
        The maximum-return vertex as a :class:`TurningPoint`, carrying the active
        inequality rows in ``active_ineq``.

    Raises:
        ValueError: If the linear program is infeasible or unbounded (the
            constraints admit no maximum-return vertex), or if that vertex is
            degenerate: the free set does not span the equality rows together with
            the active inequality rows, so the reduced KKT system would be
            singular. That case is declined here rather than left to surface as an
            opaque singular-matrix error later in the trace.
    """
    g = np.zeros((0, mean.shape[0])) if g is None else np.asarray(g, dtype=np.float64)
    h = np.zeros(0) if h is None else np.asarray(h, dtype=np.float64)

    # maximize mean @ w  ==  minimize -mean @ w, subject to A w = b, G w <= h, box.
    result = linprog(
        c=-np.asarray(mean, dtype=np.float64),
        A_eq=np.asarray(a, dtype=np.float64),
        b_eq=np.asarray(b, dtype=np.float64),
        A_ub=g if g.shape[0] else None,
        b_ub=h if g.shape[0] else None,
        bounds=list(zip(lower_bounds, upper_bounds, strict=True)),
        method="highs",
    )
    if not result.success:
        msg = f"Could not find a maximum-return vertex (linear program: {result.message})"
        raise ValueError(msg)

    weights = np.asarray(result.x, dtype=np.float64)
    free = (weights > lower_bounds + tol) & (weights < upper_bounds - tol)
    active_ineq = (g @ weights >= h - tol) if g.shape[0] else np.zeros(0, dtype=bool)

    # The free set must span the equality rows together with the active inequality
    # rows: C = [A ; G_active] restricted to the free assets must have full row
    # rank, or the reduced KKT solve is singular. A degenerate maximum-return
    # vertex (a basic asset pinned on a bound) violates this; decline it with an
    # actionable diagnosis instead of letting it surface as an opaque "Singular
    # matrix" error downstream.
    c = np.vstack([a, g[active_ineq]])
    mc = c.shape[0]
    n_free = int(np.count_nonzero(free))
    # rank(C[:, free]) <= min(mc, n_free), so fewer free assets than active rows
    # is degenerate by itself. Testing this first also keeps matrix_rank off a
    # zero-column block, whose empty singular-value reduction raises on numpy 2.0.
    if n_free < mc or np.linalg.matrix_rank(c[:, free]) < mc:
        msg = (
            f"The maximum-return vertex is degenerate (free-set size {n_free}, "
            f"active constraints {mc}): a basic asset sits exactly on a box bound, so the free set "
            "does not span the active equality and inequality rows and the reduced KKT system is "
            "singular. Tracing a frontier from a degenerate first vertex is not yet supported; perturb "
            "the bounds or the constraints so the maximum-return vertex is non-degenerate."
        )
        raise ValueError(msg)

    return TurningPoint(free=free, weights=weights, active_ineq=active_ineq)


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

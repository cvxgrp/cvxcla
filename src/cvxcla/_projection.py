"""Feasibility projection for Critical Line Algorithm turning points.

Accumulated floating-point round-off over the many turning points of a long
trace can place a candidate a hair outside its box even at a well-conditioned
vertex: the covariance there has near-flat directions (its small eigenvalues)
and the round-off lies in exactly those directions, so the candidate is optimal
to solver precision but not exactly feasible. These pure helpers project such a
candidate back onto the feasible region ``{w : lower <= w <= upper, C w = d}``.

They are a strict no-op for the well-posed, already-feasible turning points that
are the common case, so they never perturb the exact frontier. ``CLA._emit``
calls :func:`project_feasible` at every turning point.
"""

from __future__ import annotations

import numpy as np
from cvx.linalg import AffineProjection
from numpy.typing import NDArray


def project_feasible(
    weights: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    g: NDArray[np.float64],
    h: NDArray[np.float64],
    active_ineq: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Project ``weights`` onto the feasible region, clearing round-off.

    Well-posed turning points are already strictly feasible and are returned
    unchanged so the projection never perturbs the exact frontier. Otherwise the
    candidate is dispatched to the closed-form capped-simplex projection for the
    canonical all-ones budget with no active inequality row (see
    :func:`project_capped_simplex`), and to the general alternating projection
    otherwise (see :func:`project_alternating`).

    Args:
        weights: The candidate weight vector to project.
        lower: Per-asset lower bounds.
        upper: Per-asset upper bounds.
        a: Equality-constraint matrix ``A`` of ``A w = b``.
        b: Equality-constraint right-hand side ``b``.
        g: Inequality-constraint matrix ``G`` of ``G w <= h`` (``(p, n)``).
        h: Inequality-constraint right-hand side ``h`` (length ``p``).
        active_ineq: Boolean mask (length ``p``) of the active inequality rows.

    Returns:
        The projected weight vector, feasible to the box and the constraints.
    """
    # Well-posed turning points are already strictly feasible: return them
    # unchanged so the projection never perturbs the exact frontier.
    if np.all(weights >= lower) and np.all(weights <= upper):
        return weights

    if not active_ineq.any() and a.shape[0] == 1 and np.allclose(a, 1.0):
        return project_capped_simplex(weights, lower, upper, float(b[0]))

    c = np.vstack([a, g[active_ineq]])
    d = np.concatenate([b, h[active_ineq]])
    return project_alternating(weights, lower, upper, c, d)


def project_capped_simplex(
    weights: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    budget: float,
) -> NDArray[np.float64]:
    """Euclidean projection onto the capped simplex for the all-ones budget.

    Computed by water-filling a single shift ``theta`` so that
    ``sum(clip(w - theta, lower, upper)) = budget``. A plain clip-then-rescale is
    deliberately *not* used: rescaling the clipped weights to restore the budget
    can push capped weights back over their bound when many assets are capped at
    once (heavy ties under a tight cap), re-introducing the very infeasibility
    the projection is meant to clear.

    Args:
        weights: The candidate weight vector, known to violate its box.
        lower: Per-asset lower bounds.
        upper: Per-asset upper bounds.
        budget: The all-ones budget right-hand side ``sum(w)``.

    Returns:
        The projected weight vector, on the budget and inside the box.
    """
    # sum(clip(w - theta, lower, upper)) is non-increasing in theta; bisect
    # for the theta that hits the budget. The bracket clips to all-upper
    # (sum at least the budget) at theta_lo and all-lower (at most the
    # budget) at theta_hi, so a root is guaranteed for any feasible problem.
    theta_lo = float((weights - upper).min()) - 1.0
    theta_hi = float((weights - lower).max()) + 1.0
    for _ in range(100):
        theta = 0.5 * (theta_lo + theta_hi)
        if float(np.clip(weights - theta, lower, upper).sum()) > budget:
            theta_lo = theta
        else:
            theta_hi = theta
    return np.clip(weights - 0.5 * (theta_lo + theta_hi), lower, upper)


def project_alternating(
    weights: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Alternating projection onto the box and the affine set ``{C w = d}``.

    ``C`` stacks the equality rows ``A`` and the active inequality rows ``G_S``
    (held at equality ``g_i w = h_i``). The candidate already satisfies
    ``C w = d`` (the reduced KKT solve enforces it) and the inactive inequality
    rows keep a margin, so a few iterations alternating a box clip with the
    affine projection converge to a point feasible to the box, the equalities,
    and every inequality.

    Args:
        weights: The candidate weight vector, known to violate its box.
        lower: Per-asset lower bounds.
        upper: Per-asset upper bounds.
        c: Stacked equality/active-inequality matrix.
        d: Stacked equality/active-inequality right-hand side.

    Returns:
        The projected weight vector, feasible to the box and the constraints.
    """
    affine = AffineProjection(c, d)
    projected = weights
    for _ in range(100):
        projected = np.clip(projected, lower, upper)
        projected = affine.project(projected)
    return np.clip(projected, lower, upper)

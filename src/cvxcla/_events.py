"""Critical-lambda event scans for the Critical Line Algorithm.

Along a critical-line segment ``w(lam) = r_alpha + lam * r_beta`` two families of
events end the segment's validity: a *box* event (a free weight reaching a bound,
or a blocked weight's multiplier changing sign) and an *inequality-row* event (an
inactive ``G w <= h`` row's slack reaching zero, or an active row's multiplier
changing sign). Both reduce to the same ``-intercept / slope`` critical-lambda
ratio, computed here as pure functions and stacked by ``CLA.event_matrix`` into
the ``(n + p, 4)`` matrix the generic path tracer scans.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def event_ratios(
    r_alpha: NDArray[np.float64],
    r_beta: NDArray[np.float64],
    gamma: NDArray[np.float64],
    delta: NDArray[np.float64],
    free_in: NDArray[np.bool_],
    at_upper: NDArray[np.bool_],
    at_lower: NDArray[np.bool_],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Critical lambda for every candidate box event, as an ``(n, 4)`` matrix.

    Along the segment ``w(lam) = r_alpha + lam * r_beta`` a free weight can
    reach a box bound (columns 0/1, "moves to a bound") and a blocked weight's
    multiplier can change sign so it re-enters the free set (columns 2/3,
    "leaves a bound"). Entries with no event are ``-inf``.

    A free weight moves with even a tiny slope, so given a long enough lam
    range it still crosses a bound; filtering slopes at the classification
    tolerance would miss such crossings and let weights drift out of bounds.
    Only slopes at floating-point noise level are excluded: below
    ``sqrt(machine epsilon)`` a slope is indistinguishable from solve noise, and
    the huge ratios it would produce only amplify rounding errors.

    Args:
        r_alpha: Segment intercept ``w(0)``.
        r_beta: Segment slope ``dw/dlam``.
        gamma: Multiplier gradient for the alpha system.
        delta: Multiplier gradient for the beta system.
        free_in: Mask of assets in the reduced solve.
        at_upper: Mask of assets blocked at their upper bound.
        at_lower: Mask of assets blocked at their lower bound.
        lower: Per-asset lower bounds.
        upper: Per-asset upper bounds.

    Returns:
        The ``(n, 4)`` matrix of critical lambdas.
    """
    ns = len(r_alpha)
    eps = np.sqrt(np.finfo(np.float64).eps)
    # 4 columns = the 4 event types; extra unused columns are harmless.
    l_mat = np.full((ns, 4), -np.inf)  # pragma: no mutate

    # Precompute each event mask exactly once. The <,> vs <=,>= choice at
    # the eps boundary is numerically irrelevant — a slope/derivative
    # landing exactly on +/-sqrt(machine-eps) never occurs with real
    # data — so those boundary comparisons are marked no-mutate.
    beta_down = free_in & (r_beta < -eps)  # pragma: no mutate
    beta_up = free_in & (r_beta > +eps)  # pragma: no mutate
    delta_down = at_upper & (delta < -eps)  # pragma: no mutate
    delta_up = at_lower & (delta > +eps)  # pragma: no mutate

    # Columns 0,1 are "moves to a bound" (free->blocked) and 2,3 are
    # "leaves a bound" (blocked->free); the next-free update only tests
    # dirchg >= 2, so swapping a column *within* a group (0<->1 or 2<->3) is
    # behaviourally identical and marked no-mutate. Crossing the 1<->2 group
    # boundary IS exercised by the frontier tests.
    l_mat[beta_down, 0] = (upper[beta_down] - r_alpha[beta_down]) / r_beta[beta_down]  # pragma: no mutate
    l_mat[beta_up, 1] = (lower[beta_up] - r_alpha[beta_up]) / r_beta[beta_up]
    l_mat[delta_down, 2] = -gamma[delta_down] / delta[delta_down]  # pragma: no mutate
    l_mat[delta_up, 3] = -gamma[delta_up] / delta[delta_up]
    return l_mat


def ineq_event_ratios(
    r_alpha: NDArray[np.float64],
    r_beta: NDArray[np.float64],
    eta_alpha: NDArray[np.float64],
    eta_beta: NDArray[np.float64],
    active_ineq: NDArray[np.bool_],
    g: NDArray[np.float64],
    h: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Critical lambda for every inequality-row event, as a ``(p, 4)`` matrix.

    The row analogue of :func:`event_ratios`. Along the segment an *inactive*
    row ``i`` becomes active when its slack ``s_i(lam) = g_i w(lam) - h_i``
    rises to zero from the feasible (negative) side (column 0); an *active*
    row releases when its multiplier ``eta_i(lam)`` falls to zero (column 1).
    Both are affine in ``lam``, so the critical lambda is the same
    ``-intercept / slope`` ratio used for the box events, with the same
    ``sqrt(machine eps)`` slope floor: a slope below noise level is
    indistinguishable from solve round-off and would only produce a huge,
    rounding-dominated ratio. Entries with no event are ``-inf``. Columns 2
    and 3 are unused (kept so the block stacks onto the ``(n, 4)`` box block).

    Args:
        r_alpha: Segment intercept ``w(0)``.
        r_beta: Segment slope ``dw/dlam``.
        eta_alpha: Affine inequality-multiplier intercept (length ``p``).
        eta_beta: Affine inequality-multiplier slope (length ``p``).
        active_ineq: Boolean mask (length ``p``) of the active inequality rows.
        g: Inequality-constraint matrix ``G`` of ``G w <= h`` (``(p, n)``).
        h: Inequality-constraint right-hand side ``h`` (length ``p``).

    Returns:
        The ``(p, 4)`` matrix of critical lambdas.
    """
    p = g.shape[0]
    l_mat = np.full((p, 4), -np.inf)  # pragma: no mutate
    if p == 0:
        return l_mat

    eps = np.sqrt(np.finfo(np.float64).eps)
    inactive = ~active_ineq

    # Enter: an inactive row's slack rises to zero. The slope/intercept split
    # comes straight from the affine weights; the slope sign mirrors the box
    # "moves to a bound" event (decreasing lam must raise the slack).
    s_alpha = g @ r_alpha - h
    s_beta = g @ r_beta
    enter = inactive & (s_beta < -eps)  # pragma: no mutate
    l_mat[enter, 0] = -s_alpha[enter] / s_beta[enter]

    # Release: an active row's non-negative multiplier falls back to zero,
    # the row analogue of a blocked multiplier changing sign.
    release = active_ineq & (eta_beta > +eps)  # pragma: no mutate
    l_mat[release, 1] = -eta_alpha[release] / eta_beta[release]
    return l_mat

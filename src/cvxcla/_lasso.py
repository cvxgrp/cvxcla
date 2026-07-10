"""Per-segment numeric kernels for the LASSO regularisation path.

The LASSO homotopy is driven by the same generic ``cvxcla.pathtracer.trace`` loop
as the Critical Line Algorithm; the two problem-specific numeric kernels live here
as pure functions, mirroring the CLA's ``cvxcla._kkt`` (the segment solve) and
``cvxcla._events`` (the critical-lambda scan). :func:`solve_segment` solves the
affine path valid on one segment; :func:`scan_events` stacks every candidate
critical lambda into the ``(n + p, 4)`` matrix the tracer scans.

Both are pure: they take arrays (and the ``QuadraticForm`` operator) in and return
arrays, so ``cvxcla.lasso.Lasso`` reduces to the thin ``ParametricProblem`` glue
that wires them into the tracer. The two ``NamedTuple`` carriers passed between the
tracer and these kernels -- :class:`LassoState` (the vertex) and
:class:`LassoSegment` (the affine piece) -- are defined here alongside the kernels
that produce and consume them.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .operators import QuadraticForm, bordered_solve


class LassoState(NamedTuple):
    """The support, sign pattern, active inequality rows, and current penalty.

    ``lam`` is the penalty at the segment's upper end. The event scan uses it to
    require *strict* progress to a smaller penalty: right after a coefficient
    enters it sits at zero, so its leave event lies at the current ``lam``; without
    the strict window the shared selector would re-fire it and the walk would cycle
    between entering and leaving the same coordinate.
    """

    active: NDArray[np.bool_]
    signs: NDArray[np.float64]
    rows_active: NDArray[np.bool_]
    lam: float


class LassoSegment(NamedTuple):
    """The affine path ``beta(lam) = alpha - lam * beta_slope`` and its multipliers.

    ``eta_alpha``/``eta_slope`` give the active-row multiplier path
    ``eta(lam) = eta_alpha + lam * eta_slope`` (length ``p``, nonzero only on active
    rows); ``p``/``q`` give the generalised correlation ``p + lam * q``.
    """

    alpha: NDArray[np.float64]
    beta_slope: NDArray[np.float64]
    p: NDArray[np.float64]
    q: NDArray[np.float64]
    eta_alpha: NDArray[np.float64]
    eta_slope: NDArray[np.float64]


def solve_segment(
    quad: QuadraticForm,
    xty: NDArray[np.float64],
    g_matrix: NDArray[np.float64],
    h_vector: NDArray[np.float64],
    state: LassoState,
) -> LassoSegment:
    """Solve the affine segment for the current support, signs, and active rows.

    With no active rows this is the plain LASSO solve against the Gram
    submatrix. With active rows it is the bordered Schur solve of the CLA: the
    active rows ``G_S`` enter the reduced KKT system as extra equality rows.

    Args:
        quad: The quadratic form ``H`` (``X^T X``) as a :class:`QuadraticForm`.
        xty: The linear term ``X^T y`` of shape ``(n,)``.
        g_matrix: Inequality matrix ``G`` of ``G beta <= h`` (``(p, n)``).
        h_vector: Inequality right-hand side ``h`` (length ``p``).
        state: The current support, signs, and active-row masks.

    Returns:
        The :class:`LassoSegment` affine path and its multipliers.
    """
    n = xty.shape[0]
    active, signs, rows_active = state.active, state.signs, state.rows_active
    alpha = np.zeros(n)
    beta_slope = np.zeros(n)
    eta_alpha = np.zeros(g_matrix.shape[0])
    eta_slope = np.zeros(g_matrix.shape[0])

    xty_s = xty[active]
    signs_s = signs[active]
    if not np.any(active):
        # Empty support (e.g. the non-negative path when no correlation is
        # positive): beta = 0, correlation = X^T y, and there is nothing to solve.
        return LassoSegment(alpha, beta_slope, xty.copy(), np.zeros(n), eta_alpha, eta_slope)

    # The active rows G_RS act as equality rows in the reduced KKT system, exactly
    # the CLA's bordered Schur solve (operators.bordered_solve). With no active rows
    # (|R| = 0) this reduces to the plain LASSO solve beta_S(lam) = H_SS^{-1}(xty_S -
    # lam s_S). The slope's constraint right-hand side is zero, and the slope
    # multiplier nu carries the opposite sign convention to eta(lam) (beta = alpha -
    # lam beta_slope here, vs w = r_alpha + lam r_beta in the CLA), hence the flip.
    g_rs = g_matrix[np.ix_(rows_active, active)]  # |R| x |S|
    h_r = h_vector[rows_active]
    x_const, x_slope, eta_a, nu_slope = bordered_solve(quad, active, g_rs, xty_s, signs_s, h_r, np.zeros(g_rs.shape[0]))
    alpha[active] = x_const
    beta_slope[active] = x_slope
    eta_alpha[rows_active] = eta_a
    eta_slope[rows_active] = -nu_slope

    # Generalised correlation c(lam) = xty - H beta(lam) - G_R^T eta(lam) = p + lam q.
    g_r = g_matrix[rows_active]
    eta_a_r = eta_alpha[rows_active]
    eta_s_r = eta_slope[rows_active]
    p = xty - quad.matvec(alpha) - g_r.T @ eta_a_r
    q = quad.matvec(beta_slope) - g_r.T @ eta_s_r
    return LassoSegment(alpha, beta_slope, p, q, eta_alpha, eta_slope)


def scan_events(
    n: int,
    g_matrix: NDArray[np.float64],
    h_vector: NDArray[np.float64],
    tol: float,
    nonneg: bool,
    state: LassoState,
    segment: LassoSegment,
) -> NDArray[np.float64]:
    """Return the ``(n + p, 4)`` matrix of candidate critical lambdas.

    Rows ``0..n-1`` are coordinate events (col 0 leave, col 1 enter ``+``, col 2
    enter ``-``); rows ``n..n+p-1`` are inequality-row events (col 0 activate, col
    1 release). Entries are ``-inf`` where the event cannot occur, and only
    events strictly inside ``(tol, lam - tol)`` are kept.

    Args:
        n: The problem dimension (number of features).
        g_matrix: Inequality matrix ``G`` of ``G beta <= h`` (``(p, n)``).
        h_vector: Inequality right-hand side ``h`` (length ``p``).
        tol: Tolerance for event selection and the validity window.
        nonneg: When ``True`` the enter-negative events are disabled (``beta >= 0``).
        state: The current support, signs, and active-row masks.
        segment: The affine path and multipliers from :func:`solve_segment`.

    Returns:
        The ``(n + p, 4)`` matrix of critical lambdas.
    """
    rows = g_matrix.shape[0]
    active = state.active
    inactive = ~active
    alpha, beta_slope, p, q, eta_alpha, eta_slope = segment
    rows_active = state.rows_active

    l_mat = np.full((n + rows, 4), -np.inf)

    # leave: alpha_j - lam beta_slope_j = 0
    leaves = active & (np.abs(beta_slope) > tol)  # pragma: no mutate
    l_mat[:n][leaves, 0] = alpha[leaves] / beta_slope[leaves]

    # enter (+): p_j + lam q_j = +lam -> lam = p_j / (1 - q_j)
    denom_pos = 1.0 - q
    enters_pos = inactive & (np.abs(denom_pos) > tol)  # pragma: no mutate
    l_mat[:n][enters_pos, 1] = p[enters_pos] / denom_pos[enters_pos]

    # enter (-): p_j + lam q_j = -lam -> lam = -p_j / (1 + q_j). Disabled under the
    # non-negative restriction beta >= 0, where only positive entries are allowed.
    if not nonneg:
        denom_neg = 1.0 + q
        enters_neg = inactive & (np.abs(denom_neg) > tol)  # pragma: no mutate
        l_mat[:n][enters_neg, 2] = -p[enters_neg] / denom_neg[enters_neg]

    if rows:
        slope_row = g_matrix @ beta_slope  # d/d(-lam) of the row value
        level_row = g_matrix @ alpha - h_vector  # G_r alpha - h_r
        # activate: the row value G_r beta(lam) = level + h_r - lam slope rises to
        # the cap h_r as lam decreases when its slope d(value)/d(-lam) = slope_row
        # is positive; the crossing is lam = (G_r alpha - h_r) / (G_r beta_slope).
        inactive_rows = ~rows_active & (slope_row > tol)  # pragma: no mutate
        l_mat[n:][inactive_rows, 0] = level_row[inactive_rows] / slope_row[inactive_rows]
        # release: eta_r(lam) = eta_alpha + lam eta_slope -> 0 from eta > 0, i.e. eta_slope > 0.
        releasing = rows_active & (eta_slope > tol)  # pragma: no mutate
        l_mat[n:][releasing, 1] = -eta_alpha[releasing] / eta_slope[releasing]

    # Keep only events that make strict progress to a smaller, positive penalty.
    l_mat[(l_mat <= tol) | (l_mat >= state.lam - tol)] = -np.inf
    return l_mat

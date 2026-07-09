"""Reduced KKT machinery for a Critical Line Algorithm turning point.

At each turning point the active set is identified (which box bounds are held,
which assets are free) and the reduced KKT system is solved by block elimination
to produce the affine critical-line segment ``w(lam) = r_alpha + lam * r_beta``.
Both steps are pure functions of the problem data and the active set, so they
live here rather than on the ``CLA`` class; the covariance only ever enters
through the ``QuadraticForm`` interface, so structured backends never
materialise an ``n x n`` matrix.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .operators import QuadraticForm, bordered_solve, cross


def active_set(
    free: NDArray[np.bool_],
    weights: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    tol: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]:
    """Identify the active set at a turning point and the weights pinned to bounds.

    A blocked asset sitting (to tolerance) on a bound is held fixed there and
    excluded from the reduced KKT solve; every other asset is *in*. Returns the
    upper-bound mask, the lower-bound mask, the in-set mask, and the full-length
    vector of weights fixed at their bounds.

    Args:
        free: Boolean mask of the assets free at the turning point.
        weights: The turning point's weight vector.
        lower: Per-asset lower bounds.
        upper: Per-asset upper bounds.
        tol: Tolerance for classifying a weight as sitting on a bound.

    Returns:
        ``(at_upper, at_lower, free_in, fixed_weights)``.

    Raises:
        RuntimeError: If every asset is blocked, which makes the reduced system
            singular.
    """
    blocked = ~free
    if np.all(blocked):
        msg = "All variables cannot be blocked"
        raise RuntimeError(msg)

    at_upper = blocked & (np.abs(weights - upper) <= tol)  # pragma: no mutate
    at_lower = blocked & (np.abs(weights - lower) <= tol)  # pragma: no mutate
    free_in = ~(at_upper | at_lower)

    fixed_weights = np.zeros(len(weights))
    fixed_weights[at_upper] = upper[at_upper]
    fixed_weights[at_lower] = lower[at_lower]
    return at_upper, at_lower, free_in, fixed_weights


def solve_kkt(
    cov: QuadraticForm,
    mean: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    g: NDArray[np.float64],
    h: NDArray[np.float64],
    free_in: NDArray[np.bool_],
    fixed_weights: NDArray[np.float64],
    active_ineq: NDArray[np.bool_],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Solve the reduced KKT system for the current critical-line segment.

    Block elimination over the *stacked* constraint matrix ``C = [A ; G_S]``,
    where ``G_S`` are the currently-active inequality rows held at equality.
    Because an active inequality row enters the stationarity/feasibility system
    exactly as an equality row does, the same elimination handles both: a
    multi-right-hand-side solve against the free covariance block ``Sigma_FF``
    (via the backend, so structured covariances never materialise an
    ``n x n`` matrix) feeds an ``(m + |S|) x (m + |S|)`` Schur complement
    ``C_F Sigma_FF^{-1} C_F.T``. With no active inequality rows this is the
    plain equality solve.

    Args:
        cov: The covariance as a ``QuadraticForm`` backend.
        mean: Vector of expected returns.
        a: Equality-constraint matrix ``A`` of ``A w = b``.
        b: Equality-constraint right-hand side ``b``.
        g: Inequality-constraint matrix ``G`` of ``G w <= h`` (``(p, n)``).
        h: Inequality-constraint right-hand side ``h`` (length ``p``).
        free_in: Boolean mask of the assets in the reduced solve.
        fixed_weights: Full-length weights of the assets held at their bounds.
        active_ineq: Boolean mask (length ``p``) of the active inequality rows.

    Returns:
        ``(r_alpha, r_beta, gamma, delta, eta_alpha, eta_beta)``: the affine
        segment ``w(lam) = r_alpha + lam * r_beta``, the box-multiplier
        gradients ``gamma``/``delta`` that drive the leave-a-bound events, and
        the affine inequality multipliers ``eta_alpha + lam * eta_beta``
        (length ``p``, non-zero only on active rows) that drive the
        release-a-row events.
    """
    m = a.shape[0]
    ns = len(mean)
    p = g.shape[0]
    out = ~free_in
    # Stack the active inequality rows beneath the equality rows; the active
    # rows are held at equality (g_i w = h_i), so C/d is the equality system
    # of the reduced QP at this vertex.
    c = np.vstack([a, g[active_ineq]])
    d = np.concatenate([b, h[active_ineq]])
    c_free = c[:, free_in]

    # The reduced KKT system is the shared bordered solve: the constant system
    # carries the blocked-weight shift -Sigma_FB w_B and the reduced constraint
    # right-hand side d - C_B w_B; the slope system carries the mean with a zero
    # constraint right-hand side (A r_beta = 0).
    x_alpha, x_beta, nu_alpha, nu_beta = bordered_solve(
        cov,
        free_in,
        c_free,
        -cross(cov, free_in, fixed_weights),
        mean[free_in],
        d - c[:, out] @ fixed_weights[out],
        np.zeros(c.shape[0]),
    )

    r_alpha = fixed_weights.copy()
    r_alpha[free_in] = x_alpha
    r_beta = np.zeros(ns)
    r_beta[free_in] = x_beta

    gamma = cov.matvec(r_alpha) + c.T @ nu_alpha
    delta = cov.matvec(r_beta) + c.T @ nu_beta - mean

    # The tail of the stacked multiplier is the inequality multiplier eta(lam)
    # = eta_alpha + lam eta_beta, scattered back to full length p (zero on the
    # inactive rows, which have no release event).
    eta_alpha = np.zeros(p)
    eta_beta = np.zeros(p)
    eta_alpha[active_ineq] = nu_alpha[m:]
    eta_beta[active_ineq] = nu_beta[m:]
    return r_alpha, r_beta, gamma, delta, eta_alpha, eta_beta

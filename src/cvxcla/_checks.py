"""Acceptance checks for a Critical Line Algorithm turning point.

Before a candidate turning point is stored it must pass two independent tests:
the free-asset covariance block must be numerically non-singular, so the solve
that produced the candidate can be trusted (:func:`guard_degeneracy`), and the
weights must satisfy every constraint of the problem to tolerance
(:func:`check_feasible`). :func:`well_conditioned` decides once, up front,
whether the first test can ever fire. Both are pure functions of the problem data and the
candidate, so they live here rather than on the ``CLA`` class; ``CLA._emit`` and
``CLA._append`` call them at every turning point.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog  # type: ignore[import-untyped]

from ._kkt import Segment
from .errors import DegenerateProblemError, FeasibilityError
from .operators import RCOND_FLOOR, QuadraticForm


def check_feasible(
    weights: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    g: NDArray[np.float64],
    h: NDArray[np.float64],
    leverage: float | None,
    tol: float,
) -> None:
    """Refuse weights that violate any constraint of the problem.

    The box, ``G w <= h`` and leverage checks use ``tol``; the equality
    ``A w = b`` is checked to a fixed ``1e-7`` absolute tolerance. An empty
    ``g`` makes the inequality check vacuously true, so it never fires when there
    are no inequality rows; ``leverage=None`` skips the gross-exposure check.

    Args:
        weights: The candidate weight vector.
        lower: Per-asset lower bounds.
        upper: Per-asset upper bounds.
        a: Equality-constraint matrix ``A`` of ``A w = b``.
        b: Equality-constraint right-hand side ``b``.
        g: Inequality-constraint matrix ``G`` of ``G w <= h`` (``(p, n)``).
        h: Inequality-constraint right-hand side ``h`` (length ``p``).
        leverage: The gross-exposure cap ``||w||_1 <= leverage``, or ``None``.
        tol: Tolerance for the box, inequality and leverage checks.

    Raises:
        FeasibilityError: Naming the first violated constraint. A turning point
            that fails here after projection signals a numerical failure.
    """
    # (constraint holds?, message if it does not).
    checks: tuple[tuple[bool, str], ...] = (
        (bool(np.all(weights >= (lower - tol))), "Weights below lower bounds"),  # pragma: no mutate
        (bool(np.all(weights <= (upper + tol))), "Weights above upper bounds"),  # pragma: no mutate
        (bool(np.allclose(a @ weights, b, atol=1e-7)), "Weights violate the equality constraint A w = b"),
        (bool(np.all(g @ weights <= h + tol)), "Weights violate the inequality constraint G w <= h"),
        (
            leverage is None or bool(np.abs(weights).sum() <= leverage + tol),
            "Weights violate the leverage constraint ||w||_1 <= leverage",
        ),
    )
    for ok, message in checks:
        if not ok:
            raise FeasibilityError(message)


def well_conditioned(cov: QuadraticForm) -> bool:
    """Whether every free-block solve along the trace is numerically safe.

    By Cauchy's interlacing theorem every principal submatrix of the symmetric
    PSD covariance is at least as well conditioned as the whole matrix --
    deleting rows/columns cannot decrease the smallest eigenvalue nor increase
    the largest -- so the reciprocal condition number of any free block is
    ``>=`` that of the full covariance. Hence if the full covariance clears the
    singularity floor, no free block encountered along the trace can be
    singular, and :func:`guard_degeneracy` is provably never triggered. The
    caller then skips it, paying one conditioning test up front instead of one
    at every turning point (the latter is a full eigendecomposition of the free
    block, as costly as the KKT solve, so it otherwise dominates the trace).

    When the full covariance is itself near-singular (for example a sample
    covariance from fewer observations than assets) this is ``False`` and the
    per-step guard runs unchanged, preserving the degeneracy diagnosis exactly.

    Args:
        cov: The covariance as a ``QuadraticForm`` backend.

    Returns:
        Whether the full covariance clears the singularity floor.
    """
    return float(cov.rcond_free(np.arange(cov.n))) >= RCOND_FLOOR


def guard_degeneracy(cov: QuadraticForm, lamb: float, free: NDArray[np.bool_]) -> None:
    """Refuse the turning point when the free-asset block is numerically singular.

    We distinguish two regimes by the conditioning of the free-asset block.
    While that block stays numerically full rank its solve is reliable and any
    box violation is round-off, which :func:`cvxcla._projection.project_feasible`
    clears. Once the free set grows past the covariance rank the block is
    numerically singular and its solve is unreliable; whatever weights it produces
    (feasible or not) cannot be trusted, so we refuse and raise an actionable
    diagnosis instead of silently returning a possibly-suboptimal frontier.

    The discriminator is the free block's reciprocal condition number, read
    from its symmetric eigenvalues. Unlike the magnitude of the box violation,
    which is the residual of a singular solve and therefore varies with the
    BLAS/LAPACK build, the conditioning is deterministic and portable, so the
    completed-vs-declined boundary is the same on every platform.

    The caller skips this check entirely when the full covariance is well
    conditioned: by interlacing no free block can then be singular, so the check
    is provably redundant (see :func:`well_conditioned`).

    Args:
        cov: The covariance as a ``QuadraticForm`` backend.
        lamb: Lambda value of the candidate turning point, used in the message.
        free: Boolean mask of the free assets at the candidate.

    Raises:
        DegenerateProblemError: With a degeneracy-specific message when the
            free-asset block is numerically singular (an unreliable solve);
            otherwise returns without effect.
    """
    rcond = cov.rcond_free(np.flatnonzero(free))
    if rcond < RCOND_FLOOR:
        n_free = int(np.count_nonzero(free))
        msg = (
            f"Critical Line Algorithm hit a degeneracy at lambda={lamb:.4g} "
            f"(free-set size {n_free}): the free-asset covariance block is "
            f"numerically singular (reciprocal condition number {rcond:.2g}), "
            "so its solve is unreliable and the turning point cannot be "
            "trusted. The trace was stopped rather than risk silently "
            "returning a suboptimal frontier. This happens when the free set "
            "grows past the covariance rank (for example a sample covariance "
            "from far fewer days than assets). Use a well-conditioned, "
            "full-rank estimate (ample history), or a FactorCovariance backend "
            "(diagonal-plus-low-rank), which is positive definite by construction."
        )
        raise DegenerateProblemError(msg)


def check_held_bounds(
    segment: Segment,
    lamb: float,
    event: int,
    rows: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    tol: float,
    mu_scale: float,
    cov_scale: float,
) -> None:
    """Refuse a segment that ends at a point off the efficient frontier.

    On a segment the multiplier of a blocked asset is ``gamma + lambda * delta``:
    non-negative on a lower bound and non-positive on an upper one wherever the
    segment is optimal. A sign change inside the segment is an event and would have
    ended it sooner, so the sign is checked where the segment ends, at ``lamb``. The
    event scan cannot report a sign that is wrong on the whole segment. That happens
    when the maximum-return vertex is degenerate and its resolution picks a partition
    from the returns alone: the bounds that tie in the returns have ``delta = 0``, so
    they fire no event, and their ``gamma`` (set by ``q - Sigma w``) can have the wrong
    sign. The trace would then run on a partition that is not optimal anywhere and
    return a wrong path without error.

    A wrong sign is not proof by itself: when a free asset sits on a bound the point
    is primal degenerate, its multipliers are not unique, and another choice may
    certify it. So a flagged point is confirmed by :func:`_certified`, which looks for
    any multipliers of the active rows that make the point optimal.

    Args:
        segment: The segment that ends at ``lamb``.
        lamb: The finite ``lambda`` at which the segment ends.
        event: The asset whose event ends the segment (its multiplier is zero there),
            or ``-1`` when no asset event ends it.
        rows: The active constraint rows ``[A ; G_active]`` of the segment.
        lower: Per-asset lower bounds.
        upper: Per-asset upper bounds.
        tol: Tolerance for classifying a weight as sitting on a bound.
        mu_scale: The scale of the returns, which sets the scale of ``delta``.
        cov_scale: The scale of ``Sigma w``, which sets the scale of ``gamma``.

    Raises:
        DegenerateProblemError: If the point at ``lamb`` is not optimal.
    """
    multiplier = segment.gamma + lamb * segment.delta
    scale = max(cov_scale, lamb * mu_scale, float(np.max(np.abs(segment.gamma), initial=0.0)))
    floor = float(np.sqrt(np.finfo(np.float64).eps)) * scale
    wrong = (segment.at_lower & (multiplier < -floor)) | (segment.at_upper & (multiplier > floor))
    if event >= 0:
        wrong[event] = False
    if not np.any(wrong):
        return
    weights = segment.r_alpha + lamb * segment.r_beta
    eta = (segment.eta_alpha + lamb * segment.eta_beta)[segment.active_ineq]
    if _certified(multiplier, eta, weights, segment.free_in, rows, lower, upper, tol, floor):
        return
    msg = (
        f"The segment ending at lambda={lamb:.4g} held {int(np.count_nonzero(wrong))} asset(s) on a "
        "bound it should have left, so it is not on the efficient frontier. This comes from a "
        "degenerate maximum-return vertex whose bounds tie in the returns alone (tied returns at "
        "the top of the frontier, or a homogeneous equality row b = 0 whose maximum-return vertex "
        "is w = 0): which of them start the path depends on q - Sigma w, which the vertex "
        "resolution does not see. Perturb the returns or the constraints so the maximum-return "
        "vertex is non-degenerate."
    )
    raise DegenerateProblemError(msg)


def _certified(
    multiplier: NDArray[np.float64],
    eta: NDArray[np.float64],
    weights: NDArray[np.float64],
    free: NDArray[np.bool_],
    rows: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    tol: float,
    floor: float,
) -> bool:
    """Whether some multipliers of the active rows make the point optimal.

    The segment's multipliers ``y`` leave a blocked asset the multiplier ``m`` (zero
    on the free assets). Shifting them by ``dy`` changes every asset's multiplier by
    ``C^T dy`` and the inequality multipliers by ``dy`` on their rows. The point is
    optimal if some shift leaves the free assets that are strictly inside their box
    at zero, every asset on a lower bound non-negative and every asset on an upper
    bound non-positive -- free assets that sit on a bound included, which is what
    the segment's own multipliers do not allow -- and the active inequality
    multipliers non-negative. That is a linear feasibility problem in ``dy``.

    Args:
        multiplier: The asset multipliers at the point (zero on the free assets).
        eta: The active inequality multipliers at the point.
        weights: The point.
        free: Mask of the free assets.
        rows: The active constraint rows ``C = [A ; G_active]`` (equalities first).
        lower: Per-asset lower bounds.
        upper: Per-asset upper bounds.
        tol: Tolerance for classifying a weight as sitting on a bound.
        floor: Tolerance on the multipliers.

    Returns:
        ``True`` when the point is optimal.
    """
    on_lower = weights <= lower + tol
    on_upper = weights >= upper - tol
    m = np.where(free, 0.0, multiplier)
    interior = free & ~on_lower & ~on_upper
    # m + C^T dy >= -floor on a lower bound, <= floor on an upper one, |.| <= floor inside.
    up = on_lower | interior
    down = on_upper | interior
    k = rows.shape[0]
    p = eta.shape[0]
    a_ub = np.vstack([-rows.T[up], rows.T[down], -np.eye(k)[k - p :]])
    b_ub = np.concatenate([m[up] + floor, floor - m[down], eta + floor])
    result = linprog(np.zeros(k), A_ub=a_ub, b_ub=b_ub, bounds=[(None, None)] * k, method="highs")
    return bool(result.status == 0)

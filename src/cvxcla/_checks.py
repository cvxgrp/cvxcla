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

from .operators import QuadraticForm
from .operators._core import _RCOND_FLOOR


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
        ValueError: Naming the first violated constraint.
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
            raise ValueError(message)


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
    return float(cov.rcond_free(np.arange(cov.n))) >= _RCOND_FLOOR


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
        ValueError: With a degeneracy-specific message when the free-asset
            block is numerically singular (an unreliable solve); otherwise
            returns without effect.
    """
    rcond = cov.rcond_free(np.flatnonzero(free))
    if rcond < _RCOND_FLOOR:
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
        raise ValueError(msg)

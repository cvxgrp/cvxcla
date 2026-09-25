"""The equality-constrained LASSO ``A beta = 0``, traced through the leverage CLA.

The LASSO homotopy of :mod:`cvxcla.lasso` starts from ``beta = 0`` and needs every
constraint row slack there, so it cannot carry equality rows: an equality row is
active from the start and the path cannot be seeded one coordinate at a time. The
Critical Line Algorithm has no such restriction, and under ``Sigma = X^T X`` and
``mu = X^T y`` it traces the same curve (Schmelzer and Hastie, "The Critical Line
Algorithm and the Constrained LASSO: One Curve, Two Literatures",
arXiv:2609.25704).

With homogeneous constraints the route is Corollary 2 of that note. At a fixed
gross-exposure cap ``c`` the tilt sweep of the capped program

    min 1/2 w^T Sigma w - lam mu^T w   s.t.  ||w||_1 <= c,  A w = 0

is the budget-indexed path rescaled, ``w(lam) = lam * beta`` with
``||beta||_1 = c / lam``, and by Theorem 1 the budget-indexed path is the
constrained LASSO path. So one leverage-capped CLA trace with ``c = 1`` gives
every LASSO breakpoint as ``beta = w / lam``, in order: ``lam = inf`` is
``beta = 0``, and the cap's release is the constrained least-squares fit. A box of
``+/- 2 c`` makes the CLA well posed and never binds, since ``|w_i| <= ||w||_1 <= c``.
Under ``beta >= 0`` the lower bound is ``0``, which is homogeneous as well.

The CLA's tilt is not the LASSO penalty. The penalty at a breakpoint is read off
the KKT conditions of the LASSO on the segment that leaves it:
``(X^T y - X^T X beta)_F = lam_L s_F + A_F^T nu``, over the segment's free set
``F`` with signs ``s_F``. The matrix ``[s_F, A_F^T]`` has full column rank at
every non-degenerate active set, so the least-squares solution is exact.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog  # type: ignore[import-untyped]

from .cla import CLA
from .operators import QuadraticForm
from .operators._core import _RCOND_FLOOR

#: One LASSO breakpoint as ``(lam, beta, active)``; :mod:`cvxcla.lasso` wraps it.
BreakpointData = tuple[float, NDArray[np.float64], NDArray[np.bool_]]


def _max_correlation(xty: NDArray[np.float64], a: NDArray[np.float64], nonneg: bool) -> float:
    """Return ``max xty^T beta`` over ``||beta||_1 <= 1``, ``A beta = 0`` (and ``beta >= 0``).

    Zero means ``beta = 0`` is optimal at every penalty, so the path is one point.
    """
    n = xty.shape[0]
    legs = np.hstack([np.eye(n), -np.eye(n)])[:, : n if nonneg else 2 * n]
    result = linprog(
        c=-(xty @ legs),
        A_ub=np.ones((1, legs.shape[1])),
        b_ub=np.ones(1),
        A_eq=a @ legs,
        b_eq=np.zeros(a.shape[0]),
        bounds=(0.0, None),
        method="highs",
    )
    return float(-result.fun)


def _penalty(
    quad: QuadraticForm,
    xty: NDArray[np.float64],
    a: NDArray[np.float64],
    beta: NDArray[np.float64],
    free: NDArray[np.bool_],
    signs: NDArray[np.float64],
) -> float:
    """Solve the LASSO stationarity on ``free`` for the penalty ``lam_L``.

    Args:
        quad: The Gram form ``X^T X``.
        xty: The linear term ``X^T y``.
        a: The equality rows.
        beta: The coefficients at the breakpoint.
        free: The support of the segment leaving the breakpoint.
        signs: The signs of ``beta`` on that segment (length ``n``).

    Returns:
        The penalty, clamped at zero against round-off at the least-squares end.
    """
    residual = (xty - quad.matvec(beta))[free]
    system = np.hstack([signs[free][:, None], a[:, free].T])
    solution = np.linalg.lstsq(system, residual, rcond=None)[0]
    return max(float(solution[0]), 0.0)


def equality_path(
    quad: QuadraticForm,
    xty: NDArray[np.float64],
    a: NDArray[np.float64],
    nonneg: bool,
    tol: float,
) -> list[BreakpointData]:
    """Trace the LASSO path under ``A beta = 0`` via one leverage-capped CLA.

    Args:
        quad: The Gram form ``X^T X`` as a :class:`QuadraticForm`.
        xty: The linear term ``X^T y``.
        a: The equality rows ``(m, n)`` of ``A beta = 0``.
        nonneg: Restrict to ``beta >= 0``.
        tol: Below this, the largest feasible correlation counts as zero.

    Returns:
        The breakpoints ``(lam, beta, active)`` from ``lam_max`` down to the
        constrained least-squares fit at ``lam = 0``.

    Raises:
        ValueError: If the Gram form is singular, so the constrained least-squares
            fit is not unique (e.g. more features than observations), or if the CLA
            cannot trace the problem (e.g. a degenerate first vertex).
    """
    n = xty.shape[0]
    if quad.rcond_free(np.arange(n)) < _RCOND_FLOOR:
        msg = (
            "the equality-constrained LASSO needs a positive-definite Gram X^T X "
            "(more observations than features in general position), so that the "
            "constrained least-squares end of the path is unique"
        )
        raise ValueError(msg)
    if _max_correlation(xty, a, nonneg) <= tol:
        return [(0.0, np.zeros(n), np.zeros(n, dtype=bool))]

    try:
        cla = CLA(
            mean=xty,
            covariance=quad,
            lower_bounds=np.zeros(n) if nonneg else np.full(n, -2.0),
            upper_bounds=np.full(n, 2.0),
            a=a,
            b=np.zeros(a.shape[0]),
            leverage=1.0,
        )
    except ValueError as err:
        msg = f"the equality-constrained LASSO could not be traced through the leverage CLA: {err}"
        raise ValueError(msg) from err

    tps = cla.turning_points
    path: list[BreakpointData] = []
    # Segment k runs from turning point k to k + 1; the last one ends at the CLA's
    # lambda = 0 endpoint w = 0, which has no beta of its own and is not recorded.
    for hi, lo in itertools.pairwise(tps):
        if np.isinf(hi.lamb):
            # The first segment holds the maximum-return vertex, so beta runs along
            # the ray from 0 through it and carries the vertex's signs.
            beta, signs = np.zeros(n), np.sign(hi.weights)
        else:
            # w is affine in lambda on the segment, so this is beta at its midpoint,
            # where every free coordinate is strictly nonzero.
            beta = hi.weights / hi.lamb
            signs = np.sign((hi.weights + lo.weights) / (hi.lamb + lo.lamb))
        path.append((_penalty(quad, xty, a, beta, hi.free, signs), beta, hi.free.copy()))
    return path

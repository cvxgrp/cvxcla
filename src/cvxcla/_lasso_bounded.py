"""The LASSO under coefficient bounds, equality rows ``A beta = b`` and rows ``G beta <= h``, as one CLA.

The homotopy of :mod:`cvxcla.lasso` starts from ``beta = 0`` with every row slack,
and the leverage route of :mod:`cvxcla._lasso_cla` needs every constraint
homogeneous. Neither takes a sign on only some of the coefficients together with a
cap, an equality row with a nonzero right-hand side, or inequality rows together with
either. This module traces those
with the CLA directly, in the *penalty form* (Schmelzer, "The Constrained LASSO by
the Critical Line Algorithm"): the penalty ``lam ||beta||_1`` is read as a cost of
``lam`` per unit of gross position.

Split ``beta = S z`` with ``z >= 0``, where ``S`` has a column ``e_j`` for every
``j`` with ``u_j > 0`` and a column ``-e_j`` for every ``j`` with ``l_j < 0``. With
``G = X^T X`` and ``c = X^T y`` the LASSO

    min 1/2 ||y - X beta||^2 + lam ||beta||_1   s.t.  A beta = b,  G beta <= h,  l <= beta <= u

is, up to a constant, the CLA program

    min 1/2 z^T (S^T X^T X S) z - (S^T c + lam * (-1))^T z
    s.t.  A S z = b,  G S z <= h,  0 <= z <= u_bar

at tilt ``lam``, where ``u_bar`` stacks the bounds ``u_j`` and ``-l_j`` of the
columns. The CLA's tilt *is* the penalty, so every turning point is a knot,
``beta = S z`` at ``lam``. At ``lam = inf`` the CLA starts from the vertex of
smallest ``l1`` norm on the constraints, which is ``z = 0`` without equality rows
(every row of ``G beta <= h`` is slack there, since ``h > 0``). A homogeneous
equality row (``b = 0``) makes that vertex degenerate; such problems belong to the
leverage route instead, and the CLA declines them here.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from ._lasso_cla import BreakpointData
from .cla import CLA
from .errors import CLAError, InfeasibleProblemError
from .operators import QuadraticForm


def _split(lower: NDArray[np.float64], upper: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the legs ``S`` (one column per allowed sign) and their bounds ``u_bar``.

    A coefficient whose bounds are both zero has no column: it is fixed at zero.
    """
    n = lower.shape[0]
    keep = np.concatenate([upper > 0, lower < 0])
    legs = np.hstack([np.eye(n), -np.eye(n)])[:, keep]
    return legs, np.concatenate([upper, -lower])[keep]


def bounded_path(
    quad: QuadraticForm,
    xty: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    tol: float,
    g: NDArray[np.float64] | None = None,
    h: NDArray[np.float64] | None = None,
) -> list[BreakpointData]:
    """Trace the LASSO path under ``A beta = b``, ``G beta <= h`` and ``lower <= beta <= upper``.

    Args:
        quad: The Gram form ``X^T X`` as a :class:`QuadraticForm`.
        xty: The linear term ``X^T y``.
        a: The equality rows ``(m, n)``; ``m`` may be zero.
        b: Their right-hand side ``(m,)``.
        lower: Lower bounds, each ``<= 0`` (``-inf`` allowed).
        upper: Upper bounds, each ``>= 0`` (``inf`` allowed).
        tol: Coefficients below ``tol`` times the path's scale count as zero in the
            ``active`` masks.
        g: Optional inequality rows ``(p, n)`` of ``G beta <= h``.
        h: Their right-hand side ``(p,)``, strictly positive.

    Returns:
        The breakpoints ``(lam, beta, active)`` from the first knot down to the
        constrained least-squares fit at ``lam = 0``.

    Raises:
        InfeasibleProblemError: If the bounds fix every coefficient at zero and
            ``b`` is nonzero.
        ValueError: If the CLA cannot trace the problem, e.g. a degenerate or
            non-unique first vertex (see the module docstring).
    """
    n = xty.shape[0]
    legs, cap = _split(lower, upper)
    k = legs.shape[1]
    if k == 0:
        # Every coefficient is fixed at zero, so the path is that one point, if feasible.
        if b.any():
            msg = "the bounds fix beta = 0, which A beta = b with b != 0 excludes"
            raise InfeasibleProblemError(msg)
        return [(0.0, np.zeros(n), np.zeros(n, dtype=bool))]
    # S^T G S through the operator, one leg at a time; k <= 2 n, so this is O(n) matvecs.
    sigma = legs.T @ np.column_stack([quad.matvec(legs[:, i]) for i in range(k)])
    try:
        cla = CLA(
            mean=-np.ones(k),
            covariance=sigma,
            lower_bounds=np.zeros(k),
            upper_bounds=cap,
            a=a @ legs,
            b=b,
            g=None if g is None else g @ legs,
            h=h,
            q=legs.T @ xty,
        )
    except ValueError as err:
        msg = f"the bounded LASSO could not be traced through the CLA: {err}"
        # Keep the error's kind (infeasible, degenerate, ...) when the CLA classified it.
        raise (type(err) if isinstance(err, CLAError) else ValueError)(msg) from err

    # The lam = inf vertex is the first knot again, held still until a coefficient enters;
    # the trace always ends with a finite one, the constrained least-squares fit at lam = 0.
    knots = [(float(tp.lamb), legs @ tp.weights) for tp in cla.turning_points if np.isfinite(tp.lamb)]
    scale = max(float(np.abs(beta).max()) for _, beta in knots) or 1.0

    def support(beta: NDArray[np.float64]) -> NDArray[np.bool_]:
        return np.abs(beta) > tol * scale

    # The support on the segment leaving a knot is that of the segment's midpoint.
    path: list[BreakpointData] = [
        (lam, beta, support(0.5 * (beta + after))) for (lam, beta), (_, after) in itertools.pairwise(knots)
    ]
    lam, beta = knots[-1]
    path.append((lam, beta, support(beta)))
    return path

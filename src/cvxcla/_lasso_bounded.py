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
(every row of ``G beta <= h`` with ``h > 0`` is slack there). Homogeneous rows with
signs only go to the leverage route; the cases below are traced here.

**Homogeneous inequality rows** (``h_r = 0``, such as the ordering ``beta_j <=
beta_{j+1}`` of a monotone fit) are all tight at ``z = 0``, so the start is degenerate
too. They are resolved by relaxation and extrapolation. On a fixed sequence of active
sets the solution on each segment is affine in ``h`` and its slope in ``lam`` does not
depend on ``h``, so the parameter at which a coordinate reaches a bound, a knot, is
affine in ``h`` as well. Tracing the rows at ``h_r = eps`` and ``h_r = 2 eps``, both
nondegenerate, therefore gives the exact path at ``h_r = 0`` as
``2 path(eps) - path(2 eps)``, provided both runs take the same sequence. Segments that
shrink to a point at ``h_r = 0`` (rows the relaxed path met one by one, and the
exact path meets together) are dropped. The result is then *certified*, not trusted: a
linear program finds the multipliers that best satisfy the LASSO's optimality conditions
at every knot and segment midpoint, and the extrapolated path is returned only if they
hold to rounding. Otherwise the problem is declined.

**Homogeneous equality rows** (``A beta = 0``, such as a zero sum) together with a finite
bound or inequality rows are resolved the same way, relaxing ``b`` to ``eps A d`` for a
generic point ``d`` the bounds and the homogeneous rows admit, alongside any ``h_r = 0``.
At ``b = 0`` the start ``z = 0`` is degenerate; relaxed it is a whole face of
smallest-``l1`` points (every same-signed leg of a ones row ties), and the CLA starts on
that face where its quadratic term is smallest.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog  # type: ignore[import-untyped]

from ._lasso_cla import BreakpointData
from .cla import CLA
from .errors import CLAError, DegenerateProblemError, InfeasibleProblemError
from .operators import QuadraticForm


def _split(lower: NDArray[np.float64], upper: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the legs ``S`` (one column per allowed sign) and their bounds ``u_bar``.

    A coefficient whose bounds are both zero has no column: it is fixed at zero.
    """
    n = lower.shape[0]
    keep = np.concatenate([upper > 0, lower < 0])
    legs = np.hstack([np.eye(n), -np.eye(n)])[:, keep]
    return legs, np.concatenate([upper, -lower])[keep]


#: One knot of the path as ``(lam, beta)``.
Knot = tuple[float, NDArray[np.float64]]

#: The relaxation of a homogeneous row, relative to the scale ``||X^T y||_inf / max_j (X^T X)_jj``
#: of a coefficient: large enough that the CLA starts cleanly, small enough that the relaxed
#: path takes the exact path's sequence of active sets.
RELAXATION = 1e-6

#: The largest optimality violation, relative to ``||X^T y||_inf``, at which an
#: extrapolated path is accepted.
CERTIFIED = 1e-9


def _trace(
    sigma: NDArray[np.float64],
    q: NDArray[np.float64],
    legs: NDArray[np.float64],
    cap: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    g: NDArray[np.float64] | None,
    h: NDArray[np.float64] | None,
) -> list[Knot]:
    """Run the penalty-form CLA and return its knots, from the first one down to ``lam = 0``."""
    k = legs.shape[1]
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
            q=q,
        )
    except ValueError as err:
        msg = f"the bounded LASSO could not be traced through the CLA: {err}"
        # Keep the error's kind (infeasible, degenerate, ...) when the CLA classified it.
        raise (type(err) if isinstance(err, CLAError) else ValueError)(msg) from err
    # The lam = inf vertex is the first knot again, held still until a coefficient enters;
    # the trace always ends with a finite one, the constrained least-squares fit at lam = 0.
    return [(float(tp.lamb), legs @ tp.weights) for tp in cla.turning_points if np.isfinite(tp.lamb)]


def kkt_violation(
    gram: NDArray[np.float64],
    xty: NDArray[np.float64],
    lam: float,
    beta: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    g: NDArray[np.float64],
    h: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> float:
    """Largest violation of the LASSO's optimality conditions at ``beta``, relative to ``||X^T y||_inf``.

    With ``r = X^T X beta - X^T y + A^T nu + G_R^T eta`` over the rows ``R`` of
    ``G beta <= h`` that are active, a nonzero coefficient strictly inside its bounds needs
    ``r_j = -lam sign(beta_j)``, one at a cap ``u_j > 0`` needs ``r_j <= -lam``, one at a
    floor ``l_j < 0`` needs ``r_j >= lam``, and a zero one needs ``-lam <= r_j <= lam``
    less the side a sign bound takes away; ``eta >= 0``. A linear program picks ``nu`` and
    ``eta`` to make the worst violation as small as it can, which is zero exactly at the
    optimum, and the violation is then evaluated again exactly at the multipliers it found,
    so the LP's own tolerance cannot hide one. Primal feasibility is included.

    Args:
        gram: ``X^T X`` as a dense ``(n, n)`` array.
        xty: ``X^T y``.
        lam: The penalty.
        beta: The coefficients to certify.
        a: Equality rows ``(m, n)``; ``m`` may be zero.
        b: Their right-hand side.
        g: Inequality rows ``(p, n)``; ``p`` may be zero.
        h: Their right-hand side.
        lower: Lower bounds.
        upper: Upper bounds.

    Returns:
        The violation, relative to ``||X^T y||_inf`` (absolute when ``X^T y = 0``).
    """
    tol = 1e-9 * max(1.0, float(np.abs(beta).max()))
    zero = np.abs(beta) <= tol
    at_up, at_lo = ~zero & (beta >= upper - tol), ~zero & (beta <= lower + tol)
    inner = ~zero & ~at_up & ~at_lo
    rows = np.vstack([a, g[g @ beta >= h - tol * max(1.0, float(np.abs(h).max(initial=1.0)))]])
    m, k = a.shape[0], rows.shape[0]
    r0 = gram @ beta - xty
    lam_sign = lam * np.sign(beta)
    full = np.full_like(beta, lam)
    # Each condition is  side * r_j + offset <= worst, over the coordinates in its mask.
    parts = [
        (inner, 1.0, lam_sign),
        (inner, -1.0, -lam_sign),
        (at_up, 1.0, full),
        (at_lo, -1.0, full),
        (zero & (lower < 0), 1.0, -full),
        (zero & (upper > 0), -1.0, -full),
    ]
    coord = np.concatenate([np.flatnonzero(mask) for mask, _, _ in parts]).astype(int)
    side = np.concatenate([np.full(int(mask.sum()), sgn) for mask, sgn, _ in parts])
    offset = np.concatenate([off[mask] for mask, _, off in parts])
    lp = linprog(
        np.append(np.zeros(k), 1.0),
        A_ub=np.hstack([side[:, None] * rows.T[coord], -np.ones((len(coord), 1))]),
        b_ub=-(side * r0[coord] + offset),
        bounds=[(None, None)] * m + [(0.0, None)] * (k - m + 1),
        method="highs",
    )
    mult = np.concatenate([lp.x[:m], np.maximum(lp.x[m:k], 0.0)])
    worst = float(np.max(side * (r0 + rows.T @ mult)[coord] + offset, initial=0.0))
    feasibility = np.concatenate(
        [
            np.abs(a @ beta - b),
            np.maximum(g @ beta - h, 0.0),
            np.maximum(beta - upper, 0.0),
            np.maximum(lower - beta, 0.0),
        ]
    )
    # With X^T y = 0 there is no scale to be relative to; the violation is then absolute.
    return max(worst, float(feasibility.max(initial=0.0))) / (float(np.abs(xty).max()) or 1.0)


def _relaxation_direction(
    a: NDArray[np.float64],
    g: NDArray[np.float64],
    h: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """A generic point ``d`` whose image ``A d`` relaxes ``A beta = 0`` (module docstring).

    Each coefficient gets a pseudo-random magnitude in ``[0.5, 1.5]`` and a sign its bounds
    allow, so ``A d`` is generic and the relaxed start a unique vertex. ``d`` is then scaled
    so that ``G_r d <= 1/2`` on the homogeneous rows, which keeps ``eps d`` strictly inside
    their relaxation ``g_r beta <= eps``; every other constraint has room of order one.
    """
    rng = np.random.default_rng(0)
    n = a.shape[1]
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    sign = np.where(upper > 0, np.where(lower < 0, sign, 1.0), np.where(lower < 0, -1.0, 0.0))
    d: NDArray[np.float64] = sign * rng.uniform(0.5, 1.5, n)
    reach = float(np.max(g[h == 0] @ d, initial=0.0))
    return d * min(1.0, 0.5 / reach) if reach > 0 else d


def _resolve_homogeneous(
    gram: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xty: NDArray[np.float64],
    legs: NDArray[np.float64],
    cap: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    g: NDArray[np.float64],
    h: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> list[Knot]:
    """The exact knots with homogeneous rows ``h_r = 0`` or ``A beta = 0``, from two relaxed traces.

    Raises:
        DegenerateProblemError: If the two relaxed traces take different sequences, or the
            extrapolated path fails its optimality certificate.
    """
    h_dir = (h == 0).astype(np.float64)
    b_dir = a @ _relaxation_direction(a, g, h, lower, upper) if a.shape[0] and not b.any() else np.zeros_like(b)
    eps = RELAXATION * float(np.abs(xty).max()) / max(float(np.diag(gram).max()), 1e-300)
    q = legs.T @ xty
    rows = (g, h) if g.shape[0] else (None, None)

    def relaxed(t: float) -> list[Knot]:
        return _trace(sigma, q, legs, cap, a, b + t * b_dir, rows[0], None if rows[1] is None else h + t * h_dir)

    once, twice = relaxed(eps), relaxed(2 * eps)
    if len(once) != len(twice):
        msg = (
            f"the homogeneous rows could not be resolved: relaxed by {eps:.1e} and {2 * eps:.1e} they "
            f"trace {len(once)} and {len(twice)} knots, so the sequence of active sets depends on the relaxation"
        )
        raise DegenerateProblemError(msg)
    knots: list[Knot] = [(max(2 * l1 - l2, 0.0), 2 * b1 - b2) for (l1, b1), (l2, b2) in zip(once, twice, strict=True)]
    # A segment the relaxed path needed to meet the rows one by one has no length at h_r = 0.
    floor = 1e-12 * knots[0][0]
    kept = [knots[0]]
    for lam, beta in knots[1:]:
        if kept[-1][0] - lam > floor:
            kept.append((lam, beta))
    # The path ends at the least-squares fit, lam = 0, whichever of the coinciding knots was kept.
    kept[-1] = knots[-1] if kept[-1][0] - knots[-1][0] <= floor else kept[-1]
    points = kept + [(0.5 * (l1 + l2), 0.5 * (b1 + b2)) for (l1, b1), (l2, b2) in itertools.pairwise(kept)]
    worst = max(kkt_violation(gram, xty, lam, beta, a, b, g, h, lower, upper) for lam, beta in points)
    if worst > CERTIFIED:
        msg = (
            f"the homogeneous rows could not be resolved: the extrapolated path violates the "
            f"optimality conditions by {worst:.1e} (relative), above {CERTIFIED:g}"
        )
        raise DegenerateProblemError(msg)
    return kept


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
        h: Their right-hand side ``(p,)``, nonnegative; a zero entry is a homogeneous row.

    Returns:
        The breakpoints ``(lam, beta, active)`` from the first knot down to the
        constrained least-squares fit at ``lam = 0``.

    Raises:
        InfeasibleProblemError: If the bounds fix every coefficient at zero and
            ``b`` is nonzero.
        DegenerateProblemError: If homogeneous rows cannot be resolved (see the module
            docstring).
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
    homogeneous_rows = g is not None and h is not None and bool(np.any(h == 0))
    if homogeneous_rows or (a.shape[0] and not b.any()):
        gram = np.column_stack([quad.matvec(e) for e in np.eye(n)])
        g_all = np.zeros((0, n)) if g is None else g
        h_all = np.zeros(0) if h is None else h
        knots = _resolve_homogeneous(gram, sigma, xty, legs, cap, a, b, g_all, h_all, lower, upper)
    else:
        knots = _trace(sigma, legs.T @ xty, legs, cap, a, b, g, h)
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

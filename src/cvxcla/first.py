"""First turning point computation for the Critical Line Algorithm.

This module provides functions to compute the first turning point on the efficient frontier,
which is the portfolio with the highest expected return that satisfies the constraints.
Two implementations are provided: a direct algorithm and a linear programming approach.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog  # type: ignore[import-untyped]

from .errors import DegenerateProblemError, InfeasibleProblemError, NumericalError
from .operators import orthonormal_rows
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

    Raises:
        InfeasibleProblemError: If a lower bound exceeds its upper bound, or the
            bounds cannot sum to ``total``.
    """
    if np.any(lower_bounds > upper_bounds):
        msg = "Lower bounds must be less than or equal to upper bounds"
        raise InfeasibleProblemError(msg)

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
        raise InfeasibleProblemError(msg)

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

    A degenerate vertex (every weight on a bound, or a duplicated active row) is
    resolved from the set of the linear program's optimal duals, as the greedy fill
    of :func:`init_algo` resolves it for the budget: see :func:`classify_vertex`.

    Raises:
        InfeasibleProblemError: If the linear program is infeasible.
        DegenerateProblemError: If it is unbounded, or its vertex is degenerate
            beyond what its duals resolve: no optimal dual solution yields a free
            set that spans the equality rows together with the active inequality rows.
        NumericalError: If the linear program fails for another reason.
    """
    g = np.zeros((0, mean.shape[0])) if g is None else np.asarray(g, dtype=np.float64)
    h = np.zeros(0) if h is None else np.asarray(h, dtype=np.float64)

    weights, gap, eta = _solve_max_return_lp(mean, lower_bounds, upper_bounds, a, b, g, h)
    mean = np.asarray(mean, dtype=np.float64)
    # Duals carry the units of mu, so their zero test is relative to its scale.
    dual_tol = float(np.sqrt(np.finfo(np.float64).eps)) * max(float(np.max(np.abs(mean), initial=0.0)), 1e-300)
    return classify_vertex(
        weights,
        lower_bounds,
        upper_bounds,
        a,
        g,
        h,
        tol,
        duals=VertexDuals(mean=mean, gap=gap, ineq=eta, tol=dual_tol),
    )


def first_turning_point(
    mean: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    g: NDArray[np.float64],
    h: NDArray[np.float64],
    tol: float,
) -> TurningPoint:
    """Calculate the first turning point on the efficient frontier.

    The first turning point is the maximum-return vertex of the feasible
    polytope. For the all-ones budget constraint with no inequality rows and finite
    bounds it is found by the greedy fill of :func:`init_algo`; for a general equality system
    ``A w = b`` or any ``G w <= h`` it is found by solving the linear program
    in :func:`first_vertex_lp`, which also reports the initially-active rows.

    Args:
        mean: Vector of expected returns.
        lower_bounds: Lower box bounds.
        upper_bounds: Upper box bounds.
        a: Equality-constraint matrix ``A`` of ``A w = b``.
        b: Equality-constraint right-hand side ``b``.
        g: Inequality-constraint matrix ``G`` of ``G w <= h`` (``(p, n)``).
        h: Inequality-constraint right-hand side ``h`` (length ``p``).
        tol: Tolerance for the linear-programming vertex classification.

    Returns:
        A TurningPoint object representing the first point on the efficient frontier.
    """
    # The greedy fill needs finite bounds (it starts every weight at its lower bound);
    # infinite ones go to the linear program, which reports an unbounded problem.
    finite = bool(np.all(np.isfinite(lower_bounds)) and np.all(np.isfinite(upper_bounds)))
    if g.shape[0] == 0 and a.shape[0] == 1 and np.allclose(a, 1.0) and finite:
        return init_algo(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds, total=float(b[0]))
    return first_vertex_lp(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds, a=a, b=b, tol=tol, g=g, h=h)


class VertexDuals(NamedTuple):
    """An optimal dual solution of the maximum-return linear program.

    The duals satisfy ``mean = A^T nu + G^T eta + gap``: ``eta >= 0`` holds the
    inequality multipliers and ``gap`` the reduced costs of the box bounds (``<= 0``
    on a lower bound, ``>= 0`` on an upper bound, ``0`` off them). The equality
    multipliers ``nu`` are not carried: they follow from the rest.

    Attributes:
        mean: The expected returns (the linear program's objective).
        gap: The reduced cost of each variable, in the sign above.
        ineq: The inequality multipliers ``eta`` (length ``p``).
        tol: Zero tolerance for the duals, in the units of ``mean``.
    """

    mean: NDArray[np.float64]
    gap: NDArray[np.float64]
    ineq: NDArray[np.float64]
    tol: float


def classify_vertex(
    weights: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    a: NDArray[np.float64],
    g: NDArray[np.float64],
    h: NDArray[np.float64],
    tol: float,
    duals: VertexDuals | None = None,
) -> TurningPoint:
    """Read the free set and the active rows off a maximum-return vertex.

    An asset is free when it sits strictly inside its box (by more than ``tol``)
    and an inequality row is active when it is tight to ``tol``. The vertex is
    then checked for degeneracy (see :func:`_reject_degenerate_vertex`).

    With the linear program's ``duals`` a degenerate vertex -- one whose interior
    assets do not span the tight rows -- is resolved first. Its optimal duals are
    not unique, and the one the solver happens to report need not show which
    partition works, so the partition is chosen from the whole optimal dual set:
    :func:`_dual_vertex` moves the reported duals to a vertex of that set, and
    :func:`_vertex_partition` reads a partition off it -- the tight rows with a
    zero multiplier stay inactive, and assets on a bound with a zero reduced cost
    are freed, in order of decreasing return, until the free block of the active
    rows is square and nonsingular. The segment then reproduces the vertex, and the
    multipliers it implies are those of the dual vertex, which carry the right
    signs. This generalises the greedy fill of :func:`init_algo`, which marks its
    last asset free even when it lands on a bound.

    Args:
        weights: The vertex weights.
        lower_bounds: Lower box bounds.
        upper_bounds: Upper box bounds.
        a: Equality-constraint matrix (``m x n``).
        g: Inequality-constraint matrix (``p x n``); empty ``(0, n)`` when none.
        h: Inequality-constraint right-hand side (length ``p``).
        tol: Classification tolerance.
        duals: An optimal dual solution of the linear program, or ``None`` to
            classify without resolving degeneracy.

    Returns:
        The vertex as a :class:`TurningPoint` carrying its active rows.

    Raises:
        DegenerateProblemError: If the vertex is degenerate and cannot be resolved.
    """
    free = (weights > lower_bounds + tol) & (weights < upper_bounds - tol)
    active_ineq = (g @ weights >= h - tol) if g.shape[0] else np.zeros(0, dtype=bool)

    if duals is not None and not _spans(np.vstack([a, g[active_ineq]]), free):
        free, active_ineq = _resolve_vertex(weights, lower_bounds, upper_bounds, a, g, tol, free, active_ineq, duals)

    _reject_degenerate_vertex(a, g, free, active_ineq)
    return TurningPoint(free=free, weights=weights, active_ineq=active_ineq)


def _spans(c: NDArray[np.float64], free: NDArray[np.bool_]) -> bool:
    """Whether the free block ``c[:, free]`` has full row rank.

    ``rank(C[:, free]) <= min(rows, n_free)``, so fewer free assets than rows fails
    by itself. Testing this first also keeps ``matrix_rank`` off a zero-column
    block, whose empty singular-value reduction raises on numpy 2.0.

    Args:
        c: The active constraint rows (``[A ; G_active]``).
        free: Boolean mask of the free assets.

    Returns:
        ``True`` when the free assets span the rows.
    """
    rows = c.shape[0]
    return bool(int(np.count_nonzero(free)) >= rows and int(np.linalg.matrix_rank(c[:, free])) == rows)


def _resolve_vertex(
    weights: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    a: NDArray[np.float64],
    g: NDArray[np.float64],
    tol: float,
    free: NDArray[np.bool_],
    tight: NDArray[np.bool_],
    duals: VertexDuals,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Choose the free set and the active rows of a degenerate vertex from its optimal duals.

    Args:
        weights: The vertex weights.
        lower_bounds: Lower box bounds.
        upper_bounds: Upper box bounds.
        a: Equality-constraint matrix (``m x n``).
        g: Inequality-constraint matrix (``p x n``).
        tol: Classification tolerance.
        free: Boolean mask of the assets strictly inside their box.
        tight: Boolean mask of the tight rows of ``g``.
        duals: An optimal dual solution of the linear program.

    Returns:
        ``(free, active_ineq)``: the completed free set and the active rows of ``g``.
    """
    m = a.shape[0]
    rows = np.vstack([a, g[tight]])
    # The sign the reduced cost keeps on a bound: +1 on an upper bound, -1 on a
    # lower one, and none for an asset pinned at both (it is never freed).
    sign = (~free & (weights >= upper_bounds - tol)).astype(np.float64) - (
        ~free & (weights <= lower_bounds + tol)
    ).astype(np.float64)
    # The equality multipliers follow from the rest; the rows are independent.
    eta = np.maximum(duals.ineq[tight], 0.0)
    nu = np.linalg.lstsq(a.T, duals.mean - duals.gap - g[tight].T @ eta, rcond=None)[0]
    y = _dual_vertex(rows, duals.mean, free, sign, m, np.r_[nu, eta], duals.tol)
    gap = duals.mean - rows.T @ y
    freed, released = _vertex_partition(
        rows,
        free,
        (sign != 0) & (sign * gap <= duals.tol),
        (np.arange(rows.shape[0]) >= m) & (y <= duals.tol),
        -duals.mean,
    )
    active_ineq = tight.copy()
    active_ineq[tight] = ~released[m:]
    return free | freed, active_ineq


def _dual_vertex(
    rows: NDArray[np.float64],
    mean: NDArray[np.float64],
    free: NDArray[np.bool_],
    sign: NDArray[np.float64],
    m: int,
    start: NDArray[np.float64],
    tol: float,
) -> NDArray[np.float64]:
    """Move an optimal dual solution to a vertex of the optimal dual set.

    The multipliers ``y`` of the rows ``C = [A ; G_tight]`` are optimal when
    ``gap = mean - C^T y`` vanishes on the free assets, has the ``sign`` of each
    bound on the others (``sign * gap >= 0``; no condition where ``sign`` is zero)
    and the inequality multipliers ``y[m:]`` are non-negative. While the
    constraints tight at ``y`` do not determine it, ``y`` moves along a direction
    that keeps them tight until another one becomes tight; each move adds a
    constraint independent of the tight ones, so at most ``len(y)`` moves reach a
    vertex. The solver's own dual need not be one: at a degenerate vertex it can
    sit inside a face of the optimal set.

    Args:
        rows: The tight constraint rows ``C`` (``k x n``), equalities first.
        mean: Vector of expected returns.
        free: Boolean mask of the assets strictly inside their box.
        sign: ``+1`` on an upper bound, ``-1`` on a lower bound, ``0`` otherwise.
        m: Number of equality rows (the leading rows of ``C``).
        start: An optimal dual solution (length ``k``).
        tol: Zero tolerance for the duals.

    Returns:
        The multipliers at a vertex of the optimal dual set.

    Raises:
        DegenerateProblemError: If the optimal dual set has no vertex (the assets
            that can leave their bounds do not determine the multipliers).
    """
    k = rows.shape[0]
    y = start.copy()
    ineq = np.arange(k) >= m
    for _ in range(k + 1):
        gap = mean - rows.T @ y
        tight = np.hstack([rows[:, free | ((sign != 0) & (sign * gap <= tol))], np.eye(k)[:, ineq & (y <= tol)]])
        # A zero row keeps the factorisation defined when nothing is tight yet.
        _, sv, vt = np.linalg.svd(np.vstack([tight.T, np.zeros(k)]))
        rank = int(np.count_nonzero(sv > k * np.finfo(np.float64).eps * max(float(sv[0]), 1.0)))
        if rank == k:
            return y
        step = _dual_step(rows, gap, y, sign, ineq, vt[rank])
        if step is None:
            break
        y = y + step
    msg = (
        "The maximum-return vertex is degenerate and its optimal duals have no vertex: the assets "
        "that can leave their bounds do not determine the multipliers of the active rows. Perturb "
        "the bounds or the constraints so the maximum-return vertex is non-degenerate."
    )
    raise DegenerateProblemError(msg)


def _dual_step(
    rows: NDArray[np.float64],
    gap: NDArray[np.float64],
    y: NDArray[np.float64],
    sign: NDArray[np.float64],
    ineq: NDArray[np.bool_],
    direction: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """The move along ``+direction`` or ``-direction`` until a further dual constraint is tight.

    Along ``y + t d`` the constraint value ``sign_i * gap_i`` of asset ``i`` falls at
    rate ``sign_i c_i^T d`` and the multiplier of an inequality row at rate
    ``-d_j``; the first value to reach zero ends the move. ``+d`` is tried first.

    Args:
        rows: The tight constraint rows ``C`` (``k x n``).
        gap: The current reduced costs ``mean - C^T y``.
        y: The current multipliers.
        sign: ``+1`` on an upper bound, ``-1`` on a lower bound, ``0`` otherwise.
        ineq: Boolean mask of the inequality rows among the ``k``.
        direction: A unit direction that keeps every tight constraint tight.

    Returns:
        The step ``t d``, or ``None`` when neither direction meets a constraint.
    """
    floor = float(np.sqrt(np.finfo(np.float64).eps))
    scale = np.maximum(np.linalg.norm(rows, axis=0), 1.0)
    for d in (direction, -direction):
        rate_asset = sign * (rows.T @ d)
        rate_row = np.where(ineq, -d, 0.0)
        moving = rate_asset > floor * scale
        falling = rate_row > floor
        steps = np.r_[(sign * gap)[moving] / rate_asset[moving], y[falling] / rate_row[falling]]
        if steps.size:
            return float(np.min(np.maximum(steps, 0.0))) * d
    return None


def _vertex_partition(
    rows: NDArray[np.float64],
    free: NDArray[np.bool_],
    candidates: NDArray[np.bool_],
    releasable: NDArray[np.bool_],
    key: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Read a partition off a vertex of the optimal dual set.

    The constraints tight at a dual vertex span the multiplier space, so a basis
    can be drawn from them: the free assets' columns of ``C`` first, then the unit
    vectors of the zero-multiplier inequality rows, then the columns of the
    zero-reduced-cost assets in order of ``key``, each kept while it raises the
    rank. Rows whose unit vector is in the basis are released (inactive), assets
    whose column is in the basis are freed, and the free block of the remaining
    rows is square and nonsingular.

    Args:
        rows: The tight constraint rows ``C`` (``k x n``).
        free: Boolean mask of the assets strictly inside their box.
        candidates: Assets on a bound with zero reduced cost.
        releasable: Rows (of the ``k``) that are inequalities with a zero multiplier.
        key: Ordering key; smaller values are freed first.

    Returns:
        ``(freed, released)``: the assets to free and the rows to release.
    """
    k = rows.shape[0]
    order = np.flatnonzero(candidates)[np.argsort(key[candidates], kind="stable")]
    basis = rows[:, free]
    rank = int(np.linalg.matrix_rank(basis)) if basis.shape[1] else 0
    freed = np.zeros_like(free)
    released = np.zeros(k, dtype=bool)
    columns = [(np.eye(k)[:, j], released, j) for j in np.flatnonzero(releasable)]
    columns += [(rows[:, i], freed, i) for i in order]
    for column, chosen, index in columns:
        trial = np.column_stack([basis, column])
        if int(np.linalg.matrix_rank(trial)) > rank:
            basis, rank = trial, rank + 1
            chosen[index] = True
    return freed, released


# Equality rows whose row-normalised reciprocal condition number is below this are
# nearly dependent enough that the linear program enforces the direction they carry
# only loosely; they are restated in an orthonormal basis first. Above it the rows go
# in as given, so a well-conditioned problem hands HiGHS the system the user wrote.
_LP_ROW_RCOND = 1e-6  # pragma: no mutate


def _lp_equalities(a: NDArray[np.float64], b: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The equality system handed to the linear program: ``(a, b)``, or orthonormal rows if nearly dependent.

    Two nearly parallel rows (``1^T w = 1`` and ``(1 + eps v)^T w = 1``) carry the
    constraint ``v^T w = 0`` only at scale ``eps``, and an absolute feasibility
    tolerance enforces it to ``tol / eps``; from ``eps ~ 1e-7`` the vertex comes back
    infeasible for the hidden row and is misread as degenerate. The orthonormal form
    ``Q^T w = R^{-T} b`` (:func:`cvxcla.operators.orthonormal_rows`) has the same
    feasible set with every direction at unit scale.

    Args:
        a: Equality-constraint matrix (``m x n``).
        b: Equality-constraint right-hand side (length ``m``).

    Returns:
        The rows and right-hand side to pass to ``linprog``.
    """
    if a.shape[0] < 2:
        return a, b
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    sv = np.linalg.svd(a / np.where(norms > 0, norms, 1.0), compute_uv=False)
    if sv[-1] >= _LP_ROW_RCOND * sv[0]:
        return a, b
    return orthonormal_rows(a, b)


def _solve_max_return_lp(
    mean: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    g: NDArray[np.float64],
    h: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve the maximum-return linear program and return its vertex weights and duals.

    ``maximize mean @ w`` (as ``minimize -mean @ w``) subject to ``A w = b``,
    ``G w <= h`` and the box bounds, via HiGHS. The inequality rows are passed
    only when ``g`` is non-empty.

    Args:
        mean: Vector of expected returns.
        lower_bounds: Lower box bounds.
        upper_bounds: Upper box bounds.
        a: Equality-constraint matrix (``m x n``).
        b: Equality-constraint right-hand side (length ``m``).
        g: Inequality-constraint matrix (``p x n``); empty ``(0, n)`` when none.
        h: Inequality-constraint right-hand side (length ``p``).

    Returns:
        ``(w, gap, eta)``: the vertex weights, the reduced cost of each variable
        (the bound marginals, zero for a basic variable) and the inequality
        multipliers, in the signs of :class:`VertexDuals`.

    Raises:
        InfeasibleProblemError: If the linear program is infeasible.
        DegenerateProblemError: If it is unbounded (no maximum-return vertex).
        NumericalError: If HiGHS stops for another reason.
    """
    has_ineq = g.shape[0] > 0
    a_eq, b_eq = _lp_equalities(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    result = linprog(
        c=-np.asarray(mean, dtype=np.float64),
        A_eq=a_eq,
        b_eq=b_eq,
        A_ub=g if has_ineq else None,
        b_ub=h if has_ineq else None,
        bounds=list(zip(lower_bounds, upper_bounds, strict=True)),
        method="highs",
    )
    if not result.success:
        msg = f"Could not find a maximum-return vertex (linear program: {result.message})"
        error = {2: InfeasibleProblemError, 3: DegenerateProblemError}.get(result.status, NumericalError)
        raise error(msg)
    # HiGHS reports the marginals of the minimisation, ``-mean = A^T y + G^T z + s``;
    # the gap and the inequality multipliers of the maximisation are their negatives.
    gap = -(np.asarray(result.lower.marginals, dtype=np.float64) + np.asarray(result.upper.marginals, dtype=np.float64))
    eta = -np.asarray(result.ineqlin.marginals, dtype=np.float64) if has_ineq else np.zeros(0)
    return np.asarray(result.x, dtype=np.float64), gap, eta


def _reject_degenerate_vertex(
    a: NDArray[np.float64],
    g: NDArray[np.float64],
    free: NDArray[np.bool_],
    active_ineq: NDArray[np.bool_],
) -> None:
    """Decline a maximum-return vertex whose free set cannot span the active rows.

    The free set must span the equality rows together with the active inequality
    rows: ``C = [A ; G_active]`` restricted to the free assets must have full row
    rank, or the reduced KKT solve is singular. A degenerate maximum-return
    vertex (a basic asset pinned on a bound) violates this; decline it with an
    actionable diagnosis instead of letting it surface as an opaque "Singular
    matrix" error downstream.

    Args:
        a: Equality-constraint matrix (``m x n``).
        g: Inequality-constraint matrix (``p x n``); empty ``(0, n)`` when none.
        free: Boolean mask of the assets strictly inside their box bounds.
        active_ineq: Boolean mask of the tight (active) inequality rows.

    Raises:
        DegenerateProblemError: If the free set does not span the active equality
            and inequality rows.
    """
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
            "singular, even after freeing the vertex's degenerate basic assets. Perturb the bounds or "
            "the constraints so the maximum-return vertex is non-degenerate."
        )
        raise DegenerateProblemError(msg)


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

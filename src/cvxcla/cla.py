"""Markowitz implementation of the Critical Line Algorithm.

This module provides the CLA class, which implements the Critical Line Algorithm
as described by Harry Markowitz and colleagues. The algorithm computes the entire
efficient frontier by finding all turning points, which are the points where the
set of assets at their bounds changes.
"""

import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from cvx.linalg import AffineProjection
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .builder import ProblemBuilder

from .first import first_vertex_lp, init_algo
from .operators import DenseCovariance, QuadraticForm, bordered_solve, cross
from .pathtracer import InequalityConstrained, trace
from .types import Frontier, FrontierPoint, TurningPoint

# A genuinely rank-deficient free block has a reciprocal condition number at
# round-off level (~1e-16); a well-posed or merely near-degenerate block sits
# many orders above it (>= ~1e-4 across the degeneracy sweep in
# experiments/degeneracy_boundary.py). The 1e-12 cut sits in the wide gap between
# the two and is the conventional numerical-singularity scale.
_RCOND_FLOOR = 1e-12  # pragma: no mutate


class _Segment(NamedTuple):
    """The affine critical-line segment valid at one turning point.

    Bundles the affine path ``w(lam) = r_alpha + lam * r_beta``, the multiplier
    gradients ``gamma``/``delta`` that drive the leave-a-bound events, and the
    active-set masks the event scan needs. This is what ``CLA.segment`` returns
    to the generic path tracer.

    For general inequality constraints ``G w <= h`` the segment also carries the
    affine inequality multipliers ``eta_alpha + lam * eta_beta`` (one entry per
    inequality row; meaningful for *active* rows, which release when the
    multiplier crosses zero) and the active-row mask ``active_ineq``. The slacks
    that drive an *inactive* row becoming active are recomputed from
    ``r_alpha``/``r_beta`` directly in ``_ineq_event_ratios``.
    """

    r_alpha: NDArray[np.float64]
    r_beta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    delta: NDArray[np.float64]
    at_upper: NDArray[np.bool_]
    at_lower: NDArray[np.bool_]
    free_in: NDArray[np.bool_]
    active_ineq: NDArray[np.bool_]
    eta_alpha: NDArray[np.float64]
    eta_beta: NDArray[np.float64]


@dataclass(frozen=True)
class CLA(InequalityConstrained):
    """Critical Line Algorithm implementation based on Markowitz's approach.

    This class implements the Critical Line Algorithm as described by Harry Markowitz
    and colleagues. It computes the entire efficient frontier by finding all turning
    points, which are the points where the set of assets at their bounds changes.

    The algorithm starts with the first turning point (the portfolio with the highest
    expected return) and then iteratively computes the next turning point with a lower
    expected return until it reaches the minimum variance portfolio.

    Attributes:
        mean: Vector of expected returns for each asset.
        covariance: Covariance matrix of asset returns, either as a plain
            ``numpy`` array or as a ``CovarianceOperator`` backend
            (see ``cvxcla.operators``).
        lower_bounds: Vector of lower bounds for asset weights.
        upper_bounds: Vector of upper bounds for asset weights.
        a: Equality-constraint matrix ``A`` of ``A w = b`` (``m x n``). The
            canonical case is the single all-ones budget row (``sum(w) = b``),
            but an arbitrary equality system is supported: weighted single rows
            and ``m > 1`` rows (e.g. budget plus sector- or factor-neutrality).
            The all-ones budget (any right-hand side, including ``0`` for
            dollar-neutral) uses the greedy first vertex of
            :func:`cvxcla.first.init_algo`; a general ``A`` uses the
            linear-programming first vertex of
            :func:`cvxcla.first.first_vertex_lp`.
        b: Equality-constraint right-hand side ``b`` (length ``m``); ``[1]`` for
            the fully-invested budget, ``[0]`` for dollar-neutral, and so on.
        g: Optional inequality-constraint matrix ``G`` of ``G w <= h``
            (``p x n``), e.g. a group- or sector-exposure cap. ``None`` (the
            default) means no inequality rows, recovering the equality-only
            problem exactly. A ``>=`` constraint is expressed by negating both
            ``g`` and ``h``. Each *active* row (held at equality ``g_i w = h_i``)
            enters the reduced KKT system as an extra equality row, so the
            covariance is still touched only through the ``QuadraticForm``
            interface; box bounds remain a separate per-variable active set.
        h: Optional inequality-constraint right-hand side ``h`` (length ``p``).
        turning_points: List of turning points on the efficient frontier.
        tol: Tolerance for numerical calculations.
        logger: Logger instance for logging information and errors.

    """

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64] | QuadraticForm
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    g: NDArray[np.float64] | None = None
    h: NDArray[np.float64] | None = None
    turning_points: list[TurningPoint] = field(default_factory=list)
    tol: float = 1e-5  # pragma: no mutate
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    @classmethod
    def problem(cls, mean: NDArray[np.float64], covariance: NDArray[np.float64] | QuadraticForm) -> "ProblemBuilder":
        """Start a fluent :class:`cvxcla.builder.ProblemBuilder` for this problem.

        A readability convenience over the explicit constructor: chain
        ``.long_only()``/``.budget()``/``.equality()``/``.inequality()`` and finish
        with ``.trace()``. The builder maps one-to-one onto the constructor
        arguments and adds no modelling power.

        Args:
            mean: Vector of expected returns of length ``n``.
            covariance: Covariance matrix or ``QuadraticForm`` backend.

        Returns:
            A ``ProblemBuilder`` ready to accept constraints.
        """
        from .builder import ProblemBuilder

        return ProblemBuilder(mean, covariance)

    @cached_property
    def covariance_operator(self) -> QuadraticForm:
        """Return the covariance as a ``QuadraticForm`` backend.

        A plain ``numpy`` covariance matrix is wrapped in ``DenseCovariance``;
        an object already implementing the protocol is passed through. This is
        the single point where the input form is normalised.
        """
        if isinstance(self.covariance, QuadraticForm):
            return self.covariance
        return DenseCovariance(self.covariance)

    @property
    def dimension(self) -> int:
        """Number of assets ``n`` (the problem dimension for the path tracer)."""
        return len(self.mean)

    @cached_property
    def _free_blocks_well_conditioned(self) -> bool:
        """Whether every free-block solve along the trace is numerically safe.

        Decided once, up front. By Cauchy's interlacing theorem every principal
        submatrix of the symmetric PSD covariance is at least as well conditioned
        as the whole matrix -- deleting rows/columns cannot decrease the smallest
        eigenvalue nor increase the largest -- so the reciprocal condition number
        of any free block is ``>=`` that of the full covariance. Hence if the full
        covariance clears the singularity floor, no free block encountered along
        the trace can be singular, and the per-turning-point conditioning guard in
        :meth:`_emit` is provably never triggered. We then skip it, paying one
        conditioning test here instead of one at every turning point (the latter
        is a full eigendecomposition of the free block, as costly as the KKT solve,
        so it otherwise dominates the trace). The up-front test computes the
        reciprocal condition number of the full covariance once, via
        :meth:`~cvx.linalg.SymmetricOperator.rcond_free`.

        When the full covariance is itself near-singular (for example a sample
        covariance from fewer observations than assets) this is ``False`` and the
        per-step guard in :meth:`_emit` runs unchanged, preserving the degeneracy
        diagnosis exactly.
        """
        full = np.arange(self.dimension)
        return self.covariance_operator.rcond_free(full) >= _RCOND_FLOOR

    def __post_init__(self) -> None:
        """Initialize the CLA object and compute the efficient frontier.

        This method is automatically called after initialization. It computes
        the entire efficient frontier by finding all turning points, starting
        from the first turning point (highest expected return) and iteratively
        computing the next turning point with a lower expected return until
        it reaches the minimum variance portfolio.

        The actual walk is driven by the generic ``cvxcla.pathtracer.trace``
        loop; this class supplies the portfolio-specific hooks (``begin``,
        ``segment``, ``event_matrix``, ``step``, ``finish``) it calls.

        The reduced KKT system at each turning point is solved by block
        elimination: a single multi-RHS solve against the free covariance block
        (via the covariance backend), covering the constraint columns and the
        alpha and beta systems together so ``Sigma_FF`` is factorised once, then a
        small Schur-complement solve ``A_F @ Sigma_FF^{-1} @ A_F.T`` over the
        equality (and active inequality) rows. The covariance only enters through
        the ``QuadraticForm`` interface, so structured backends (e.g.
        ``FactorCovariance``) never materialise an n x n matrix.

        Raises:
            RuntimeError: If all variables are blocked, which would make the
                          system of equations singular.
            ValueError: If the inequality matrix ``g`` and vector ``h`` have
                          mismatched or wrong shapes.

        """
        if self.g_matrix.shape[1] != self.dimension:
            msg = f"g must have {self.dimension} columns, got shape {self.g_matrix.shape}"
            raise ValueError(msg)
        if self.h_vector.shape[0] != self.g_matrix.shape[0]:
            msg = f"h must have {self.g_matrix.shape[0]} entries, got {self.h_vector.shape[0]}"
            raise ValueError(msg)
        trace(self)

    def begin(self) -> tuple[float, TurningPoint]:
        """Record the first turning point and start the trace at ``lambda = inf``.

        Returns:
            ``(inf, first_turning_point)``: the starting lambda bound and the
            initial state for the path tracer.
        """
        first = self._first_turning_point()
        self._append(first)
        return np.inf, first

    def segment(self, state: TurningPoint) -> _Segment:
        """Solve the reduced KKT system for the critical-line segment at ``state``."""
        at_upper, at_lower, free_in, fixed_weights = self._active_set(state)
        r_alpha, r_beta, gamma, delta, eta_alpha, eta_beta = self._solve_kkt(free_in, fixed_weights, state.active_ineq)
        return _Segment(
            r_alpha, r_beta, gamma, delta, at_upper, at_lower, free_in, state.active_ineq, eta_alpha, eta_beta
        )

    def event_matrix(self, state: TurningPoint, segment: _Segment) -> NDArray[np.float64]:  # noqa: ARG002
        """Return the ``(n + p, 4)`` matrix of candidate critical lambdas for ``segment``.

        The first ``n`` rows are the box events (a free weight reaching a bound, a
        blocked multiplier changing sign); the trailing ``p`` rows are the
        inequality-row events (an inactive row's slack reaching zero, an active
        row's multiplier changing sign). The generic tracer treats the two blocks
        uniformly; ``step`` decodes a row index ``>= n`` as a row event.

        ``state`` is part of the uniform ``ParametricProblem`` signature; the CLA
        does not need it here because ``segment`` already bundles the active-set
        masks derived from it.
        """
        box = self._event_ratios(
            segment.r_alpha,
            segment.r_beta,
            segment.gamma,
            segment.delta,
            segment.free_in,
            segment.at_upper,
            segment.at_lower,
        )
        ineq = self._ineq_event_ratios(segment)
        return np.vstack([box, ineq])

    def step(self, state: TurningPoint, segment: _Segment, sec: int, direction: int, lam: float) -> TurningPoint:
        """Emit the turning point at ``lam`` after flipping the activity at ``sec``.

        ``sec < n`` is a box event on asset ``sec``: a "leaves a bound" event
        (``direction`` in {2, 3}) makes it free, a "moves to a bound" event
        (``direction`` in {0, 1}) blocks it. ``sec >= n`` is an inequality-row
        event on row ``sec - n``: ``direction == 0`` activates the row (its slack
        reached zero), ``direction == 1`` releases it (its multiplier reached
        zero). The weight vector is continuous across either event.
        """
        n = self.dimension
        free = state.free
        active_ineq = state.active_ineq
        if sec < n:
            free = free.copy()
            free[sec] = direction >= 2
        else:
            active_ineq = active_ineq.copy()
            active_ineq[sec - n] = direction == 0
        self._emit(lam, segment.r_alpha + lam * segment.r_beta, free, active_ineq)
        return self.turning_points[-1]

    def finish(self, state: TurningPoint, segment: _Segment) -> None:
        """Emit the minimum-variance endpoint at ``lambda = 0``."""
        self._emit(0.0, segment.r_alpha, state.free, state.active_ineq)

    def _active_set(
        self, last: TurningPoint
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]:
        """Identify the active set at ``last`` and the weights pinned to bounds.

        A blocked asset sitting (to tolerance) on a bound is held fixed there and
        excluded from the reduced KKT solve; every other asset is *in*. Returns
        the upper-bound mask, the lower-bound mask, the in-set mask, and the
        full-length vector of weights fixed at their bounds.

        Args:
            last: The most recent turning point, whose free set and weights define
                the active set.

        Returns:
            ``(at_upper, at_lower, free_in, fixed_weights)``.

        Raises:
            RuntimeError: If every asset is blocked, which makes the reduced
                system singular.
        """
        blocked = ~last.free
        if np.all(blocked):
            msg = "All variables cannot be blocked"
            raise RuntimeError(msg)

        at_upper = blocked & (np.abs(last.weights - self.upper_bounds) <= self.tol)  # pragma: no mutate
        at_lower = blocked & (np.abs(last.weights - self.lower_bounds) <= self.tol)  # pragma: no mutate
        free_in = ~(at_upper | at_lower)

        fixed_weights = np.zeros(len(self.mean))
        fixed_weights[at_upper] = self.upper_bounds[at_upper]
        fixed_weights[at_lower] = self.lower_bounds[at_lower]
        return at_upper, at_lower, free_in, fixed_weights

    def _solve_kkt(
        self,
        free_in: NDArray[np.bool_],
        fixed_weights: NDArray[np.float64],
        active_ineq: NDArray[np.bool_] | None = None,
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
            free_in: Boolean mask of the assets in the reduced solve.
            fixed_weights: Full-length weights of the assets held at their bounds.
            active_ineq: Boolean mask (length ``p``) of the active inequality rows;
                ``None`` means no inequality rows.

        Returns:
            ``(r_alpha, r_beta, gamma, delta, eta_alpha, eta_beta)``: the affine
            segment ``w(lam) = r_alpha + lam * r_beta``, the box-multiplier
            gradients ``gamma``/``delta`` that drive the leave-a-bound events, and
            the affine inequality multipliers ``eta_alpha + lam * eta_beta``
            (length ``p``, non-zero only on active rows) that drive the
            release-a-row events.
        """
        if active_ineq is None:
            active_ineq = np.zeros(self.g_matrix.shape[0], dtype=bool)
        m = self.a.shape[0]
        ns = len(self.mean)
        cov = self.covariance_operator
        out = ~free_in
        # Stack the active inequality rows beneath the equality rows; the active
        # rows are held at equality (g_i w = h_i), so C/d is the equality system
        # of the reduced QP at this vertex.
        c = np.vstack([self.a, self.g_matrix[active_ineq]])
        d = np.concatenate([self.b, self.h_vector[active_ineq]])
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
            self.mean[free_in],
            d - c[:, out] @ fixed_weights[out],
            np.zeros(c.shape[0]),
        )

        r_alpha = fixed_weights.copy()
        r_alpha[free_in] = x_alpha
        r_beta = np.zeros(ns)
        r_beta[free_in] = x_beta

        gamma = cov.matvec(r_alpha) + c.T @ nu_alpha
        delta = cov.matvec(r_beta) + c.T @ nu_beta - self.mean

        # The tail of the stacked multiplier is the inequality multiplier eta(lam)
        # = eta_alpha + lam eta_beta, scattered back to full length p (zero on the
        # inactive rows, which have no release event).
        eta_alpha = np.zeros(self.g_matrix.shape[0])
        eta_beta = np.zeros(self.g_matrix.shape[0])
        eta_alpha[active_ineq] = nu_alpha[m:]
        eta_beta[active_ineq] = nu_beta[m:]
        return r_alpha, r_beta, gamma, delta, eta_alpha, eta_beta

    def _event_ratios(
        self,
        r_alpha: NDArray[np.float64],
        r_beta: NDArray[np.float64],
        gamma: NDArray[np.float64],
        delta: NDArray[np.float64],
        free_in: NDArray[np.bool_],
        at_upper: NDArray[np.bool_],
        at_lower: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Critical lambda for every candidate event, as an ``(n, 4)`` matrix.

        Along the segment ``w(lam) = r_alpha + lam * r_beta`` a free weight can
        reach a box bound (columns 0/1, "moves to a bound") and a blocked weight's
        multiplier can change sign so it re-enters the free set (columns 2/3,
        "leaves a bound"). Entries with no event are ``-inf``.

        A free weight moves with even a tiny slope, so given a long enough lam
        range it still crosses a bound; filtering slopes at ``self.tol`` would
        miss such crossings and let weights drift out of bounds. Only slopes at
        floating-point noise level are excluded: below ``sqrt(machine epsilon)`` a
        slope is indistinguishable from solve noise, and the huge ratios it would
        produce only amplify rounding errors.

        Args:
            r_alpha: Segment intercept ``w(0)``.
            r_beta: Segment slope ``dw/dlam``.
            gamma: Multiplier gradient for the alpha system.
            delta: Multiplier gradient for the beta system.
            free_in: Mask of assets in the reduced solve.
            at_upper: Mask of assets blocked at their upper bound.
            at_lower: Mask of assets blocked at their lower bound.

        Returns:
            The ``(n, 4)`` matrix of critical lambdas.
        """
        ns = len(self.mean)
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
        l_mat[beta_down, 0] = (  # pragma: no mutate
            self.upper_bounds[beta_down] - r_alpha[beta_down]
        ) / r_beta[beta_down]
        l_mat[beta_up, 1] = (self.lower_bounds[beta_up] - r_alpha[beta_up]) / r_beta[beta_up]
        l_mat[delta_down, 2] = -gamma[delta_down] / delta[delta_down]  # pragma: no mutate
        l_mat[delta_up, 3] = -gamma[delta_up] / delta[delta_up]
        return l_mat

    def _ineq_event_ratios(self, segment: _Segment) -> NDArray[np.float64]:
        """Critical lambda for every inequality-row event, as a ``(p, 4)`` matrix.

        The row analogue of :meth:`_event_ratios`. Along the segment an *inactive*
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
            segment: The current segment, carrying the affine weights, the
                inequality multipliers, and the active-row mask.

        Returns:
            The ``(p, 4)`` matrix of critical lambdas.
        """
        p = self.g_matrix.shape[0]
        l_mat = np.full((p, 4), -np.inf)  # pragma: no mutate
        if p == 0:
            return l_mat

        eps = np.sqrt(np.finfo(np.float64).eps)
        active = segment.active_ineq
        inactive = ~active

        # Enter: an inactive row's slack rises to zero. The slope/intercept split
        # comes straight from the affine weights; the slope sign mirrors the box
        # "moves to a bound" event (decreasing lam must raise the slack).
        s_alpha = self.g_matrix @ segment.r_alpha - self.h_vector
        s_beta = self.g_matrix @ segment.r_beta
        enter = inactive & (s_beta < -eps)  # pragma: no mutate
        l_mat[enter, 0] = -s_alpha[enter] / s_beta[enter]

        # Release: an active row's non-negative multiplier falls back to zero,
        # the row analogue of a blocked multiplier changing sign.
        release = active & (segment.eta_beta > +eps)  # pragma: no mutate
        l_mat[release, 1] = -segment.eta_alpha[release] / segment.eta_beta[release]
        return l_mat

    def __len__(self) -> int:
        """Get the number of turning points in the efficient frontier.

        Returns:
            The number of turning points currently stored in the object.

        """
        return len(self.turning_points)

    def _first_turning_point(self) -> TurningPoint:
        """Calculate the first turning point on the efficient frontier.

        The first turning point is the maximum-return vertex of the feasible
        polytope. For the all-ones budget constraint with no inequality rows it is
        found by the greedy fill of ``init_algo``; for a general equality system
        ``A w = b`` or any ``G w <= h`` it is found by solving the linear program
        in ``first_vertex_lp``, which also reports the initially-active rows.

        Returns:
            A TurningPoint object representing the first point on the efficient frontier.

        """
        if self.g_matrix.shape[0] == 0 and self.a.shape[0] == 1 and np.allclose(self.a, 1.0):
            return init_algo(
                mean=self.mean,
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
                total=float(self.b[0]),
            )
        return first_vertex_lp(
            mean=self.mean,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            a=self.a,
            b=self.b,
            tol=self.tol,
            g=self.g_matrix,
            h=self.h_vector,
        )

    def _append(self, tp: TurningPoint, tol: float | None = None) -> None:
        """Append a turning point to the list of turning points.

        This method validates that the turning point satisfies the constraints
        before adding it to the list.

        Args:
            tp: The turning point to append.
            tol: Tolerance for constraint validation. If None, uses the class's
                tol attribute. Pass 0 for exact validation.

        Raises:
            ValueError: If the turning point violates any constraints.

        """
        tol = self.tol if tol is None else tol

        if not np.all(tp.weights >= (self.lower_bounds - tol)):  # pragma: no mutate
            msg = "Weights below lower bounds"
            raise ValueError(msg)
        if not np.all(tp.weights <= (self.upper_bounds + tol)):  # pragma: no mutate
            msg = "Weights above upper bounds"
            raise ValueError(msg)
        if not np.allclose(self.a @ tp.weights, self.b, atol=1e-7):
            msg = "Weights violate the equality constraint A w = b"
            raise ValueError(msg)
        if self.g_matrix.shape[0] and not np.all(self.g_matrix @ tp.weights <= self.h_vector + tol):
            msg = "Weights violate the inequality constraint G w <= h"
            raise ValueError(msg)

        self.turning_points.append(tp)

    def _emit(
        self,
        lamb: float,
        weights: NDArray[np.float64],
        free: NDArray[np.bool_],
        active_ineq: NDArray[np.bool_],
    ) -> None:
        """Build and store a turning point, projecting away sub-tolerance round-off.

        Orchestrates the three steps taken at every turning point: refuse the point
        if the free-asset block is numerically singular (see
        :meth:`_guard_degeneracy`); project the candidate back onto the feasible
        set to clear sub-tolerance round-off (see :meth:`_project_feasible`); then
        validate and store it (see :meth:`_append`).

        On tie-heavy or near-degenerate problems (a short, near-rank-deficient
        sample covariance, duplicated assets, or many coincident events) accumulated
        floating-point round-off over the many turning points of a large trace can
        place a free weight a hair outside its box. The covariance there has
        near-flat directions (its small eigenvalues) and the round-off lies in
        exactly those directions, so the candidate is optimal to solver precision
        but not exactly feasible; the projection clears it and is a strict no-op for
        the well-posed turning points that are already feasible.
        """
        self._guard_degeneracy(lamb, free)
        weights = self._project_feasible(weights, active_ineq)
        self._append(TurningPoint(lamb=lamb, weights=weights, free=free, active_ineq=active_ineq))

    def _guard_degeneracy(self, lamb: float, free: NDArray[np.bool_]) -> None:
        """Refuse the turning point when the free-asset block is numerically singular.

        We distinguish two regimes by the conditioning of the free-asset block.
        While that block stays numerically full rank its solve is reliable and any
        box violation is round-off, which :meth:`_project_feasible` clears. Once the
        free set grows past the covariance rank the block is numerically singular
        and its solve is unreliable; whatever weights it produces (feasible or not)
        cannot be trusted, so we refuse and raise an actionable diagnosis instead of
        silently returning a possibly-suboptimal frontier.

        The discriminator is the free block's reciprocal condition number, read
        from its symmetric eigenvalues. Unlike the magnitude of the box violation,
        which is the residual of a singular solve and therefore varies with the
        BLAS/LAPACK build, the conditioning is deterministic and portable, so the
        completed-vs-declined boundary is the same on every platform.

        The per-turning-point conditioning check is skipped entirely when the full
        covariance is well conditioned: by interlacing no free block can then be
        singular, so the check is provably redundant (see
        :attr:`_free_blocks_well_conditioned`). It runs only when the full
        covariance is itself near-singular, which is exactly the regime that can
        produce an untrustworthy free-block solve.

        Args:
            lamb: Lambda value of the candidate turning point, used in the message.
            free: Boolean mask of the free assets at the candidate.

        Raises:
            ValueError: With a degeneracy-specific message when the free-asset
                block is numerically singular (an unreliable solve); otherwise
                returns without effect.
        """
        # When the full covariance clears the floor, interlacing guarantees every
        # free block does too, so the per-step guard can never fire -- skip it and
        # the costly per-step rcond. Only a near-singular full covariance needs the
        # check, and there it runs exactly as before.
        if not self._free_blocks_well_conditioned:
            rcond = self.covariance_operator.rcond_free(np.flatnonzero(free))
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

    def _project_feasible(
        self, weights: NDArray[np.float64], active_ineq: NDArray[np.bool_] | None = None
    ) -> NDArray[np.float64]:
        """Project ``weights`` onto the feasible region, clearing round-off.

        On tie-heavy or near-degenerate problems accumulated round-off can place a
        candidate a hair outside its box at a well-conditioned vertex. Projecting it
        back onto ``{w : lower <= w <= upper, C w = d}`` clears that round-off and
        lets the trace continue. The projection is a strict no-op for well-posed,
        already-feasible turning points (the common case), so it does not perturb
        the exact frontier.

        Dispatches to the closed-form capped-simplex projection for the canonical
        all-ones budget with no active inequality row (see
        :meth:`_project_capped_simplex`), and to the general alternating projection
        otherwise (see :meth:`_project_alternating`).

        Args:
            weights: The candidate weight vector to project.
            active_ineq: Boolean mask (length ``p``) of the active inequality rows;
                ``None`` means no inequality rows.

        Returns:
            The projected weight vector, feasible to the box and the constraints.
        """
        if active_ineq is None:
            active_ineq = np.zeros(self.g_matrix.shape[0], dtype=bool)
        lower, upper = self.lower_bounds, self.upper_bounds
        # Well-posed turning points are already strictly feasible: return them
        # unchanged so the projection never perturbs the exact frontier.
        if np.all(weights >= lower) and np.all(weights <= upper):
            return weights

        if not active_ineq.any() and self.a.shape[0] == 1 and np.allclose(self.a, 1.0):
            return self._project_capped_simplex(weights)
        return self._project_alternating(weights, active_ineq)

    def _project_capped_simplex(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Euclidean projection onto the capped simplex for the all-ones budget.

        Computed by water-filling a single shift ``theta`` so that
        ``sum(clip(w - theta, lower, upper)) = b``. A plain clip-then-rescale is
        deliberately *not* used: rescaling the clipped weights to restore the budget
        can push capped weights back over their bound when many assets are capped at
        once (heavy ties under a tight cap), re-introducing the very infeasibility
        the projection is meant to clear.

        Args:
            weights: The candidate weight vector, known to violate its box.

        Returns:
            The projected weight vector, on the budget and inside the box.
        """
        lower, upper = self.lower_bounds, self.upper_bounds
        # sum(clip(w - theta, lower, upper)) is non-increasing in theta; bisect
        # for the theta that hits the budget. The bracket clips to all-upper
        # (sum at least the budget) at theta_lo and all-lower (at most the
        # budget) at theta_hi, so a root is guaranteed for any feasible problem.
        budget = float(self.b[0])
        theta_lo = float((weights - upper).min()) - 1.0
        theta_hi = float((weights - lower).max()) + 1.0
        for _ in range(100):
            theta = 0.5 * (theta_lo + theta_hi)
            if float(np.clip(weights - theta, lower, upper).sum()) > budget:
                theta_lo = theta
            else:
                theta_hi = theta
        return np.clip(weights - 0.5 * (theta_lo + theta_hi), lower, upper)

    def _project_alternating(self, weights: NDArray[np.float64], active_ineq: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Alternating projection onto the box and the affine set ``{C w = d}``.

        ``C`` stacks the equality rows ``A`` and the active inequality rows ``G_S``
        (held at equality ``g_i w = h_i``). The candidate already satisfies
        ``C w = d`` (the reduced KKT solve enforces it) and the inactive inequality
        rows keep a margin, so a few iterations alternating a box clip with the
        affine projection converge to a point feasible to the box, the equalities,
        and every inequality.

        Args:
            weights: The candidate weight vector, known to violate its box.
            active_ineq: Boolean mask (length ``p``) of the active inequality rows.

        Returns:
            The projected weight vector, feasible to the box and the constraints.
        """
        lower, upper = self.lower_bounds, self.upper_bounds
        c = np.vstack([self.a, self.g_matrix[active_ineq]])
        d = np.concatenate([self.b, self.h_vector[active_ineq]])
        affine = AffineProjection(c, d)
        projected = weights
        for _ in range(100):
            projected = np.clip(projected, lower, upper)
            projected = affine.project(projected)
        return np.clip(projected, lower, upper)

    @property
    def frontier(self) -> Frontier:
        """Get the efficient frontier constructed from the turning points.

        This property creates a Frontier object from the list of turning points,
        which can be used to analyze the risk-return characteristics of the
        efficient portfolios.

        Returns:
            A Frontier object representing the efficient frontier.

        """
        return Frontier(
            covariance=self.covariance,
            mean=self.mean,
            frontier=[FrontierPoint(point.weights) for point in self.turning_points],
        )

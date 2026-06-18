"""Markowitz implementation of the Critical Line Algorithm.

This module provides the CLA class, which implements the Critical Line Algorithm
as described by Harry Markowitz and colleagues. The algorithm computes the entire
efficient frontier by finding all turning points, which are the points where the
set of assets at their bounds changes.
"""

import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .first import init_algo
from .operators import DenseCovariance, QuadraticForm
from .pathtracer import trace
from .types import Frontier, FrontierPoint, TurningPoint


class _Segment(NamedTuple):
    """The affine critical-line segment valid at one turning point.

    Bundles the affine path ``w(lam) = r_alpha + lam * r_beta``, the multiplier
    gradients ``gamma``/``delta`` that drive the leave-a-bound events, and the
    active-set masks the event scan needs. This is what ``CLA.segment`` returns
    to the generic path tracer.
    """

    r_alpha: NDArray[np.float64]
    r_beta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    delta: NDArray[np.float64]
    at_upper: NDArray[np.bool_]
    at_lower: NDArray[np.bool_]
    free_in: NDArray[np.bool_]


@dataclass(frozen=True)
class CLA:
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
        a: Equality-constraint matrix. The supported constraint is the
            fully-invested budget ``sum(w) = 1``: ``a`` is the single all-ones
            row and ``b`` is ``[1]``. The per-turning-point KKT solve is written
            for a general ``m``-row ``A`` (through the Schur complement), but the
            first turning point (:func:`cvxcla.first.init_algo`) and the
            fully-invested ``sum(w) = 1`` invariant of
            :class:`cvxcla.types.FrontierPoint` assume the budget constraint, so
            other equality systems (a different total, weighted coefficients, or
            ``m > 1`` rows) are not supported end to end.
        b: Equality-constraint right-hand side; ``[1]`` for the fully-invested
            budget.
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
    turning_points: list[TurningPoint] = field(default_factory=list)
    tol: float = 1e-5  # pragma: no mutate
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

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
        elimination: two multi-RHS solves against the free covariance block
        (via the covariance backend) and a small m x m Schur complement
        ``A_F @ Sigma_FF^{-1} @ A_F.T``, where m is the number of equality
        constraints. The covariance only enters through the ``QuadraticForm``
        interface, so structured backends (e.g. ``FactorCovariance``) never
        materialise an n x n matrix.

        Raises:
            RuntimeError: If all variables are blocked, which would make the
                          system of equations singular.

        """
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
        r_alpha, r_beta, gamma, delta = self._solve_kkt(free_in, fixed_weights)
        return _Segment(r_alpha, r_beta, gamma, delta, at_upper, at_lower, free_in)

    def event_matrix(self, state: TurningPoint, segment: _Segment) -> NDArray[np.float64]:  # noqa: ARG002
        """Return the ``(n, 4)`` matrix of candidate critical lambdas for ``segment``.

        ``state`` is part of the uniform ``ParametricProblem`` signature; the CLA
        does not need it here because ``segment`` already bundles the active-set
        masks derived from it.
        """
        return self._event_ratios(
            segment.r_alpha,
            segment.r_beta,
            segment.gamma,
            segment.delta,
            segment.free_in,
            segment.at_upper,
            segment.at_lower,
        )

    def step(self, state: TurningPoint, segment: _Segment, sec: int, direction: int, lam: float) -> TurningPoint:
        """Emit the turning point at ``lam`` after flipping asset ``sec``'s activity.

        A "leaves a bound" event (``direction`` in {2, 3}) makes the asset free; a
        "moves to a bound" event (``direction`` in {0, 1}) blocks it.
        """
        free = state.free.copy()
        free[sec] = direction >= 2
        self._emit(lam, segment.r_alpha + lam * segment.r_beta, free)
        return self.turning_points[-1]

    def finish(self, state: TurningPoint, segment: _Segment) -> None:
        """Emit the minimum-variance endpoint at ``lambda = 0``."""
        self._emit(0.0, segment.r_alpha, state.free)

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
        self, free_in: NDArray[np.bool_], fixed_weights: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Solve the reduced KKT system for the current critical-line segment.

        Block elimination: a multi-right-hand-side solve against the free
        covariance block ``Sigma_FF`` (via the backend, so structured covariances
        never materialise an ``n x n`` matrix) feeds an ``m x m`` Schur complement
        ``A_F Sigma_FF^{-1} A_F.T`` that handles the equality constraints.

        Args:
            free_in: Boolean mask of the assets in the reduced solve.
            fixed_weights: Full-length weights of the assets held at their bounds.

        Returns:
            ``(r_alpha, r_beta, gamma, delta)``: the affine segment
            ``w(lam) = r_alpha + lam * r_beta`` and the multiplier gradients
            ``gamma`` and ``delta`` that drive the leave-a-bound events.
        """
        m = self.a.shape[0]
        ns = len(self.mean)
        cov = self.covariance_operator
        out = ~free_in
        a_free = self.a[:, free_in]

        # [Sigma_FF  A_F.T] [x ]   [r1]   solved for the alpha (weights) and beta
        # [A_F       0    ] [nu] = [r2]   systems via Sigma_FF^{-1} [A_F.T | r1_a | r1_b]
        rhs_free = np.column_stack([a_free.T, -cov.cross(free_in, fixed_weights), self.mean[free_in]])
        solved = cov.solve_free(free_in, rhs_free)
        y = solved[:, :m]  # Sigma_FF^{-1} A_F.T
        z_alpha = solved[:, m]
        z_beta = solved[:, m + 1]

        # Schur complement A_F Sigma_FF^{-1} A_F.T and equality multipliers
        schur = a_free @ y
        r2_alpha = self.b - self.a[:, out] @ fixed_weights[out]
        nu = np.linalg.solve(schur, np.column_stack([a_free @ z_alpha - r2_alpha, a_free @ z_beta]))
        nu_alpha, nu_beta = nu[:, 0], nu[:, 1]

        # Back-substitute the free weights
        r_alpha = fixed_weights.copy()
        r_alpha[free_in] = z_alpha - y @ nu_alpha
        r_beta = np.zeros(ns)
        r_beta[free_in] = z_beta - y @ nu_beta

        gamma = cov.matvec(r_alpha) + self.a.T @ nu_alpha
        delta = cov.matvec(r_beta) + self.a.T @ nu_beta - self.mean
        return r_alpha, r_beta, gamma, delta

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

    def __len__(self) -> int:
        """Get the number of turning points in the efficient frontier.

        Returns:
            The number of turning points currently stored in the object.

        """
        return len(self.turning_points)

    def _first_turning_point(self) -> TurningPoint:
        """Calculate the first turning point on the efficient frontier.

        This method uses the init_algo function to find the first turning point
        based on the mean returns and the bounds on asset weights.

        Returns:
            A TurningPoint object representing the first point on the efficient frontier.

        """
        first = init_algo(
            mean=self.mean,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )
        return first

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
        if not np.allclose(np.sum(tp.weights), 1.0):
            msg = "Weights do not sum to 1"
            raise ValueError(msg)

        self.turning_points.append(tp)

    def _emit(self, lamb: float, weights: NDArray[np.float64], free: NDArray[np.bool_]) -> None:
        """Build and store a turning point, projecting away sub-tolerance round-off.

        On tie-heavy or near-degenerate problems (a short, near-rank-deficient
        sample covariance, duplicated assets, or many coincident events) the walk
        can reach a degenerate vertex at which a free weight sits essentially on
        one of its bounds. Accumulated floating-point round-off over the many
        turning points of a large trace then places that weight a hair outside its
        box. The covariance there has near-flat directions (its small eigenvalues),
        and the round-off lies in exactly those directions, so the candidate is
        optimal to solver precision but not exactly feasible.

        We distinguish two regimes by the conditioning of the free-asset block.
        While that block stays numerically full rank its solve is reliable and any
        box violation is round-off: we project the candidate onto the feasible box
        and, for the canonical budget constraint, rescale to restore the budget
        exactly. The projected point is then exactly feasible while remaining
        optimal (its objective matches a reference QP solve to roughly ``1e-8``;
        the weight difference is the problem's own non-uniqueness along the flat
        directions, not suboptimality). This is a no-op for well-posed turning
        points, which are already strictly feasible. Once the free set grows past
        the covariance rank the block is numerically singular and its solve is
        unreliable; whatever weights it produces (feasible or not) cannot be
        trusted, so we refuse and raise an actionable diagnosis instead of
        silently returning a possibly-suboptimal frontier.

        The discriminator is the free block's reciprocal condition number, read
        from its symmetric eigenvalues. Unlike the magnitude of the box violation,
        which is the residual of a singular solve and therefore varies with the
        BLAS/LAPACK build, the conditioning is deterministic and portable, so the
        completed-vs-declined boundary is the same on every platform.

        Raises:
            ValueError: With a degeneracy-specific message when the free-asset
                block is numerically singular (an unreliable solve); otherwise
                propagates nothing.
        """
        # A genuinely rank-deficient free block has a reciprocal condition number
        # at round-off level (~1e-16); a well-posed or merely near-degenerate
        # block sits many orders above it (>= ~1e-4 across the degeneracy sweep in
        # experiments/degeneracy_boundary.py). The 1e-12 cut sits in the wide gap
        # between the two and is the conventional numerical-singularity scale.
        rcond_floor = 1e-12  # pragma: no mutate
        rcond = self.covariance_operator.rcond_free(free)
        if rcond < rcond_floor:
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
        # Full-rank regime: the box violation is sub-tolerance round-off in the
        # covariance's flat directions, so project the candidate back onto the
        # feasible set and let the trace continue.
        weights = self._project_feasible(weights)
        self._append(TurningPoint(lamb=lamb, weights=weights, free=free))

    def _project_feasible(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project ``weights`` onto the feasible region, clearing round-off.

        This is the Euclidean projection onto the capped simplex
        ``{w : lower <= w <= upper, sum(w) = b}`` for the budget constraint the
        algorithm traces, computed by water-filling a single shift ``theta`` so
        that ``sum(clip(w - theta, lower, upper)) = b``.

        A plain clip-then-rescale is deliberately *not* used: rescaling the clipped
        weights to restore the budget can push capped weights back over their bound
        when many assets are capped at once (heavy ties under a tight cap),
        re-introducing the very infeasibility the projection is meant to clear and
        aborting the trace with a spurious "weights above upper bounds" error. The
        capped-simplex projection respects both the box and the budget
        simultaneously, and is a strict no-op for well-posed, already-feasible
        turning points (the common case), so it does not perturb the exact frontier.

        Args:
            weights: The candidate weight vector to project.

        Returns:
            The projected weight vector, feasible to the box and the budget.
        """
        lower, upper = self.lower_bounds, self.upper_bounds
        # Well-posed turning points are already strictly feasible: return them
        # unchanged so the projection never perturbs the exact frontier.
        if np.all(weights >= lower) and np.all(weights <= upper):
            return weights

        # sum(clip(w - theta, lower, upper)) is non-increasing in theta; bisect for
        # the theta that hits the budget. The bracket clips to all-upper (sum at
        # least the budget) at theta_lo and all-lower (at most the budget) at
        # theta_hi, so a root is guaranteed for any feasible problem.
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

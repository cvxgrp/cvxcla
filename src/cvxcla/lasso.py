"""LASSO / LARS regularisation path as a parametric active-set problem.

This module shows that the Critical Line Algorithm's machinery is not specific to
portfolios: the *same* ``cvxcla.pathtracer.trace`` loop, the *same*
``QuadraticForm`` operator, and the *same* Bland event selection trace the LASSO
homotopy. Only the problem-specific glue (the segment solve and what an event
means) differs.

The LASSO solves, for a response ``y`` and design matrix ``X``,

    minimize  1/2 ||y - X beta||^2 + lam ||beta||_1

and its minimiser ``beta(lam)`` is continuous and piecewise linear in the penalty
``lam``. On a segment where the active set ``A`` (the support) and the signs
``s_A`` are fixed,

    beta_A(lam)      = (X_A^T X_A)^{-1} (X_A^T y - lam s_A) = alpha_A - lam * beta_slope_A
    correlation(lam) = X^T (y - X beta(lam))               = p + lam * q

with ``|correlation_j| <= lam`` off the support and ``correlation_j = lam s_j`` on
it. The role played by the covariance ``Sigma`` and mean ``mu`` in the CLA is
played here by the Gram matrix ``H = X^T X`` (wrapped in ``DenseCovariance``) and
the vector ``X^T y``.

**Constraints.** Like the CLA, the path tracer admits general linear inequality
constraints ``G beta <= h``. An active row enters the reduced KKT system exactly as
in the CLA (the bordered Schur complement of ``cla.py``), and the generalised
correlation that drives the enter/leave events carries the active-row multipliers,
``correlation(lam) = X^T y - H beta(lam) - G_S^T eta(lam)``. The constrained path is
still piecewise linear (a quadratic loss under a polyhedral penalty *and* polyhedral
constraints; cf. Rosset and Zhu). We require ``h > 0`` so the path can start from
``beta = 0`` with every row slack -- the same first vertex as the unconstrained
LASSO. (Equality constraints, or ``h`` with a zero entry, need a feasibility seed
analogous to the CLA's linear-programming first vertex, and are left to future work.)

Event families, mirroring the CLA's "move to / leave a bound":

* **leave** -- an active coefficient reaches zero: ``lam = alpha_j / beta_slope_j``.
* **enter** -- an inactive (generalised) correlation reaches ``+/-lam``.
* **activate** -- a slack inequality row's residual ``G_r beta - h_r`` reaches zero.
* **release** -- an active row's multiplier ``eta_r`` reaches zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .operators import DenseCovariance, QuadraticForm
from .pathtracer import trace

if TYPE_CHECKING:
    from .builder import LassoBuilder


class _LassoState(NamedTuple):
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


class _LassoSegment(NamedTuple):
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


@dataclass(frozen=True)
class Breakpoint:
    """A vertex of the piecewise-linear LASSO path.

    Attributes:
        lam: The penalty value at this breakpoint.
        beta: The coefficient vector ``beta(lam)``.
        active: Boolean mask of the support (non-zero coefficients) on the
            segment leaving this breakpoint towards smaller ``lam``.
    """

    lam: float
    beta: NDArray[np.float64]
    active: NDArray[np.bool_]


@dataclass
class Lasso:
    """The LASSO regularisation path, traced as a parametric active-set problem.

    Constructing a ``Lasso`` traces the entire path from ``lam_max`` (where
    ``beta = 0``) down to ``lam = 0`` (the least-squares fit on the final support,
    subject to any active constraints), storing the breakpoints in ``path``. The
    walk is driven by the same ``cvxcla.pathtracer.trace`` loop as the Critical Line
    Algorithm.

    Optional linear inequality constraints ``G beta <= h`` (with ``h > 0``) are
    traced through the same bordered solve as the CLA's ``G w <= h`` rows.

    Attributes:
        x: Design matrix of shape ``(m, n)``.
        y: Response vector of shape ``(m,)``.
        g: Optional inequality matrix ``(p, n)`` of ``G beta <= h``; ``None`` means
            the plain LASSO.
        h: Optional inequality right-hand side ``(p,)``; must be strictly positive.
        tol: Tolerance for event selection and the validity window.
        path: The discovered breakpoints, populated on construction.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    g: NDArray[np.float64] | None = None
    h: NDArray[np.float64] | None = None
    nonneg: bool = False  # pragma: no mutate
    tol: float = 1e-9  # pragma: no mutate
    path: list[Breakpoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate shapes and trace the full LASSO path.

        Raises:
            ValueError: If ``x`` is not 2d, ``y``'s length does not match ``x``, the
                constraint shapes are inconsistent, or any ``h`` entry is not
                strictly positive (which would make ``beta = 0`` infeasible).
        """
        if self.x.ndim != 2:
            msg = f"x must be a 2d design matrix, got shape {self.x.shape}"
            raise ValueError(msg)
        if self.y.shape != (self.x.shape[0],):
            msg = f"y must have shape ({self.x.shape[0]},), got {self.y.shape}"
            raise ValueError(msg)
        if self.g is not None or self.h is not None:
            if self.g is None or self.h is None:
                msg = "g and h must be provided together"
                raise ValueError(msg)
            if self.g.shape != (self.h.shape[0], self.dimension):
                msg = f"g must have shape ({self.h.shape[0]}, {self.dimension}), got {self.g.shape}"
                raise ValueError(msg)
            if np.any(self.h <= self.tol):
                msg = "h must be strictly positive so beta = 0 is feasible (equality/zero-h needs a feasibility seed)"
                raise ValueError(msg)
        trace(self)

    @classmethod
    def problem(cls, x: NDArray[np.float64], y: NDArray[np.float64]) -> LassoBuilder:
        """Start a fluent :class:`cvxcla.builder.LassoBuilder` for a LASSO path.

        The LASSO counterpart of :meth:`cvxcla.cla.CLA.problem`: chain
        ``.inequality(G, h)`` and finish with ``.trace()``. The builder maps onto the
        constructor arguments and adds no modelling power.

        Args:
            x: Design matrix of shape ``(m, n)``.
            y: Response vector of shape ``(m,)``.

        Returns:
            A :class:`cvxcla.builder.LassoBuilder`.
        """
        from .builder import LassoBuilder

        return LassoBuilder(x, y)

    @property
    def g_matrix(self) -> NDArray[np.float64]:
        """Inequality matrix ``G`` as a ``(p, n)`` array (empty ``(0, n)`` if none)."""
        if self.g is None:
            return np.zeros((0, self.dimension))
        return np.atleast_2d(self.g)

    @property
    def h_vector(self) -> NDArray[np.float64]:
        """Inequality right-hand side ``h`` as a ``(p,)`` array (empty if none)."""
        if self.h is None:
            return np.zeros(0)
        return np.atleast_1d(self.h)

    @cached_property
    def quad(self) -> QuadraticForm:
        """The Gram matrix ``X^T X`` as a ``QuadraticForm`` backend (cached: ``X`` is fixed)."""
        return DenseCovariance(self.x.T @ self.x)

    @cached_property
    def xty(self) -> NDArray[np.float64]:
        """The linear data ``X^T y`` (the analogue of the CLA's expected returns; cached)."""
        return self.x.T @ self.y

    @property
    def dimension(self) -> int:
        """Number of features ``n`` (the problem dimension for the path tracer)."""
        return int(self.x.shape[1])

    @property
    def event_dimension(self) -> int:
        """Coordinate count for the path-length cap: ``n`` features + ``p`` rows."""
        return self.dimension + int(self.g_matrix.shape[0])

    @property
    def lam_max(self) -> float:
        """The smallest penalty at which ``beta = 0`` is optimal: ``||X^T y||_inf``.

        With ``h > 0`` every inequality row is slack at ``beta = 0`` (zero
        multiplier), so the unconstrained threshold is unchanged.
        """
        return float(np.max(np.abs(self.xty)))

    def begin(self) -> tuple[float, _LassoState]:
        """Record the all-zero solution at the start penalty and enter the first coordinate.

        For the plain or inequality-constrained LASSO the start is
        ``lam_max = ||X^T y||_inf`` and the most-correlated coordinate enters with its
        sign. Under the non-negative restriction ``beta >= 0`` the l1 penalty becomes
        the linear term ``lam * 1^T beta``, only positive correlations can enter, so
        the start is ``lam_max = max_j (X^T y)_j`` and the coordinate enters with sign
        ``+``. When no coordinate can enter (e.g. every correlation is non-positive
        under ``beta >= 0``), ``beta = 0`` is optimal for all ``lambda`` and the path
        is the single point.
        """
        n = self.dimension
        xty = self.xty
        rows_active = np.zeros(self.g_matrix.shape[0], dtype=bool)
        if self.nonneg:
            lam_max = float(np.max(xty)) if n else 0.0
            j0, s0 = int(np.argmax(xty)), 1.0
        else:
            lam_max = self.lam_max
            j0 = int(np.argmax(np.abs(xty)))
            s0 = float(np.sign(xty[j0]))

        self.path.append(Breakpoint(max(lam_max, 0.0), np.zeros(n), np.zeros(n, dtype=bool)))
        active = np.zeros(n, dtype=bool)
        signs = np.zeros(n)
        if lam_max > self.tol:
            active[j0] = True
            signs[j0] = s0
        return max(lam_max, 0.0), _LassoState(active, signs, rows_active, max(lam_max, 0.0))

    def segment(self, state: _LassoState) -> _LassoSegment:
        """Solve the affine segment for the current support, signs, and active rows.

        With no active rows this is the plain LASSO solve against the Gram
        submatrix. With active rows it is the bordered Schur solve of the CLA: the
        active rows ``G_S`` enter the reduced KKT system as extra equality rows.
        """
        n = self.dimension
        active, signs, rows_active = state.active, state.signs, state.rows_active
        xty = self.xty
        alpha = np.zeros(n)
        beta_slope = np.zeros(n)
        eta_alpha = np.zeros(self.g_matrix.shape[0])
        eta_slope = np.zeros(self.g_matrix.shape[0])

        xty_s = xty[active]
        signs_s = signs[active]
        if not np.any(active):
            # Empty support (e.g. the non-negative path when no correlation is
            # positive): beta = 0, correlation = X^T y, and there is nothing to solve.
            return _LassoSegment(alpha, beta_slope, xty.copy(), np.zeros(n), eta_alpha, eta_slope)
        if not np.any(rows_active):
            # Plain LASSO solve: beta_S(lam) = H_SS^{-1}(xty_S - lam s_S). Solve both
            # right-hand sides at once so H_SS is factorised a single time per
            # breakpoint (one np.linalg.solve, not two).
            sol = self.quad.solve_free(active, np.column_stack([xty_s, signs_s]))
            alpha[active] = sol[:, 0]
            beta_slope[active] = sol[:, 1]
        else:
            # Bordered solve over (beta_S, eta_R): the active rows G_RS act as
            # equality rows, exactly the CLA's Schur complement (cla.py).
            g_rs = self.g_matrix[np.ix_(rows_active, active)]  # |R| x |S|
            h_r = self.h_vector[rows_active]
            rhs = np.column_stack([xty_s, signs_s, g_rs.T])  # |S| x (2 + |R|)
            sol = self.quad.solve_free(active, rhs)
            u0, u1, big_y = sol[:, 0], sol[:, 1], sol[:, 2:]  # big_y = H_SS^{-1} G_RS^T
            schur = g_rs @ big_y  # |R| x |R|
            eta_a = np.linalg.solve(schur, g_rs @ u0 - h_r)
            eta_s = -np.linalg.solve(schur, g_rs @ u1)
            alpha[active] = u0 - big_y @ eta_a
            beta_slope[active] = u1 + big_y @ eta_s
            eta_alpha[rows_active] = eta_a
            eta_slope[rows_active] = eta_s

        # Generalised correlation c(lam) = xty - H beta(lam) - G_R^T eta(lam) = p + lam q.
        g_r = self.g_matrix[rows_active]
        eta_a_r = eta_alpha[rows_active]
        eta_s_r = eta_slope[rows_active]
        p = xty - self.quad.matvec(alpha) - g_r.T @ eta_a_r
        q = self.quad.matvec(beta_slope) - g_r.T @ eta_s_r
        return _LassoSegment(alpha, beta_slope, p, q, eta_alpha, eta_slope)

    def event_matrix(self, state: _LassoState, segment: _LassoSegment) -> NDArray[np.float64]:
        """Return the ``(n + p, 4)`` matrix of candidate critical lambdas.

        Rows ``0..n-1`` are coordinate events (col 0 leave, col 1 enter ``+``, col 2
        enter ``-``); rows ``n..n+p-1`` are inequality-row events (col 0 activate, col
        1 release). Entries are ``-inf`` where the event cannot occur, and only
        events strictly inside ``(tol, lam - tol)`` are kept.
        """
        n = self.dimension
        rows = self.g_matrix.shape[0]
        active = state.active
        inactive = ~active
        alpha, beta_slope, p, q, eta_alpha, eta_slope = segment
        rows_active = state.rows_active

        l_mat = np.full((n + rows, 4), -np.inf)

        # leave: alpha_j - lam beta_slope_j = 0
        leaves = active & (np.abs(beta_slope) > self.tol)  # pragma: no mutate
        l_mat[:n][leaves, 0] = alpha[leaves] / beta_slope[leaves]

        # enter (+): p_j + lam q_j = +lam -> lam = p_j / (1 - q_j)
        denom_pos = 1.0 - q
        enters_pos = inactive & (np.abs(denom_pos) > self.tol)  # pragma: no mutate
        l_mat[:n][enters_pos, 1] = p[enters_pos] / denom_pos[enters_pos]

        # enter (-): p_j + lam q_j = -lam -> lam = -p_j / (1 + q_j). Disabled under the
        # non-negative restriction beta >= 0, where only positive entries are allowed.
        if not self.nonneg:
            denom_neg = 1.0 + q
            enters_neg = inactive & (np.abs(denom_neg) > self.tol)  # pragma: no mutate
            l_mat[:n][enters_neg, 2] = -p[enters_neg] / denom_neg[enters_neg]

        if rows:
            g_mat = self.g_matrix
            slope_row = g_mat @ beta_slope  # d/d(-lam) of the row value
            level_row = g_mat @ alpha - self.h_vector  # G_r alpha - h_r
            # activate: the row value G_r beta(lam) = level + h_r - lam slope rises to
            # the cap h_r as lam decreases when its slope d(value)/d(-lam) = slope_row
            # is positive; the crossing is lam = (G_r alpha - h_r) / (G_r beta_slope).
            inactive_rows = ~rows_active & (slope_row > self.tol)  # pragma: no mutate
            l_mat[n:][inactive_rows, 0] = level_row[inactive_rows] / slope_row[inactive_rows]
            # release: eta_r(lam) = eta_alpha + lam eta_slope -> 0 from eta > 0, i.e. eta_slope > 0.
            releasing = rows_active & (eta_slope > self.tol)  # pragma: no mutate
            l_mat[n:][releasing, 1] = -eta_alpha[releasing] / eta_slope[releasing]

        # Keep only events that make strict progress to a smaller, positive penalty.
        l_mat[(l_mat <= self.tol) | (l_mat >= state.lam - self.tol)] = -np.inf
        return l_mat

    def step(self, state: _LassoState, segment: _LassoSegment, sec: int, direction: int, lam: float) -> _LassoState:
        """Record the breakpoint at ``lam`` after flipping coordinate or row ``sec``.

        For a coordinate (``sec < n``): direction 0 removes it from the support, 1/2
        add it with sign ``+1``/``-1``. For an inequality row (``sec >= n``):
        direction 0 activates the row, 1 releases it. The path is continuous across
        the flip, so the recorded coefficients are the old segment at ``lam``.
        """
        n = self.dimension
        active = state.active.copy()
        signs = state.signs.copy()
        rows_active = state.rows_active.copy()
        if sec < n:
            if direction == 0:
                active[sec] = False
                signs[sec] = 0.0
            else:
                active[sec] = True
                signs[sec] = 1.0 if direction == 1 else -1.0
        else:
            rows_active[sec - n] = direction == 0

        beta = segment.alpha - lam * segment.beta_slope
        self.path.append(Breakpoint(lam, beta, active.copy()))
        return _LassoState(active, signs, rows_active, lam)

    def finish(self, state: _LassoState, segment: _LassoSegment) -> None:
        """Record the ``lam = 0`` endpoint: the least-squares fit on the final support."""
        self.path.append(Breakpoint(0.0, segment.alpha.copy(), state.active.copy()))

    def solution(self, lam: float) -> NDArray[np.float64]:
        """Evaluate the piecewise-linear path at penalty ``lam``.

        Args:
            lam: The penalty value at which to evaluate ``beta``.

        Returns:
            The coefficient vector ``beta(lam)``, by linear interpolation between
            the bracketing breakpoints (clamped to the path's endpoints).
        """
        ordered = sorted(self.path, key=lambda bp: bp.lam)
        if lam <= ordered[0].lam:
            return ordered[0].beta
        if lam >= ordered[-1].lam:
            return ordered[-1].beta
        for lo, hi in pairwise(ordered):
            if lo.lam <= lam <= hi.lam:
                weight = (lam - lo.lam) / (hi.lam - lo.lam)
                return (1.0 - weight) * lo.beta + weight * hi.beta
        msg = "lam lies within the path range but no bracketing segment was found"  # pragma: no cover
        raise AssertionError(msg)  # pragma: no cover

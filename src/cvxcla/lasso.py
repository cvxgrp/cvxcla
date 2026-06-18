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

Two coordinate-indexed event families, mirroring the CLA's "move to / leave a
bound":

* **leave** -- an active coefficient reaches zero: ``lam = alpha_j / beta_slope_j``.
* **enter** -- an inactive correlation reaches ``+/-lam``: ``p_j + lam q_j = +/-lam``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .operators import DenseCovariance, QuadraticForm
from .pathtracer import trace


class _LassoState(NamedTuple):
    """The active set, sign pattern, and current penalty defining the segment.

    ``lam`` is the penalty at the segment's upper end. The event scan uses it to
    require *strict* progress to a smaller penalty: right after a coefficient
    enters it sits at zero, so its leave event lies at the current ``lam``; without
    the strict window the shared selector would re-fire it and the walk would cycle
    between entering and leaving the same coordinate.
    """

    active: NDArray[np.bool_]
    signs: NDArray[np.float64]
    lam: float


class _LassoSegment(NamedTuple):
    """The affine path ``beta(lam) = alpha - lam * beta_slope`` and correlation ``p + lam * q``."""

    alpha: NDArray[np.float64]
    beta_slope: NDArray[np.float64]
    p: NDArray[np.float64]
    q: NDArray[np.float64]


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
    ``beta = 0``) down to ``lam = 0`` (the ordinary least-squares fit on the final
    support), storing the breakpoints in ``path``. The walk is driven by the same
    ``cvxcla.pathtracer.trace`` loop as the Critical Line Algorithm.

    Attributes:
        x: Design matrix of shape ``(m, n)``.
        y: Response vector of shape ``(m,)``.
        tol: Tolerance for event selection and the validity window.
        path: The discovered breakpoints, populated on construction.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    tol: float = 1e-9  # pragma: no mutate
    path: list[Breakpoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate shapes and trace the full LASSO path.

        Raises:
            ValueError: If ``x`` is not 2d or ``y``'s length does not match the
                number of rows of ``x``.
        """
        if self.x.ndim != 2:
            msg = f"x must be a 2d design matrix, got shape {self.x.shape}"
            raise ValueError(msg)
        if self.y.shape != (self.x.shape[0],):
            msg = f"y must have shape ({self.x.shape[0]},), got {self.y.shape}"
            raise ValueError(msg)
        trace(self)

    @property
    def quad(self) -> QuadraticForm:
        """The Gram matrix ``X^T X`` as a ``QuadraticForm`` backend."""
        return DenseCovariance(self.x.T @ self.x)

    @property
    def xty(self) -> NDArray[np.float64]:
        """The linear data ``X^T y`` (the analogue of the CLA's expected returns)."""
        return self.x.T @ self.y

    @property
    def dimension(self) -> int:
        """Number of features ``n`` (the problem dimension for the path tracer)."""
        return int(self.x.shape[1])

    @property
    def lam_max(self) -> float:
        """The smallest penalty at which ``beta = 0`` is optimal: ``||X^T y||_inf``."""
        return float(np.max(np.abs(self.xty)))

    def begin(self) -> tuple[float, _LassoState]:
        """Record the all-zero solution at ``lam_max`` and enter the first coordinate.

        Returns:
            ``(lam_max, state)`` where ``state`` already has the most-correlated
            coordinate in the active set, mirroring the CLA's first turning point.
        """
        n = self.dimension
        xty = self.xty
        lam_max = self.lam_max
        j0 = int(np.argmax(np.abs(xty)))
        self.path.append(Breakpoint(lam_max, np.zeros(n), np.zeros(n, dtype=bool)))

        active = np.zeros(n, dtype=bool)
        active[j0] = True
        signs = np.zeros(n)
        signs[j0] = np.sign(xty[j0])
        return lam_max, _LassoState(active, signs, lam_max)

    def segment(self, state: _LassoState) -> _LassoSegment:
        """Solve the affine segment for the current ``(active, signs)`` state.

        Both solves go through the ``QuadraticForm`` principal-submatrix solver,
        exactly as the CLA solves against the free covariance block.
        """
        n = self.dimension
        active, signs = state.active, state.signs
        alpha = np.zeros(n)
        beta_slope = np.zeros(n)
        alpha[active] = self.quad.solve_free(active, self.xty[active])
        beta_slope[active] = self.quad.solve_free(active, signs[active])

        # correlation(lam) = X^T y - H beta(lam) = (xty - H alpha) + lam (H beta_slope)
        p = self.xty - self.quad.matvec(alpha)
        q = self.quad.matvec(beta_slope)
        return _LassoSegment(alpha, beta_slope, p, q)

    def event_matrix(self, state: _LassoState, segment: _LassoSegment) -> NDArray[np.float64]:
        """Return the ``(n, 3)`` matrix of candidate critical lambdas.

        Columns: 0 = leave (active coefficient hits zero), 1 = enter with sign
        ``+1`` (inactive correlation reaches ``+lam``), 2 = enter with sign ``-1``
        (reaches ``-lam``). Only events strictly above ``tol`` are kept, so the
        trace stops cleanly as ``lam`` approaches zero.
        """
        n = self.dimension
        active = state.active
        inactive = ~active
        alpha, beta_slope, p, q = segment

        l_mat = np.full((n, 3), -np.inf)

        # leave: alpha_j - lam beta_slope_j = 0
        leaves = active & (np.abs(beta_slope) > self.tol)  # pragma: no mutate
        l_mat[leaves, 0] = alpha[leaves] / beta_slope[leaves]

        # enter (+): p_j + lam q_j = +lam -> lam = p_j / (1 - q_j)
        denom_pos = 1.0 - q
        enters_pos = inactive & (np.abs(denom_pos) > self.tol)  # pragma: no mutate
        l_mat[enters_pos, 1] = p[enters_pos] / denom_pos[enters_pos]

        # enter (-): p_j + lam q_j = -lam -> lam = -p_j / (1 + q_j)
        denom_neg = 1.0 + q
        enters_neg = inactive & (np.abs(denom_neg) > self.tol)  # pragma: no mutate
        l_mat[enters_neg, 2] = -p[enters_neg] / denom_neg[enters_neg]

        # Keep only events that make strict progress to a smaller, positive penalty:
        # at or above the current lam they are the spurious inverse of the coordinate
        # just flipped (which would cycle); at or below ~0 the path is complete.
        l_mat[(l_mat <= self.tol) | (l_mat >= state.lam - self.tol)] = -np.inf
        return l_mat

    def step(self, state: _LassoState, segment: _LassoSegment, sec: int, direction: int, lam: float) -> _LassoState:
        """Record the breakpoint at ``lam`` after flipping coordinate ``sec``.

        ``direction`` 0 removes ``sec`` from the support; 1/2 add it with sign
        ``+1``/``-1``. The recorded coefficients are the old segment evaluated at
        ``lam`` (the path is continuous across the flip).
        """
        active = state.active.copy()
        signs = state.signs.copy()
        if direction == 0:
            active[sec] = False
            signs[sec] = 0.0
        else:
            active[sec] = True
            signs[sec] = 1.0 if direction == 1 else -1.0

        beta = segment.alpha - lam * segment.beta_slope
        self.path.append(Breakpoint(lam, beta, active.copy()))
        return _LassoState(active, signs, lam)

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

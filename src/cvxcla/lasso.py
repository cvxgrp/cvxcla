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
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._lasso import LassoSegment, LassoState, scan_events, solve_segment
from ._lasso_validate import validate_constraints, validate_design_inputs, validate_operator_inputs
from .operators import DenseCovariance, GramCovariance, QuadraticForm
from .pathtracer import InequalityConstrained, trace


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
class Lasso(InequalityConstrained):
    """The LASSO regularisation path, traced as a parametric active-set problem.

    Constructing a ``Lasso`` traces the entire path from ``lam_max`` (where
    ``beta = 0``) down to ``lam = 0`` (the least-squares fit on the final support,
    subject to any active constraints), storing the breakpoints in ``path``. The
    walk is driven by the same ``cvxcla.pathtracer.trace`` loop as the Critical Line
    Algorithm.

    Optional linear inequality constraints ``G beta <= h`` (with ``h > 0``) are
    traced through the same bordered solve as the CLA's ``G w <= h`` rows.

    The quadratic form may be given either as a dense design ``(x, y)`` (the usual
    case, ``H = X^T X``) or, via :meth:`from_operator`, as a ``QuadraticForm``
    operator with the linear term ``X^T y``. The operator route lets a structured
    form, a diagonal-plus-low-rank factor model or a kernel, drive the path in
    ``O(nk)`` per step without forming the ``n x n`` Gram matrix, exactly as on the
    portfolio side.

    Attributes:
        x: Design matrix of shape ``(m, n)`` (``None`` in operator mode).
        y: Response vector of shape ``(m,)`` (``None`` in operator mode).
        g: Optional inequality matrix ``(p, n)`` of ``G beta <= h``; ``None`` means
            the plain LASSO.
        h: Optional inequality right-hand side ``(p,)``; must be strictly positive.
        nonneg: When ``True``, restrict to the non-negative LASSO ``beta >= 0``;
            the default ``False`` traces the ordinary signed path.
        gram: When ``True``, drive the path with the ``GramCovariance`` data-matrix
            backend (Woodbury solves in the ``m``-dimensional observation space),
            never materialising the ``n x n`` Gram ``X^T X`` — the win in the
            ``n >> m`` regime. The default ``False`` forms the dense Gram.
        tol: Tolerance for event selection and the validity window.
        path: The discovered breakpoints, populated on construction.
        quad_form: Optional ``QuadraticForm`` operator ``H`` (operator mode).
        linear: Optional linear term ``X^T y`` of shape ``(n,)`` (operator mode).
    """

    x: NDArray[np.float64] | None = None
    y: NDArray[np.float64] | None = None
    g: NDArray[np.float64] | None = None
    h: NDArray[np.float64] | None = None
    nonneg: bool = False  # pragma: no mutate
    gram: bool = False  # pragma: no mutate
    tol: float = 1e-9  # pragma: no mutate
    path: list[Breakpoint] = field(default_factory=list)
    quad_form: QuadraticForm | None = None  # pragma: no mutate
    linear: NDArray[np.float64] | None = None  # pragma: no mutate

    def __post_init__(self) -> None:
        """Validate shapes and trace the full LASSO path.

        Raises:
            ValueError: If ``x`` is not 2d, ``y``'s length does not match ``x``, the
                constraint shapes are inconsistent, or any ``h`` entry is not
                strictly positive (which would make ``beta = 0`` infeasible).
        """
        if self.quad_form is not None or self.linear is not None:
            self.linear = validate_operator_inputs(self.quad_form, self.linear, self.x, self.y)
        else:
            validate_design_inputs(self.x, self.y)
        validate_constraints(self.g, self.h, self.dimension, self.tol)
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
        return LassoBuilder(x, y)

    @classmethod
    def from_operator(
        cls,
        quad: QuadraticForm,
        xty: NDArray[np.float64],
        *,
        g: NDArray[np.float64] | None = None,
        h: NDArray[np.float64] | None = None,
        nonneg: bool = False,
        tol: float = 1e-9,
    ) -> Lasso:
        """Trace a LASSO path with the quadratic form supplied as an operator.

        The regression counterpart of :class:`cvxcla.cla.CLA` accepting a
        ``QuadraticForm`` covariance. Instead of a dense design ``X``, pass the Gram
        operator ``H`` (anything implementing :class:`QuadraticForm`, for example a
        :class:`cvxcla.operators.FactorCovariance` or a kernel) together with the
        linear term ``X^T y``. The homotopy reaches ``H`` only through ``matvec`` and
        ``solve_free``, so a diagonal-plus-low-rank factor model or a kernel traces
        the path in ``O(nk)`` per step without ever forming the ``n x n`` matrix,
        exactly as on the portfolio side. For the path to coincide with the
        design-matrix LASSO one needs ``H = X^T X`` and ``xty = X^T y``
        (Theorem 1); any positive-semidefinite operator whose free blocks are
        positive definite traces a well-defined path.

        Args:
            quad: The quadratic form ``H`` as a :class:`QuadraticForm` operator.
            xty: The linear term ``X^T y`` of shape ``(n,)``.
            g: Optional inequality matrix of ``G beta <= h``.
            h: Optional inequality right-hand side; entries must be strictly positive.
            nonneg: Restrict the path to ``beta >= 0``.
            tol: Tolerance for event selection and the validity window.

        Returns:
            A traced :class:`Lasso` whose ``path`` holds the breakpoints.
        """
        return cls(
            quad_form=quad,
            linear=np.asarray(xty, dtype=np.float64),
            g=g,
            h=h,
            nonneg=nonneg,
            tol=tol,
        )

    @cached_property
    def quad(self) -> QuadraticForm:
        """The Gram matrix ``X^T X`` as a ``QuadraticForm`` backend (cached: ``X`` is fixed).

        With ``gram=True`` the data-matrix backend is used instead of forming the
        ``n x n`` Gram: it solves through the Woodbury identity in the
        ``m``-dimensional observation space and never materialises an ``n x n``
        matrix, the win in the high-dimensional ``p >> n`` regime (more features than
        observations). ``GramCovariance`` represents ``X_c^T X_c / (m-1)``, so scaling
        the data by ``sqrt(m-1)`` recovers ``X^T X`` exactly for a **centred** design
        (the standard LASSO convention; pass a column-centred ``x``).
        """
        if self.quad_form is not None:
            return self.quad_form
        # Not operator mode, so __post_init__ guarantees a design matrix.
        x = cast("NDArray[np.float64]", self.x)
        if self.gram:
            m = x.shape[0]
            return GramCovariance(x * np.sqrt(m - 1.0))
        return DenseCovariance(x.T @ x)

    @cached_property
    def xty(self) -> NDArray[np.float64]:
        """The linear data ``X^T y`` (the analogue of the CLA's expected returns; cached)."""
        if self.linear is not None:
            return self.linear
        # Not operator mode, so __post_init__ guarantees a design (x, y).
        x = cast("NDArray[np.float64]", self.x)
        y = cast("NDArray[np.float64]", self.y)
        return x.T @ y

    @property
    def dimension(self) -> int:
        """Number of features ``n`` (the problem dimension for the path tracer)."""
        if self.x is not None:
            return int(self.x.shape[1])
        return int(self.xty.shape[0])

    @property
    def lam_max(self) -> float:
        """The smallest penalty at which ``beta = 0`` is optimal: ``||X^T y||_inf``.

        With ``h > 0`` every inequality row is slack at ``beta = 0`` (zero
        multiplier), so the unconstrained threshold is unchanged.
        """
        return float(np.max(np.abs(self.xty)))

    def begin(self) -> tuple[float, LassoState]:
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
        return max(lam_max, 0.0), LassoState(active, signs, rows_active, max(lam_max, 0.0))

    def segment(self, state: LassoState) -> LassoSegment:
        """Solve the affine segment for the current support, signs, and active rows.

        Delegates to :func:`cvxcla._lasso.solve_segment`; see there for the
        bordered Schur solve that admits active inequality rows.
        """
        return solve_segment(self.quad, self.xty, self.g_matrix, self.h_vector, state)

    def event_matrix(self, state: LassoState, segment: LassoSegment) -> NDArray[np.float64]:
        """Return the ``(n + p, 4)`` matrix of candidate critical lambdas.

        Delegates to :func:`cvxcla._lasso.scan_events`; see there for the
        coordinate (leave/enter) and inequality-row (activate/release) events.
        """
        return scan_events(self.dimension, self.g_matrix, self.h_vector, self.tol, self.nonneg, state, segment)

    def step(self, state: LassoState, segment: LassoSegment, sec: int, direction: int, lam: float) -> LassoState:
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
        return LassoState(active, signs, rows_active, lam)

    def finish(self, state: LassoState, segment: LassoSegment) -> None:
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


class LassoBuilder:
    """Chainable builder for a LASSO regularisation-path problem.

    The LASSO counterpart of :class:`cvxcla.cla.ProblemBuilder`. Construct one via
    :meth:`Lasso.problem`, optionally add inequality constraints with
    :meth:`inequality`, and finish with :meth:`trace`, which builds the
    :class:`Lasso` and traces the entire regularisation path. Like the CLA builder
    it adds no modelling power: it accepts the same ``G beta <= h`` rows the
    ``Lasso`` already supports and nothing else.

    Examples:
        >>> import numpy as np
        >>> from cvxcla import Lasso
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal((30, 5))
        >>> y = rng.standard_normal(30)
        >>> lasso = Lasso.problem(x, y).trace()
        >>> len(lasso.path) > 0
        True
    """

    def __init__(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Start a builder for design matrix ``x`` and response ``y``.

        Args:
            x: Design matrix of shape ``(m, n)``.
            y: Response vector of shape ``(m,)``.
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self._g_blocks: list[NDArray[np.float64]] = []
        self._h_blocks: list[NDArray[np.float64]] = []
        self._nonneg = False

    def non_negative(self) -> LassoBuilder:
        """Restrict the coefficients to ``beta >= 0`` (the non-negative LASSO).

        Under ``beta >= 0`` the l1 penalty collapses to the linear term
        ``lam * sum(beta)``, so the path is the standard one restricted to positive
        signs -- structurally the CLA's box-bounded parametric QP.

        Returns:
            ``self``, for chaining.
        """
        self._nonneg = True
        return self

    def inequality(self, g: NDArray[np.float64], h: float | NDArray[np.float64]) -> LassoBuilder:
        """Add one or more inequality rows ``G beta <= h`` (repeated calls accumulate).

        Args:
            g: A length-``n`` row vector or a ``(p, n)`` matrix.
            h: The matching right-hand side: a scalar for a single row, or a
                length-``p`` vector. Each entry must be strictly positive (so
                ``beta = 0`` stays feasible), checked when the path is traced.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If ``g``'s column count is not ``n`` or ``h``'s length does
                not match the rows of ``g``.
        """
        g_block = np.atleast_2d(np.asarray(g, dtype=np.float64))
        h_block = np.atleast_1d(np.asarray(h, dtype=np.float64))
        if self.x.ndim == 2 and g_block.shape[1] != self.x.shape[1]:
            msg = f"inequality: coefficient matrix must have {self.x.shape[1]} columns, got shape {g_block.shape}"
            raise ValueError(msg)
        if h_block.shape[0] != g_block.shape[0]:
            msg = f"inequality: h must have {g_block.shape[0]} entries to match the rows, got {h_block.shape[0]}"
            raise ValueError(msg)
        self._g_blocks.append(g_block)
        self._h_blocks.append(h_block)
        return self

    def trace(self) -> Lasso:
        """Assemble the pieces, build the ``Lasso``, and trace the full path.

        Returns:
            The traced :class:`Lasso`, whose ``path`` holds the breakpoints of the
            (constrained) regularisation path.
        """
        g = np.vstack(self._g_blocks) if self._g_blocks else None
        h = np.concatenate(self._h_blocks) if self._h_blocks else None
        return Lasso(x=self.x, y=self.y, g=g, h=h, nonneg=self._nonneg)

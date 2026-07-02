"""Fluent builder for assembling a Critical Line Algorithm problem.

A thin, chainable convenience layer over the explicit :class:`cvxcla.cla.CLA`
constructor. It exists purely for readability: portfolio practitioners expect to
say "long-only, fully invested" rather than to remember that the budget is
encoded as ``a=np.ones((1, n)), b=np.ones(1)``. Every method maps one-to-one onto
a constructor argument, so the builder adds no modelling power and imposes no
expression algebra: it accepts the same polyhedral pieces the CLA already
supports (a quadratic objective, box bounds, linear equalities ``A w = b``, and
linear inequalities ``G w <= h``) and nothing else. Anything the explicit
constructor cannot trace, the builder cannot express either.

The terminal :meth:`ProblemBuilder.trace` builds the ``CLA`` and runs the full
parametric trace, returning the solved object whose ``frontier`` and
``turning_points`` describe the entire efficient frontier (not a single optimum,
which is the distinction from a one-shot convex solver).

Examples:
    >>> import numpy as np
    >>> from cvxcla import CLA
    >>> rng = np.random.default_rng(0)
    >>> mean = rng.uniform(0.0, 1.0, 4)
    >>> covariance = np.eye(4)
    >>> cla = CLA.problem(mean, covariance).long_only().budget().trace()
    >>> len(cla) > 0
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .cla import CLA
from .lasso import Lasso
from .operators import QuadraticForm


class ProblemBuilder:
    """Chainable builder that assembles the polyhedral pieces of a CLA problem.

    Construct one via :meth:`cvxcla.cla.CLA.problem`, chain the constraint
    methods (each returns ``self``), and finish with :meth:`trace`. The builder
    is a convenience over the explicit ``CLA(...)`` constructor and validates
    shapes with actionable messages as pieces are added.

    Attributes:
        mean: Vector of expected returns, fixing the problem dimension ``n``.
        covariance: The covariance, either a plain ``numpy`` array or a
            ``QuadraticForm`` backend (e.g. ``FactorCovariance``), passed through
            to ``CLA`` unchanged so the structured backends keep their advantage.
    """

    def __init__(self, mean: NDArray[np.float64], covariance: NDArray[np.float64] | QuadraticForm) -> None:
        """Start a builder for an ``n``-asset problem.

        Args:
            mean: Vector of expected returns of length ``n``.
            covariance: Covariance matrix or ``QuadraticForm`` backend.
        """
        self.mean = np.asarray(mean, dtype=np.float64)
        self.covariance = covariance
        self._lower: NDArray[np.float64] | None = None
        self._upper: NDArray[np.float64] | None = None
        self._a_blocks: list[NDArray[np.float64]] = []
        self._b_blocks: list[NDArray[np.float64]] = []
        self._g_blocks: list[NDArray[np.float64]] = []
        self._h_blocks: list[NDArray[np.float64]] = []

    @property
    def _n(self) -> int:
        """Number of assets ``n``, fixed by ``mean``."""
        return int(self.mean.shape[0])

    def _as_vector(self, value: float | NDArray[np.float64], name: str) -> NDArray[np.float64]:
        """Broadcast a scalar or length-``n`` array to a length-``n`` vector.

        Args:
            value: A scalar (applied to every asset) or a length-``n`` array.
            name: Argument name, used in the error message.

        Returns:
            A fresh length-``n`` float array.

        Raises:
            ValueError: If an array is passed whose length is not ``n``.
        """
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            return np.full(self._n, float(array))
        if array.shape != (self._n,):
            msg = f"{name} must be a scalar or a length-{self._n} vector, got shape {array.shape}"
            raise ValueError(msg)
        return array.astype(np.float64, copy=True)

    def bounds(self, lower: float | NDArray[np.float64], upper: float | NDArray[np.float64]) -> ProblemBuilder:
        """Set the box bounds ``lower <= w <= upper``.

        Args:
            lower: Lower bound, a scalar (same for every asset) or length-``n`` array.
            upper: Upper bound, a scalar or length-``n`` array.

        Returns:
            ``self``, for chaining.
        """
        self._lower = self._as_vector(lower, "lower")
        self._upper = self._as_vector(upper, "upper")
        return self

    def long_only(self, upper: float | NDArray[np.float64] = 1.0) -> ProblemBuilder:
        """Set long-only box bounds ``0 <= w <= upper`` (``upper`` defaults to ``1``).

        Args:
            upper: Upper bound, a scalar or length-``n`` array; defaults to ``1.0``.

        Returns:
            ``self``, for chaining.
        """
        return self.bounds(0.0, upper)

    def budget(self, total: float = 1.0) -> ProblemBuilder:
        """Add the fully-invested budget constraint ``sum(w) = total``.

        This is the canonical all-ones equality row; ``total=0`` gives a
        dollar-neutral book. Equivalent to ``equality(np.ones(n), total)``.

        Args:
            total: The right-hand side of ``sum(w) = total``; defaults to ``1.0``.

        Returns:
            ``self``, for chaining.
        """
        return self.equality(np.ones(self._n), total)

    def equality(self, a: NDArray[np.float64], b: float | NDArray[np.float64]) -> ProblemBuilder:
        """Add one or more equality rows ``A w = b``.

        Accepts a single row (a length-``n`` vector with a scalar right-hand side)
        or a block of rows (an ``(m, n)`` matrix with a length-``m`` right-hand
        side). Repeated calls accumulate rows, so a budget plus a sector-neutrality
        block can be added separately.

        Args:
            a: A length-``n`` row vector or an ``(m, n)`` matrix.
            b: The matching right-hand side: a scalar for a single row, or a
                length-``m`` vector for a block.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If ``a`` does not have ``n`` columns, or ``b``'s length
                does not match the number of rows of ``a``.
        """
        a_block = np.atleast_2d(np.asarray(a, dtype=np.float64))
        b_block = np.atleast_1d(np.asarray(b, dtype=np.float64))
        self._validate_rows(a_block, b_block, "equality", "b")
        self._a_blocks.append(a_block)
        self._b_blocks.append(b_block)
        return self

    def inequality(self, g: NDArray[np.float64], h: float | NDArray[np.float64]) -> ProblemBuilder:
        """Add one or more inequality rows ``G w <= h``.

        Like :meth:`equality` but for ``<=`` rows (e.g. a group- or
        sector-exposure cap). A ``>=`` row is expressed by negating both ``g`` and
        ``h``. Repeated calls accumulate rows.

        Args:
            g: A length-``n`` row vector or a ``(p, n)`` matrix.
            h: The matching right-hand side: a scalar for a single row, or a
                length-``p`` vector for a block.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If ``g`` does not have ``n`` columns, or ``h``'s length
                does not match the number of rows of ``g``.
        """
        g_block = np.atleast_2d(np.asarray(g, dtype=np.float64))
        h_block = np.atleast_1d(np.asarray(h, dtype=np.float64))
        self._validate_rows(g_block, h_block, "inequality", "h")
        self._g_blocks.append(g_block)
        self._h_blocks.append(h_block)
        return self

    def _validate_rows(self, lhs: NDArray[np.float64], rhs: NDArray[np.float64], method: str, rhs_name: str) -> None:
        """Check a constraint block has ``n`` columns and a matching right-hand side.

        Args:
            lhs: The ``(m, n)`` coefficient block.
            rhs: The length-``m`` right-hand side.
            method: The calling method name, used in error messages.
            rhs_name: The right-hand-side argument name, used in error messages.

        Raises:
            ValueError: If the column count is not ``n`` or the lengths disagree.
        """
        if lhs.shape[1] != self._n:
            msg = f"{method}: coefficient matrix must have {self._n} columns, got shape {lhs.shape}"
            raise ValueError(msg)
        if rhs.shape[0] != lhs.shape[0]:
            msg = f"{method}: {rhs_name} must have {lhs.shape[0]} entries to match the rows, got {rhs.shape[0]}"
            raise ValueError(msg)

    def trace(self) -> CLA:
        """Assemble the pieces, build the ``CLA``, and run the full trace.

        Returns:
            The solved :class:`cvxcla.cla.CLA`, whose ``frontier`` and
            ``turning_points`` describe the entire efficient frontier.

        Raises:
            ValueError: If no box bounds were set (call :meth:`bounds` or
                :meth:`long_only`), or no equality constraint was added (call
                :meth:`budget` or :meth:`equality`).
        """
        if self._lower is None or self._upper is None:
            msg = "set box bounds before tracing: call .long_only() or .bounds(lower, upper)"
            raise ValueError(msg)
        if not self._a_blocks:
            msg = "a CLA problem needs an equality constraint: call .budget() or .equality(A, b)"
            raise ValueError(msg)

        g = np.vstack(self._g_blocks) if self._g_blocks else None
        h = np.concatenate(self._h_blocks) if self._h_blocks else None
        return CLA(
            mean=self.mean,
            covariance=self.covariance,
            lower_bounds=self._lower,
            upper_bounds=self._upper,
            a=np.vstack(self._a_blocks),
            b=np.concatenate(self._b_blocks),
            g=g,
            h=h,
        )


class LassoBuilder:
    """Chainable builder for a LASSO regularisation-path problem.

    The LASSO counterpart of :class:`ProblemBuilder`. Construct one via
    :meth:`cvxcla.lasso.Lasso.problem`, optionally add inequality constraints with
    :meth:`inequality`, and finish with :meth:`trace`, which builds the
    :class:`cvxcla.lasso.Lasso` and traces the entire regularisation path. Like the
    CLA builder it adds no modelling power: it accepts the same ``G beta <= h`` rows
    the ``Lasso`` already supports and nothing else.

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
            The traced :class:`cvxcla.lasso.Lasso`, whose ``path`` holds the
            breakpoints of the (constrained) regularisation path.
        """
        g = np.vstack(self._g_blocks) if self._g_blocks else None
        h = np.concatenate(self._h_blocks) if self._h_blocks else None
        return Lasso(x=self.x, y=self.y, g=g, h=h, nonneg=self._nonneg)

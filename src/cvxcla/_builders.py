"""Chainable builders that assemble the polyhedral pieces of a traced problem.

This module is a **leaf**: it imports numpy and :mod:`cvxcla.operators` and
nothing else from the package. In particular it never imports :mod:`cvxcla.cla`
or :mod:`cvxcla.lasso`, not even under ``TYPE_CHECKING``. The solver a builder
finally constructs is *injected* -- :meth:`cvxcla.cla.CLA.problem` passes ``CLA``
and :meth:`cvxcla.lasso.Lasso.problem` passes ``Lasso`` -- and each builder is
generic in that type, so ``.trace()`` still returns the precise solver class
without this module ever naming it.

That injection is what keeps the internal import graph acyclic while letting the
builders live apart from the solvers they build. The dependency runs one way,
``cla``/``lasso`` -> ``_builders``, and :mod:`tests.test_import_graph` pins it.

Neither builder adds modelling power. Each accepts exactly the polyhedral pieces
its solver already supports -- box bounds, linear equalities ``A w = b``, linear
inequalities ``G w <= h`` -- and maps them one-to-one onto constructor arguments.
Anything the explicit constructor cannot trace, the builder cannot express either.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from .operators import QuadraticForm

#: The solver a builder constructs. Bound at the call site by ``CLA.problem`` /
#: ``Lasso.problem``, so ``trace()`` returns the concrete class rather than
#: ``Any`` -- without this module importing either of them.
SolverT = TypeVar("SolverT")


def _as_block(
    lhs: NDArray[np.float64], rhs: float | NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalise a constraint row or block to ``(m, n)`` and ``(m,)`` float arrays.

    A single row may be given as a length-``n`` vector with a scalar right-hand
    side; both are promoted so the accumulation logic sees only blocks.

    Args:
        lhs: A length-``n`` row vector or an ``(m, n)`` coefficient matrix.
        rhs: A scalar or a length-``m`` right-hand side.

    Returns:
        The ``(lhs, rhs)`` pair as ``(m, n)`` and ``(m,)`` float arrays.
    """
    return np.atleast_2d(np.asarray(lhs, dtype=np.float64)), np.atleast_1d(np.asarray(rhs, dtype=np.float64))


def _validate_block(
    lhs: NDArray[np.float64],
    rhs: NDArray[np.float64],
    n: int | None,
    method: str,
    rhs_name: str,
) -> None:
    """Check a constraint block has ``n`` columns and a matching right-hand side.

    Args:
        lhs: The ``(m, n)`` coefficient block.
        rhs: The length-``m`` right-hand side.
        n: The expected column count, or ``None`` to skip the column check (the
            LASSO builder cannot know ``n`` until the design matrix is 2-D, and
            defers that diagnosis to :class:`cvxcla.lasso.Lasso`).
        method: The calling method name, used in error messages.
        rhs_name: The right-hand-side argument name, used in error messages.

    Raises:
        ValueError: If the column count is not ``n`` or the lengths disagree.
    """
    if n is not None and lhs.shape[1] != n:
        msg = f"{method}: coefficient matrix must have {n} columns, got shape {lhs.shape}"
        raise ValueError(msg)
    if rhs.shape[0] != lhs.shape[0]:
        msg = f"{method}: {rhs_name} must have {lhs.shape[0]} entries to match the rows, got {rhs.shape[0]}"
        raise ValueError(msg)


def _stack(
    lhs_blocks: list[NDArray[np.float64]], rhs_blocks: list[NDArray[np.float64]]
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Stack accumulated constraint blocks, or ``(None, None)`` when none were added.

    ``None`` is the solvers' own encoding of "no rows of this kind", so an unused
    builder method costs nothing downstream.

    Args:
        lhs_blocks: The accumulated ``(m_i, n)`` coefficient blocks.
        rhs_blocks: The accumulated length-``m_i`` right-hand sides.

    Returns:
        The vertically stacked ``(lhs, rhs)``, or ``(None, None)`` if empty.
    """
    if not lhs_blocks:
        return None, None
    return np.vstack(lhs_blocks), np.concatenate(rhs_blocks)


class ProblemBuilder(Generic[SolverT]):
    """Chainable builder that assembles the polyhedral pieces of a CLA problem.

    A thin, chainable convenience layer over the explicit
    :class:`cvxcla.cla.CLA` constructor. It exists purely for readability:
    portfolio practitioners expect to say "long-only, fully invested" rather than
    to remember that the budget is encoded as ``a=np.ones((1, n)), b=np.ones(1)``.
    Every method maps one-to-one onto a constructor argument, so the builder adds
    no modelling power and imposes no expression algebra: it accepts the same
    polyhedral pieces the CLA already supports (a quadratic objective, box bounds,
    linear equalities ``A w = b``, and linear inequalities ``G w <= h``) and
    nothing else. Anything the explicit constructor cannot trace, the builder
    cannot express either.

    Construct one via :meth:`cvxcla.cla.CLA.problem`, chain the constraint methods
    (each returns ``self``), and finish with :meth:`trace`, which builds the
    ``CLA`` and runs the full parametric trace, returning the solved object whose
    ``frontier`` and ``turning_points`` describe the entire efficient frontier
    (not a single optimum, which is the distinction from a one-shot convex solver).

    Attributes:
        mean: Vector of expected returns, fixing the problem dimension ``n``.
        covariance: The covariance, either a plain ``numpy`` array or a
            ``QuadraticForm`` backend (e.g. ``FactorCovariance``), passed through
            to ``CLA`` unchanged so the structured backends keep their advantage.

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

    def __init__(
        self,
        mean: NDArray[np.float64],
        covariance: NDArray[np.float64] | QuadraticForm,
        solver: Callable[..., SolverT],
    ) -> None:
        """Start a builder for an ``n``-asset problem.

        Args:
            mean: Vector of expected returns of length ``n``.
            covariance: Covariance matrix or ``QuadraticForm`` backend.
            solver: The class :meth:`trace` constructs, injected by
                :meth:`cvxcla.cla.CLA.problem` so this module never imports it.
        """
        self.mean = np.asarray(mean, dtype=np.float64)
        self.covariance = covariance
        self._solver = solver
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

    def bounds(self, lower: float | NDArray[np.float64], upper: float | NDArray[np.float64]) -> ProblemBuilder[SolverT]:
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

    def long_only(self, upper: float | NDArray[np.float64] = 1.0) -> ProblemBuilder[SolverT]:
        """Set long-only box bounds ``0 <= w <= upper`` (``upper`` defaults to ``1``).

        Args:
            upper: Upper bound, a scalar or length-``n`` array; defaults to ``1.0``.

        Returns:
            ``self``, for chaining.
        """
        return self.bounds(0.0, upper)

    def budget(self, total: float = 1.0) -> ProblemBuilder[SolverT]:
        """Add the fully-invested budget constraint ``sum(w) = total``.

        This is the canonical all-ones equality row; ``total=0`` gives a
        dollar-neutral book. Equivalent to ``equality(np.ones(n), total)``.

        Args:
            total: The right-hand side of ``sum(w) = total``; defaults to ``1.0``.

        Returns:
            ``self``, for chaining.
        """
        return self.equality(np.ones(self._n), total)

    def equality(self, a: NDArray[np.float64], b: float | NDArray[np.float64]) -> ProblemBuilder[SolverT]:
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
        a_block, b_block = _as_block(a, b)
        _validate_block(a_block, b_block, self._n, "equality", "b")
        self._a_blocks.append(a_block)
        self._b_blocks.append(b_block)
        return self

    def inequality(self, g: NDArray[np.float64], h: float | NDArray[np.float64]) -> ProblemBuilder[SolverT]:
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
        g_block, h_block = _as_block(g, h)
        _validate_block(g_block, h_block, self._n, "inequality", "h")
        self._g_blocks.append(g_block)
        self._h_blocks.append(h_block)
        return self

    def trace(self) -> SolverT:
        """Assemble the pieces, build the ``CLA``, and run the full trace.

        Returns:
            The solved :class:`cvxcla.cla.CLA`, whose ``frontier`` and
            ``turning_points`` describe the entire efficient frontier.

        Raises:
            ValueError: If no box bounds were set (call :meth:`bounds` or
                :meth:`long_only`), or no equality constraint was added (call
                :meth:`budget` or :meth:`equality`).
        """
        lower, upper = self._resolved_bounds()
        if not self._a_blocks:
            msg = "a CLA problem needs an equality constraint: call .budget() or .equality(A, b)"
            raise ValueError(msg)

        g, h = _stack(self._g_blocks, self._h_blocks)
        return self._solver(
            mean=self.mean,
            covariance=self.covariance,
            lower_bounds=lower,
            upper_bounds=upper,
            a=np.vstack(self._a_blocks),
            b=np.concatenate(self._b_blocks),
            g=g,
            h=h,
        )

    def _resolved_bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the box bounds, raising if they were never set.

        Returns:
            The ``(lower, upper)`` box-bound vectors.

        Raises:
            ValueError: If no box bounds were set (call :meth:`bounds` or
                :meth:`long_only`).
        """
        if self._lower is None or self._upper is None:
            msg = "set box bounds before tracing: call .long_only() or .bounds(lower, upper)"
            raise ValueError(msg)
        return self._lower, self._upper


class LassoBuilder(Generic[SolverT]):
    """Chainable builder for a LASSO regularisation-path problem.

    The LASSO counterpart of :class:`ProblemBuilder`. Construct one via
    :meth:`cvxcla.lasso.Lasso.problem`, optionally add inequality constraints with
    :meth:`inequality`, and finish with :meth:`trace`, which builds the
    :class:`cvxcla.lasso.Lasso` and traces the entire regularisation path. Like the
    CLA builder it adds no modelling power: it accepts the same ``G beta <= h``
    rows the ``Lasso`` already supports and nothing else.

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

    def __init__(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        solver: Callable[..., SolverT],
    ) -> None:
        """Start a builder for design matrix ``x`` and response ``y``.

        Args:
            x: Design matrix of shape ``(m, n)``.
            y: Response vector of shape ``(m,)``.
            solver: The class :meth:`trace` constructs, injected by
                :meth:`cvxcla.lasso.Lasso.problem` so this module never imports it.
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self._solver = solver
        self._g_blocks: list[NDArray[np.float64]] = []
        self._h_blocks: list[NDArray[np.float64]] = []
        self._nonneg = False

    def non_negative(self) -> LassoBuilder[SolverT]:
        """Restrict the coefficients to ``beta >= 0`` (the non-negative LASSO).

        Under ``beta >= 0`` the l1 penalty collapses to the linear term
        ``lam * sum(beta)``, so the path is the standard one restricted to positive
        signs -- structurally the CLA's box-bounded parametric QP.

        Returns:
            ``self``, for chaining.
        """
        self._nonneg = True
        return self

    def inequality(self, g: NDArray[np.float64], h: float | NDArray[np.float64]) -> LassoBuilder[SolverT]:
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
        g_block, h_block = _as_block(g, h)
        # A 1-D design has no column count to check against; ``Lasso`` rejects it
        # with its own (better) diagnosis when the path is traced.
        n = int(self.x.shape[1]) if self.x.ndim == 2 else None
        _validate_block(g_block, h_block, n, "inequality", "h")
        self._g_blocks.append(g_block)
        self._h_blocks.append(h_block)
        return self

    def trace(self) -> SolverT:
        """Assemble the pieces, build the ``Lasso``, and trace the full path.

        Returns:
            The traced :class:`cvxcla.lasso.Lasso`, whose ``path`` holds the
            breakpoints of the (constrained) regularisation path.
        """
        g, h = _stack(self._g_blocks, self._h_blocks)
        return self._solver(x=self.x, y=self.y, g=g, h=h, nonneg=self._nonneg)

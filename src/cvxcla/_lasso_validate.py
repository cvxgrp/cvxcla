"""Input validation for :class:`cvxcla.lasso.Lasso` construction.

The LASSO accepts its quadratic form either as a dense design ``(x, y)`` or as a
``QuadraticForm`` operator plus the linear term ``X^T y``, optionally with linear
inequality constraints ``G beta <= h``. The shape and consistency checks for those
two input modes, plus the constraint checks, are pure functions here so
``Lasso.__post_init__`` reduces to dispatching to them before handing off to the
tracer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .operators import QuadraticForm


def validate_operator_inputs(
    quad_form: QuadraticForm | None,
    linear: NDArray[np.float64] | None,
    x: NDArray[np.float64] | None,
    y: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Validate the operator-mode inputs ``quad_form`` and ``linear`` (``X^T y``).

    Args:
        quad_form: The quadratic form ``H`` as a :class:`QuadraticForm` operator.
        linear: The linear term ``X^T y`` (must be the 1d vector).
        x: The dense design matrix, which must be absent in operator mode.
        y: The dense response vector, which must be absent in operator mode.

    Returns:
        The validated ``linear`` term as a 1d ``float64`` array.

    Raises:
        ValueError: If only one of ``quad_form``/``linear`` is given, a design
            ``(x, y)`` is also supplied, or ``linear`` is not the 1d ``X^T y``.
    """
    if quad_form is None or linear is None:
        msg = "quad_form and linear (X^T y) must be provided together"
        raise ValueError(msg)
    _reject_conflicting_design(x, y)
    linear = np.asarray(linear, dtype=np.float64)
    if linear.ndim != 1:
        msg = f"linear must be the 1d vector X^T y, got shape {linear.shape}"
        raise ValueError(msg)
    return linear


def _reject_conflicting_design(x: NDArray[np.float64] | None, y: NDArray[np.float64] | None) -> None:
    """Reject a dense design ``(x, y)`` supplied alongside the operator inputs.

    Args:
        x: The dense design matrix, which must be absent in operator mode.
        y: The dense response vector, which must be absent in operator mode.

    Raises:
        ValueError: If either ``x`` or ``y`` is provided.
    """
    if x is not None or y is not None:
        msg = "supply either a design (x, y) or an operator (quad_form, linear), not both"
        raise ValueError(msg)


def validate_design_inputs(x: NDArray[np.float64] | None, y: NDArray[np.float64] | None) -> None:
    """Validate the dense-design inputs ``x`` and ``y``.

    Args:
        x: The design matrix of shape ``(m, n)``.
        y: The response vector of shape ``(m,)``.

    Raises:
        ValueError: If ``x``/``y`` are missing, ``x`` is not a 2d design matrix,
            or ``y``'s length does not match ``x``'s row count.
    """
    if x is None or y is None:
        msg = "provide a design (x, y) or an operator (quad_form, linear)"
        raise ValueError(msg)
    if x.ndim != 2:
        msg = f"x must be a 2d design matrix, got shape {x.shape}"
        raise ValueError(msg)
    if y.shape != (x.shape[0],):
        msg = f"y must have shape ({x.shape[0]},), got {y.shape}"
        raise ValueError(msg)


def validate_constraints(
    g: NDArray[np.float64] | None,
    h: NDArray[np.float64] | None,
    dimension: int,
    tol: float,
) -> None:
    """Validate the optional inequality constraints ``G beta <= h``.

    Args:
        g: Inequality matrix ``G`` of ``G beta <= h`` (``None`` for the plain LASSO).
        h: Inequality right-hand side ``h`` (``None`` for the plain LASSO).
        dimension: The problem dimension ``n`` (number of features).
        tol: Tolerance below which an ``h`` entry counts as non-positive.

    Raises:
        ValueError: If only one of ``g``/``h`` is given, their shapes are
            inconsistent with the problem dimension, or any ``h`` entry is not
            strictly positive (which would make ``beta = 0`` infeasible).
    """
    if g is None and h is None:
        return
    if g is None or h is None:
        msg = "g and h must be provided together"
        raise ValueError(msg)
    _validate_constraint_shapes(g, h, dimension, tol)


def _validate_constraint_shapes(
    g: NDArray[np.float64],
    h: NDArray[np.float64],
    dimension: int,
    tol: float,
) -> None:
    """Validate the shapes and positivity of a fully-provided ``G beta <= h``.

    Args:
        g: Inequality matrix ``G`` (both ``g`` and ``h`` known to be present).
        h: Inequality right-hand side ``h``.
        dimension: The problem dimension ``n`` (number of features).
        tol: Tolerance below which an ``h`` entry counts as non-positive.

    Raises:
        ValueError: If ``g``'s shape is inconsistent with ``h`` and the problem
            dimension, or any ``h`` entry is not strictly positive.
    """
    if g.shape != (h.shape[0], dimension):
        msg = f"g must have shape ({h.shape[0]}, {dimension}), got {g.shape}"
        raise ValueError(msg)
    if np.any(h <= tol):
        msg = "h must be strictly positive so beta = 0 is feasible (equality/zero-h needs a feasibility seed)"
        raise ValueError(msg)


def validate_equality(
    a: NDArray[np.float64],
    g: NDArray[np.float64] | None,
    dimension: int,
) -> NDArray[np.float64]:
    """Validate the homogeneous equality constraints ``A beta = 0``.

    Args:
        a: Equality matrix ``A`` of ``A beta = 0``.
        g: The inequality matrix, which must be absent alongside ``a``.
        dimension: The problem dimension ``n`` (number of features).

    Returns:
        ``a`` as a 2d ``float64`` array.

    Raises:
        ValueError: If ``a`` is not a 2d matrix with ``n`` columns, or ``g`` is
            also given (the leverage-CLA route needs homogeneous rows, and
            ``G beta <= 0`` rows are all tight at ``beta = 0``, which makes the
            first vertex degenerate).
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != dimension:
        msg = f"a must have shape (m, {dimension}), got {a.shape}"
        raise ValueError(msg)
    if g is not None:
        msg = "equality rows a cannot be combined with inequality rows g"
        raise ValueError(msg)
    return a


def validate_bounds(
    lower: NDArray[np.float64] | None,
    upper: NDArray[np.float64] | None,
    b: NDArray[np.float64] | None,
    a: NDArray[np.float64] | None,
    g: NDArray[np.float64] | None,
    nonneg: bool,
    dimension: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate per-coefficient bounds ``lower <= beta <= upper`` and the right-hand side ``b``.

    Args:
        lower: Lower bounds, each ``<= 0`` (``-inf`` allowed); ``None`` for none.
        upper: Upper bounds, each ``>= 0`` (``inf`` allowed); ``None`` for none.
        b: Right-hand side of ``A beta = b``; ``None`` for ``b = 0``.
        a: The (validated) equality matrix, or ``None``.
        g: The inequality matrix, which must be absent alongside bounds or ``b``.
        nonneg: Whether ``beta >= 0`` was requested, which tightens ``lower`` to ``0``.
        dimension: The problem dimension ``n`` (number of features).

    Returns:
        ``(lower, upper, b)`` as ``float64`` arrays, with absent bounds infinite and
        an absent ``b`` zero.

    Raises:
        ValueError: If a bound has the wrong shape or excludes ``beta_j = 0``, ``b``
            is given without ``a`` or with the wrong length, or ``g`` is also given.
    """
    if g is not None and (lower is not None or upper is not None or b is not None):
        msg = "bounds and a right-hand side b cannot be combined with inequality rows g"
        raise ValueError(msg)
    lo = np.full(dimension, -np.inf) if lower is None else np.asarray(lower, dtype=np.float64)
    up = np.full(dimension, np.inf) if upper is None else np.asarray(upper, dtype=np.float64)
    if lo.shape != (dimension,) or up.shape != (dimension,):
        msg = f"lower and upper must have shape ({dimension},), got {lo.shape} and {up.shape}"
        raise ValueError(msg)
    if np.any(lo > 0) or np.any(up < 0):
        msg = "bounds must admit beta_j = 0: every lower bound <= 0 and every upper bound >= 0"
        raise ValueError(msg)
    if nonneg:
        lo = np.maximum(lo, 0.0)
    rows = 0 if a is None else a.shape[0]
    if b is not None and a is None:
        msg = "a right-hand side b needs equality rows a"
        raise ValueError(msg)
    rhs = np.zeros(rows) if b is None else np.asarray(b, dtype=np.float64).reshape(-1)
    if rhs.shape != (rows,):
        msg = f"b must have shape ({rows},), got {rhs.shape}"
        raise ValueError(msg)
    return lo, up, rhs

"""Input validation for :class:`cvxcla.lasso.Lasso` construction.

The LASSO accepts its quadratic form either as a dense design ``(x, y)`` or as a
``QuadraticForm`` operator plus the linear term ``X^T y``, optionally with linear
inequality constraints ``G beta <= h``. The shape and consistency checks for those
two input modes, plus the constraint check, are pure functions here so
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

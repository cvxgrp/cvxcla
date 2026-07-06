"""Operator protocol alias and the parametric-path helpers built on cvx-linalg.

The parametric active-set path tracer reaches its Hessian through the cvx-linalg
symmetric-operator protocol (``matvec`` / ``block_matvec`` / ``solve_free`` /
``rcond_free``). :data:`QuadraticForm` is that contract; :data:`CovarianceOperator`
is a backward-compatible alias for the portfolio (covariance) setting. The
concrete backends live in :mod:`cvx.linalg`; :mod:`cvxcla.operators.builders`
assembles them from CLA / LASSO inputs.

The generic linear algebra -- the bordered KKT (Schur complement) solve and the
affine projection -- also lives in :mod:`cvx.linalg`. What remains here is the
thin homotopy-specific glue: adapting the loop's boolean masks to the operators'
integer-index API, and packing the constant / ``lambda``-slope pair of a
parametric segment into the shared multi-RHS solve.
"""

from __future__ import annotations

import numpy as np
from cvx.linalg import SymmetricOperator
from cvx.linalg import bordered_solve as _bordered_solve
from numpy.typing import NDArray

# The Hessian contract for a parametric active-set path. In the CLA it is the
# covariance ``Sigma``; in a LASSO / LARS path it is the Gram matrix ``X.T @ X``.
QuadraticForm = SymmetricOperator
CovarianceOperator = SymmetricOperator


def cross(operator: SymmetricOperator, free: NDArray[np.bool_], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Free-to-blocked cross product ``H[free][:, ~free] @ x[~free]`` from a boolean mask.

    Adapts the boolean-mask indexing the path tracers use to the integer-index
    :meth:`~cvx.linalg.SymmetricOperator.block_matvec` of a symmetric operator.

    Args:
        operator: The symmetric operator (Hessian) backend.
        free: Boolean mask of shape ``(n,)`` selecting the free coordinates.
        x: Full-length vector of shape ``(n,)``; only ``x[~free]`` enters the product.

    Returns:
        Vector of shape ``(n_free,)``.
    """
    result: NDArray[np.float64] = operator.block_matvec(np.flatnonzero(free), np.flatnonzero(~free), x[~free])
    return result


def bordered_solve(
    quad: SymmetricOperator,
    free: NDArray[np.bool_],
    c_free: NDArray[np.float64],
    rhs_const: NDArray[np.float64],
    rhs_slope: NDArray[np.float64],
    d_const: NDArray[np.float64],
    d_slope: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Solve the bordered KKT system for a parametric segment's constant and slope parts.

    A thin adapter over :func:`cvx.linalg.bordered_solve`: it converts the loop's
    boolean *free* mask to integer indices and packs the constant and ``lambda``-slope
    right-hand sides as the two columns of one multi-RHS solve, so ``H_FF`` and the
    Schur complement are factorised once. Returns
    ``(x_const, x_slope, nu_const, nu_slope)`` (multipliers empty when there are no
    constraint rows).
    """
    x, nu = _bordered_solve(
        quad,
        np.flatnonzero(free),
        c_free,
        np.column_stack([rhs_const, rhs_slope]),
        np.column_stack([d_const, d_slope]),
    )
    return x[:, 0], x[:, 1], nu[:, 0], nu[:, 1]

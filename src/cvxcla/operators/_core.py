"""Operator protocol alias and the bordered KKT solve shared by the CLA and LASSO paths.

The parametric active-set path tracer reaches its Hessian through the cvx-linalg
symmetric-operator protocol (``matvec`` / ``block_matvec`` / ``solve_free`` /
``rcond_free``). :data:`QuadraticForm` is that contract; :data:`CovarianceOperator`
is a backward-compatible alias for the portfolio (covariance) setting. The
concrete backends live in :mod:`cvx.linalg`; :mod:`cvxcla.operators.builders`
assembles them from CLA / LASSO inputs.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from cvx.linalg import SymmetricOperator
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
    """Solve the bordered KKT system ``[[H_FF, C_F^T], [C_F, 0]] @ [x; nu] = [rhs; d]``.

    The shared segment solve of ``cvxcla.cla`` and ``cvxcla.lasso``. ``H_FF`` is the
    free block of the quadratic form, ``C_F`` the constraint rows on the free columns.
    The constant and ``lambda``-slope right-hand sides are solved together so ``H_FF``
    is factorised once; ``mc == 0`` (no constraint rows) is the plain free-block solve.
    ``H_FF`` is reached only through :meth:`~cvx.linalg.SymmetricOperator.solve_free`, so
    structured backends never materialise an ``n x n`` matrix. Returns
    ``(x_const, x_slope, nu_const, nu_slope)``: the free solution and constraint
    multipliers per system (multipliers empty if ``mc == 0``).
    """
    free_idx = np.flatnonzero(free)
    mc = c_free.shape[0]
    if mc == 0:
        sol = quad.solve_free(free_idx, np.column_stack([rhs_const, rhs_slope]))
        empty: NDArray[np.float64] = np.zeros(0)
        return sol[:, 0], sol[:, 1], empty, empty

    # One multi-RHS solve against H_FF covers the constraint columns C_F^T and both
    # the constant and slope systems, so H_FF is factorised a single time.
    solved = quad.solve_free(free_idx, np.column_stack([c_free.T, rhs_const, rhs_slope]))
    y = solved[:, :mc]  # H_FF^{-1} C_F^T
    z_const = solved[:, mc]
    z_slope = solved[:, mc + 1]

    # Schur complement C_F H_FF^{-1} C_F^T and the stacked multipliers.
    schur = c_free @ y
    nu = cast(
        "NDArray[np.float64]",
        np.linalg.solve(schur, np.column_stack([c_free @ z_const - d_const, c_free @ z_slope - d_slope])),
    )
    nu_const, nu_slope = nu[:, 0], nu[:, 1]

    # Back-substitute the free solution.
    return z_const - y @ nu_const, z_slope - y @ nu_slope, nu_const, nu_slope

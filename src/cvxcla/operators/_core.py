"""Shared contract and solver primitives for the covariance backends.

This module defines the :class:`QuadraticForm` protocol that every backend
implements, the deterministic symmetric reciprocal-condition helper they share,
and :func:`bordered_solve`, the bordered KKT solve used by both
``cvxcla.cla`` and ``cvxcla.lasso``. The concrete backends live in sibling
modules (``dense``, ``factor``, ``gram``) and are re-exported from the package
root ``cvxcla.operators``.
"""

from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# When the cheap LAPACK 1-norm condition *estimate* clears the singularity floor
# by this margin, the decision is conclusive; an estimate landing within the
# margin falls back to the exact symmetric rcond. The margin generously covers
# the gap between the 1-norm estimate and the true 2-norm reciprocal condition
# number, so the boolean answer is identical to the exact comparison.
_RCOND_ESTIMATE_MARGIN = 1.0e3


def _rcond_symmetric(block: NDArray[np.float64]) -> float:
    """Reciprocal 2-norm condition number of a symmetric (PSD) ``block``.

    Returns a value in ``[0, 1]``: ``1`` is perfectly conditioned and ``0`` is
    numerically singular. The condition number is read off the symmetric
    eigenvalues (``lambda_min / lambda_max``), so the result is deterministic
    and independent of the BLAS/LAPACK build, unlike the residual of a solve
    against a singular block. An empty block is treated as well conditioned.
    """
    if block.shape[0] == 0:
        return 1.0
    eigenvalues = np.linalg.eigvalsh(block)
    lam_max = float(eigenvalues[-1])
    if lam_max <= 0.0:
        return 0.0
    return max(float(eigenvalues[0]), 0.0) / lam_max


@runtime_checkable
class QuadraticForm(Protocol):
    """Operations a parametric active-set path tracer needs from its Hessian.

    The turning-point loop touches the symmetric positive (semi-)definite matrix
    ``H`` of the quadratic objective through a small number of operations: full
    matrix-vector products, solves against the *free* (active) principal block,
    and cross-products between free and blocked coordinates. In the Critical Line
    Algorithm ``H`` is the covariance ``Sigma``; in a LASSO / LARS path it is the
    Gram matrix ``X.T @ X``. Any object implementing this protocol can serve as
    that backend.

    The reference implementation is ``DenseCovariance``, which wraps an explicit
    matrix; structured backends (e.g. diagonal-plus-low-rank via the Woodbury
    identity) implement the same contract without ever materialising an
    ``n x n`` matrix.

    ``CovarianceOperator`` is kept as a backward-compatible alias of this
    protocol (see ``CovarianceOperator`` below).
    """

    @property
    def n(self) -> int:
        """Number of variables (the dimension of the quadratic form)."""
        ...  # pragma: no cover

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the matrix-vector product ``Sigma @ x``.

        Args:
            x: Vector of shape ``(n,)`` or matrix of shape ``(n, r)``.

        Returns:
            ``Sigma @ x`` with the same trailing shape as ``x``.
        """
        ...  # pragma: no cover

    def solve_free(self, free: NDArray[np.bool_], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve the free-block system ``Sigma[free][:, free] @ y = rhs``.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.
            rhs: Right-hand side of shape ``(n_free,)`` or ``(n_free, r)``.

        Returns:
            The solution ``y`` with the same shape as ``rhs``.
        """
        ...  # pragma: no cover

    def cross(self, free: NDArray[np.bool_], x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the free-to-blocked cross product ``Sigma[free][:, ~free] @ x[~free]``.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.
            x: Full-length vector of shape ``(n,)``; only the blocked entries
               ``x[~free]`` enter the product.

        Returns:
            Vector of shape ``(n_free,)``.
        """
        ...  # pragma: no cover

    def rcond_free(self, free: NDArray[np.bool_]) -> float:
        """Reciprocal condition number of the free block ``Sigma[free][:, free]``.

        Returns a value in ``[0, 1]`` (``1`` well conditioned, ``0`` singular).
        The CLA uses this to detect a rank-deficient free block, whose solve is
        unreliable, directly and portably, rather than inferring it from the
        (backend-dependent) magnitude of the resulting box violation.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.

        Returns:
            The reciprocal 2-norm condition number of the free block.
        """
        ...  # pragma: no cover


def bordered_solve(
    quad: QuadraticForm,
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
    ``H_FF`` is reached only through ``solve_free``, so structured backends never
    materialise an ``n x n`` matrix. Returns ``(x_const, x_slope, nu_const, nu_slope)``:
    the free solution and constraint multipliers per system (multipliers empty if
    ``mc == 0``). Callers assemble the right-hand sides and interpret the outputs.
    """
    mc = c_free.shape[0]
    if mc == 0:
        sol = quad.solve_free(free, np.column_stack([rhs_const, rhs_slope]))
        empty: NDArray[np.float64] = np.zeros(0)
        return sol[:, 0], sol[:, 1], empty, empty

    # One multi-RHS solve against H_FF covers the constraint columns C_F^T and both
    # the constant and slope systems, so H_FF is factorised a single time.
    solved = quad.solve_free(free, np.column_stack([c_free.T, rhs_const, rhs_slope]))
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


# Backward-compatible alias. The protocol was introduced as ``CovarianceOperator``
# for the portfolio (covariance) setting; it is the Hessian contract for any
# parametric active-set path, so the canonical name is now ``QuadraticForm``.
# Existing imports of ``CovarianceOperator`` keep working unchanged.
CovarianceOperator = QuadraticForm

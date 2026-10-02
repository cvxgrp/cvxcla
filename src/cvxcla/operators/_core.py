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
from scipy.linalg import solve_triangular  # type: ignore[import-untyped]

# The Hessian contract for a parametric active-set path. In the CLA it is the
# covariance ``Sigma``; in a LASSO / LARS path it is the Gram matrix ``X.T @ X``.
QuadraticForm = SymmetricOperator
CovarianceOperator = SymmetricOperator

# The singularity threshold for ``rcond_free``. A genuinely rank-deficient free
# block has a reciprocal condition number at round-off level (~1e-16); a
# well-posed or merely near-degenerate block sits many orders above it (>= ~1e-4
# across the degeneracy sweep in experiments/degeneracy_boundary.py). The 1e-12
# cut sits in the wide gap between the two and is the conventional
# numerical-singularity scale.
RCOND_FLOOR = 1e-12  # pragma: no mutate


def cross(operator: SymmetricOperator, free: NDArray[np.bool_], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Free-to-blocked cross product ``H[free][:, ~free] @ x[~free]`` from a boolean mask.

    Taken as one full product on ``x`` with its free entries zeroed,
    ``(H @ x_B)[free]``, which equals the cross block since the zeroed entries
    contribute nothing. A dense backend then runs a single contiguous matvec
    instead of first copying the ``(free, ~free)`` block out by fancy indexing,
    which dominated this product along the trace.

    Args:
        operator: The symmetric operator (Hessian) backend.
        free: Boolean mask of shape ``(n,)`` selecting the free coordinates.
        x: Full-length vector of shape ``(n,)``; only ``x[~free]`` enters the product.

    Returns:
        Vector of shape ``(n_free,)``.
    """
    result: NDArray[np.float64] = operator.matvec(np.where(free, 0.0, x))[free]
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

    The constraint rows are first replaced by an orthonormal basis of their span,
    ``C^T = Q R``, so the system solved is ``[[H_FF, Q], [Q^T, 0]]`` with target
    ``R^{-T} d`` and the multipliers are recovered as ``R^{-1} nu'``. Both describe
    the same solution, but the Schur complement ``C H_FF^{-1} C^T`` has condition
    number up to ``cond(H_FF) cond(C)^2``, so nearly dependent rows (several caps
    almost parallel on the free set) lose accuracy twice over, while
    ``Q^T H_FF^{-1} Q`` is no worse conditioned than ``H_FF`` itself. The weights
    are then accurate to about ``cond(C)`` times the round-off, the sensitivity of
    the constraint data, and the two factors are exactly what the free-block and
    active-row guards bound. With more rows than free coordinates the rows cannot
    be independent; the plain solve is used and reports the singular system. A
    single row is its own orthogonal basis, so it too takes the plain solve.
    """
    rhs = np.column_stack([rhs_const, rhs_slope])
    d = np.column_stack([d_const, d_slope])
    mc, n_free = c_free.shape
    if mc <= 1 or mc > n_free:
        x, nu = _bordered_solve(quad, np.flatnonzero(free), c_free, rhs, d)
    else:
        q, r = np.linalg.qr(c_free.T)
        x, nu_q = _bordered_solve(quad, np.flatnonzero(free), q.T, rhs, solve_triangular(r.T, d, lower=True))
        nu = solve_triangular(r, nu_q)
    return x[:, 0], x[:, 1], nu[:, 0], nu[:, 1]


def orthonormal_rows(a: NDArray[np.float64], b: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rewrite ``A w = b`` with orthonormal rows spanning the same space, ``Q^T w = R^{-T} b``.

    The affine set is unchanged, but nearly dependent rows stop hurting. Two nearly
    parallel rows (``1^T w = 1`` and ``(1 + eps v)^T w = 1`` encode ``v^T w = 0`` at
    scale ``eps``) give a linear program with an absolute feasibility tolerance that
    enforces the hidden row only to ``tol / eps``, and a projection whose Gram matrix
    ``C C^T`` has condition number ``cond(C)^2``; in the orthonormal form every
    direction of the span is enforced to the same tolerance and the Gram matrix is
    the identity. ``CLA`` drops redundant rows before tracing, so ``R`` is
    invertible there; rows that are dependent to round-off are passed through
    unchanged, for the caller to judge.

    Args:
        a: Equality-constraint matrix (``m x n``).
        b: Equality-constraint right-hand side (length ``m``).

    Returns:
        ``(Q^T, R^{-T} b)``; the input itself when there are no rows or they are
        dependent.
    """
    if a.shape[0] == 0 or a.shape[0] > a.shape[1]:
        return a, b
    q, r = np.linalg.qr(a.T)
    diag = np.abs(np.diag(r))
    if diag.min() <= a.shape[1] * np.finfo(np.float64).eps * diag.max():  # pragma: no mutate
        return a, b
    return q.T, solve_triangular(r.T, b, lower=True)

"""Builders that assemble cvx-linalg symmetric operators from CLA / LASSO inputs.

These replace the former operator *classes*. The numerics -- matrix-vector and
block products, free-block solves (Cholesky, Woodbury, maintained inverse), and
the reciprocal-condition check -- now live in :mod:`cvx.linalg`. Here we only
build the right operator from the domain inputs: a covariance matrix, a
returns / data matrix, or a factor model. The backward-compatible names
``DenseCovariance`` / ``IncrementalDenseCovariance`` / ``GramCovariance`` /
``FactorCovariance`` are kept as aliases of these builders.
"""

from __future__ import annotations

import numpy as np
from cvx.linalg import DenseOperator, FactorOperator, GramOperator, IncrementalDenseOperator
from numpy.typing import NDArray


def _symmetric_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return *matrix* as a float array after checking it is square and symmetric.

    Raises:
        ValueError: If *matrix* is not square or not symmetric to tolerance.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        msg = f"Covariance must be a square matrix, got shape {matrix.shape}"
        raise ValueError(msg)
    if not np.allclose(matrix, matrix.T):
        msg = "Covariance must be symmetric"
        raise ValueError(msg)
    return matrix


def dense_covariance(matrix: NDArray[np.float64]) -> DenseOperator:
    """Build a :class:`~cvx.linalg.DenseOperator` from an explicit symmetric covariance.

    Args:
        matrix: A symmetric ``(n, n)`` covariance matrix.

    Returns:
        A dense symmetric operator wrapping *matrix*.
    """
    return DenseOperator(_symmetric_matrix(matrix))


def incremental_dense_covariance(matrix: NDArray[np.float64]) -> IncrementalDenseOperator:
    """Build an :class:`~cvx.linalg.IncrementalDenseOperator` (maintained free-block inverse).

    A drop-in alternative to :func:`dense_covariance` for a loop that changes its
    free set one index at a time; see the operator's own caveats on numerics.

    Args:
        matrix: A symmetric ``(n, n)`` covariance matrix.

    Returns:
        A dense symmetric operator that maintains the free-block inverse.
    """
    return IncrementalDenseOperator(_symmetric_matrix(matrix))


def gram_covariance(returns: NDArray[np.float64], ridge: float = 0.0) -> GramOperator:
    """Build a :class:`~cvx.linalg.GramOperator` for the sample covariance of *returns*.

    The sample covariance ``X_c.T X_c / (T - 1)`` (``X_c`` the column-centered
    ``(T, n)`` data) is realised by absorbing the scale into the factor:
    ``M = sqrt(1 / (T - 1)) * X_c``, so the operator represents
    ``M.T M + ridge * I`` and never forms the ``n x n`` covariance.

    Args:
        returns: The ``(T, n)`` matrix of observations (``T >= 2``).
        ridge: A non-negative diagonal loading added to the covariance.

    Returns:
        A matrix-free Gram operator for the (ridged) sample covariance.

    Raises:
        ValueError: If *returns* is not a ``(T, n)`` matrix with ``T >= 2``.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2 or returns.shape[0] < 2:
        msg = f"returns must be a (T, n) matrix with T >= 2 observations, got shape {returns.shape}"
        raise ValueError(msg)
    t = returns.shape[0]
    centered = returns - returns.mean(axis=0, keepdims=True)
    factor = centered / np.sqrt(t - 1.0)
    return GramOperator(factor, ridge=ridge)


def factor_covariance(
    d: NDArray[np.float64],
    u: NDArray[np.float64],
    delta: NDArray[np.float64],
) -> FactorOperator:
    """Build a :class:`~cvx.linalg.FactorOperator` for ``Sigma = diag(d) + U Delta U.T``.

    Args:
        d: Positive idiosyncratic variances of shape ``(n,)``.
        u: Factor loadings of shape ``(n, k)``.
        delta: Factor covariance, either ``(k,)`` eigenvalues (a diagonal ``Delta``)
            or a symmetric positive-definite ``(k, k)`` matrix.

    Returns:
        A diagonal-plus-low-rank operator with Woodbury free-block solves.

    Raises:
        ValueError: If *delta* is neither a ``(k,)`` vector nor a ``(k, k)`` matrix.
    """
    d = np.asarray(d, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim == 1:
        inner = np.diag(delta)
    elif delta.ndim == 2:
        inner = delta
    else:
        msg = f"delta must be a (k,) vector or (k, k) matrix, got ndim {delta.ndim}"
        raise ValueError(msg)
    return FactorOperator(d, u, inner)


# Backward-compatible names: the operator *classes* are gone, but the familiar
# constructor-style names remain as builders returning cvx-linalg operators.
DenseCovariance = dense_covariance
IncrementalDenseCovariance = incremental_dense_covariance
GramCovariance = gram_covariance
FactorCovariance = factor_covariance

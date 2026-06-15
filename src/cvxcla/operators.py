#    Copyright 2023 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Covariance backend abstraction for the Critical Line Algorithm.

The turning-point loop of the CLA touches the covariance matrix through a
small number of operations: full matrix-vector products, solves against the
free-asset block, and cross-products between free and blocked assets. This
module defines that contract as the ``CovarianceOperator`` protocol, together
with two implementations:

- ``DenseCovariance``: the adapter that reproduces the behaviour of a plain
  ``numpy`` covariance matrix.
- ``FactorCovariance``: a diagonal-plus-low-rank covariance
  ``Sigma = diag(d) + U @ Delta @ U.T`` whose solves go through the Woodbury
  identity, so no ``n x n`` matrix is ever materialised. Memory and per-solve
  cost are ``O(n * k)`` instead of ``O(n^2)``.

See https://github.com/cvxgrp/cvxcla/issues/646 for the roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class CovarianceOperator(Protocol):
    """Operations the Critical Line Algorithm needs from a covariance matrix.

    Any object implementing this protocol can serve as the covariance backend
    of the CLA. The reference implementation is ``DenseCovariance``, which
    wraps an explicit matrix; structured backends (e.g. diagonal-plus-low-rank
    via the Woodbury identity) implement the same contract without ever
    materialising an ``n x n`` matrix.
    """

    @property
    def n(self) -> int:
        """Number of assets (the dimension of the covariance)."""
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


@dataclass(frozen=True)
class DenseCovariance:
    """Dense reference implementation of the ``CovarianceOperator`` protocol.

    Wraps an explicit symmetric positive (semi-)definite covariance matrix and
    implements the protocol operations with plain ``numpy`` calls, reproducing
    exactly what the CLA does with a raw matrix.

    Attributes:
        matrix: The covariance matrix of shape ``(n, n)``.

    Examples:
        >>> import numpy as np
        >>> cov = DenseCovariance(np.eye(2))
        >>> cov.n
        2
        >>> cov.matvec(np.array([1.0, 2.0]))
        array([1., 2.])
    """

    matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate that the wrapped matrix is a square, symmetric 2d array.

        Raises:
            ValueError: If the matrix is not two-dimensional and square, or
                not symmetric to numerical tolerance.
        """
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            msg = f"Covariance must be a square matrix, got shape {self.matrix.shape}"
            raise ValueError(msg)
        if not np.allclose(self.matrix, self.matrix.T):
            msg = "Covariance must be symmetric"
            raise ValueError(msg)

    @property
    def n(self) -> int:
        """Number of assets (the dimension of the covariance)."""
        return int(self.matrix.shape[0])

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the matrix-vector product ``Sigma @ x``.

        Args:
            x: Vector of shape ``(n,)`` or matrix of shape ``(n, r)``.

        Returns:
            ``Sigma @ x`` with the same trailing shape as ``x``.
        """
        return self.matrix @ x

    def solve_free(self, free: NDArray[np.bool_], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve the free-block system ``Sigma[free][:, free] @ y = rhs``.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.
            rhs: Right-hand side of shape ``(n_free,)`` or ``(n_free, r)``.

        Returns:
            The solution ``y`` with the same shape as ``rhs``.
        """
        return cast("NDArray[np.float64]", np.linalg.solve(self.matrix[np.ix_(free, free)], rhs))

    def cross(self, free: NDArray[np.bool_], x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the free-to-blocked cross product ``Sigma[free][:, ~free] @ x[~free]``.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.
            x: Full-length vector of shape ``(n,)``; only the blocked entries
               ``x[~free]`` enter the product.

        Returns:
            Vector of shape ``(n_free,)``.
        """
        blocked = ~free
        return self.matrix[np.ix_(free, blocked)] @ x[blocked]


@dataclass(frozen=True)
class FactorCovariance:
    """Diagonal-plus-low-rank covariance ``Sigma = diag(d) + U @ Delta @ U.T``.

    This backend implements the ``CovarianceOperator`` protocol for factor
    risk models and RMT-cleaned covariances (constant-residual-eigenvalue /
    eigenvalue clipping) without ever forming an ``n x n`` matrix. Solves
    against the free block go through the Woodbury identity

    ``Sigma_FF^{-1} = D_F^{-1} - D_F^{-1} U_F W^{-1} U_F^T D_F^{-1}``

    with the ``k x k`` correction matrix ``W = Delta^{-1} + U_F^T D_F^{-1} U_F``,
    so a solve costs ``O(n_F k^2 + k^3)`` instead of ``O(n_F^3)``. The
    cross-product against blocked assets only involves the low-rank part,
    since the diagonal contributes nothing off the diagonal.

    Attributes:
        d: Positive idiosyncratic variances of shape ``(n,)``.
        u: Factor loadings of shape ``(n, k)``.
        delta: Factor covariance, either as eigenvalues/variances of shape
            ``(k,)`` (interpreted as a diagonal matrix) or as a symmetric
            positive definite matrix of shape ``(k, k)``.

    Examples:
        >>> import numpy as np
        >>> cov = FactorCovariance(d=np.array([1.0, 1.0]), u=np.eye(2), delta=np.array([1.0, 1.0]))
        >>> cov.n
        2
        >>> cov.matvec(np.array([1.0, 2.0]))
        array([2., 4.])
    """

    d: NDArray[np.float64]
    u: NDArray[np.float64]
    delta: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate shapes, positivity of the diagonal, and symmetry of ``delta``.

        Raises:
            ValueError: If ``d`` is not a positive vector, ``u`` is not an
                ``(n, k)`` matrix matching ``d``, or ``delta`` is neither a
                ``(k,)`` vector nor a symmetric ``(k, k)`` matrix.
        """
        if self.d.ndim != 1 or not np.all(self.d > 0):
            msg = "d must be a vector of positive idiosyncratic variances"
            raise ValueError(msg)
        if self.u.ndim != 2 or self.u.shape[0] != self.d.shape[0]:
            msg = f"u must have shape (n, k) with n = {self.d.shape[0]}, got {self.u.shape}"
            raise ValueError(msg)
        k = self.u.shape[1]
        if self.delta.ndim == 1:
            if self.delta.shape[0] != k:
                msg = f"delta must have {k} entries, got {self.delta.shape[0]}"
                raise ValueError(msg)
            if not np.all(self.delta > 0):
                msg = "A diagonal delta must have positive entries"
                raise ValueError(msg)
        elif self.delta.ndim == 2:
            if self.delta.shape != (k, k):
                msg = f"delta must have shape ({k}, {k}), got {self.delta.shape}"
                raise ValueError(msg)
            if not np.allclose(self.delta, self.delta.T):
                msg = "delta must be symmetric"
                raise ValueError(msg)
        else:
            msg = f"delta must be a (k,) vector or (k, k) matrix, got ndim {self.delta.ndim}"
            raise ValueError(msg)

    @property
    def n(self) -> int:
        """Number of assets (the dimension of the covariance)."""
        return int(self.d.shape[0])

    @property
    def k(self) -> int:
        """Number of factors (the rank of the low-rank part)."""
        return int(self.u.shape[1])

    @cached_property
    def _delta_matrix(self) -> NDArray[np.float64]:
        """The factor covariance ``Delta`` as a ``(k, k)`` matrix."""
        if self.delta.ndim == 1:
            return np.diag(self.delta)
        return self.delta

    @cached_property
    def _delta_inv(self) -> NDArray[np.float64]:
        """The inverse of the ``(k, k)`` factor covariance."""
        if self.delta.ndim == 1:
            return np.diag(1.0 / self.delta)
        return cast("NDArray[np.float64]", np.linalg.inv(self.delta))

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the matrix-vector product ``Sigma @ x``.

        Args:
            x: Vector of shape ``(n,)`` or matrix of shape ``(n, r)``.

        Returns:
            ``Sigma @ x`` with the same trailing shape as ``x``, computed in
            ``O(n k)`` per column.
        """
        low_rank = self.u @ (self._delta_matrix @ (self.u.T @ x))
        diagonal = self.d * x if x.ndim == 1 else self.d[:, None] * x
        return diagonal + low_rank

    def solve_free(self, free: NDArray[np.bool_], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve the free-block system ``Sigma[free][:, free] @ y = rhs`` via Woodbury.

        Forms the ``k x k`` correction matrix
        ``W = Delta^{-1} + U_F^T D_F^{-1} U_F`` and solves against it, so the
        cost is ``O(n_F k^2 + k^3 + n_F k r)`` for ``r`` right-hand sides.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.
            rhs: Right-hand side of shape ``(n_free,)`` or ``(n_free, r)``.

        Returns:
            The solution ``y`` with the same shape as ``rhs``.
        """
        d_free = self.d[free]
        u_free = self.u[free]

        rhs_2d = rhs if rhs.ndim == 2 else rhs[:, None]
        dinv_rhs = rhs_2d / d_free[:, None]
        dinv_u = u_free / d_free[:, None]

        w = self._delta_inv + u_free.T @ dinv_u
        solution = dinv_rhs - dinv_u @ np.linalg.solve(w, u_free.T @ dinv_rhs)
        return solution if rhs.ndim == 2 else solution[:, 0]

    def cross(self, free: NDArray[np.bool_], x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the free-to-blocked cross product ``Sigma[free][:, ~free] @ x[~free]``.

        Only the low-rank part contributes, since the diagonal of ``Sigma``
        has no off-diagonal block: the result is
        ``U_F @ Delta @ (U_out^T @ x_out)``, an ``O(n k)`` computation.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.
            x: Full-length vector of shape ``(n,)``; only the blocked entries
               ``x[~free]`` enter the product.

        Returns:
            Vector of shape ``(n_free,)``.
        """
        blocked = ~free
        return self.u[free] @ (self._delta_matrix @ (self.u[blocked].T @ x[blocked]))

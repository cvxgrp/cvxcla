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
with ``DenseCovariance``, the adapter that reproduces the behaviour of a plain
``numpy`` covariance matrix.

Structured implementations of the protocol (for example diagonal-plus-low-rank
covariances solved via the Woodbury identity) can then avoid forming any
``n x n`` matrix. See https://github.com/cvxgrp/cvxcla/issues/646 for the
roadmap; this module is Phase 1 (abstraction only, no behaviour change).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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
        return self.matrix.shape[0]

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
        return np.linalg.solve(self.matrix[np.ix_(free, free)], rhs)

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

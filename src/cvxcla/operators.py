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


class IncrementalDenseCovariance:
    """Dense backend that maintains ``Sigma_FF^{-1}`` across turning points.

    A drop-in alternative to ``DenseCovariance`` for the dense case. Both wrap an
    explicit covariance and return identical results; they differ only in how the
    free-block solve is computed. ``DenseCovariance`` factorises ``Sigma_FF`` from
    scratch at every turning point (``O(n_F^3)`` each). The CLA changes the free
    set by exactly one asset between consecutive turning points, so this backend
    instead *maintains* the explicit inverse ``Sigma_FF^{-1}`` and updates it with
    a rank-one bordered (asset enters) or deletion (asset leaves) formula
    (``O(n_F^2)`` each). Empirically this is roughly ``2x`` faster on dense
    problems of a few hundred assets (see ``experiments/inverse_cla.py``).

    Two caveats motivate this being opt-in rather than the default:

    - **Numerics.** A maintained inverse accumulates floating-point error over the
      ``O(n)`` rank-one updates of a trace, whereas a fresh solve is clean each
      step. On well-conditioned, positive-definite problems the difference is
      negligible, but for ill-conditioned or near-degenerate inputs the plain
      ``DenseCovariance`` (or a ``FactorCovariance``) is the safer choice. As a
      guard, any non-positive or non-finite pivot triggers a full refactorisation
      of that step, and any free-set change that is not a single-asset flip
      (e.g. the first solve of a trace) is computed from scratch.
    - **Scope.** The win is dense-only. The structured ``FactorCovariance`` never
      forms ``Sigma_FF`` --- it solves through Woodbury --- and is already faster
      than a maintained dense inverse once the universe is large.

    Attributes:
        matrix: The covariance matrix of shape ``(n, n)``.

    Examples:
        >>> import numpy as np
        >>> cov = IncrementalDenseCovariance(np.eye(3))
        >>> free = np.array([True, True, False])
        >>> rhs = np.array([1.0, 2.0])
        >>> np.allclose(cov.solve_free(free, rhs), rhs)
        True
    """

    def __init__(self, matrix: NDArray[np.float64]) -> None:
        """Wrap ``matrix`` and validate it is square and symmetric.

        Args:
            matrix: The covariance matrix of shape ``(n, n)``.

        Raises:
            ValueError: If the matrix is not a square, symmetric 2d array.
        """
        self._dense = DenseCovariance(matrix)
        self.matrix = self._dense.matrix
        # Cache of the maintained inverse and the free indices it is aligned to,
        # both in ascending index order. ``None`` until the first solve.
        self._free_idx: NDArray[np.intp] | None = None
        self._inv: NDArray[np.float64] | None = None

    @property
    def n(self) -> int:
        """Number of assets (the dimension of the covariance)."""
        return self._dense.n

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the matrix-vector product ``Sigma @ x`` (delegates to dense)."""
        return self._dense.matvec(x)

    def cross(self, free: NDArray[np.bool_], x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the free-to-blocked cross product (delegates to dense)."""
        return self._dense.cross(free, x)

    def solve_free(self, free: NDArray[np.bool_], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve ``Sigma[free][:, free] @ y = rhs`` using the maintained inverse.

        Updates the cached ``Sigma_FF^{-1}`` for the single-asset change since the
        previous call (or recomputes it from scratch when the change is not a lone
        flip or a pivot is degenerate), then returns ``Sigma_FF^{-1} @ rhs``.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.
            rhs: Right-hand side of shape ``(n_free,)`` or ``(n_free, r)``.

        Returns:
            The solution ``y`` with the same shape as ``rhs``.
        """
        cur = np.flatnonzero(free)
        inv = self._inverse_for(cur)
        self._free_idx, self._inv = cur, inv
        return cast("NDArray[np.float64]", inv @ rhs)

    def _inverse_for(self, cur: NDArray[np.intp]) -> NDArray[np.float64]:
        """Return ``Sigma[cur][:, cur]^{-1}``, updating the cache incrementally."""
        prev, prev_inv = self._free_idx, self._inv
        if prev is None or prev_inv is None:
            return self._refactor(cur)

        added = np.setdiff1d(cur, prev, assume_unique=True)
        removed = np.setdiff1d(prev, cur, assume_unique=True)
        if added.size == 1 and removed.size == 0:
            updated = self._insert(prev, prev_inv, int(added[0]), cur)
        elif removed.size == 1 and added.size == 0:
            updated = self._delete(prev, prev_inv, int(removed[0]))
        else:
            updated = None  # not a single-asset flip; fall back

        return updated if updated is not None else self._refactor(cur)

    def _refactor(self, cur: NDArray[np.intp]) -> NDArray[np.float64]:
        """Invert ``Sigma[cur][:, cur]`` from scratch."""
        if cur.size == 0:
            return np.zeros((0, 0))
        return cast("NDArray[np.float64]", np.linalg.inv(self.matrix[np.ix_(cur, cur)]))

    def _insert(
        self,
        prev: NDArray[np.intp],
        prev_inv: NDArray[np.float64],
        asset: int,
        cur: NDArray[np.intp],
    ) -> NDArray[np.float64] | None:
        """Rank-one bordered update for one asset entering the free set.

        Returns the inverse aligned to ascending ``cur`` order, or ``None`` if the
        Schur pivot is non-positive/non-finite (caller refactorises instead).
        """
        c = self.matrix[prev, asset]
        v = prev_inv @ c
        schur = float(self.matrix[asset, asset] - c @ v)
        if not np.isfinite(schur) or schur <= 0.0:
            return None
        k = prev.shape[0]
        aug = np.empty((k + 1, k + 1))
        aug[:k, :k] = prev_inv + np.outer(v, v) / schur
        aug[:k, k] = -v / schur
        aug[k, :k] = -v / schur
        aug[k, k] = 1.0 / schur
        # ``aug`` is ordered [prev..., asset]; permute to ascending ``cur`` order.
        perm = np.empty(k + 1, dtype=np.intp)
        is_new = cur == asset
        perm[is_new] = k
        perm[~is_new] = np.searchsorted(prev, cur[~is_new])
        return aug[np.ix_(perm, perm)]

    def _delete(
        self,
        prev: NDArray[np.intp],
        prev_inv: NDArray[np.float64],
        asset: int,
    ) -> NDArray[np.float64] | None:
        """Rank-one deletion update for one asset leaving the free set.

        Returns the inverse aligned to the remaining (ascending) order, or ``None``
        if the pivot is non-positive/non-finite (caller refactorises instead).
        """
        p = int(np.searchsorted(prev, asset))
        pivot = float(prev_inv[p, p])
        if not np.isfinite(pivot) or pivot <= 0.0:
            return None
        mask = np.ones(prev.shape[0], dtype=bool)
        mask[p] = False
        col = prev_inv[mask, p]
        return prev_inv[np.ix_(mask, mask)] - np.outer(col, col) / pivot


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

"""Dense covariance backends.

``DenseCovariance`` is the reference implementation of the
:class:`~cvxcla.operators._core.QuadraticForm` protocol, wrapping an explicit
``n x n`` matrix. ``IncrementalDenseCovariance`` is a drop-in alternative that
maintains the explicit free-block inverse across turning points with rank-one
updates, trading numerical robustness for speed on dense problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import cast

import numpy as np
from cvx.linalg import DenseOperator
from numpy.typing import NDArray
from scipy.linalg import cho_factor  # type: ignore[import-untyped]
from scipy.linalg.lapack import get_lapack_funcs  # type: ignore[import-untyped]

from ._core import _RCOND_ESTIMATE_MARGIN, _rcond_symmetric


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

    @cached_property
    def _operator(self) -> DenseOperator:
        """The shared ``cvx.linalg`` operator backing the products (never re-formed)."""
        return DenseOperator(self.matrix)

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
        return cast("NDArray[np.float64]", self._operator.matvec(x))

    def solve_free(self, free: NDArray[np.bool_], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve the free-block system ``Sigma[free][:, free] @ y = rhs``.

        Delegates to the shared ``DenseOperator`` (Cholesky with an LU fallback);
        the CLA's ``rcond_free`` guard keeps rank-deficient blocks away from here.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.
            rhs: Right-hand side of shape ``(n_free,)`` or ``(n_free, r)``.

        Returns:
            The solution ``y`` with the same shape as ``rhs``.
        """
        return cast("NDArray[np.float64]", self._operator.solve_free(np.flatnonzero(free), rhs))

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
        return cast(
            "NDArray[np.float64]",
            self._operator.block_matvec(np.flatnonzero(free), np.flatnonzero(blocked), x[blocked]),
        )

    def rcond_free(self, free: NDArray[np.bool_]) -> float:
        """Reciprocal condition number of the free block; see the protocol."""
        return _rcond_symmetric(self.matrix[np.ix_(free, free)])

    def rcond_floor_cleared(self, floor: float) -> bool:
        """Fast up-front conditioning test: is the whole matrix's rcond at least ``floor``?

        An optional optimisation hook (not part of the :class:`QuadraticForm`
        protocol, so backends need not provide it): the CLA calls it once, when
        present, to decide whether the per-turning-point conditioning guard can be
        skipped, and otherwise falls back to the exact ``rcond_free`` on the full
        mask. The boolean is identical to ``rcond_free(<all free>) >= floor``.

        Settles the common well-conditioned case with a Cholesky factorisation
        plus a LAPACK 1-norm condition estimate (``?pocon``) -- ``O(n^3)`` for the
        factor but only ``O(n^2)`` for the estimate -- instead of the full
        eigendecomposition the exact :meth:`rcond_free` would run. A
        non-positive-definite matrix (Cholesky fails) is below any positive floor;
        an estimate within :data:`_RCOND_ESTIMATE_MARGIN` of the floor defers to
        the exact symmetric rcond, so the answer matches ``rcond_free(all) >= floor``.
        """
        try:
            cho, _lower = cho_factor(self.matrix, check_finite=False)
        except np.linalg.LinAlgError:
            return False  # not numerically positive definite: below any positive floor
        (pocon,) = get_lapack_funcs(("pocon",), (self.matrix,))
        anorm = float(np.linalg.norm(self.matrix, 1))
        rcond_estimate = float(pocon(cho, anorm)[0])
        if rcond_estimate >= floor * _RCOND_ESTIMATE_MARGIN:
            return True
        return _rcond_symmetric(self.matrix) >= floor


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

    def rcond_free(self, free: NDArray[np.bool_]) -> float:
        """Reciprocal condition number of the free block (delegates to dense)."""
        return self._dense.rcond_free(free)

    def rcond_floor_cleared(self, floor: float) -> bool:
        """Fast up-front conditioning test (delegates to the wrapped dense matrix)."""
        return self._dense.rcond_floor_cleared(floor)

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
        return inv @ rhs

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
        return cast("NDArray[np.float64]", aug[np.ix_(perm, perm)])

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
        return cast("NDArray[np.float64]", prev_inv[np.ix_(mask, mask)] - np.outer(col, col) / pivot)

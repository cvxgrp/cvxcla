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
    protocol (see the bottom of this module).
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

    def rcond_free(self, free: NDArray[np.bool_]) -> float:
        """Reciprocal condition number of the free block; see the protocol."""
        return _rcond_symmetric(self.matrix[np.ix_(free, free)])


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

    def rcond_free(self, free: NDArray[np.bool_]) -> float:
        """Lower bound on the free block's reciprocal condition number.

        The free block ``Sigma_FF = diag(d_F) + U_F @ Delta @ U_F^T`` is
        positive definite by construction (the positive idiosyncratic floor
        ``d`` keeps it full rank), which is exactly why this backend is the
        documented remedy for rank deficiency. Rather than form the
        ``n_F x n_F`` block, we bound the conditioning from the structure via
        Weyl's inequalities:

        * ``lambda_min(Sigma_FF) >= min(d_F)`` (the low-rank part is PSD), and
        * ``lambda_max(Sigma_FF) <= max(d_F) + ||U_F||_2^2 * lambda_max(Delta)``.

        Their ratio is a guaranteed *lower* bound on the true reciprocal
        condition number, computed in ``O(n_F k^2 + k^3)`` without ever
        materialising an ``n x n`` matrix. For a well-floored factor model it
        sits far above the CLA's singularity threshold, so the guard never
        trips; a pathologically tiny floor correctly drives it toward zero.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.

        Returns:
            A lower bound on the reciprocal 2-norm condition number, in ``[0, 1]``.
        """
        if not np.any(free):
            return 1.0
        d_free = self.d[free]
        u_free = self.u[free]
        u_spectral_norm = float(np.linalg.svd(u_free, compute_uv=False)[0])
        delta_max = float(np.linalg.eigvalsh(self._delta_matrix)[-1])
        lam_max_upper = float(np.max(d_free)) + u_spectral_norm**2 * max(delta_max, 0.0)
        return float(np.min(d_free)) / lam_max_upper


@dataclass(frozen=True)
class GramCovariance:
    """Sample covariance backed by the data matrix, never forming ``Sigma``.

    A sample covariance *is* a (scaled, centered) Gram matrix:
    ``Sigma = X_c^T X_c / (T - 1)`` where ``X_c`` is the column-centered
    ``(T, n)`` returns matrix. This backend stores only ``X_c`` (``O(T n)``
    memory) and implements the ``QuadraticForm`` protocol straight from it, so
    the ``n x n`` covariance is never materialised: ``matvec`` is two thin
    products ``X_c^T (X_c v)``, and ``solve_free`` works on the free columns of
    ``X_c`` alone.

    An optional ridge ``delta >= 0`` represents ``Sigma = X_c^T X_c / (T - 1) +
    delta I``. It matters because ``X_c^T X_c`` has rank at most ``T - 1``: when
    there are fewer observations than assets (``T <= n``) the raw covariance is
    singular and the free block becomes unsolvable once the free set outgrows the
    data rank (the degeneracy the CLA detects via ``rcond_free``). A positive
    ridge restores positive definiteness, and the free-block solve is then done by
    the Woodbury identity in the ``T``-dimensional observation space
    (``O(n_F T^2 + T^3)``) whenever that is cheaper than the ``n_F x n_F`` normal
    equations -- i.e. this is a factor model whose factors are the observations.

    When to use:
        This is a *memory* play, not an unconditional speed-up. Its win is the
        short-sample regime ``T < n`` (a sample covariance from far fewer periods
        than assets), where it never forms the ``n x n`` matrix and the
        ridge + Woodbury solve works in the small ``T``-space. When ``T >= n`` and
        the dense covariance fits in memory, ``DenseCovariance`` is faster per
        turning point: it slices a *precomputed* ``Sigma_FF`` once, whereas this
        backend re-forms ``X_cF^T X_cF`` (``O(n_F^2 T)``) at every solve. The same
        caution applies to ``FactorCovariance`` with many factors: a
        diagonal-plus-low-rank solve only pays off while the rank stays well below
        ``n`` -- at full rank the structured route is slower than the plain dense
        matrix.

    Note:
        The covariance is built from the *centered* data; the raw Gram
        ``X^T X`` is the second-moment matrix, which differs by the rank-one mean
        correction ``T x_bar x_bar^T``. This backend centers for you.

    Attributes:
        returns: The ``(T, n)`` matrix of observations (e.g. asset returns).
        ridge: A non-negative diagonal loading added to the covariance.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> returns = rng.standard_normal((50, 4))
        >>> gram = GramCovariance(returns)
        >>> bool(np.allclose(gram.matvec(np.ones(4)), np.cov(returns, rowvar=False) @ np.ones(4)))
        True
    """

    returns: NDArray[np.float64]
    ridge: float = 0.0

    def __post_init__(self) -> None:
        """Validate that ``returns`` is a ``(T, n)`` matrix with ``T >= 2`` and ``ridge >= 0``.

        Raises:
            ValueError: If ``returns`` is not two-dimensional with at least two
                rows, or if ``ridge`` is negative.
        """
        if self.returns.ndim != 2 or self.returns.shape[0] < 2:
            msg = f"returns must be a (T, n) matrix with T >= 2 observations, got shape {self.returns.shape}"
            raise ValueError(msg)
        if self.ridge < 0.0:
            msg = f"ridge must be non-negative, got {self.ridge}"
            raise ValueError(msg)

    @cached_property
    def _centered(self) -> NDArray[np.float64]:
        """The column-centered data matrix ``X_c`` of shape ``(T, n)``."""
        return cast("NDArray[np.float64]", self.returns - self.returns.mean(axis=0, keepdims=True))

    @property
    def _t(self) -> int:
        """Number of observations ``T``."""
        return int(self.returns.shape[0])

    @property
    def _scale(self) -> float:
        """The sample-covariance scale ``1 / (T - 1)`` (``ddof = 1``, matching ``np.cov``)."""
        return 1.0 / (self._t - 1)

    @property
    def n(self) -> int:
        """Number of assets (the dimension of the covariance)."""
        return int(self.returns.shape[1])

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute ``Sigma @ x`` as ``scale * X_c^T (X_c @ x) + ridge * x``.

        Args:
            x: Vector of shape ``(n,)`` or matrix of shape ``(n, r)``.

        Returns:
            ``Sigma @ x`` with the same trailing shape as ``x``, in ``O(T n)`` per
            column and without forming the ``n x n`` covariance.
        """
        xc = self._centered
        product = self._scale * (xc.T @ (xc @ x))
        return product + self.ridge * x

    def cross(self, free: NDArray[np.bool_], x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the free-to-blocked cross product ``Sigma[free][:, ~free] @ x[~free]``.

        The ridge is diagonal, so it contributes nothing to this off-diagonal
        block; the result is ``scale * X_cF^T (X_cB @ x_B)``.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.
            x: Full-length vector of shape ``(n,)``; only the blocked entries
               ``x[~free]`` enter the product.

        Returns:
            Vector of shape ``(n_free,)``.
        """
        blocked = ~free
        xc_free = self._centered[:, free]
        xc_blocked = self._centered[:, blocked]
        return self._scale * (xc_free.T @ (xc_blocked @ x[blocked]))

    def solve_free(self, free: NDArray[np.bool_], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve ``Sigma[free][:, free] @ y = rhs`` from the free columns of ``X_c``.

        Without a ridge the free block ``scale * X_cF^T X_cF`` is formed and solved
        directly (it is singular, and the solve unreliable, once the free set
        outgrows the data rank -- which ``rcond_free`` reports so the CLA can
        refuse). With a positive ridge the block is positive definite and, when
        there are more free assets than observations, the solve is done by the
        Woodbury identity in ``T``-space instead of inverting an ``n_F x n_F``
        matrix.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.
            rhs: Right-hand side of shape ``(n_free,)`` or ``(n_free, r)``.

        Returns:
            The solution ``y`` with the same shape as ``rhs``.
        """
        xc_free = self._centered[:, free]
        n_free = xc_free.shape[1]

        if self.ridge > 0.0 and self._t < n_free:
            # Woodbury in the T-dimensional observation space:
            # (ridge I + W^T W)^{-1} = (1/ridge) I - (1/ridge^2) W^T (I + W W^T / ridge)^{-1} W,
            # with W = sqrt(scale) X_cF (shape (T, n_free)).
            w = np.sqrt(self._scale) * xc_free
            correction_system = np.eye(self._t) + (w @ w.T) / self.ridge
            correction = w.T @ np.linalg.solve(correction_system, w @ rhs)
            return cast("NDArray[np.float64]", (rhs - correction / self.ridge) / self.ridge)

        block = self._scale * (xc_free.T @ xc_free)
        if self.ridge > 0.0:
            block = block + self.ridge * np.eye(n_free)
        return cast("NDArray[np.float64]", np.linalg.solve(block, rhs))

    def rcond_free(self, free: NDArray[np.bool_]) -> float:
        """Reciprocal condition number of the free block, from the SVD of ``X_cF``.

        The eigenvalues of ``scale * X_cF^T X_cF + ridge I`` are
        ``scale * sigma_i^2 + ridge`` for the singular values ``sigma_i`` of
        ``X_cF``, plus extra ``ridge`` eigenvalues when there are more free assets
        than the rank of ``X_cF``. This reads the conditioning off the SVD of the
        ``(T, n_free)`` data block without forming the ``n_F x n_F`` matrix, so a
        rank-deficient (``T``-limited) free set correctly drives the result toward
        zero.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.

        Returns:
            The reciprocal 2-norm condition number of the free block, in ``[0, 1]``.
        """
        n_free = int(np.count_nonzero(free))
        if n_free == 0:
            return 1.0
        singular_values = np.linalg.svd(self._centered[:, free], compute_uv=False)
        eig = self._scale * singular_values**2
        lam_max = float(eig[0]) + self.ridge
        if lam_max <= 0.0:
            return 0.0
        lam_min = self.ridge if n_free > singular_values.shape[0] else float(eig[-1]) + self.ridge
        return max(lam_min, 0.0) / lam_max


# Backward-compatible alias. The protocol was introduced as ``CovarianceOperator``
# for the portfolio (covariance) setting; it is the Hessian contract for any
# parametric active-set path, so the canonical name is now ``QuadraticForm``.
# Existing imports of ``CovarianceOperator`` keep working unchanged.
CovarianceOperator = QuadraticForm

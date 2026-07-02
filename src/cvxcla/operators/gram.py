"""Data-matrix-backed sample covariance.

``GramCovariance`` implements the :class:`~cvxcla.operators._core.QuadraticForm`
protocol straight from the ``(T, n)`` returns matrix, storing only the centered
data (``O(T n)`` memory) and never forming the ``n x n`` covariance. An optional
ridge restores positive definiteness in the short-sample (``T < n``) regime,
with the free-block solve done by Woodbury in ``T``-space.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import cast

import numpy as np
from numpy.typing import NDArray


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

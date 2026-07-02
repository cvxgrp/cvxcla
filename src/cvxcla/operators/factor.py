"""Diagonal-plus-low-rank covariance backend.

``FactorCovariance`` implements the :class:`~cvxcla.operators._core.QuadraticForm`
protocol for factor risk models and RMT-cleaned covariances of the form
``Sigma = diag(d) + U @ Delta @ U.T``, solving against the free block through the
Woodbury identity so no ``n x n`` matrix is ever materialised.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import cast

import numpy as np
from cvx.linalg import FactorOperator
from numpy.typing import NDArray


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
    def _operator(self) -> FactorOperator:
        """Shared ``cvx.linalg`` operator for ``Sigma = diag(d) + U Delta U.T`` (Woodbury solves)."""
        return FactorOperator(self.d, self.u, self._delta_matrix)

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the matrix-vector product ``Sigma @ x``.

        Args:
            x: Vector of shape ``(n,)`` or matrix of shape ``(n, r)``.

        Returns:
            ``Sigma @ x`` with the same trailing shape as ``x``, computed in
            ``O(n k)`` per column.
        """
        return cast("NDArray[np.float64]", self._operator.matvec(x))

    def solve_free(self, free: NDArray[np.bool_], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve the free-block system ``Sigma[free][:, free] @ y = rhs`` via Woodbury.

        Delegates to the shared ``FactorOperator``, which forms the ``k x k``
        correction matrix ``W = Delta^{-1} + U_F^T D_F^{-1} U_F`` and solves
        against it, so the cost is ``O(n_F k^2 + k^3 + n_F k r)`` for ``r``
        right-hand sides.

        Args:
            free: Boolean mask of shape ``(n,)`` selecting the free assets.
            rhs: Right-hand side of shape ``(n_free,)`` or ``(n_free, r)``.

        Returns:
            The solution ``y`` with the same shape as ``rhs``.
        """
        return cast("NDArray[np.float64]", self._operator.solve_free(np.flatnonzero(free), rhs))

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
        return cast(
            "NDArray[np.float64]",
            self._operator.block_matvec(np.flatnonzero(free), np.flatnonzero(blocked), x[blocked]),
        )

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

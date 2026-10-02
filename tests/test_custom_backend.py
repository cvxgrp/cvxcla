"""A user-defined covariance backend, written against the public protocol only.

``CLA`` reaches the covariance through :data:`cvxcla.QuadraticForm` (the
:class:`cvx.linalg.SymmetricOperator` contract). A backend subclasses it and
implements five members -- ``n``, ``matvec``, ``block_matvec``, ``solve_free`` and
``rcond_free`` -- and ``CLA`` then uses it in place of a dense matrix without any
change to the algorithm. The example here is a block-diagonal covariance (assets in
groups with no cross-group covariance), a structure the package does not ship:
its free-block solve splits into one small Cholesky solve per block.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import block_diag, cho_factor, cho_solve

from cvxcla import CLA, QuadraticForm


class BlockDiagonalCovariance(QuadraticForm):
    """``Sigma = blockdiag(S_1, ..., S_m)`` with symmetric positive-definite blocks."""

    def __init__(self, blocks: list[np.ndarray]) -> None:
        """Store the blocks and the index range each one covers."""
        self._blocks = [np.asarray(b, dtype=float) for b in blocks]
        sizes = [len(b) for b in self._blocks]
        self._starts = np.cumsum([0, *sizes])
        self._block_of = np.repeat(np.arange(len(sizes)), sizes)

    @property
    def n(self) -> int:
        """Total number of assets."""
        return int(self._starts[-1])

    def matvec(self, x):
        """``Sigma @ x``, one block at a time (``x`` a vector or a matrix of columns)."""
        return np.concatenate(
            [b @ x[s:e] for b, s, e in zip(self._blocks, self._starts[:-1], self._starts[1:], strict=True)]
        )

    def block_matvec(self, rows, cols, v):
        """``Sigma[rows, cols] @ v``, as the product with ``v`` scattered into ``cols``."""
        rows, cols = np.asarray(rows), np.asarray(cols)
        x = np.zeros((self.n, *np.shape(v)[1:]))
        x[cols] = v
        return self.matvec(x)[rows]

    def _free_blocks(self, free):
        """Yield ``(positions in free, sub-block)`` for each block the free set meets."""
        free = np.asarray(free)
        for k in np.unique(self._block_of[free]):
            pos = np.flatnonzero(self._block_of[free] == k)
            local = free[pos] - self._starts[k]
            yield pos, self._blocks[k][np.ix_(local, local)]

    def solve_free(self, free, rhs):
        """``Sigma[free, free]^{-1} @ rhs`` by one Cholesky solve per block."""
        out = np.empty_like(np.asarray(rhs, dtype=float))
        for pos, sub in self._free_blocks(free):
            out[pos] = cho_solve(cho_factor(sub), rhs[pos])
        return out

    def rcond_free(self, free):
        """Reciprocal condition number of ``Sigma[free, free]`` from the blocks' spectra."""
        eig = np.concatenate([np.linalg.eigvalsh(sub) for _, sub in self._free_blocks(free)])
        return float(eig.min() / eig.max())


@pytest.fixture(scope="module")
def blocks() -> list[np.ndarray]:
    """Four groups of five assets, each with its own positive-definite covariance."""
    rng = np.random.default_rng(4)
    out = []
    for _ in range(4):
        m = rng.standard_normal((5, 5))
        out.append(m @ m.T / 5 + 0.2 * np.eye(5))
    return out


def _weights(covariance, mean, **constraints) -> np.ndarray:
    """Turning-point weights of a trace with the given covariance argument."""
    n = len(mean)
    problem = {
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    } | constraints
    return np.array([tp.weights for tp in CLA(mean=mean, covariance=covariance, **problem).turning_points])


def test_backend_satisfies_the_protocol(blocks):
    """The backend is a QuadraticForm and agrees with the dense matrix it represents."""
    op = BlockDiagonalCovariance(blocks)
    dense = block_diag(*blocks)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(op.n)
    free = np.array([0, 2, 3, 7, 11, 12, 19])
    assert isinstance(op, QuadraticForm)
    np.testing.assert_allclose(op.matvec(x), dense @ x)
    np.testing.assert_allclose(op.block_matvec(free, [1, 5], x[:2]), dense[np.ix_(free, [1, 5])] @ x[:2])
    np.testing.assert_allclose(
        op.solve_free(free, x[: len(free)]), np.linalg.solve(dense[np.ix_(free, free)], x[: len(free)])
    )
    assert op.rcond_free(free) == pytest.approx(1.0 / np.linalg.cond(dense[np.ix_(free, free)]))


@pytest.mark.parametrize("constraints", ["budget", "sector caps", "gross-exposure cap"])
def test_custom_backend_traces_the_dense_frontier(blocks, constraints):
    """CLA runs unchanged on the custom backend and reproduces the dense frontier."""
    n = sum(len(b) for b in blocks)
    mean = np.random.default_rng(1).uniform(0.0, 1.0, n)
    kwargs: dict = {}
    if constraints == "sector caps":
        kwargs = {"g": np.kron(np.eye(4), np.ones((1, 5))), "h": np.full(4, 0.35)}
    elif constraints == "gross-exposure cap":  # long/short: exercises block_matvec through the lift
        kwargs = {"lower_bounds": np.full(n, -0.3), "leverage": 1.3}
    custom = _weights(BlockDiagonalCovariance(blocks), mean, **kwargs)
    dense = _weights(block_diag(*blocks), mean, **kwargs)
    assert custom.shape == dense.shape
    np.testing.assert_allclose(custom, dense, atol=1e-10)

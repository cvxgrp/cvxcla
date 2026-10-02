"""The covariance-backend contract, checked on independently written backends.

``docs/custom-backend.md`` states what ``CLA`` relies on from a
:data:`cvxcla.QuadraticForm`: float64 arrays, duplicate-free but possibly unsorted
index sets with results aligned to their order, vector and matrix right-hand
sides, no mutation or aliasing of arguments, and an ``rcond_free`` that never
overstates the conditioning. Each convention is checked here, for four
structures written only against the public protocol and for the bundled
backends. Randomized problems -- the budget, general equalities with group caps,
and a gross-exposure cap -- are then traced through every custom backend and
compared with the dense trace of the same matrix, which shows the algorithm uses
the covariance only through the documented members.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.linalg import block_diag, cho_factor, cho_solve

from cvxcla import (
    CLA,
    DenseCovariance,
    FactorCovariance,
    GramCovariance,
    IncrementalDenseCovariance,
    QuadraticForm,
)

N = 24
SEEDS = range(4)


def _spd(rng: np.random.Generator, n: int) -> np.ndarray:
    """A random symmetric positive-definite matrix with condition number in the hundreds."""
    m = rng.standard_normal((n, n))
    return m @ m.T / n + 0.1 * np.eye(n)


def _rcond(block: np.ndarray) -> float:
    """Reciprocal 2-norm condition number of a symmetric positive-definite block."""
    if block.shape[0] == 0:
        return 1.0
    eig = np.linalg.eigvalsh(block)
    return float(eig[0] / eig[-1])


class _ExplicitBlockSolve(QuadraticForm):
    """Shared ``block_matvec`` / ``solve_free`` / ``rcond_free`` for backends that can form one entry block."""

    def _block(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def block_matvec(self, rows, cols, v):
        """``Sigma[rows, cols] @ v`` from the entries of the block."""
        return self._block(np.asarray(rows), np.asarray(cols)) @ v

    def solve_free(self, free, rhs):
        """``Sigma[free, free]^{-1} @ rhs`` by Cholesky of the formed block."""
        free = np.asarray(free)
        return cho_solve(cho_factor(self._block(free, free)), rhs)

    def rcond_free(self, free):
        """Exact reciprocal condition number of the formed block."""
        free = np.asarray(free)
        return _rcond(self._block(free, free))


class BlockDiagonal(QuadraticForm):
    """``Sigma = blockdiag(S_1, ..., S_m)``: one small Cholesky solve per block the free set meets."""

    def __init__(self, blocks: list[np.ndarray]) -> None:
        """Store the blocks and the asset range of each."""
        self._blocks = blocks
        sizes = [len(b) for b in blocks]
        self._starts = np.cumsum([0, *sizes])
        self._owner = np.repeat(np.arange(len(blocks)), sizes)

    @property
    def n(self) -> int:
        """Number of assets."""
        return int(self._starts[-1])

    def matvec(self, x):
        """Blockwise product."""
        return np.concatenate(
            [b @ x[s:e] for b, s, e in zip(self._blocks, self._starts[:-1], self._starts[1:], strict=True)]
        )

    def block_matvec(self, rows, cols, v):
        """The product with ``v`` scattered into ``cols``, read off at ``rows``."""
        x = np.zeros((self.n, *np.shape(v)[1:]))
        x[np.asarray(cols)] = v
        return self.matvec(x)[np.asarray(rows)]

    def _pieces(self, free: np.ndarray):
        for k in np.unique(self._owner[free]):
            pos = np.flatnonzero(self._owner[free] == k)
            local = free[pos] - self._starts[k]
            yield pos, self._blocks[k][np.ix_(local, local)]

    def solve_free(self, free, rhs):
        """One Cholesky solve per block, results scattered back in the order of ``free``."""
        out = np.empty(np.shape(rhs))
        for pos, sub in self._pieces(np.asarray(free)):
            out[pos] = cho_solve(cho_factor(sub), rhs[pos])
        return out

    def rcond_free(self, free):
        """Exact, from the spectra of the blocks (the free block is their direct sum)."""
        free = np.asarray(free)
        if free.size == 0:
            return 1.0
        eig = np.concatenate([np.linalg.eigvalsh(sub) for _, sub in self._pieces(free)])
        return float(eig.min() / eig.max())


class LowRankWoodbury(QuadraticForm):
    """``Sigma = diag(d) + V V'``, solved by a hand-written Woodbury identity; ``rcond_free`` is a bound."""

    def __init__(self, d: np.ndarray, v: np.ndarray) -> None:
        """Store the diagonal and the loadings."""
        self._d, self._v = d, v

    @property
    def n(self) -> int:
        """Number of assets."""
        return int(self._d.shape[0])

    def matvec(self, x):
        """``d * x + V (V' x)`` without forming Sigma."""
        return (self._d * x.T).T + self._v @ (self._v.T @ x)

    def block_matvec(self, rows, cols, v):
        """The diagonal acts only where ``rows`` and ``cols`` share an asset."""
        rows, cols = np.asarray(rows), np.asarray(cols)
        out = self._v[rows] @ (self._v[cols].T @ v)
        shared = rows[:, None] == cols[None, :]
        return out + (shared * self._d[rows][:, None]) @ v

    def solve_free(self, free, rhs):
        """``(D + V V')^{-1} = D^{-1} - D^{-1} V (I + V' D^{-1} V)^{-1} V' D^{-1}`` on the free block."""
        free = np.asarray(free)
        d, v = self._d[free], self._v[free]
        scaled = (rhs.T / d).T
        capacitance = np.eye(v.shape[1]) + v.T @ (v / d[:, None])
        return scaled - (v @ np.linalg.solve(capacitance, v.T @ scaled)) / (d if rhs.ndim == 1 else d[:, None])

    def rcond_free(self, free):
        """Weyl lower bound ``min(d_F) / (max(d_F) + ||V_F||^2)``, like the bundled factor backend."""
        free = np.asarray(free)
        if free.size == 0:
            return 1.0
        d = self._d[free]
        return float(d.min() / (d.max() + np.linalg.norm(self._v[free], 2) ** 2))


class Kronecker(_ExplicitBlockSolve):
    """``Sigma = A kron B`` (two asset dimensions), with ``matvec`` by the reshape identity."""

    def __init__(self, a: np.ndarray, b: np.ndarray) -> None:
        """Store the two factors."""
        self._a, self._b = a, b
        self._q = b.shape[0]

    @property
    def n(self) -> int:
        """Number of assets, ``p * q``."""
        return int(self._a.shape[0] * self._q)

    def matvec(self, x):
        """``(A kron B) vec(X) = vec(A X B')`` for row-major ``X``, one column of ``x`` at a time."""
        cols = x.reshape(self.n, -1)
        p = self._a.shape[0]
        out = np.stack(
            [(self._a @ c.reshape(p, self._q) @ self._b.T).ravel() for c in cols.T],
            axis=1,
        )
        return out.reshape(x.shape)

    def _block(self, rows, cols):
        """Entries ``A[i // q, j // q] * B[i % q, j % q]``."""
        q = self._q
        return self._a[np.ix_(rows // q, cols // q)] * self._b[np.ix_(rows % q, cols % q)]


class PermutedDense(_ExplicitBlockSolve):
    """A dense matrix stored in a shuffled asset order, so every index set is remapped."""

    def __init__(self, sigma: np.ndarray, rng: np.random.Generator) -> None:
        """Store ``Sigma[perm][:, perm]`` and the map from asset to storage position."""
        perm = rng.permutation(sigma.shape[0])
        self._stored = sigma[np.ix_(perm, perm)]
        self._pos = np.argsort(perm)  # asset i lives at stored position _pos[i]

    @property
    def n(self) -> int:
        """Number of assets."""
        return int(self._stored.shape[0])

    def matvec(self, x):
        """Scatter into storage order, multiply, gather back."""
        stored_x = np.empty_like(x)
        stored_x[self._pos] = x
        return (self._stored @ stored_x)[self._pos]

    def _block(self, rows, cols):
        return self._stored[np.ix_(self._pos[rows], self._pos[cols])]


def _custom(name: str, seed: int) -> tuple[QuadraticForm, np.ndarray]:
    """One custom backend and the dense matrix it represents."""
    rng = np.random.default_rng(100 + seed)
    if name == "block-diagonal":
        blocks = [_spd(rng, 6) for _ in range(4)]
        return BlockDiagonal(blocks), block_diag(*blocks)
    if name == "low-rank Woodbury":
        d, v = rng.uniform(0.05, 0.2, N), rng.standard_normal((N, 3)) / np.sqrt(N)
        return LowRankWoodbury(d, v), np.diag(d) + v @ v.T
    if name == "Kronecker":
        a, b = _spd(rng, 4), _spd(rng, 6)
        return Kronecker(a, b), np.kron(a, b)
    sigma = _spd(rng, N)
    return PermutedDense(sigma, rng), sigma


CUSTOM = ["block-diagonal", "low-rank Woodbury", "Kronecker", "permuted dense"]


def _bundled(name: str) -> tuple[QuadraticForm, np.ndarray]:
    """One bundled backend and its dense matrix."""
    rng = np.random.default_rng(7)
    if name == "factor":
        d, u, delta = rng.uniform(0.05, 0.2, N), rng.standard_normal((N, 3)), np.array([0.5, 0.2, 0.1])
        return FactorCovariance(d=d, u=u, delta=delta), np.diag(d) + (u * delta) @ u.T
    if name == "gram":
        returns = rng.standard_normal((3 * N, N))
        return GramCovariance(returns, ridge=0.01), np.cov(returns, rowvar=False) + 0.01 * np.eye(N)
    sigma = _spd(rng, N)
    builder: Callable[[np.ndarray], QuadraticForm] = DenseCovariance if name == "dense" else IncrementalDenseCovariance
    return builder(sigma), sigma


BUNDLED = ["dense", "incremental dense", "factor", "gram"]


@pytest.fixture(params=[("custom", c) for c in CUSTOM] + [("bundled", b) for b in BUNDLED], ids=lambda p: p[1])
def backend(request) -> tuple[QuadraticForm, np.ndarray]:
    """Every backend under test with the dense matrix it represents."""
    kind, name = request.param
    return _custom(name, 0) if kind == "custom" else _bundled(name)


def _unsorted(rng: np.random.Generator, size: int) -> np.ndarray:
    """A duplicate-free, deliberately unsorted index set."""
    idx = rng.choice(N, size=size, replace=False).astype(np.intp)
    return idx if not np.all(np.diff(idx) > 0) else idx[::-1]


def _calls(op: QuadraticForm, rng: np.random.Generator) -> list[tuple[str, tuple, Callable[[np.ndarray], np.ndarray]]]:
    """Each protocol call with its arguments and the dense result it must reproduce."""
    rows, cols, free = _unsorted(rng, 7), _unsorted(rng, 5), _unsorted(rng, 9)
    calls = []
    for trailing in [(), (3,)]:
        x = rng.standard_normal((N, *trailing))
        v = rng.standard_normal((len(cols), *trailing))
        rhs = rng.standard_normal((len(free), *trailing))
        calls += [
            ("matvec", (x,), lambda s, x=x: s @ x),
            ("block_matvec", (rows, cols, v), lambda s, v=v: s[np.ix_(rows, cols)] @ v),
            ("solve_free", (free, rhs), lambda s, rhs=rhs: np.linalg.solve(s[np.ix_(free, free)], rhs)),
        ]
    return calls


def test_products_and_solves_follow_the_conventions(backend):
    """Unsorted indices, both shapes, float64 out, aligned results, no mutation, no aliasing."""
    op, dense = backend
    assert isinstance(op, QuadraticForm)
    assert op.n == N
    rng = np.random.default_rng(11)
    for name, args, expected in _calls(op, rng):
        before = [np.copy(a) for a in args]
        result = np.asarray(getattr(op, name)(*args))
        assert result.dtype == np.float64, name
        assert result.shape == expected(dense).shape, name
        np.testing.assert_allclose(result, expected(dense), rtol=1e-9, atol=1e-11, err_msg=name)
        for arg, copy in zip(args, before, strict=True):
            np.testing.assert_array_equal(arg, copy, err_msg=f"{name} modified an argument")
            assert not np.shares_memory(result, arg), f"{name} returned a view of an argument"


def test_rcond_free_never_overstates_the_conditioning(backend):
    """``rcond_free`` is the reciprocal condition number or a lower bound, in [0, 1], and 1 when empty."""
    op, dense = backend
    rng = np.random.default_rng(12)
    for free in [_unsorted(rng, 1), _unsorted(rng, 9), np.arange(N, dtype=np.intp)]:
        rcond = op.rcond_free(free)
        assert 0.0 < rcond <= 1.0
        assert rcond <= _rcond(dense[np.ix_(free, free)]) * (1 + 1e-9)
    assert op.rcond_free(np.zeros(0, dtype=np.intp)) == 1.0


def test_results_do_not_depend_on_the_call_history(backend):
    """A solve repeated after other solves gives the same answer (stateful backends included)."""
    op, _ = backend
    rng = np.random.default_rng(13)
    free, rhs = _unsorted(rng, 8), rng.standard_normal((8, 2))
    first = np.copy(op.solve_free(free, rhs))
    for size in (7, 9, 8):
        op.solve_free(_unsorted(rng, size), rng.standard_normal((size, 2)))
    np.testing.assert_allclose(op.solve_free(free, rhs), first, rtol=1e-10, atol=1e-12)


def _constraints(kind: str, rng: np.random.Generator) -> dict:
    """A feasible constraint set of one of three kinds."""
    if kind == "budget":
        return {"lower_bounds": np.zeros(N), "upper_bounds": np.ones(N), "a": np.ones((1, N)), "b": np.ones(1)}
    if kind == "general":
        s = rng.uniform(0.5, 1.5, N)  # a weighted second equality, met by the equal-weight portfolio
        return {
            "lower_bounds": np.zeros(N),
            "upper_bounds": np.full(N, 0.3),
            "a": np.vstack([np.ones(N), s]),
            "b": np.array([1.0, s.mean()]),
            "g": np.kron(np.eye(4), np.ones((1, N // 4))),
            "h": np.full(4, 0.35),
        }
    return {
        "lower_bounds": np.full(N, -0.3),
        "upper_bounds": np.full(N, 0.5),
        "a": np.ones((1, N)),
        "b": np.ones(1),
        "leverage": 1.4,
    }


@pytest.mark.parametrize("constraints", ["budget", "general", "gross exposure"])
@pytest.mark.parametrize("name", CUSTOM)
@pytest.mark.parametrize("seed", SEEDS)
def test_custom_backends_trace_the_dense_frontier(name, seed, constraints):
    """On random problems every custom backend reproduces the dense trace of its matrix."""
    op, dense = _custom(name, seed)
    rng = np.random.default_rng(seed)
    mean = rng.uniform(0.0, 1.0, N)
    kwargs = _constraints(constraints, rng)
    custom = CLA(mean=mean, covariance=op, **kwargs)
    reference = CLA(mean=mean, covariance=dense, **kwargs)
    assert len(custom) == len(reference)
    np.testing.assert_allclose(
        [tp.weights for tp in custom.turning_points],
        [tp.weights for tp in reference.turning_points],
        atol=1e-9,
    )

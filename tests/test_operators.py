"""Tests for the cvxcla operator layer: builders, helpers, and CLA integration.

The operator numerics -- block products, free-block solves (Cholesky, Woodbury,
maintained inverse) and the reciprocal-condition check -- live in ``cvx.linalg``
and are tested there. Here we test only what cvxcla adds: builders that assemble
the right ``cvx.linalg`` operator from CLA / LASSO inputs, the ``cross`` and
``bordered_solve`` helpers, and that the CLA runs on a ``cvx.linalg`` operator
identically to a raw matrix.
"""

from __future__ import annotations

import numpy as np
import pytest
from cvx.linalg import DenseOperator, FactorOperator, GramOperator, IncrementalDenseOperator, SymmetricOperator

from cvxcla import (
    CLA,
    CovarianceOperator,
    DenseCovariance,
    FactorCovariance,
    GramCovariance,
    IncrementalDenseCovariance,
)
from cvxcla.operators import (
    bordered_solve,
    cross,
    dense_covariance,
    factor_covariance,
    gram_covariance,
    incremental_dense_covariance,
)

# --- builders -----------------------------------------------------------------


def test_dense_covariance_builds_dense_operator() -> None:
    """dense_covariance wraps a symmetric matrix in a DenseOperator."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal((6, 6))
    a = b @ b.T + 6 * np.eye(6)
    op = dense_covariance(a)
    assert isinstance(op, DenseOperator)
    assert np.allclose(op.matvec(np.ones(6)), a @ np.ones(6))


def test_incremental_dense_covariance_builds_incremental_operator() -> None:
    """incremental_dense_covariance yields the maintained-inverse dense operator."""
    op = incremental_dense_covariance(np.eye(4))
    assert isinstance(op, IncrementalDenseOperator)
    assert np.allclose(op.solve_free(np.array([0, 2]), np.array([1.0, 2.0])), [1.0, 2.0])


def test_gram_covariance_matches_sample_covariance() -> None:
    """gram_covariance reproduces the centered 1/(T-1)-scaled sample covariance."""
    rng = np.random.default_rng(1)
    returns = rng.standard_normal((40, 5))
    op = gram_covariance(returns)
    assert isinstance(op, GramOperator)
    sigma = np.cov(returns, rowvar=False)
    x = rng.standard_normal(5)
    assert np.allclose(op.matvec(x), sigma @ x)


def test_gram_covariance_ridge() -> None:
    """The ridge adds delta * I to the sample covariance."""
    rng = np.random.default_rng(2)
    returns = rng.standard_normal((30, 4))
    op = gram_covariance(returns, ridge=0.5)
    sigma = np.cov(returns, rowvar=False) + 0.5 * np.eye(4)
    x = rng.standard_normal(4)
    assert np.allclose(op.matvec(x), sigma @ x)


def test_factor_covariance_matches_dense() -> None:
    """factor_covariance reproduces diag(d) + U Delta U.T, with a vector delta."""
    rng = np.random.default_rng(3)
    n, k = 8, 3
    d = rng.uniform(0.1, 0.5, n)
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k)
    op = factor_covariance(d, u, delta)
    assert isinstance(op, FactorOperator)
    a = np.diag(d) + (u * delta) @ u.T
    x = rng.standard_normal(n)
    assert np.allclose(op.matvec(x), a @ x)


def test_factor_covariance_accepts_matrix_delta() -> None:
    """A full (k, k) delta matrix is accepted."""
    rng = np.random.default_rng(4)
    n, k = 6, 2
    d = rng.uniform(0.1, 0.5, n)
    u = rng.standard_normal((n, k))
    c = rng.standard_normal((k, k))
    delta = c @ c.T + np.eye(k)
    op = factor_covariance(d, u, delta)
    a = np.diag(d) + u @ delta @ u.T
    x = rng.standard_normal(n)
    assert np.allclose(op.matvec(x), a @ x)


def test_factor_covariance_rejects_bad_delta_shape() -> None:
    """A 3-D delta is rejected."""
    with pytest.raises(ValueError, match="delta"):
        factor_covariance(np.ones(3), np.ones((3, 2)), np.ones((2, 2, 2)))


def test_dense_covariance_rejects_nonsymmetric() -> None:
    """A non-symmetric matrix is rejected by the dense builder."""
    with pytest.raises(ValueError, match="symmetric"):
        dense_covariance(np.array([[1.0, 2.0], [0.0, 1.0]]))


def test_dense_covariance_rejects_non_square() -> None:
    """A non-square matrix is rejected by the dense builder."""
    with pytest.raises(ValueError, match="square"):
        dense_covariance(np.ones((2, 3)))


def test_gram_covariance_rejects_too_few_rows() -> None:
    """gram_covariance needs at least two observations."""
    with pytest.raises(ValueError, match="T >= 2"):
        gram_covariance(np.ones((1, 4)))


def test_backward_compatible_aliases() -> None:
    """The PascalCase names remain as aliases of the snake_case builders."""
    assert DenseCovariance is dense_covariance
    assert IncrementalDenseCovariance is incremental_dense_covariance
    assert GramCovariance is gram_covariance
    assert FactorCovariance is factor_covariance


# --- helpers ------------------------------------------------------------------


def test_cross_matches_dense_cross() -> None:
    """cross(op, free, x) equals the free-to-blocked block product."""
    rng = np.random.default_rng(5)
    b = rng.standard_normal((7, 7))
    a = b @ b.T + 7 * np.eye(7)
    op = DenseOperator(a)
    free = np.array([True, False, True, True, False, False, True])
    x = rng.standard_normal(7)
    expected = a[np.ix_(free, ~free)] @ x[~free]
    assert np.allclose(cross(op, free, x), expected)


def test_bordered_solve_without_constraints() -> None:
    """With no constraint rows, bordered_solve is the plain free-block solve."""
    rng = np.random.default_rng(6)
    b = rng.standard_normal((6, 6))
    a = b @ b.T + 6 * np.eye(6)
    op = DenseOperator(a)
    free = np.array([True, False, True, True, False, True])
    idx = np.flatnonzero(free)
    rhs_const = rng.standard_normal(idx.size)
    rhs_slope = rng.standard_normal(idx.size)
    c_free = np.zeros((0, idx.size))
    empty = np.zeros(0)
    x_const, x_slope, nu_const, nu_slope = bordered_solve(op, free, c_free, rhs_const, rhs_slope, empty, empty)
    block = a[np.ix_(idx, idx)]
    assert np.allclose(block @ x_const, rhs_const)
    assert np.allclose(block @ x_slope, rhs_slope)
    assert nu_const.size == 0
    assert nu_slope.size == 0


def test_bordered_solve_with_constraint() -> None:
    """With a constraint row, bordered_solve matches the assembled KKT solve."""
    rng = np.random.default_rng(7)
    b = rng.standard_normal((5, 5))
    a = b @ b.T + 5 * np.eye(5)
    op = DenseOperator(a)
    free = np.array([True, True, True, False, True])
    idx = np.flatnonzero(free)
    nf = idx.size
    c_free = rng.standard_normal((1, nf))
    rhs_const = rng.standard_normal(nf)
    d_const = rng.standard_normal(1)
    zero_slope = np.zeros(nf)
    zero_d = np.zeros(1)
    x_const, _x_slope, nu_const, _nu_slope = bordered_solve(op, free, c_free, rhs_const, zero_slope, d_const, zero_d)
    # Assemble and solve the bordered KKT system directly.
    block = a[np.ix_(idx, idx)]
    kkt = np.block([[block, c_free.T], [c_free, np.zeros((1, 1))]])
    sol = np.linalg.solve(kkt, np.concatenate([rhs_const, d_const]))
    assert np.allclose(x_const, sol[:nf])
    assert np.allclose(nu_const, sol[nf:])


# --- CLA integration ----------------------------------------------------------


class TestCLAWithOperator:
    """CLA accepts a cvx-linalg operator backend in place of a raw matrix."""

    @pytest.fixture
    def problem(self) -> dict:
        """A portfolio problem large enough to produce several turning points."""
        rng = np.random.default_rng(42)
        n = 8
        l_matrix = rng.standard_normal((n, n))
        covariance = l_matrix @ l_matrix.T + n * np.eye(n)
        return {
            "mean": rng.uniform(0.05, 0.2, n),
            "covariance": covariance,
            "lower_bounds": np.zeros(n),
            "upper_bounds": np.full(n, 0.6),
            "a": np.ones((1, n)),
            "b": np.ones(1),
        }

    def test_dense_operator_matches_raw_matrix(self, problem: dict) -> None:
        """CLA(dense_covariance(S)) and CLA(S) produce identical turning points."""
        cla_raw = CLA(**problem)
        cla_op = CLA(**{**problem, "covariance": dense_covariance(problem["covariance"])})
        assert len(cla_raw) == len(cla_op)
        for tp_raw, tp_op in zip(cla_raw.turning_points, cla_op.turning_points, strict=True):
            assert tp_raw.lamb == pytest.approx(tp_op.lamb, abs=1e-12)
            np.testing.assert_allclose(tp_raw.weights, tp_op.weights, atol=1e-12)
            np.testing.assert_array_equal(tp_raw.free, tp_op.free)

    def test_frontier_with_operator(self, problem: dict) -> None:
        """The frontier property works when a backend is passed."""
        cla_op = CLA(**{**problem, "covariance": dense_covariance(problem["covariance"])})
        frontier = cla_op.frontier
        assert len(frontier) == len(cla_op)
        assert np.all(frontier.variance > 0)

    def test_factor_backend_matches_raw_matrix(self, problem: dict) -> None:
        """A factor backend traces the same frontier as its dense equivalent."""
        rng = np.random.default_rng(11)
        n = problem["covariance"].shape[0]
        u = rng.standard_normal((n, 2)) / np.sqrt(n)
        delta = rng.uniform(0.5, 2.0, 2) * n
        d = rng.uniform(0.5, 1.5, n)
        dense = np.diag(d) + (u * delta) @ u.T
        cla_raw = CLA(**{**problem, "covariance": dense})
        cla_op = CLA(**{**problem, "covariance": factor_covariance(d, u, delta)})
        assert len(cla_raw) == len(cla_op)
        for tp_raw, tp_op in zip(cla_raw.turning_points, cla_op.turning_points, strict=True):
            np.testing.assert_allclose(tp_raw.weights, tp_op.weights, atol=1e-9)

    def test_custom_operator_subclass_supported(self, problem: dict) -> None:
        """A user SymmetricOperator subclass traces the same frontier as the raw matrix."""

        class CustomOperator(SymmetricOperator):
            """Minimal explicit-matrix operator implementing the protocol."""

            def __init__(self, matrix: np.ndarray) -> None:
                self._a = matrix

            @property
            def n(self) -> int:
                return int(self._a.shape[0])

            def matvec(self, x: np.ndarray) -> np.ndarray:
                return self._a @ x

            def block_matvec(self, rows: object, cols: object, v: np.ndarray) -> np.ndarray:
                return self._a[np.ix_(np.asarray(rows), np.asarray(cols))] @ v

            def solve_free(self, free: object, rhs: np.ndarray) -> np.ndarray:
                idx = np.asarray(free)
                return np.linalg.solve(self._a[np.ix_(idx, idx)], rhs)

            def rcond_free(self, free: object) -> float:
                idx = np.asarray(free)
                if idx.size == 0:
                    return 1.0
                eig = np.linalg.eigvalsh(self._a[np.ix_(idx, idx)])
                return float(eig[0] / eig[-1])

        backend = CustomOperator(problem["covariance"])
        assert isinstance(backend, CovarianceOperator)
        cla_raw = CLA(**problem)
        cla_op = CLA(**{**problem, "covariance": backend})
        assert len(cla_raw) == len(cla_op)
        for tp_raw, tp_op in zip(cla_raw.turning_points, cla_op.turning_points, strict=True):
            np.testing.assert_allclose(tp_raw.weights, tp_op.weights, atol=1e-12)

    def test_covariance_operator_property(self, problem: dict) -> None:
        """The property wraps a raw array in a DenseOperator and passes backends through."""
        cla_raw = CLA(**problem)
        assert isinstance(cla_raw.covariance_operator, DenseOperator)
        backend = dense_covariance(problem["covariance"])
        cla_op = CLA(**{**problem, "covariance": backend})
        assert cla_op.covariance_operator is backend

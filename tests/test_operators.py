"""Tests for the covariance backend abstraction (Phase 1 of issue #646).

This module tests the CovarianceOperator protocol and its dense reference
implementation, and verifies that the CLA produces identical results whether
it is given a raw covariance matrix or a DenseCovariance backend.
"""

import numpy as np
import pytest

from cvxcla import CLA, CovarianceOperator, DenseCovariance


@pytest.fixture
def random_spd():
    """Create a random symmetric positive definite matrix and a free mask."""
    rng = np.random.default_rng(7)
    n = 12
    l_matrix = rng.standard_normal((n, n))
    matrix = l_matrix @ l_matrix.T + n * np.eye(n)
    free = np.zeros(n, dtype=bool)
    free[[0, 3, 4, 7, 11]] = True
    return matrix, free, rng


class TestDenseCovariance:
    """Tests for the dense reference implementation of the protocol."""

    def test_satisfies_protocol(self, random_spd):
        """DenseCovariance is a runtime instance of CovarianceOperator."""
        matrix, _, _ = random_spd
        assert isinstance(DenseCovariance(matrix), CovarianceOperator)

    def test_n(self, random_spd):
        """The dimension reflects the wrapped matrix."""
        matrix, _, _ = random_spd
        assert DenseCovariance(matrix).n == matrix.shape[0]

    def test_matvec_matches_numpy(self, random_spd):
        """Matvec reproduces the plain matrix product, vector and multi-RHS."""
        matrix, _, rng = random_spd
        cov = DenseCovariance(matrix)
        x = rng.standard_normal(cov.n)
        np.testing.assert_allclose(cov.matvec(x), matrix @ x)
        xs = rng.standard_normal((cov.n, 3))
        np.testing.assert_allclose(cov.matvec(xs), matrix @ xs)

    def test_solve_free_matches_numpy(self, random_spd):
        """solve_free reproduces a direct solve on the free block, vector and multi-RHS."""
        matrix, free, rng = random_spd
        cov = DenseCovariance(matrix)
        reduced = matrix[np.ix_(free, free)]
        rhs = rng.standard_normal(int(free.sum()))
        np.testing.assert_allclose(cov.solve_free(free, rhs), np.linalg.solve(reduced, rhs))
        rhs2 = rng.standard_normal((int(free.sum()), 2))
        np.testing.assert_allclose(cov.solve_free(free, rhs2), np.linalg.solve(reduced, rhs2))

    def test_cross_matches_numpy(self, random_spd):
        """Cross reproduces the free-to-blocked product with a full-length vector."""
        matrix, free, rng = random_spd
        cov = DenseCovariance(matrix)
        x = rng.standard_normal(cov.n)
        expected = matrix[np.ix_(free, ~free)] @ x[~free]
        np.testing.assert_allclose(cov.cross(free, x), expected)

    def test_rejects_non_square(self):
        """A non-square matrix is rejected."""
        with pytest.raises(ValueError, match="square"):
            DenseCovariance(np.ones((2, 3)))

    def test_rejects_non_symmetric(self):
        """A non-symmetric matrix is rejected."""
        matrix = np.array([[1.0, 2.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            DenseCovariance(matrix)


class TestCLAWithOperator:
    """Tests that CLA accepts a covariance backend in place of a raw matrix."""

    @pytest.fixture
    def problem(self):
        """Create a portfolio problem large enough to produce several turning points."""
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

    def test_dense_operator_matches_raw_matrix(self, problem):
        """CLA(DenseCovariance(S)) and CLA(S) produce identical turning points."""
        cla_raw = CLA(**problem)
        cla_op = CLA(**{**problem, "covariance": DenseCovariance(problem["covariance"])})

        assert len(cla_raw) == len(cla_op)
        for tp_raw, tp_op in zip(cla_raw.turning_points, cla_op.turning_points, strict=True):
            assert tp_raw.lamb == pytest.approx(tp_op.lamb, abs=1e-12)
            np.testing.assert_allclose(tp_raw.weights, tp_op.weights, atol=1e-12)
            np.testing.assert_array_equal(tp_raw.free, tp_op.free)

    def test_frontier_with_operator(self, problem):
        """The frontier property works when a backend is passed."""
        cla_op = CLA(**{**problem, "covariance": DenseCovariance(problem["covariance"])})
        frontier = cla_op.frontier
        assert len(frontier) == len(cla_op)
        assert np.all(frontier.variance > 0)

    def test_structured_operator_not_yet_supported(self, problem):
        """A backend without an explicit matrix raises a clear Phase 2 pointer."""

        class MatrixFreeCovariance:
            """Minimal protocol implementation without a dense matrix."""

            def __init__(self, matrix):
                self._matrix = matrix

            @property
            def n(self):
                return self._matrix.shape[0]

            def matvec(self, x):
                return self._matrix @ x

            def solve_free(self, free, rhs):
                return np.linalg.solve(self._matrix[np.ix_(free, free)], rhs)

            def cross(self, free, x):
                return self._matrix[np.ix_(free, ~free)] @ x[~free]

        backend = MatrixFreeCovariance(problem["covariance"])
        assert isinstance(backend, CovarianceOperator)
        with pytest.raises(NotImplementedError, match="Phase 2"):
            CLA(**{**problem, "covariance": backend})

    def test_covariance_operator_property(self, problem):
        """The covariance_operator property wraps arrays and passes backends through."""
        cla_raw = CLA(**problem)
        assert isinstance(cla_raw.covariance_operator, DenseCovariance)

        backend = DenseCovariance(problem["covariance"])
        cla_op = CLA(**{**problem, "covariance": backend})
        assert cla_op.covariance_operator is backend

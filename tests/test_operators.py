"""Tests for the covariance backends of the CLA (issue #646).

This module tests the CovarianceOperator protocol and its implementations:
DenseCovariance (the dense reference), IncrementalDenseCovariance (the dense
backend that maintains Sigma_FF^{-1} across turning points), and
FactorCovariance (diagonal plus low rank, solved via the Woodbury identity).
It verifies that the CLA produces identical results whether it is given a raw
covariance matrix, a DenseCovariance backend, or a matrix-free backend, and
that the incremental and factor backends trace the same frontier as the dense
one.
"""

import tracemalloc

import numpy as np
import pytest

from cvxcla import (
    CLA,
    CovarianceOperator,
    DenseCovariance,
    FactorCovariance,
    GramCovariance,
    IncrementalDenseCovariance,
)


def random_factor_model(rng, n, k):
    """Create a random diagonal-plus-low-rank covariance and its dense equivalent."""
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.05, 0.5, n)
    dense = np.diag(d) + (u * delta) @ u.T
    return FactorCovariance(d=d, u=u, delta=delta), dense


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

    def test_rcond_free_matches_numpy(self, random_spd):
        """rcond_free is the free block's reciprocal 2-norm condition number."""
        matrix, free, _ = random_spd
        cov = DenseCovariance(matrix)
        block = matrix[np.ix_(free, free)]
        assert cov.rcond_free(free) == pytest.approx(1.0 / np.linalg.cond(block))

    def test_rcond_free_singular_block_is_zero(self):
        """A rank-deficient free block reports a reciprocal condition number of ~0."""
        # Two identical assets make the 2x2 free block exactly singular.
        matrix = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
        cov = DenseCovariance(matrix)
        free = np.array([True, True, False])
        assert cov.rcond_free(free) < 1e-12

    def test_rcond_free_empty_block_is_one(self, random_spd):
        """An empty free set is treated as perfectly conditioned."""
        matrix, _, _ = random_spd
        cov = DenseCovariance(matrix)
        assert cov.rcond_free(np.zeros(matrix.shape[0], dtype=bool)) == 1.0

    def test_rcond_free_zero_block_is_zero(self):
        """A zero (degenerate) free block has a non-positive top eigenvalue and maps to 0."""
        cov = DenseCovariance(np.zeros((2, 2)))
        assert cov.rcond_free(np.array([True, True])) == 0.0

    def test_rcond_floor_cleared_well_conditioned_fast_path(self, random_spd):
        """A well-conditioned matrix clears the floor via the cheap Cholesky estimate."""
        matrix, _, _ = random_spd
        assert DenseCovariance(matrix).rcond_floor_cleared(1e-12) is True

    def test_rcond_floor_cleared_non_positive_definite(self):
        """A symmetric indefinite matrix (Cholesky fails) is below any positive floor."""
        assert DenseCovariance(np.diag([1.0, -1.0])).rcond_floor_cleared(1e-12) is False

    def test_rcond_floor_cleared_borderline_above_floor(self):
        """An estimate within the margin of the floor defers to the exact rcond (clears)."""
        # rcond 5e-11: below margin*floor (1e-9) so it falls back, but >= the 1e-12 floor.
        assert DenseCovariance(np.diag([1.0, 5e-11])).rcond_floor_cleared(1e-12) is True

    def test_rcond_floor_cleared_borderline_below_floor(self):
        """An estimate within the margin of the floor defers to the exact rcond (fails)."""
        # rcond 5e-13: positive-definite so Cholesky succeeds, but below the 1e-12 floor.
        assert DenseCovariance(np.diag([1.0, 5e-13])).rcond_floor_cleared(1e-12) is False

    @pytest.mark.parametrize("diagonal", [[1.0, 1.0], [1.0, 1e-6], [1.0, 5e-11], [1.0, 5e-13], [2.0, 0.0]])
    def test_rcond_floor_cleared_matches_exact(self, diagonal):
        """The fast boolean equals the exact full-mask ``rcond_free`` comparison everywhere."""
        cov = DenseCovariance(np.diag(diagonal))
        full = np.ones(len(diagonal), dtype=bool)
        for floor in (1e-12, 1e-9, 1e-3):
            assert cov.rcond_floor_cleared(floor) == (cov.rcond_free(full) >= floor)

    def test_rejects_non_square(self):
        """A non-square matrix is rejected."""
        with pytest.raises(ValueError, match="square"):
            DenseCovariance(np.ones((2, 3)))

    def test_rejects_non_symmetric(self):
        """A non-symmetric matrix is rejected."""
        matrix = np.array([[1.0, 2.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            DenseCovariance(matrix)


class TestIncrementalDenseCovariance:
    """Tests for the dense backend that maintains Sigma_FF^{-1} across solves."""

    def test_satisfies_protocol(self, random_spd):
        """IncrementalDenseCovariance is a runtime instance of CovarianceOperator."""
        matrix, _, _ = random_spd
        assert isinstance(IncrementalDenseCovariance(matrix), CovarianceOperator)

    def test_n_matvec_cross_match_dense(self, random_spd):
        """n, matvec and cross delegate to and match the dense reference."""
        matrix, free, rng = random_spd
        cov = IncrementalDenseCovariance(matrix)
        ref = DenseCovariance(matrix)
        assert cov.n == ref.n
        x = rng.standard_normal(cov.n)
        np.testing.assert_allclose(cov.matvec(x), ref.matvec(x))
        np.testing.assert_allclose(cov.cross(free, x), ref.cross(free, x))

    def test_solve_free_incremental_matches_numpy(self, random_spd):
        """A sequence of single-asset flips keeps the maintained inverse correct.

        The first call recomputes from scratch; each later call exercises a
        rank-one insert (asset enters) or deletion (asset leaves), including the
        permutation back to ascending free-index order. Both vector and multi-RHS
        right-hand sides are checked.
        """
        matrix, _, rng = random_spd
        n = matrix.shape[0]
        cov = IncrementalDenseCovariance(matrix)

        free = np.zeros(n, dtype=bool)
        free[[0, 3, 4, 7, 11]] = True
        masks = [free.copy()]
        cur = free.copy()
        for flip in (2, 7, 9, 0, 5, 3, 10):  # mix of adds and removes
            cur = cur.copy()
            cur[flip] = not cur[flip]
            masks.append(cur.copy())

        for mask in masks:
            idx = np.flatnonzero(mask)
            reduced = matrix[np.ix_(idx, idx)]
            rhs = rng.standard_normal(idx.size)
            np.testing.assert_allclose(cov.solve_free(mask, rhs), np.linalg.solve(reduced, rhs), atol=1e-10)
            rhs2 = rng.standard_normal((idx.size, 3))
            np.testing.assert_allclose(cov.solve_free(mask, rhs2), np.linalg.solve(reduced, rhs2), atol=1e-10)

    def test_solve_free_multi_change_refactors(self, random_spd):
        """A change that is not a single-asset flip falls back to a fresh factorisation."""
        matrix, _, rng = random_spd
        n = matrix.shape[0]
        cov = IncrementalDenseCovariance(matrix)

        first = np.zeros(n, dtype=bool)
        first[[0, 1, 2]] = True
        cov.solve_free(first, rng.standard_normal(3))

        disjoint = np.zeros(n, dtype=bool)
        disjoint[[5, 6, 7, 8]] = True  # shares no asset with `first`
        idx = np.flatnonzero(disjoint)
        rhs = rng.standard_normal(idx.size)
        np.testing.assert_allclose(
            cov.solve_free(disjoint, rhs),
            np.linalg.solve(matrix[np.ix_(idx, idx)], rhs),
            atol=1e-10,
        )

    def test_solve_free_empty_refactors(self, random_spd):
        """An all-False free mask refactorises to a 0x0 inverse and a length-0 solution."""
        matrix, _, _ = random_spd
        cov = IncrementalDenseCovariance(matrix)
        empty = np.zeros(matrix.shape[0], dtype=bool)
        result = cov.solve_free(empty, np.zeros(0))
        assert result.shape == (0,)

    def test_insert_nonpositive_schur_refactors(self):
        """A non-positive Schur pivot on insert falls back to a fresh factorisation.

        ``[[1, 2], [2, 1]]`` is symmetric but indefinite, so bordering the cached
        ``{0}`` inverse with asset 1 yields a negative Schur complement; the update
        bails out and the full block is refactorised instead.
        """
        matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
        cov = IncrementalDenseCovariance(matrix)
        cov.solve_free(np.array([True, False]), np.zeros(1))  # seed cache with {0}
        rhs = np.array([1.0, -1.0])
        np.testing.assert_allclose(
            cov.solve_free(np.array([True, True]), rhs),
            np.linalg.solve(matrix, rhs),
        )

    def test_delete_nonpositive_pivot_refactors(self):
        """A non-positive pivot on delete falls back to a fresh factorisation.

        The inverse of the indefinite ``[[1, 2], [2, 1]]`` has negative diagonal
        entries, so removing asset 1 from the cached ``{0, 1}`` inverse hits a
        non-positive pivot and the remaining block is refactorised instead.
        """
        matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
        cov = IncrementalDenseCovariance(matrix)
        cov.solve_free(np.array([True, True]), np.zeros(2))  # seed cache with {0, 1}
        rhs = np.array([3.0])
        np.testing.assert_allclose(
            cov.solve_free(np.array([True, False]), rhs),
            np.linalg.solve(matrix[:1, :1], rhs),
        )

    def test_rcond_free_matches_dense(self, random_spd):
        """rcond_free delegates to and matches the dense reference."""
        matrix, free, _ = random_spd
        cov = IncrementalDenseCovariance(matrix)
        assert cov.rcond_free(free) == DenseCovariance(matrix).rcond_free(free)

    def test_rejects_non_square_and_non_symmetric(self):
        """Validation is inherited from the dense reference."""
        with pytest.raises(ValueError, match="square"):
            IncrementalDenseCovariance(np.ones((2, 3)))
        with pytest.raises(ValueError, match="symmetric"):
            IncrementalDenseCovariance(np.array([[1.0, 2.0], [0.0, 1.0]]))


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

    def test_matrix_free_backend_supported(self, problem):
        """A backend without an explicit matrix traces the same frontier as the raw matrix."""

        class MatrixFreeCovariance:
            """Minimal protocol implementation without a dense matrix attribute."""

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

            def rcond_free(self, free):
                block = self._matrix[np.ix_(free, free)]
                eigenvalues = np.linalg.eigvalsh(block)
                return float(eigenvalues[0] / eigenvalues[-1])

        backend = MatrixFreeCovariance(problem["covariance"])
        assert isinstance(backend, CovarianceOperator)

        cla_raw = CLA(**problem)
        cla_op = CLA(**{**problem, "covariance": backend})

        assert len(cla_raw) == len(cla_op)
        for tp_raw, tp_op in zip(cla_raw.turning_points, cla_op.turning_points, strict=True):
            assert tp_raw.lamb == pytest.approx(tp_op.lamb, abs=1e-12)
            np.testing.assert_allclose(tp_raw.weights, tp_op.weights, atol=1e-12)
            np.testing.assert_array_equal(tp_raw.free, tp_op.free)

    def test_covariance_operator_property(self, problem):
        """The covariance_operator property wraps arrays and passes backends through."""
        cla_raw = CLA(**problem)
        assert isinstance(cla_raw.covariance_operator, DenseCovariance)

        backend = DenseCovariance(problem["covariance"])
        cla_op = CLA(**{**problem, "covariance": backend})
        assert cla_op.covariance_operator is backend


class TestFactorCovariance:
    """Tests for the Woodbury-based diagonal-plus-low-rank backend."""

    @pytest.fixture
    def model(self):
        """Create a random factor model with its dense equivalent and a free mask."""
        rng = np.random.default_rng(11)
        n, k = 40, 5
        factor, dense = random_factor_model(rng, n, k)
        free = rng.random(n) < 0.5
        return factor, dense, free, rng

    def test_satisfies_protocol(self, model):
        """FactorCovariance is a runtime instance of CovarianceOperator."""
        factor, _, _, _ = model
        assert isinstance(factor, CovarianceOperator)

    def test_dimensions(self, model):
        """Dimensions n and k reflect the factor structure."""
        factor, _, _, _ = model
        assert factor.n == 40
        assert factor.k == 5

    def test_matvec_matches_dense(self, model):
        """Matvec agrees with the dense product, vector and multi-RHS."""
        factor, dense, _, rng = model
        x = rng.standard_normal(factor.n)
        np.testing.assert_allclose(factor.matvec(x), dense @ x)
        xs = rng.standard_normal((factor.n, 3))
        np.testing.assert_allclose(factor.matvec(xs), dense @ xs)

    def test_solve_free_matches_dense(self, model):
        """solve_free agrees with a dense solve on the free block, vector and multi-RHS."""
        factor, dense, free, rng = model
        reduced = dense[np.ix_(free, free)]
        rhs = rng.standard_normal(int(free.sum()))
        np.testing.assert_allclose(factor.solve_free(free, rhs), np.linalg.solve(reduced, rhs))
        rhs2 = rng.standard_normal((int(free.sum()), 4))
        np.testing.assert_allclose(factor.solve_free(free, rhs2), np.linalg.solve(reduced, rhs2))

    def test_cross_matches_dense(self, model):
        """Cross agrees with the dense free-to-blocked product."""
        factor, dense, free, rng = model
        x = rng.standard_normal(factor.n)
        expected = dense[np.ix_(free, ~free)] @ x[~free]
        np.testing.assert_allclose(factor.cross(free, x), expected)

    def test_rcond_free_lower_bounds_true_value(self, model):
        """rcond_free is a positive, valid lower bound on the true reciprocal cond.

        The factor block is positive definite by construction, so the bound is
        strictly positive (it never spuriously trips the singularity guard), and
        Weyl's inequalities make it no larger than the exact value.
        """
        factor, dense, free, _ = model
        bound = factor.rcond_free(free)
        true_rcond = 1.0 / np.linalg.cond(dense[np.ix_(free, free)])
        assert 0.0 < bound <= true_rcond + 1e-15

    def test_rcond_free_empty_block_is_one(self, model):
        """An empty free set is treated as perfectly conditioned."""
        factor, _, _, _ = model
        assert factor.rcond_free(np.zeros(factor.n, dtype=bool)) == 1.0

    def test_full_delta_matrix(self):
        """A full symmetric (k, k) delta is supported."""
        rng = np.random.default_rng(3)
        n, k = 30, 4
        u = rng.standard_normal((n, k))
        l_matrix = rng.standard_normal((k, k))
        delta = l_matrix @ l_matrix.T + k * np.eye(k)
        d = rng.uniform(0.1, 1.0, n)
        factor = FactorCovariance(d=d, u=u, delta=delta)
        dense = np.diag(d) + u @ delta @ u.T

        x = rng.standard_normal(n)
        np.testing.assert_allclose(factor.matvec(x), dense @ x)

        free = rng.random(n) < 0.6
        rhs = rng.standard_normal(int(free.sum()))
        np.testing.assert_allclose(
            factor.solve_free(free, rhs),
            np.linalg.solve(dense[np.ix_(free, free)], rhs),
        )
        np.testing.assert_allclose(factor.cross(free, x), dense[np.ix_(free, ~free)] @ x[~free])

    def test_rejects_nonpositive_d(self):
        """A non-positive idiosyncratic variance is rejected."""
        with pytest.raises(ValueError, match="positive idiosyncratic"):
            FactorCovariance(d=np.array([1.0, 0.0]), u=np.ones((2, 1)), delta=np.ones(1))

    def test_rejects_mismatched_u(self):
        """A loadings matrix not matching d is rejected."""
        with pytest.raises(ValueError, match=r"u must have shape"):
            FactorCovariance(d=np.ones(3), u=np.ones((2, 1)), delta=np.ones(1))

    def test_rejects_wrong_delta_length(self):
        """A diagonal delta with the wrong number of entries is rejected."""
        with pytest.raises(ValueError, match="entries"):
            FactorCovariance(d=np.ones(3), u=np.ones((3, 2)), delta=np.ones(3))

    def test_rejects_nonpositive_diagonal_delta(self):
        """A diagonal delta with non-positive entries is rejected."""
        with pytest.raises(ValueError, match="positive entries"):
            FactorCovariance(d=np.ones(3), u=np.ones((3, 1)), delta=np.array([-1.0]))

    def test_rejects_asymmetric_delta(self):
        """An asymmetric full delta is rejected."""
        delta = np.array([[1.0, 0.5], [0.0, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            FactorCovariance(d=np.ones(3), u=np.ones((3, 2)), delta=delta)

    def test_rejects_bad_delta_shape(self):
        """A delta of wrong shape or dimension is rejected."""
        with pytest.raises(ValueError, match=r"delta must have shape"):
            FactorCovariance(d=np.ones(3), u=np.ones((3, 2)), delta=np.ones((2, 3)))
        with pytest.raises(ValueError, match="vector or"):
            FactorCovariance(d=np.ones(3), u=np.ones((3, 2)), delta=np.ones((2, 2, 2)))


@pytest.mark.property
class TestIncrementalMatchesDense:
    """Property test: the incremental backend traces the same frontier as the dense one."""

    @pytest.mark.parametrize(
        ("seed", "n", "k"),
        [
            (1, 30, 3),
            (2, 80, 8),
            (8, 150, 15),
            (4, 300, 25),
        ],
    )
    def test_turning_points_match(self, seed, n, k):
        """Lambdas, weights, and free sets agree between the incremental and dense backends."""
        rng = np.random.default_rng(seed)
        _, dense = random_factor_model(rng, n, k)

        problem = {
            "mean": rng.uniform(0.0, 0.1, n),
            "lower_bounds": np.zeros(n),
            "upper_bounds": np.minimum(1.0, rng.uniform(3.0, 8.0, n) / n),
            "a": np.ones((1, n)),
            "b": np.ones(1),
        }

        cla_dense = CLA(covariance=DenseCovariance(dense), **problem)
        cla_incr = CLA(covariance=IncrementalDenseCovariance(dense), **problem)

        assert len(cla_dense) > 2
        assert len(cla_dense) == len(cla_incr)
        for tp_dense, tp_incr in zip(cla_dense.turning_points, cla_incr.turning_points, strict=True):
            assert tp_dense.lamb == pytest.approx(tp_incr.lamb, rel=1e-7, abs=1e-10)
            np.testing.assert_allclose(tp_dense.weights, tp_incr.weights, atol=1e-8)
            np.testing.assert_array_equal(tp_dense.free, tp_incr.free)


@pytest.mark.property
class TestFactorMatchesDense:
    """Property test: the factor backend traces the same frontier as the dense one."""

    @pytest.mark.parametrize(
        ("seed", "n", "k"),
        [
            (1, 30, 3),
            (2, 80, 8),
            (8, 150, 15),
            (4, 300, 25),
            (5, 500, 40),
        ],
    )
    def test_turning_points_match(self, seed, n, k):
        """Lambdas, weights, and free sets agree between the factor and dense backends."""
        rng = np.random.default_rng(seed)
        factor, dense = random_factor_model(rng, n, k)

        problem = {
            "mean": rng.uniform(0.0, 0.1, n),
            "lower_bounds": np.zeros(n),
            # tight upper bounds so that many assets sit at their bound,
            # exercising the cross-product path; randomised per asset so no
            # two events coincide exactly (identical bounds create exact ties
            # that the two backends may break differently)
            "upper_bounds": np.minimum(1.0, rng.uniform(3.0, 8.0, n) / n),
            "a": np.ones((1, n)),
            "b": np.ones(1),
        }

        cla_dense = CLA(covariance=dense, **problem)
        cla_factor = CLA(covariance=factor, **problem)

        assert len(cla_dense) > 2
        assert len(cla_dense) == len(cla_factor)
        for tp_dense, tp_factor in zip(cla_dense.turning_points, cla_factor.turning_points, strict=True):
            assert tp_dense.lamb == pytest.approx(tp_factor.lamb, rel=1e-7, abs=1e-10)
            np.testing.assert_allclose(tp_dense.weights, tp_factor.weights, atol=1e-8)
            np.testing.assert_array_equal(tp_dense.free, tp_factor.free)


@pytest.mark.stress
class TestFactorLargeScale:
    """The factor path never allocates O(n^2): run a frontier infeasible for the dense path."""

    def test_large_frontier_without_dense_allocation(self):
        """An n = 20,000, k = 100 frontier runs in O(n k) memory.

        A dense covariance at this size needs 3.2 GB for Sigma alone (plus as
        much again for the KKT block matrix). We assert the peak memory of the
        whole factor-backend run stays below 3% of one dense matrix.

        The problem is constructed so that the frontier only involves a small
        group of attractive low-volatility assets: every other asset loads
        twice as strongly on a common factor (and carries large idiosyncratic
        risk), so it covaries too strongly with the held portfolio to ever
        enter. This keeps the number of turning points - and therefore the
        size of the result itself - small while n is huge.
        """
        rng = np.random.default_rng(123)
        n, k, good = 20_000, 100, 50

        u = rng.standard_normal((n, k)) / np.sqrt(n)
        u[:, 0] = 1.0
        u[good:, 0] = 2.0
        delta = rng.uniform(0.5, 2.0, k)
        delta[0] = 5.0
        d = rng.uniform(5.0, 10.0, n)
        d[:good] = rng.uniform(0.01, 0.05, good)
        mean = rng.uniform(0.0, 0.01, n)
        mean[:good] += 0.05

        factor = FactorCovariance(d=d, u=u, delta=delta)

        problem = {
            "mean": mean,
            "lower_bounds": np.zeros(n),
            "upper_bounds": np.full(n, 0.1),
            "a": np.ones((1, n)),
            "b": np.ones(1),
        }

        tracemalloc.start()
        cla = CLA(covariance=factor, **problem)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert len(cla) > 2
        dense_sigma_bytes = 8 * n * n
        assert peak < 0.03 * dense_sigma_bytes


def _dense_from_returns(returns, ridge=0.0):
    """The dense covariance a GramCovariance(returns, ridge) must reproduce."""
    sigma = np.cov(returns, rowvar=False)
    if ridge:
        sigma = sigma + ridge * np.eye(returns.shape[1])
    return DenseCovariance(sigma)


class TestGramCovariance:
    """Tests for the data-matrix (Gram) backend.

    GramCovariance(returns, ridge) must behave exactly like
    DenseCovariance(np.cov(returns, rowvar=False) + ridge I), while never forming
    the n x n matrix. We check every protocol operation against that dense
    reference across the long-sample (T >= n) and short-sample (T < n) regimes.
    """

    def test_is_quadratic_form_instance(self):
        """GramCovariance satisfies the QuadraticForm (alias CovarianceOperator) protocol."""
        gram = GramCovariance(np.random.default_rng(0).standard_normal((20, 4)))
        assert isinstance(gram, CovarianceOperator)
        assert gram.n == 4

    def test_matches_numpy_cov(self):
        """The (centered, ddof=1) covariance equals np.cov, not the raw second moment."""
        returns = np.random.default_rng(1).standard_normal((40, 5))
        gram = GramCovariance(returns)
        sigma = np.cov(returns, rowvar=False)
        v = np.arange(1.0, 6.0)
        np.testing.assert_allclose(gram.matvec(v), sigma @ v)
        # the raw (uncentered) second moment differs unless the data is mean-zero
        raw = returns.T @ returns / (returns.shape[0] - 1)
        assert not np.allclose(sigma, raw)

    @pytest.mark.parametrize(("t", "n", "ridge"), [(40, 5, 0.0), (40, 5, 0.3), (4, 8, 0.5)])
    def test_operations_match_dense(self, t, n, ridge):
        """Every operation (matvec / cross / solve_free / rcond_free) agrees with the dense reference.

        Covers the long-sample dense solve (T >= n), the ridged dense solve, and
        the short-sample Woodbury solve (T < n, n_free > T).
        """
        rng = np.random.default_rng(int(t * 100 + n))
        returns = rng.standard_normal((t, n))
        gram = GramCovariance(returns, ridge=ridge)
        dense = _dense_from_returns(returns, ridge=ridge)

        free = np.array([i % 2 == 0 for i in range(n)])
        free[0] = True  # ensure a non-empty, mixed mask

        x = rng.standard_normal(n)
        np.testing.assert_allclose(gram.matvec(x), dense.matvec(x), atol=1e-10)
        np.testing.assert_allclose(gram.cross(free, x), dense.cross(free, x), atol=1e-10)

        # single- and multi-RHS solves against the free block
        rhs = rng.standard_normal(int(free.sum()))
        if ridge > 0.0 or int(free.sum()) <= t:  # solvable block
            np.testing.assert_allclose(gram.solve_free(free, rhs), dense.solve_free(free, rhs), atol=1e-8)
            rhs2 = rng.standard_normal((int(free.sum()), 3))
            np.testing.assert_allclose(gram.solve_free(free, rhs2), dense.solve_free(free, rhs2), atol=1e-8)

        np.testing.assert_allclose(gram.rcond_free(free), dense.rcond_free(free), rtol=1e-6, atol=1e-12)

    def test_woodbury_path_for_all_free_short_sample(self):
        """With T < n_free and a ridge, the all-free solve uses Woodbury yet matches dense."""
        rng = np.random.default_rng(7)
        returns = rng.standard_normal((5, 12))
        gram = GramCovariance(returns, ridge=0.2)
        dense = _dense_from_returns(returns, ridge=0.2)
        free = np.ones(12, dtype=bool)
        rhs = rng.standard_normal(12)
        np.testing.assert_allclose(gram.solve_free(free, rhs), dense.solve_free(free, rhs), atol=1e-8)

    def test_rcond_empty_free_is_one(self):
        """An empty free set is treated as perfectly conditioned."""
        gram = GramCovariance(np.random.default_rng(0).standard_normal((10, 3)))
        assert gram.rcond_free(np.zeros(3, dtype=bool)) == 1.0

    def test_rcond_singular_free_block_is_zero(self):
        """A constant (zero-variance) free column has a singular block: rcond = 0."""
        returns = np.random.default_rng(0).standard_normal((6, 3))
        returns[:, 0] = 2.5  # constant column -> centered to zero
        gram = GramCovariance(returns)  # no ridge
        assert gram.rcond_free(np.array([True, False, False])) == 0.0

    def test_rank_deficient_free_block_drives_rcond_to_zero(self):
        """Without a ridge, a free set larger than the data rank is singular (rcond ~ 0)."""
        returns = np.random.default_rng(0).standard_normal((4, 8))
        gram = GramCovariance(returns)  # rank <= T-1 = 3 < 8
        assert gram.rcond_free(np.ones(8, dtype=bool)) < 1e-12

    def test_cla_matches_dense_backend(self):
        """The CLA traces the same frontier from the data matrix as from np.cov."""
        rng = np.random.default_rng(3)
        t, n = 60, 6  # long sample -> full-rank, well-posed
        returns = rng.standard_normal((t, n))
        mean = rng.uniform(0.0, 1.0, n)
        kwargs = {
            "mean": mean,
            "lower_bounds": np.zeros(n),
            "upper_bounds": np.ones(n),
            "a": np.ones((1, n)),
            "b": np.ones(1),
        }
        cla_gram = CLA(covariance=GramCovariance(returns), **kwargs)
        cla_dense = CLA(covariance=DenseCovariance(np.cov(returns, rowvar=False)), **kwargs)

        assert len(cla_gram) == len(cla_dense)
        for tp_g, tp_d in zip(cla_gram.turning_points, cla_dense.turning_points, strict=True):
            np.testing.assert_allclose(tp_g.weights, tp_d.weights, atol=1e-6)

    def test_rejects_non_2d_returns(self):
        """A 1d returns array is rejected."""
        with pytest.raises(ValueError, match=r"must be a \(T, n\) matrix"):
            GramCovariance(np.ones(5))

    def test_rejects_too_few_observations(self):
        """A single observation cannot define a sample covariance."""
        with pytest.raises(ValueError, match="T >= 2"):
            GramCovariance(np.ones((1, 3)))

    def test_rejects_negative_ridge(self):
        """A negative ridge is rejected."""
        with pytest.raises(ValueError, match="ridge must be non-negative"):
            GramCovariance(np.random.default_rng(0).standard_normal((10, 3)), ridge=-0.1)

    def test_solve_is_matrix_free_in_n(self):
        """A short-sample free-block solve never allocates an n x n matrix.

        With ``T << n`` and a ridge, ``solve_free`` over the whole universe goes
        through the Woodbury identity in ``T``-space, so its peak allocation
        scales with ``T n`` (the data), not ``n^2`` (a dense block). This is the
        memory advantage that motivates the backend.
        """
        rng = np.random.default_rng(0)
        n, t = 800, 30
        gram = GramCovariance(rng.standard_normal((t, n)), ridge=1e-2)
        free = np.ones(n, dtype=bool)
        rhs = rng.standard_normal(n)
        _ = gram._centered  # warm the cached data matrix; measure only the solve

        tracemalloc.start()
        gram.solve_free(free, rhs)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # The solve allocates a few copies of the (T, n) data, never an n x n block.
        assert peak < 4 * 8 * t * n
        assert peak < 0.2 * 8 * n * n  # far below a single dense n x n matrix


def test_dense_solve_free_falls_back_when_not_positive_definite():
    """A symmetric indefinite block makes Cholesky fail; the LU fallback still solves it."""
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric; eigenvalues 3 and -1 (indefinite)
    cov = DenseCovariance(matrix)
    free = np.array([True, True])
    rhs = np.array([1.0, -1.0])
    np.testing.assert_allclose(cov.solve_free(free, rhs), np.linalg.solve(matrix, rhs))

"""Tests for the CLA module.

This module tests the main Critical Line Algorithm implementation,
including initialization, turning point computation, and frontier generation.
"""

import numpy as np
import pytest

from cvxcla import CLA


class TestCLA:
    """Tests for the CLA class."""

    @pytest.fixture
    def simple_problem(self):
        """Create a simple portfolio optimization problem."""
        n = 3
        np.random.seed(42)
        mean = np.array([0.1, 0.15, 0.2])
        # Create a positive definite covariance matrix
        l_matrix = np.random.randn(n, n)
        covariance = l_matrix @ l_matrix.T
        lower_bounds = np.zeros(n)
        upper_bounds = np.ones(n)
        a = np.ones((1, n))  # Fully invested constraint
        b = np.ones(1)
        return {
            "mean": mean,
            "covariance": covariance,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "a": a,
            "b": b,
        }

    def test_cla_initialization(self, simple_problem):
        """Test that CLA can be initialized with valid inputs."""
        cla = CLA(**simple_problem)
        assert cla.mean.shape == (3,)
        assert cla.covariance.shape == (3, 3)
        assert len(cla.turning_points) > 0

    def test_turning_points_generated(self, simple_problem):
        """Test that multiple turning points are generated."""
        cla = CLA(**simple_problem)
        # Should have at least 2 turning points (first at lambda=inf and last at lambda=0)
        assert len(cla) >= 2
        # First turning point should have lambda = inf
        assert cla.turning_points[0].lamb == np.inf
        # Last turning point should have lambda = 0
        assert cla.turning_points[-1].lamb == 0.0

    def test_weights_sum_to_one(self, simple_problem):
        """Test that all turning point weights sum to 1."""
        cla = CLA(**simple_problem)
        for tp in cla.turning_points:
            assert np.isclose(np.sum(tp.weights), 1.0)

    def test_weights_respect_bounds(self, simple_problem):
        """Test that all weights respect lower and upper bounds."""
        cla = CLA(**simple_problem)
        for tp in cla.turning_points:
            assert np.all(tp.weights >= simple_problem["lower_bounds"] - cla.tol)
            assert np.all(tp.weights <= simple_problem["upper_bounds"] + cla.tol)

    def test_lambda_decreasing(self, simple_problem):
        """Test that lambda values are decreasing along the frontier."""
        cla = CLA(**simple_problem)
        lambdas = [tp.lamb for tp in cla.turning_points]
        # Lambda should be monotonically decreasing
        for i in range(len(lambdas) - 1):
            assert lambdas[i] >= lambdas[i + 1]

    def test_frontier_property(self, simple_problem):
        """Test that the frontier property returns a valid Frontier object."""
        cla = CLA(**simple_problem)
        frontier = cla.frontier
        assert len(frontier) == len(cla.turning_points)
        assert np.array_equal(frontier.mean, cla.mean)
        assert np.array_equal(frontier.covariance, cla.covariance)

    def test_with_tight_bounds(self):
        """Test CLA with tight bounds on weights."""
        n = 4
        mean = np.array([0.08, 0.10, 0.12, 0.15])
        # Simple diagonal covariance
        covariance = np.diag([0.04, 0.09, 0.16, 0.25])
        lower_bounds = np.array([0.1, 0.1, 0.1, 0.1])
        upper_bounds = np.array([0.4, 0.4, 0.3, 0.3])
        a = np.ones((1, n))
        b = np.ones(1)

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            a=a,
            b=b,
        )

        assert len(cla) > 0
        for tp in cla.turning_points:
            assert np.all(tp.weights >= lower_bounds - cla.tol)
            assert np.all(tp.weights <= upper_bounds + cla.tol)

    def test_max_sharpe_ratio(self, simple_problem):
        """Test that maximum Sharpe ratio can be computed."""
        cla = CLA(**simple_problem)
        max_sr, max_weights = cla.frontier.max_sharpe

        assert isinstance(max_sr, float)
        assert max_weights.shape == (3,)
        assert np.isclose(np.sum(max_weights), 1.0)
        assert max_sr > 0  # Sharpe ratio should be positive with positive returns

    def test_with_different_tolerance(self, simple_problem):
        """Test CLA with different tolerance values."""
        cla1 = CLA(**simple_problem, tol=1e-5)
        cla2 = CLA(**simple_problem, tol=1e-8)

        # Both should produce valid frontiers
        assert len(cla1) > 0
        assert len(cla2) > 0

    def test_two_asset_problem(self):
        """Test CLA with a simple two-asset problem."""
        mean = np.array([0.1, 0.15])
        covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        lower_bounds = np.array([0.0, 0.0])
        upper_bounds = np.array([1.0, 1.0])
        a = np.ones((1, 2))
        b = np.ones(1)

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            a=a,
            b=b,
        )

        # Should generate a valid frontier
        assert len(cla) >= 2
        # First point should have all weight on higher return asset (if allowed)
        first_tp = cla.turning_points[0]
        assert first_tp.weights[1] >= first_tp.weights[0]  # Asset 1 has higher return

    def test_proj_property(self, simple_problem):
        """Test the projection matrix property."""
        cla = CLA(**simple_problem)
        proj = cla.proj

        # proj should be [covariance | a.T]
        n = simple_problem["mean"].shape[0]
        m = simple_problem["a"].shape[0]
        assert proj.shape == (n, n + m)

    def test_kkt_property(self, simple_problem):
        """Test the KKT matrix property."""
        cla = CLA(**simple_problem)
        kkt = cla.kkt

        n = simple_problem["mean"].shape[0]
        m = simple_problem["a"].shape[0]
        assert kkt.shape == (n + m, n + m)
        # KKT matrix should be symmetric
        assert np.allclose(kkt, kkt.T)

    def test_solve_static_method(self):
        """Test the static _solve method."""
        # Create a simple system: [2 1; 1 2] @ x = b
        a = np.array([[2.0, 1.0], [1.0, 2.0]])
        b = np.array([[3.0, 6.0], [3.0, 6.0]])
        free = np.array([True, True])

        alpha, beta = CLA._solve(a, b, free)

        # Check that the solution is correct
        assert np.allclose(a @ np.column_stack([alpha, beta]), b)

    def test_solve_with_fixed_variables(self):
        """Test _solve with some variables fixed."""
        a = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
        b = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        free = np.array([True, False, True])

        alpha, beta = CLA._solve(a, b, free)

        # Solution should have shape (3, 2)
        assert alpha.shape == (3,)
        assert beta.shape == (3,)


class TestCLAEdgeCases:
    """Test edge cases and special scenarios for CLA."""

    def test_single_asset(self):
        """Test with a single asset."""
        mean = np.array([0.1])
        covariance = np.array([[0.04]])
        lower_bounds = np.array([1.0])  # Must invest fully
        upper_bounds = np.array([1.0])
        a = np.ones((1, 1))
        b = np.ones(1)

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            a=a,
            b=b,
        )

        # Should have at least one turning point
        assert len(cla) >= 1
        # All weights should be 1.0
        for tp in cla.turning_points:
            assert np.isclose(tp.weights[0], 1.0)

    def test_many_assets(self):
        """Test with a larger number of assets."""
        n = 10
        np.random.seed(123)
        mean = np.random.rand(n) * 0.2
        l_matrix = np.random.randn(n, n)
        covariance = l_matrix @ l_matrix.T
        lower_bounds = np.zeros(n)
        upper_bounds = np.ones(n)
        a = np.ones((1, n))
        b = np.ones(1)

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            a=a,
            b=b,
        )

        assert len(cla) > 0
        # Verify all constraints are satisfied
        for tp in cla.turning_points:
            assert np.isclose(np.sum(tp.weights), 1.0)
            assert np.all(tp.weights >= -cla.tol)
            assert np.all(tp.weights <= 1.0 + cla.tol)

    def test_equal_returns(self):
        """Test with assets having equal expected returns."""
        mean = np.array([0.1, 0.1, 0.1])
        covariance = np.eye(3) * 0.04
        lower_bounds = np.zeros(3)
        upper_bounds = np.ones(3)
        a = np.ones((1, 3))
        b = np.ones(1)

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            a=a,
            b=b,
        )

        # Should still produce a valid frontier
        assert len(cla) > 0

    def test_no_short_selling(self):
        """Test no short selling constraint (lower bounds = 0)."""
        n = 5
        np.random.seed(456)
        mean = np.random.rand(n) * 0.15
        l_matrix = np.random.randn(n, n)
        covariance = l_matrix @ l_matrix.T
        lower_bounds = np.zeros(n)  # No short selling
        upper_bounds = np.ones(n)
        a = np.ones((1, n))
        b = np.ones(1)

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            a=a,
            b=b,
        )

        # All weights should be non-negative
        for tp in cla.turning_points:
            assert np.all(tp.weights >= -cla.tol)

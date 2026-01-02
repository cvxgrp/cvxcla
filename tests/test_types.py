"""Tests for the types module.

This module tests the core data structures used in the Critical Line Algorithm:
- FrontierPoint: Represents a point on the efficient frontier
- TurningPoint: Represents a turning point on the efficient frontier
- Frontier: Represents the entire efficient frontier
"""

import numpy as np
import pytest

from cvxcla.types import Frontier, FrontierPoint, TurningPoint


class TestFrontierPoint:
    """Tests for the FrontierPoint class."""

    def test_frontier_point_creation(self):
        """Test creating a valid frontier point with weights summing to 1."""
        weights = np.array([0.3, 0.3, 0.4])
        point = FrontierPoint(weights=weights)
        assert np.allclose(np.sum(point.weights), 1.0)
        assert np.array_equal(point.weights, weights)

    def test_frontier_point_invalid_weights(self):
        """Test that invalid weights (not summing to 1) raise an assertion error."""
        weights = np.array([0.3, 0.3, 0.3])  # Sum = 0.9, not 1.0
        with pytest.raises(AssertionError):
            FrontierPoint(weights=weights)

    def test_mean_computation(self):
        """Test computation of expected return."""
        weights = np.array([0.5, 0.5])
        point = FrontierPoint(weights=weights)
        mean = np.array([0.1, 0.2])
        expected_return = point.mean(mean)
        assert np.isclose(expected_return, 0.15)  # (0.5*0.1 + 0.5*0.2)

    def test_variance_computation(self):
        """Test computation of portfolio variance."""
        weights = np.array([0.6, 0.4])
        point = FrontierPoint(weights=weights)
        # Simple diagonal covariance matrix
        covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        variance = point.variance(covariance)
        # Expected: 0.6^2 * 0.04 + 2 * 0.6 * 0.4 * 0.01 + 0.4^2 * 0.09
        expected = 0.6**2 * 0.04 + 2 * 0.6 * 0.4 * 0.01 + 0.4**2 * 0.09
        assert np.isclose(variance, expected)


class TestTurningPoint:
    """Tests for the TurningPoint class."""

    def test_turning_point_creation(self):
        """Test creating a turning point with free and blocked assets."""
        weights = np.array([0.3, 0.3, 0.4])
        free = np.array([True, False, True])
        tp = TurningPoint(weights=weights, free=free, lamb=0.5)
        assert np.allclose(np.sum(tp.weights), 1.0)
        assert np.array_equal(tp.free, free)
        assert tp.lamb == 0.5

    def test_free_indices(self):
        """Test getting indices of free assets."""
        weights = np.array([0.3, 0.3, 0.4])
        free = np.array([True, False, True])
        tp = TurningPoint(weights=weights, free=free)
        free_idx = tp.free_indices
        assert np.array_equal(free_idx, np.array([0, 2]))

    def test_blocked_indices(self):
        """Test getting indices of blocked assets."""
        weights = np.array([0.3, 0.3, 0.4])
        free = np.array([True, False, True])
        tp = TurningPoint(weights=weights, free=free)
        blocked_idx = tp.blocked_indices
        assert np.array_equal(blocked_idx, np.array([1]))

    def test_default_lambda(self):
        """Test that default lambda is infinity."""
        weights = np.array([1.0])
        free = np.array([True])
        tp = TurningPoint(weights=weights, free=free)
        assert tp.lamb == np.inf


class TestFrontier:
    """Tests for the Frontier class."""

    @pytest.fixture
    def sample_frontier(self):
        """Create a sample frontier for testing."""
        mean = np.array([0.1, 0.15, 0.2])
        covariance = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.01], [0.0, 0.01, 0.16]])
        points = [
            FrontierPoint(weights=np.array([0.2, 0.3, 0.5])),
            FrontierPoint(weights=np.array([0.3, 0.4, 0.3])),
            FrontierPoint(weights=np.array([0.5, 0.3, 0.2])),
        ]
        return Frontier(mean=mean, covariance=covariance, frontier=points)

    def test_frontier_creation(self, sample_frontier):
        """Test creating a frontier object."""
        assert len(sample_frontier) == 3
        assert sample_frontier.mean.shape == (3,)
        assert sample_frontier.covariance.shape == (3, 3)

    def test_frontier_iteration(self, sample_frontier):
        """Test iterating over frontier points."""
        points = list(sample_frontier)
        assert len(points) == 3
        for point in points:
            assert isinstance(point, FrontierPoint)

    def test_weights_property(self, sample_frontier):
        """Test weights matrix property."""
        weights_matrix = sample_frontier.weights
        assert weights_matrix.shape == (3, 3)  # 3 points, 3 assets each
        assert np.allclose(np.sum(weights_matrix, axis=1), 1.0)

    def test_returns_property(self, sample_frontier):
        """Test expected returns property."""
        returns = sample_frontier.returns
        assert returns.shape == (3,)
        # Returns should be computed as mean @ weights.T for each point
        for i, point in enumerate(sample_frontier):
            expected = point.mean(sample_frontier.mean)
            assert np.isclose(returns[i], expected)

    def test_variance_property(self, sample_frontier):
        """Test variance property."""
        variances = sample_frontier.variance
        assert variances.shape == (3,)
        assert np.all(variances >= 0)  # Variance should be non-negative

    def test_volatility_property(self, sample_frontier):
        """Test volatility property."""
        volatility = sample_frontier.volatility
        variance = sample_frontier.variance
        assert np.allclose(volatility, np.sqrt(variance))

    def test_sharpe_ratio_property(self, sample_frontier):
        """Test Sharpe ratio property."""
        sharpe = sample_frontier.sharpe_ratio
        expected = sample_frontier.returns / sample_frontier.volatility
        assert np.allclose(sharpe, expected)

    def test_interpolate(self, sample_frontier):
        """Test frontier interpolation."""
        interpolated = sample_frontier.interpolate(num=10)
        # Should have more points after interpolation
        assert len(interpolated) > len(sample_frontier)
        # All interpolated points should have valid weights
        for point in interpolated:
            assert np.isclose(np.sum(point.weights), 1.0)

    def test_max_sharpe(self, sample_frontier):
        """Test computation of maximum Sharpe ratio."""
        max_sr, max_weights = sample_frontier.max_sharpe
        assert isinstance(max_sr, float)
        assert max_weights.shape == (3,)
        assert np.isclose(np.sum(max_weights), 1.0)
        # Max Sharpe should be at least as good as any discrete point
        assert max_sr >= np.max(sample_frontier.sharpe_ratio) - 1e-6

    def test_empty_frontier(self):
        """Test creating an empty frontier."""
        mean = np.array([0.1, 0.2])
        covariance = np.eye(2)
        frontier = Frontier(mean=mean, covariance=covariance, frontier=[])
        assert len(frontier) == 0

    def test_max_sharpe_edge_case_single_point(self):
        """Test max Sharpe with only one point on frontier."""
        mean = np.array([0.1, 0.15, 0.2])
        covariance = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.01], [0.0, 0.01, 0.16]])
        points = [FrontierPoint(weights=np.array([0.3, 0.4, 0.3]))]
        frontier = Frontier(mean=mean, covariance=covariance, frontier=points)

        max_sr, max_weights = frontier.max_sharpe
        # With only one point, should return that point
        assert np.allclose(max_weights, points[0].weights, atol=1e-4)

    def test_max_sharpe_edge_case_two_points(self):
        """Test max Sharpe with exactly two points on frontier."""
        mean = np.array([0.1, 0.15, 0.2])
        covariance = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.01], [0.0, 0.01, 0.16]])
        points = [
            FrontierPoint(weights=np.array([0.2, 0.3, 0.5])),
            FrontierPoint(weights=np.array([0.5, 0.3, 0.2])),
        ]
        frontier = Frontier(mean=mean, covariance=covariance, frontier=points)

        max_sr, max_weights = frontier.max_sharpe
        assert isinstance(max_sr, float)
        assert np.isclose(np.sum(max_weights), 1.0)

    def test_plot_with_variance(self):
        """Test plotting with variance on x-axis."""
        mean = np.array([0.1, 0.15, 0.2])
        covariance = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.01], [0.0, 0.01, 0.16]])
        points = [
            FrontierPoint(weights=np.array([0.2, 0.3, 0.5])),
            FrontierPoint(weights=np.array([0.3, 0.4, 0.3])),
        ]
        frontier = Frontier(mean=mean, covariance=covariance, frontier=points)

        fig = frontier.plot(volatility=False, markers=True)
        assert fig is not None
        # Check that the figure has the expected properties
        assert len(fig.data) > 0

    def test_plot_with_volatility(self):
        """Test plotting with volatility on x-axis."""
        mean = np.array([0.1, 0.15, 0.2])
        covariance = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.01], [0.0, 0.01, 0.16]])
        points = [
            FrontierPoint(weights=np.array([0.2, 0.3, 0.5])),
            FrontierPoint(weights=np.array([0.3, 0.4, 0.3])),
        ]
        frontier = Frontier(mean=mean, covariance=covariance, frontier=points)

        fig = frontier.plot(volatility=True, markers=False)
        assert fig is not None
        assert len(fig.data) > 0

    def test_max_sharpe_not_at_last_point(self):
        """Test max Sharpe when the maximum is not at the last point.

        This test ensures that the right-side optimization branch is covered.
        """
        # Create a frontier where max Sharpe is at the first point
        # Use high return, low variance for first point
        mean = np.array([0.3, 0.15, 0.1])  # First asset has highest return
        # Low variance for first asset
        covariance = np.array([[0.01, 0.001, 0.0], [0.001, 0.09, 0.01], [0.0, 0.01, 0.16]])

        # Create points with first having most of high-return, low-variance asset
        points = [
            FrontierPoint(weights=np.array([0.6, 0.2, 0.2])),  # Should have highest Sharpe
            FrontierPoint(weights=np.array([0.3, 0.4, 0.3])),
            FrontierPoint(weights=np.array([0.2, 0.3, 0.5])),  # Low Sharpe
        ]
        frontier = Frontier(mean=mean, covariance=covariance, frontier=points)

        # Verify max Sharpe is at first point (index 0)
        sr_position = np.argmax(frontier.sharpe_ratio)
        assert sr_position == 0, f"Expected max Sharpe at position 0, got {sr_position}"

        # Call max_sharpe - this should exercise the right > sr_position_max branch
        max_sr, max_weights = frontier.max_sharpe
        assert isinstance(max_sr, float)
        assert np.isclose(np.sum(max_weights), 1.0)
        # Max Sharpe should be at least as good as the discrete maximum
        assert max_sr >= np.max(frontier.sharpe_ratio) - 1e-6

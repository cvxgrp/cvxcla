"""Tests for the types module.

This module tests the core data structures used in the Critical Line Algorithm:
- FrontierPoint: Represents a point on the efficient frontier
- TurningPoint: Represents a turning point on the efficient frontier
- Frontier: Represents the entire efficient frontier
"""

import itertools

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
        """Test that invalid weights (not summing to 1) raise a ValueError."""
        weights = np.array([0.3, 0.3, 0.3])  # Sum = 0.9, not 1.0
        with pytest.raises(ValueError, match=r"^Weights do not sum to 1$"):
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

        _max_sr, max_weights = frontier.max_sharpe
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
        pytest.importorskip("plotly")
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
        pytest.importorskip("plotly")
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


class TestFrontierMutationHardening:
    """Targeted tests pinning behaviour that mutation testing flagged as unguarded."""

    @pytest.fixture
    def mean_cov(self):
        """A well-conditioned 3-asset mean/covariance pair."""
        mean = np.array([0.10, 0.15, 0.20])
        covariance = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.01], [0.0, 0.01, 0.16]])
        return mean, covariance

    def test_frontier_is_frozen(self, mean_cov):
        """Frontier is an immutable (frozen) dataclass."""
        import dataclasses

        mean, covariance = mean_cov
        frontier = Frontier(mean=mean, covariance=covariance, frontier=[])
        with pytest.raises(dataclasses.FrozenInstanceError):
            frontier.mean = np.zeros(3)

    def test_frontier_defaults_to_empty_list(self, mean_cov):
        """Omitting ``frontier`` yields an empty, iterable list (not None)."""
        mean, covariance = mean_cov
        frontier = Frontier(mean=mean, covariance=covariance)
        assert len(frontier) == 0
        assert list(frontier) == []

    def test_interpolate_point_count_two_points(self, mean_cov):
        """interpolate() yields exactly num-1 points per segment (one segment here)."""
        mean, covariance = mean_cov
        points = [
            FrontierPoint(weights=np.array([0.2, 0.3, 0.5])),
            FrontierPoint(weights=np.array([0.5, 0.3, 0.2])),
        ]
        frontier = Frontier(mean=mean, covariance=covariance, frontier=points)
        # linspace(0, 1, num) drops lamb == 0, so num-1 points per adjacent pair.
        assert len(frontier.interpolate()) == 99  # default num=100
        assert len(frontier.interpolate(num=10)) == 9

    def test_interpolate_point_count_three_points(self, mean_cov):
        """Three points -> two adjacent segments -> 2*(num-1) interpolated points.

        Pins the adjacent-pair slicing ``zip(weights[0:-1], weights[1:])``.
        """
        mean, covariance = mean_cov
        points = [
            FrontierPoint(weights=np.array([0.2, 0.3, 0.5])),
            FrontierPoint(weights=np.array([0.3, 0.4, 0.3])),
            FrontierPoint(weights=np.array([0.5, 0.3, 0.2])),
        ]
        frontier = Frontier(mean=mean, covariance=covariance, frontier=points)
        assert len(frontier.interpolate(num=100)) == 198

    def test_interpolated_weights_are_convex_combinations(self, mean_cov):
        """Every interpolated weight lies between the two endpoint weights.

        Pins lamb in (0, 1]: a wider lambda range (e.g. [0, 2]) would push
        weights outside the convex hull of the two adjacent points.
        """
        mean, covariance = mean_cov
        w0 = np.array([0.2, 0.3, 0.5])
        w1 = np.array([0.5, 0.3, 0.2])
        frontier = Frontier(
            mean=mean,
            covariance=covariance,
            frontier=[FrontierPoint(weights=w0), FrontierPoint(weights=w1)],
        )
        lo = np.minimum(w0, w1)
        hi = np.maximum(w0, w1)
        for point in frontier.interpolate(num=50):
            assert np.all(point.weights >= lo - 1e-9)
            assert np.all(point.weights <= hi + 1e-9)

    def test_max_sharpe_ratio_matches_weights(self, mean_cov):
        """The returned Sharpe ratio equals the true Sharpe of the returned weights.

        This pins the entire ``neg_sharpe`` objective (the convex-combination
        formula and the -returns/sqrt(var) expression): a corrupted objective
        makes the reported ratio inconsistent with the weights it returns.
        """
        mean, covariance = mean_cov
        points = [
            FrontierPoint(weights=np.array([0.2, 0.3, 0.5])),
            FrontierPoint(weights=np.array([0.3, 0.4, 0.3])),
            FrontierPoint(weights=np.array([0.5, 0.3, 0.2])),
        ]
        frontier = Frontier(mean=mean, covariance=covariance, frontier=points)

        max_sr, w = frontier.max_sharpe
        true_sr = float(mean @ w) / np.sqrt(float(w @ covariance @ w))
        assert np.isclose(max_sr, true_sr, atol=1e-8)
        # And it must be a genuine maximum: no fine-grained convex combination of
        # adjacent points beats it.
        best = max(np.max(frontier.sharpe_ratio), max_sr)
        for wl, wr in zip(frontier.weights[:-1], frontier.weights[1:], strict=False):
            for t in np.linspace(0.0, 1.0, 50):
                wt = t * wl + (1 - t) * wr
                sr = float(mean @ wt) / np.sqrt(float(wt @ covariance @ wt))
                best = max(best, sr)
        assert max_sr >= best - 1e-4


class TestMaxSharpeNeighbourSelection:
    """max_sharpe must search the segments adjacent to the best discrete point.

    Each frontier places the true (continuous) maximum-Sharpe portfolio inside a
    specific segment next to the discrete argmax. If the neighbour-index
    arithmetic is perturbed (wrong side, wrong clamp, off-by-one), the optimiser
    searches the wrong segment and returns a strictly worse ratio — so asserting
    that max_sharpe attains the global optimum pins that arithmetic.
    """

    @staticmethod
    def _global_max_sharpe(frontier, mean, covariance, n=400):
        """Brute-force the best Sharpe ratio over every adjacent segment."""
        weights = frontier.weights
        best = -np.inf
        for w_a, w_b in itertools.pairwise(weights):
            for t in np.linspace(0.0, 1.0, n):
                w = t * w_a + (1 - t) * w_b
                sr = float(mean @ w) / np.sqrt(float(w @ covariance @ w))
                best = max(best, sr)
        return best

    @pytest.mark.parametrize(
        ("mean", "diag", "weights"),
        [
            # argmax=2, len=5, optimum in segment (2, 3): pins the right neighbour
            # (right = sr+1, not sr-1 / sr+2).
            (
                [0.05, 0.20, 0.21, 0.10, 0.08],
                [0.02, 0.05, 0.20, 0.10, 0.30],
                [[1, 0, 0, 0, 0], [0, 0.5, 0.5, 0, 0], [0, 0.7, 0.3, 0, 0], [0, 0.2, 0, 0.8, 0], [0, 0, 0, 0, 1]],
            ),
            # argmax=2=len-2, len=4, optimum in the final segment: pins the right
            # clamp (len-1, not len-2).
            (
                [0.05, 0.20, 0.21, 0.10],
                [0.02, 0.05, 0.20, 0.10],
                [[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.7, 0.3, 0], [0, 0.15, 0, 0.85]],
            ),
            # argmax=1, optimum in segment (0, 1): pins the left neighbour
            # (left = max(0, sr-1), and the sr-1 offset).
            (
                [0.10, 0.13, 0.04, 0.03],
                [0.04, 0.05, 0.25, 0.40],
                [[0.85, 0.15, 0, 0], [0.35, 0.65, 0, 0], [0, 0.4, 0.6, 0], [0, 0, 0, 1]],
            ),
        ],
    )
    def test_max_sharpe_attains_global_optimum(self, mean, diag, weights):
        """max_sharpe equals the brute-force global optimum over adjacent segments."""
        mean = np.array(mean)
        covariance = np.diag(diag)
        frontier = Frontier(
            mean=mean,
            covariance=covariance,
            frontier=[FrontierPoint(weights=np.array(w, dtype=float)) for w in weights],
        )
        max_sr, _ = frontier.max_sharpe
        expected = self._global_max_sharpe(frontier, mean, covariance)
        assert np.isclose(max_sr, expected, atol=1e-3)

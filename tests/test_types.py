from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from cvx.cla.types import Frontier, FrontierPoint, TurningPoint


@pytest.fixture()
def fp() -> FrontierPoint:
    """
    Fixture that creates a FrontierPoint instance for testing.

    Returns:
        A FrontierPoint instance with predefined weights
    """
    return FrontierPoint(weights=np.array([0.3, 0.7]))


@pytest.fixture()
def tp() -> TurningPoint:
    """
    Fixture that creates a TurningPoint instance for testing.

    Returns:
        A TurningPoint instance with predefined weights and free variables
    """
    return TurningPoint(weights=np.array([0.5, 0.5]), free=np.array([True, False]))


def test_turningpoint(tp: TurningPoint) -> None:
    """
    Test the basic properties of a TurningPoint.

    Verifies that the lambda value is infinity by default and that
    the weights and free variables are correctly set.

    Args:
        tp: The TurningPoint instance to test
    """
    assert np.isinf(tp.lamb)
    assert np.allclose(tp.weights, [0.5, 0.5])
    assert np.allclose(tp.free, [True, False])


def test_indices(tp: TurningPoint) -> None:
    """
    Test the free_indices and blocked_indices properties of a TurningPoint.

    Verifies that the indices of free and blocked variables are correctly identified.

    Args:
        tp: The TurningPoint instance to test
    """
    assert np.allclose(tp.free_indices, [0])
    assert np.allclose(tp.blocked_indices, [1])


def test_mean(tp: TurningPoint) -> None:
    """
    Test the mean method of a TurningPoint.

    Verifies that the expected return is correctly calculated as the dot product
    of the weights and the mean vector.

    Args:
        tp: The TurningPoint instance to test
    """
    x = tp.mean(mean=np.array([1.0, 2.0]))
    assert x == pytest.approx(1.5)


def test_variance(tp: TurningPoint) -> None:
    """
    Test the variance method of a TurningPoint.

    Verifies that the expected variance is correctly calculated using
    the quadratic form with the covariance matrix.

    Args:
        tp: The TurningPoint instance to test
    """
    x = tp.variance(covariance=np.array([[2.0, 0.2], [0.2, 2.0]]))
    assert x == pytest.approx(1.1)


def test_frontierpoint_post_init() -> None:
    """
    Test the __post_init__ method of FrontierPoint.

    Verifies that the assertion for weights summing to 1 works correctly.
    """
    # Should work with weights that sum to 1
    fp = FrontierPoint(weights=np.array([0.3, 0.7]))
    assert np.allclose(fp.weights, [0.3, 0.7])

    # Should work with weights that are very close to 1
    fp = FrontierPoint(weights=np.array([0.3, 0.7 + 1e-10]))
    assert np.allclose(fp.weights, [0.3, 0.7 + 1e-10])

    # Should raise an assertion error if weights don't sum to 1
    with pytest.raises(AssertionError):
        FrontierPoint(weights=np.array([0.3, 0.6]))


def test_frontierpoint_mean(fp: FrontierPoint) -> None:
    """
    Test the mean method of FrontierPoint.

    Verifies that the expected return is correctly calculated as the dot product
    of the weights and the mean vector.

    Args:
        fp: The FrontierPoint instance to test
    """
    x = fp.mean(mean=np.array([1.0, 2.0]))
    assert x == pytest.approx(1.0 * 0.3 + 2.0 * 0.7)


def test_frontierpoint_variance(fp: FrontierPoint) -> None:
    """
    Test the variance method of FrontierPoint.

    Verifies that the expected variance is correctly calculated using
    the quadratic form with the covariance matrix.

    Args:
        fp: The FrontierPoint instance to test
    """
    x = fp.variance(covariance=np.array([[2.0, 0.2], [0.2, 2.0]]))
    expected = 0.3 * 0.3 * 2.0 + 0.3 * 0.7 * 0.2 + 0.7 * 0.3 * 0.2 + 0.7 * 0.7 * 2.0
    assert x == pytest.approx(expected)


@pytest.fixture()
def frontier() -> Frontier:
    """
    Fixture that creates a Frontier instance for testing.

    Creates a Frontier with mean returns, covariance matrix, and a list of
    FrontierPoint instances.

    Returns:
        A Frontier instance with predefined data
    """
    mean = np.array([0.1, 0.2, 0.3])
    covariance = np.array([[0.2, 0.05, 0.01], [0.05, 0.2, 0.05], [0.01, 0.05, 0.2]])

    # Create a list of frontier points
    points = [
        FrontierPoint(weights=np.array([0.5, 0.3, 0.2])),
        FrontierPoint(weights=np.array([0.4, 0.4, 0.2])),
        FrontierPoint(weights=np.array([0.3, 0.4, 0.3])),
        FrontierPoint(weights=np.array([0.2, 0.3, 0.5])),
    ]

    return Frontier(mean=mean, covariance=covariance, frontier=points)


def test_frontier_init(frontier: Frontier) -> None:
    """
    Test the initialization of the Frontier class.

    Verifies that the mean, covariance, and frontier points are correctly set.

    Args:
        frontier: The Frontier instance to test
    """
    assert frontier.mean.shape == (3,)
    assert frontier.covariance.shape == (3, 3)
    assert len(frontier.frontier) == 4


def test_frontier_interpolate(frontier: Frontier) -> None:
    """
    Test the interpolate method of the Frontier class.

    Verifies that the interpolate method creates the correct number of points
    and that the interpolated points have weights that sum to 1.

    Args:
        frontier: The Frontier instance to test
    """
    # Interpolate with 3 points between each pair
    interpolated = frontier.interpolate(num=3)

    # Should have (n-1) * (num-1) + n points
    # Original had 4 points, so we should have 3 * 2 + 4 = 10 points
    assert len(interpolated.frontier) == 6

    # Check that all weights sum to 1
    for point in interpolated.frontier:
        assert np.isclose(np.sum(point.weights), 1.0)

    # Check that mean and covariance are preserved
    assert np.array_equal(interpolated.mean, frontier.mean)
    assert np.array_equal(interpolated.covariance, frontier.covariance)


def test_frontier_iter(frontier: Frontier) -> None:
    """
    Test the __iter__ method of the Frontier class.

    Verifies that the Frontier class can be iterated over and yields
    the correct frontier points.

    Args:
        frontier: The Frontier instance to test
    """
    points = list(frontier)
    assert len(points) == 4
    assert all(isinstance(point, FrontierPoint) for point in points)
    assert np.allclose(points[0].weights, [0.5, 0.3, 0.2])
    assert np.allclose(points[-1].weights, [0.2, 0.3, 0.5])


def test_frontier_len(frontier: Frontier) -> None:
    """
    Test the __len__ method of the Frontier class.

    Verifies that the len() function returns the correct number of frontier points.

    Args:
        frontier: The Frontier instance to test
    """
    assert len(frontier) == 4


def test_frontier_weights(frontier: Frontier) -> None:
    """
    Test the weights property of the Frontier class.

    Verifies that the weights property returns a matrix with one row per point.

    Args:
        frontier: The Frontier instance to test
    """
    weights = frontier.weights
    assert weights.shape == (4, 3)
    assert np.allclose(weights[0], [0.5, 0.3, 0.2])
    assert np.allclose(weights[-1], [0.2, 0.3, 0.5])


def test_frontier_returns(frontier: Frontier) -> None:
    """
    Test the returns property of the Frontier class.

    Verifies that the returns property correctly calculates the expected return
    for each frontier point.

    Args:
        frontier: The Frontier instance to test
    """
    returns = frontier.returns
    assert returns.shape == (4,)

    # Calculate expected returns manually for verification
    expected_returns = np.array(
        [
            0.5 * 0.1 + 0.3 * 0.2 + 0.2 * 0.3,
            0.4 * 0.1 + 0.4 * 0.2 + 0.2 * 0.3,
            0.3 * 0.1 + 0.4 * 0.2 + 0.3 * 0.3,
            0.2 * 0.1 + 0.3 * 0.2 + 0.5 * 0.3,
        ]
    )

    assert np.allclose(returns, expected_returns)


def test_frontier_variance(frontier: Frontier) -> None:
    """
    Test the variance property of the Frontier class.

    Verifies that the variance property correctly calculates the expected variance
    for each frontier point.

    Args:
        frontier: The Frontier instance to test
    """
    variances = frontier.variance
    assert variances.shape == (4,)

    # Calculate expected variances manually for the first point
    w = frontier.weights[0]
    expected_variance = w.T @ frontier.covariance @ w

    assert np.isclose(variances[0], expected_variance)


def test_frontier_volatility(frontier: Frontier) -> None:
    """
    Test the volatility property of the Frontier class.

    Verifies that the volatility property correctly calculates the square root
    of the variance for each frontier point.

    Args:
        frontier: The Frontier instance to test
    """
    volatilities = frontier.volatility
    variances = frontier.variance

    assert volatilities.shape == (4,)
    assert np.allclose(volatilities, np.sqrt(variances))


def test_frontier_sharpe_ratio(frontier: Frontier) -> None:
    """
    Test the sharpe_ratio property of the Frontier class.

    Verifies that the sharpe_ratio property correctly calculates the ratio of
    returns to volatility for each frontier point.

    Args:
        frontier: The Frontier instance to test
    """
    sharpe_ratios = frontier.sharpe_ratio
    returns = frontier.returns
    volatilities = frontier.volatility

    assert sharpe_ratios.shape == (4,)
    assert np.allclose(sharpe_ratios, returns / volatilities)


def test_frontier_max_sharpe(frontier: Frontier) -> None:
    """
    Test the max_sharpe property of the Frontier class.

    Verifies that the max_sharpe property correctly identifies the point with
    the maximum Sharpe ratio and returns the correct Sharpe ratio and weights.

    Args:
        frontier: The Frontier instance to test
    """
    max_sharpe, max_weights = frontier.max_sharpe

    # Verify that max_sharpe is a float and max_weights is an array
    assert isinstance(max_sharpe, float)
    assert isinstance(max_weights, np.ndarray)

    # Verify that the weights sum to 1
    assert np.isclose(np.sum(max_weights), 1.0)

    # Verify that the max_sharpe is greater than or equal to all other Sharpe ratios
    assert max_sharpe >= np.max(frontier.sharpe_ratio)


def test_frontier_plot(frontier: Frontier) -> None:
    """
    Test the plot method of the Frontier class.

    Verifies that the plot method returns a plotly figure object with the
    correct data.

    Args:
        frontier: The Frontier instance to test
    """
    # Test with volatility=False (default)
    fig = frontier.plot()
    assert isinstance(fig, go.Figure)

    # Verify that the figure has one trace
    assert len(fig.data) == 1

    # Verify that the x-axis data matches the variance
    assert np.allclose(fig.data[0].x, frontier.variance)

    # Verify that the y-axis data matches the returns
    assert np.allclose(fig.data[0].y, frontier.returns)

    # Test with volatility=True
    fig = frontier.plot(volatility=True)
    assert isinstance(fig, go.Figure)

    # Verify that the x-axis data matches the volatility
    assert np.allclose(fig.data[0].x, frontier.volatility)

    # Verify that the y-axis data matches the returns
    assert np.allclose(fig.data[0].y, frontier.returns)

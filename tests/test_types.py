from __future__ import annotations

import numpy as np
import pytest

from cvx.cla.types import TurningPoint


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

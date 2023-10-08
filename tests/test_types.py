import numpy as np
import pytest

from cvx.cla.types import TurningPoint


@pytest.fixture()
def tp():
    """Fixture for turning point"""
    return TurningPoint(weights=np.array([0.5, 0.5]), free=np.array([True, False]))


def test_turningpoint(tp):
    """Test turning point"""
    assert np.isinf(tp.lamb)
    assert np.allclose(tp.weights, [0.5, 0.5])
    assert np.allclose(tp.free, [True, False])


def test_indices(tp):
    """Test indices. Both free and blocked"""
    assert np.allclose(tp.free_indices, [0])
    assert np.allclose(tp.blocked_indices, [1])


def test_mean(tp):
    """Test mean"""
    x = tp.mean(mean=np.array([1.0, 2.0]))
    assert x == pytest.approx(1.5)


def test_variance(tp):
    """Test variance"""
    x = tp.variance(covariance=np.array([[2.0, 0.2], [0.2, 2.0]]))
    assert x == pytest.approx(1.1)

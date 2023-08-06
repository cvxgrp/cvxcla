import numpy as np
import pytest

from cvx.cla.first import init_algo
from cvx.cla.types import TurningPoint


@pytest.mark.parametrize("n", [5, 20, 100, 1000, 10000, 100000])
def test_init_algo(n):
    mean = np.random.randn(n)
    lower_bounds = np.zeros(n)
    upper_bounds = np.random.rand(n)

    first = init_algo(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

    assert np.sum(first.free) == 1
    assert np.sum(first.weights) == pytest.approx(1.0)
    assert isinstance(first, TurningPoint)

def test_small():
    mean = np.array([1.0, 2.0, 3.0])
    lower_bound = np.array([0.0, 0.0, 0.0])
    upper_bound = np.array([0.4, 0.4, 0.4])
    tp = init_algo(mean=mean, lower_bounds=lower_bound, upper_bounds=upper_bound)

    assert np.allclose(tp.weights, [0.2, 0.4, 0.4])
    assert tp.lamb == np.inf
    assert np.allclose(tp.free, [True, False, False])

@pytest.mark.parametrize("n", [5, 20, 100, 1000, 10000, 100000])
def test_sorting(n):
    mean = np.random.randn(n)

    order1 = np.argsort(mean)[::-1]
    order2 = np.argsort(-mean)

    np.testing.assert_array_equal(order1, order2)
    np.testing.assert_array_equal(np.sort(mean)[::-1], mean[order2])

def test_no_free_asset():
    mean = np.array([1.0, 2.0, 3.0])
    lower_bounds = np.array([0.0, 0.0, 0.0])
    upper_bounds = np.array([0.2, 0.2, 0.2])

    with pytest.raises(
        ValueError, match="Could not construct a fully invested portfolio"
    ):
        init_algo(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)


def test_order():
    mean = np.array([1.0, 1.0, 1.0])
    order = np.argsort(mean)[::-1]
    assert np.alltrue(np.sort(mean)[::-1] == mean[order])


def test_lb_ub_mixed():
    upper_bounds = np.zeros(3)
    lower_bounds = np.ones(3)
    mean = np.ones(3)

    with pytest.raises(ValueError):
        init_algo(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

def test_no_fully_invested():
    upper_bounds = 0.2 * np.ones(3)
    lower_bounds = np.zeros(3)
    mean = np.ones(3)

    with pytest.raises(ValueError):
        init_algo(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

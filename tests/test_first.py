import numpy as np
import pytest

from cvx.cla.first import init_algo


@pytest.mark.parametrize("n", [5, 20, 100, 1000])
def test_init_algo(n):
    mean = np.random.randn(n)
    lB = np.zeros(n)
    uB = np.random.rand(n)

    first = init_algo(mean=mean, lower_bounds=lB, upper_bounds=uB)

    assert np.sum(first.free) == 1
    assert np.sum(first.weights) == pytest.approx(1.0)

def test_small():
    mean = np.array([1.0, 2.0, 3.0])
    lower_bound = np.array([0.0, 0.0, 0.0])
    upper_bound = np.array([0.4, 0.4, 0.4])
    next = init_algo(mean=mean, lower_bounds=lower_bound, upper_bounds=upper_bound)

    assert np.allclose(next.weights, [0.2, 0.4, 0.4])
    assert next.lamb == np.inf
    assert np.allclose(next.free, [True, False, False])


@pytest.mark.parametrize("n", [5, 20, 100, 1000])
def test_init_algo(n):
    mean = np.ones(n)
    lower_bounds=np.zeros(n) #array([0.0, 0.0, 0.0])
    upper_bounds=np.ones(n) #array([1.0, 1.0, 1.0])
    tp = init_algo(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

    b = np.zeros(n)
    b[0] = 1.0
    assert np.allclose(tp.weights, b)
    assert tp.lamb == np.inf

    b = np.full_like(mean, False)
    b[0] = True
    assert np.allclose(tp.free, b)


def test_no_free_asset():
    mean = np.array([1.0, 2.0, 3.0])
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([0.2, 0.2, 0.2])

    with pytest.raises(
        ValueError, match="Could not construct a fully invested portfolio"
    ):
        init_algo(mean=mean, lower_bounds=lb, upper_bounds=ub)


def test_order():
    mean = np.array([1.0, 1.0, 1.0])
    order = np.argsort(mean)[::-1]
    assert np.alltrue(np.sort(mean)[::-1] == mean[order])


def test_lb_ub_mixed():
    uB = np.zeros(3)
    lB = np.ones(3)
    mean = np.ones(3)

    with pytest.raises(ValueError):
        init_algo(mean=mean, lower_bounds=lB, upper_bounds=uB)

def test_no_fully_invested():
    uB = 0.2 * np.ones(3)
    lB = np.zeros(3)
    mean = np.ones(3)

    with pytest.raises(ValueError):
        init_algo(mean=mean, lower_bounds=lB, upper_bounds=uB)

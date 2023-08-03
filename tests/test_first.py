import numpy as np
import pytest

from cvx.cla.first import init_algo


def test_init_algo():
    mean = np.array([1.0, 2.0, 3.0])
    lower_bound = np.array([0.0, 0.0, 0.0])
    upper_bound = np.array([0.4, 0.4, 0.4])
    next = init_algo(mean=mean, lower_bounds=lower_bound, upper_bounds=upper_bound)

    assert np.allclose(next.weights, [0.2, 0.4, 0.4])
    assert next.lamb == np.inf
    assert np.allclose(next.free, [True, False, False])
    assert np.allclose(next.mean, 2.2)


def test_init_algo_border():
    mean = np.array([1.0, 1.0, 1.0])
    tp = init_algo(mean=mean)

    assert np.allclose(tp.weights, [0.0, 0.0, 1.0])
    assert tp.lamb == np.inf
    assert np.allclose(tp.free, [False, False, True])
    assert np.allclose(tp.mean, 1.0)


def test_no_free_asset():
    mean = np.array([1.0, 2.0, 3.0])
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([0.2, 0.2, 0.2])

    with pytest.raises(
        ValueError, match="Could not construct a fully invested portfolio"
    ):
        init_algo(mean=mean, lower_bounds=lb, upper_bounds=ub)

    # assert np.allclose(tp.weights, [0.2, 0.2, 0.2])
    # assert tp.lamb == np.inf
    # assert np.allclose(tp.free, [False, False, False])
    # assert np.allclose(tp.mean, 1.2)


def test_order():
    mean = np.array([1.0, 1.0, 1.0])
    order = np.argsort(mean)[::-1]
    assert np.alltrue(np.sort(mean)[::-1] == mean[order])

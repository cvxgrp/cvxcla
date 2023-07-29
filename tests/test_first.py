import numpy as np

from cvx.cla.first import init_algo


def test_init_algo():
    mean = np.array([1.0, 2.0, 3.0])
    lower_bound = np.array([0.0, 0.0, 0.0])
    upper_bound = np.array([0.4, 0.4, 0.4])
    next = init_algo(mean=mean, lower_bounds=lower_bound, upper_bounds=upper_bound)

    assert np.allclose(next.weights, [0.2, 0.4, 0.4])
    assert not next.lamb
    assert np.allclose(next.free, [True, False, False])

import numpy as np
import pytest

from cvx.cla.first import init_algo
from cvx.cla.markowitz.cla import CLA


def test_solver(input_data, results):
    cla = CLA(mean=input_data.mean,
        lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds,
        covariance=input_data.covariance, tol=1e-5)

    observed = np.array([tp.lamb for tp in cla.turning_points[1:]])
    assert np.allclose(results.lamb, observed, atol=1e-2)

    observed = np.array([tp.mean(input_data.mean) for tp in cla.turning_points[1:]])
    assert np.allclose(results.mean, observed, atol=1e-2)

    observed = np.array([tp.variance(input_data.covariance) for tp in cla.turning_points[1:]])
    assert np.allclose(results.variance, observed, atol=0.5)

def test_example(example, example_solution):
    # example from section 3.1 in the Markowitz 2019 paper
    means = example.mean(axis=0)
    std = example.std(axis=0, ddof=1)
    assert np.allclose(means.values, np.array([0.062, 0.146, 0.128]), atol=1e-3)
    assert np.allclose(np.power(std.values,2), np.array([0.016, 0.091, 0.031]), atol=1e-3)

    ns = example.shape[1]

    lower_bounds = 0.1 * np.ones(ns)
    upper_bounds = 0.5 * np.ones(ns)

    #covariance = example.cov().values
    tp = init_algo(mean=means.values, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    assert np.allclose(tp.weights, np.array([0.1, 0.5, 0.4]), atol=1e-9)
    assert np.allclose(tp.free, np.array([False, False, True]))

    cla = CLA(mean=means.values, lower_bounds=lower_bounds, upper_bounds=upper_bounds, covariance=example.cov().values)

    for row, turning_point in enumerate(cla.turning_points):
        if row > 0:
            assert turning_point.lamb == pytest.approx(example_solution["lambda"][row], abs=1e-3)

        assert np.allclose(turning_point.weights, example_solution.values[row,1:], atol=1e-3)

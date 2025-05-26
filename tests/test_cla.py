from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pandas import DataFrame

from cvx.cla import CLA
from cvx.cla.first import init_algo_lp


def test_solver(input_data: Any, results: Any) -> None:
    """
    Test the Markowitz CLA solver against expected results.

    This test verifies that the Markowitz CLA implementation produces the expected
    lambda values, mean returns, and variances for the turning points.

    Args:
        input_data: Test data containing covariance, mean, and bounds
        results: Expected results for comparison
    """
    cla = CLA(
        mean=input_data.mean,
        lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds,
        covariance=input_data.covariance,
        tol=1e-5,
        A=np.array([np.ones_like(input_data.mean)]),
        b=np.ones(1),
    )

    observed = np.array([tp.lamb for tp in cla.turning_points[1:]])
    assert np.allclose(results.lamb, observed, atol=1e-2)

    observed = np.array([tp.mean(input_data.mean) for tp in cla.turning_points[1:]])
    assert np.allclose(results.mean, observed, atol=1e-2)

    observed = np.array([tp.variance(input_data.covariance) for tp in cla.turning_points[1:]])
    assert np.allclose(results.variance, observed, atol=0.5)


def test_example(example: DataFrame, example_solution: DataFrame) -> None:
    """
    Test the Markowitz CLA solver against a known example from literature.

    This test uses the example from section 3.1 in the Markowitz 2019 paper
    to verify that the Markowitz CLA implementation produces the expected results.
    It also tests the init_algo_lp function for computing the first turning point.

    Args:
        example: DataFrame containing the example data
        example_solution: DataFrame containing the expected solution
    """
    # example from section 3.1 in the Markowitz 2019 paper
    means = example.mean(axis=0)
    std = example.std(axis=0, ddof=1)
    assert np.allclose(means.values, np.array([0.062, 0.146, 0.128]), atol=1e-3)
    assert np.allclose(np.power(std.values, 2), np.array([0.016, 0.091, 0.031]), atol=1e-3)

    ns = example.shape[1]

    lower_bounds = 0.1 * np.ones(ns)
    upper_bounds = 0.5 * np.ones(ns)

    # covariance = example.cov().values
    tp = init_algo_lp(mean=means.values, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    assert np.allclose(tp.weights, np.array([0.1, 0.5, 0.4]), atol=1e-9)
    assert np.allclose(tp.free, np.array([False, False, True]))

    cla = CLA(
        mean=means.values,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        covariance=example.cov().values,
        A=np.ones((1, len(means))),
        b=np.ones(1),
    )

    assert len(cla) == 8

    for row, turning_point in enumerate(cla.turning_points):
        if row > 0:
            assert turning_point.lamb == pytest.approx(example_solution["lambda"][row], abs=1e-3)

        assert np.allclose(turning_point.weights, example_solution.values[row, 1:], atol=1e-3)

from __future__ import annotations

import numpy as np
import pytest
from cvx.cla import Frontier

np.random.seed(42)

def test_frontier(input_data):
    f = Frontier.construct(
        mean=input_data.mean, lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds, covariance=input_data.covariance, name="test"
    )

    np.testing.assert_equal(f.covariance, input_data.covariance)
    assert len(f) == 11
    np.testing.assert_almost_equal(f.max_sharpe[0], 4.4535334766464025)

    np.testing.assert_almost_equal(f.mean, input_data.mean)
    np.testing.assert_almost_equal(
        f.returns,
        np.array(
            [
                1.19,
                1.19,
                1.1802595,
                1.1600565,
                1.1112623,
                1.1083602,
                1.0224839,
                1.0153059,
                0.9727204,
                0.9499368,
                0.8032154,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        f.variance,
        np.array(
            [
                0.9063047,
                0.9063047,
                0.2977414,
                0.1741023,
                0.0711394,
                0.070234,
                0.0527529,
                0.0519761,
                0.0482043,
                0.0466666,
                0.0421225,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        f.volatility,
        np.array(
            [
                0.9520004,
                0.9520004,
                0.5456569,
                0.4172557,
                0.2667196,
                0.265017,
                0.2296801,
                0.2279827,
                0.2195549,
                0.2160246,
                0.2052376,
            ]
        ),
    )

    f.interpolate(num=10)

@pytest.mark.parametrize("n", [5, 5, 5, 5, 10, 20, 20, 20, 20, 20, 20])
def test_frontiers(n):
    mean = np.random.randn(n)
    lower_bounds = np.zeros(n)
    upper_bounds = np.ones(n)

    cov = np.random.randn(n, n)

    covar = cov @ cov.T

    f = Frontier.construct(
        mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds, covariance=covar, name="test"
    )

    assert np.sum(f.frontier[-1].weights) == pytest.approx(1)

    print(f.max_sharpe[0])

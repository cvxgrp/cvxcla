from __future__ import annotations

import numpy as np
import pytest

from cvx.cla.markowitz.cla import CLA as MARKOWITZ
from tests.bailey.cla import CLA as BAILEY

# np.random.seed(40)


@pytest.mark.parametrize("solver", [BAILEY, MARKOWITZ])
def test_frontier(input_data, solver):
    """
    Test the frontier both for Bailey and Markowitz
    """
    f = solver(
        covariance=input_data.covariance,
        mean=input_data.mean,
        lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds,
        A=np.ones((1, len(input_data.mean))),
        b=np.ones(1),
    ).frontier

    np.testing.assert_equal(f.covariance, input_data.covariance)
    np.testing.assert_almost_equal(f.max_sharpe[0], 4.4535334766464025, decimal=5)
    np.testing.assert_almost_equal(f.mean, input_data.mean)

    g = f.interpolate(num=10)
    np.testing.assert_equal(g.covariance, input_data.covariance)
    np.testing.assert_almost_equal(g.max_sharpe[0], 4.4535334766464025, decimal=5)
    np.testing.assert_almost_equal(g.mean, input_data.mean)


@pytest.mark.parametrize("n", [3, 5, 10, 20])
def test_frontiers(n, resource_dir):
    """
    Compare the frontiers of BAILEY and MARKOWITZ for a variety of dimensions.
    """
    mean = np.random.randn(n)
    lower_bounds = np.zeros(n)
    upper_bounds = np.ones(n)

    cov = np.random.randn(n, n)

    covar = cov @ cov.T

    f_bailey = BAILEY(
        covariance=covar,
        mean=mean,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        A=np.ones((1, n)),
        b=np.ones(1),
        tol=1e-5,
    ).frontier

    f_markowitz = MARKOWITZ(
        mean=np.copy(mean),
        lower_bounds=np.copy(lower_bounds),
        upper_bounds=np.copy(upper_bounds),
        covariance=np.copy(covar),
        tol=1e-5,
        A=np.ones((1, n)),
        b=np.ones(1),
    ).frontier

    assert np.sum(f_bailey.frontier[-1].weights) == pytest.approx(1)
    assert np.sum(f_markowitz.frontier[-1].weights) == pytest.approx(1)
    assert f_bailey.max_sharpe[0] == pytest.approx(f_markowitz.max_sharpe[0])

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cvx.cla import CLA as MARKOWITZ
from tests.bailey.cla import CLA as BAILEY

# np.random.seed(40)


@pytest.mark.parametrize("solver", [BAILEY, MARKOWITZ])
def test_frontier(input_data: Any, solver: type) -> None:
    """
    Test the frontier computation for both Bailey and Markowitz implementations.

    This test verifies that both implementations correctly compute the efficient frontier
    and that the interpolation of the frontier preserves its properties.

    Args:
        input_data: Test data containing covariance, mean, and bounds
        solver: The CLA implementation to test (either BAILEY or MARKOWITZ)
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
def test_frontiers(n: int, resource_dir: Path) -> None:
    """
    Compare the frontiers computed by Bailey and Markowitz implementations.

    This test creates random portfolio optimization problems of different sizes
    and verifies that both implementations produce consistent results.

    Args:
        n: The number of assets in the portfolio
        resource_dir: Path to the test resources directory
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

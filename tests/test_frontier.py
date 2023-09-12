from __future__ import annotations

import numpy as np
import pytest

from cvx.cla import Frontier
from cvx.cla.solver import Solver

np.random.seed(40)


@pytest.mark.parametrize("solver", [Solver.BAILEY, Solver.MARKOWITZ])
def test_frontier(input_data, solver):
    f = Frontier.build(
        solver=solver,
        mean=input_data.mean,
        lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds,
        covariance=input_data.covariance,
        name="test",
        A=np.ones((1, len(input_data.mean))),
        b=np.ones(1),
    )

    np.testing.assert_equal(f.covariance, input_data.covariance)
    np.testing.assert_almost_equal(f.max_sharpe[0], 4.4535334766464025)
    np.testing.assert_almost_equal(f.mean, input_data.mean)

    f.interpolate(num=10)


@pytest.mark.parametrize(
    "n", [2, 2, 2, 3, 3, 3, 5, 5, 5, 5, 10, 20, 20, 20, 20, 20, 20]
)
def test_frontiers(n, resource_dir):
    mean = np.random.randn(n)
    lower_bounds = np.zeros(n)
    upper_bounds = np.ones(n)

    cov = np.random.randn(n, n)

    covar = cov @ cov.T

    f_bailey = Frontier.build(
        solver=Solver.BAILEY,
        mean=np.copy(mean),
        lower_bounds=np.copy(lower_bounds),
        upper_bounds=np.copy(upper_bounds),
        covariance=np.copy(covar),
        name="Bailey",
        tol=1e-5,
        A=np.ones((1, n)),
        b=np.ones(1),
    )

    f_markowitz = Frontier.build(
        solver=Solver.MARKOWITZ,
        mean=np.copy(mean),
        lower_bounds=np.copy(lower_bounds),
        upper_bounds=np.copy(upper_bounds),
        covariance=np.copy(covar),
        name="Markowitz",
        tol=1e-5,
        A=np.ones((1, n)),
        b=np.ones(1),
    )

    assert np.sum(f_bailey.frontier[-1].weights) == pytest.approx(1)
    assert np.sum(f_markowitz.frontier[-1].weights) == pytest.approx(1)

    assert len(f_bailey.frontier) == len(f_markowitz.frontier)
    print(f_bailey.max_sharpe[0], f_markowitz.max_sharpe[0])
    for pt_bailey, pt_markowitz in zip(f_bailey.frontier, f_markowitz.frontier):
        assert np.allclose(pt_bailey.weights, pt_markowitz.weights, atol=1e-5)

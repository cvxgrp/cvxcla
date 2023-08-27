from __future__ import annotations

import numpy as np
import pytest

from cvx.bson.file import read_bson, write_bson
from cvx.cla import Frontier
from cvx.cla.plotting import plot_efficient_frontiers
from cvx.cla.solver import Solver

np.random.seed(42)


@pytest.mark.parametrize("solver", [Solver.BAILEY, Solver.MARKOWITZ])
def test_frontier(input_data, solver):
    f = Frontier.build(
        solver=solver,
        mean=input_data.mean,
        lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds,
        covariance=input_data.covariance,
        name="test",
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
    )

    f_markowitz = Frontier.build(
        solver=Solver.MARKOWITZ,
        mean=np.copy(mean),
        lower_bounds=np.copy(lower_bounds),
        upper_bounds=np.copy(upper_bounds),
        covariance=np.copy(covar),
        name="Markowitz",
        tol=1e-5,
    )

    assert np.sum(f_bailey.frontier[-1].weights) == pytest.approx(1)
    assert np.sum(f_markowitz.frontier[-1].weights) == pytest.approx(1)

    if np.abs(f_markowitz.max_sharpe[0] - f_bailey.max_sharpe[0]) > 0.3:
        fig = plot_efficient_frontiers([f_markowitz, f_bailey])
        assert fig
        fig.show()
        data = {
            "mean": mean,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "covariance": covar,
        }

        write_bson(
            file=resource_dir / f"problem_{np.random.randint(low=0, high=10000)}.bson",
            data=data,
        )


def test_xxx(resource_dir):
    data = read_bson(file=resource_dir / "problem_1077.bson")

    f_markowitz = Frontier.build(
        solver=Solver.MARKOWITZ,
        mean=data["mean"],
        lower_bounds=data["lower_bounds"],
        upper_bounds=data["upper_bounds"],
        covariance=data["covariance"],
        name="Markowitz",
        tol=1e-6,
    )

    for point in f_markowitz.frontier:
        print(point.expected_variance(data["covariance"]))
        print(point.weights)

    # f_bailey = Frontier.build(
    #    solver=Solver.BAILEY,
    #    mean=data["mean"], lower_bounds=data["lower_bounds"],
    #    upper_bounds=data["upper_bounds"], covariance=data["covariance"],
    #    name="Bailey", tol=1e-10)

    # print("***************************")
    # for point in f_bailey.frontier[:3]:
    #    print(point.expected_variance(data["covariance"]))
    #    print(point.weights)

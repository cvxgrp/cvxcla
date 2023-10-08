import numpy as np
import pytest

from cvx.cla.markowitz.cla import CLA as MARKOWITZ
from tests.bailey.cla import CLA as BAILEY


@pytest.mark.parametrize("solver", [BAILEY, MARKOWITZ])
def test_solver(input_data, solver):
    x = solver(
        mean=input_data.mean,
        lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds,
        covariance=input_data.covariance,
        tol=1e-5,
        A=np.ones((1, len(input_data.mean))),
        b=np.ones(1),
    )

    assert x
    for a in x.turning_points:
        assert a


@pytest.mark.parametrize("solver", [BAILEY, MARKOWITZ])
def test_example(example, example_solution, solver):
    # example from section 3.1 in the Markowitz 2019 paper
    means = example.mean(axis=0)
    std = example.std(axis=0, ddof=1)
    assert np.allclose(means.values, np.array([0.062, 0.146, 0.128]), atol=1e-3)
    assert np.allclose(
        np.power(std.values, 2), np.array([0.016, 0.091, 0.031]), atol=1e-3
    )

    ns = example.shape[1]

    lower_bounds = 0.1 * np.ones(ns)
    upper_bounds = 0.5 * np.ones(ns)

    cla = solver(
        mean=means.values,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        covariance=example.cov().values,
        A=np.ones((1, len(means))),
        b=np.ones(1),
    )

    for row, turning_point in enumerate(cla.turning_points):
        if row > 0:
            assert turning_point.lamb == pytest.approx(
                example_solution["lambda"][row], abs=1e-3
            )

        assert np.allclose(
            turning_point.weights, example_solution.values[row, 1:], atol=1e-3
        )


@pytest.mark.parametrize("n", [2, 3, 5, 10, 20, 50])
def test_init_dimension(n):
    mean = np.random.randn(n)

    # solver = Solver.BAILEY
    tp_bailey = BAILEY(
        mean=mean,
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
        covariance=np.eye(n),
        A=np.ones((1, n)),
        b=np.ones(1),
    )

    # solver = Solver.MARKOWITZ
    tp_markowitz = MARKOWITZ(
        mean=mean,
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
        covariance=np.eye(n),
        A=np.ones((1, n)),
        b=np.ones(1),
    )

    np.testing.assert_almost_equal(
        tp_bailey.turning_points[0].weights, tp_markowitz.turning_points[0].weights
    )

    assert len(tp_bailey) == len(tp_markowitz)


@pytest.mark.parametrize("solver", [MARKOWITZ, BAILEY])
@pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64, 128])  # , 256, 512])
def test_init_solver(solver, n):
    mean = np.random.randn(n)
    A = np.random.randn(n, n)
    sigma = 0.1

    solver(
        mean=mean,
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
        covariance=A @ A.T + sigma * np.eye(n),
        A=np.ones((1, len(mean))),
        b=np.ones(1),
    )

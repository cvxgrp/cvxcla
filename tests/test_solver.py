import pytest
import numpy as np

from cvx.cla.solver import Solver

@pytest.mark.parametrize("solver", [Solver.BAILEY, Solver.MARKOWITZ])
def test_solver(input_data, solver):
    x = solver.build(mean=input_data.mean,
                 lower_bounds=input_data.lower_bounds,
                 upper_bounds=input_data.upper_bounds,
                 covariance=input_data.covariance, tol=1e-5)

    assert x
    for a in x.turning_points:
        assert a

@pytest.mark.parametrize("solver", [Solver.BAILEY, Solver.MARKOWITZ])
def test_example(example, example_solution, solver):
    # example from section 3.1 in the Markowitz 2019 paper
    means = example.mean(axis=0)
    std = example.std(axis=0, ddof=1)
    assert np.allclose(means.values, np.array([0.062, 0.146, 0.128]), atol=1e-3)
    assert np.allclose(np.power(std.values,2), np.array([0.016, 0.091, 0.031]), atol=1e-3)

    ns = example.shape[1]

    lower_bounds = 0.1 * np.ones(ns)
    upper_bounds = 0.5 * np.ones(ns)

    cla = solver.build(mean=means.values, lower_bounds=lower_bounds, upper_bounds=upper_bounds, covariance=example.cov().values)

    for row, turning_point in enumerate(cla.turning_points):
        if row > 0:
            assert turning_point.lamb == pytest.approx(example_solution["lambda"][row], abs=1e-3)

        assert np.allclose(turning_point.weights, example_solution.values[row,1:], atol=1e-3)

@pytest.mark.parametrize("n", [2,3,5,10,20,50])
def test_init(n):
    mean = np.random.randn(n)

    solver = Solver.BAILEY
    tp_bailey = solver.build(mean=mean,
                             lower_bounds=np.zeros(n),
                             upper_bounds=np.ones(n),
                             covariance=np.eye(n))

    solver = Solver.MARKOWITZ
    tp_markowitz = solver.build(mean=mean,
                                lower_bounds=np.zeros(n),
                                upper_bounds=np.ones(n),
                                covariance=np.eye(n))


    np.testing.assert_almost_equal(tp_bailey.turning_points[0].weights,
                                   tp_markowitz.turning_points[0].weights)

    assert tp_bailey.num_points == tp_markowitz.num_points


@pytest.mark.parametrize("solver", [Solver.BAILEY, Solver.MARKOWITZ])
@pytest.mark.parametrize("n", [50,100])
def test_init(solver, n):
    mean = np.random.randn(n)


    solver = solver.build(mean=mean,
                          lower_bounds=np.zeros(n),
                          upper_bounds=np.ones(n),
                          covariance=np.eye(n))

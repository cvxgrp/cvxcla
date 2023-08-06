import pytest

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

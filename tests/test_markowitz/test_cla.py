import numpy as np
from cvx.cla.markowitz.cla import CLA


def test_solver(input_data, results):
    cla = CLA(mean=input_data.mean,
        lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds,
        covariance=input_data.covariance, tol=1e-5)

    observed = np.array([tp.lamb for tp in cla.turning_points[2:]])
    assert np.allclose(results.lamb[:-1], observed, atol=1e-2)

    observed = np.array([tp.mean(input_data.mean) for tp in cla.turning_points[2:]])
    assert np.allclose(results.mean[:-1], observed, atol=1e-2)

    observed = np.array([tp.variance(input_data.covariance) for tp in cla.turning_points[2:]])
    assert np.allclose(results.variance[:-1], observed, atol=0.5)

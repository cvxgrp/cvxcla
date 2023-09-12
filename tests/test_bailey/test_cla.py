import numpy as np

from tests.bailey.cla import CLA


def test_big(input_data, results):
    cla = CLA(
        mean=input_data.mean,
        lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds,
        covariance=input_data.covariance,
        A=np.ones((1, len(input_data.mean))),
        b=np.ones(1),
    )

    observed = [tp.lamb for tp in cla.turning_points[1:]]
    np.allclose(results.lamb, np.array(observed))

    observed = [tp.mean(input_data.mean) for tp in cla.turning_points[1:]]
    np.allclose(results.mean, np.array(observed))

    observed = [tp.variance(input_data.covariance) for tp in cla.turning_points[1:]]
    np.allclose(results.variance, np.array(observed))

    observed = [tp.weights for tp in cla.turning_points[1:]]
    np.allclose(results.weights, np.array(observed))

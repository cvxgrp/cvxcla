import numpy as np

from cvx.cla.schur import Schur


def test_compute_lambda():
    # Create a Schur instance with some sample data
    covariance = np.array([[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 3]])
    mean = np.array([1, 2, 3])
    free = np.array([True, True, False])
    weights = np.array([0.3, 0.5, 0.2])
    schur = Schur(covariance, mean, free, weights)

    # Test the compute_lambda method with different inputs
    index = 1
    bi = 0.5
    expected_lambda = 0.62
    expected_gamma = 0.5
    actual_lambda, actual_gamma = schur.compute_lambda(index, bi)

    assert np.isclose(actual_lambda, expected_lambda)
    assert np.isclose(actual_gamma, expected_gamma)

    index = 1
    bi = [0.2, 0.3]
    expected_lambda = 0.02
    expected_gamma = 0.2
    actual_lambda, actual_gamma = schur.compute_lambda(index, bi)
    assert np.isclose(actual_lambda, expected_lambda)
    assert np.isclose(actual_gamma, expected_gamma)


def test_compute_weight():
    # Define some test data
    covariance = np.array([[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 3]])
    mean = np.array([1.0, 1.0, 3.0])
    free = np.array([True, True, False])
    weights = np.array([0.4, 0.6, 0.0])

    # Create a Schur object
    schur = Schur(covariance, mean, free, weights)

    # Test the compute_weight method
    lamb = 0.5

    actual_weights = schur.update_weights(lamb)
    np.testing.assert_allclose(actual_weights, np.array([0.75, 0.25, 0.0]), rtol=1e-3)

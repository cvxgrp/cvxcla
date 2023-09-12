import numpy as np
import pytest

from tests.bailey.cla import _Schur


def test_compute_lambda():
    # Create a _Schur instance with some sample data
    covariance = np.array([[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 3]])
    mean = np.array([1, 2, 3])
    free = np.array([True, True, False])
    weights = np.array([0.3, 0.5, 0.2])
    schur = _Schur(covariance, mean, free, weights)

    # Test the compute_lambda method with different inputs
    index = 1
    bi = np.array([0.5])
    expected_lambda = 0.62
    expected_gamma = 0.5
    actual_lambda, actual_gamma = schur.compute_lambda(index, bi)

    assert np.isclose(actual_lambda, expected_lambda)
    assert np.isclose(actual_gamma, expected_gamma)

    index = 1
    bi = np.array([0.2, 0.3])
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

    # Create a _Schur object
    schur = _Schur(covariance, mean, free, weights)

    # Test the compute_weight method
    lamb = 0.5

    actual_weights = schur.update_weights(lamb)
    np.testing.assert_allclose(actual_weights, np.array([0.75, 0.25, 0.0]), rtol=1e-3)


def test_special_minvar():
    mean = np.array([0.1, 0.2])

    lower_bounds = np.array([0.0, 0.0])
    upper_bounds = np.array([0.6, 0.7])
    covariance = np.array([[2.0, 1.0], [1.0, 3.0]])

    # start with the vector [0.3, 0.7]. Try to free the second asset
    schur = _Schur(
        covariance=covariance,
        mean=mean,
        free=np.array([True, True]),
        weights=np.array([0.3, 0.7]),
    )
    lamb, bi = schur.compute_lambda(index=1, bi=np.array([0.7]))
    assert lamb == pytest.approx(11.0)
    assert bi == 0.7
    w = schur.update_weights(lamb=11)
    assert np.allclose(w, np.array([0.3, 0.7]))

    schur = _Schur(
        covariance=covariance,
        mean=mean,
        free=np.array([True, True]),
        weights=np.array([0.3, 0.7]),
    )

    # try to block the first asset
    lamb, bi = schur.compute_lambda(
        index=0,
        bi=np.array([lower_bounds[0], upper_bounds[0]]),
    )

    assert lamb == pytest.approx(2.0)
    assert bi == pytest.approx(0.6)

    schur = _Schur(
        covariance=covariance,
        mean=mean,
        free=np.array([False, True]),
        weights=np.array([0.6, 0.7]),
    )
    w = schur.update_weights(lamb=2)

    assert np.allclose(w, np.array([0.6, 0.4]))

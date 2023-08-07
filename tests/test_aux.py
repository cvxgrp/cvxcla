from dataclasses import dataclass

import numpy as np
import pytest

from cvx.cla.aux import CLAUX
from cvx.cla.types import TurningPoint


@dataclass(frozen=True)
class TestCla(CLAUX):
    pass


@pytest.fixture()
def cla(input_data):
    cla = TestCla(covariance=input_data.covariance,
                  mean=input_data.mean,
                  lower_bounds=input_data.lower_bounds,
                  upper_bounds=input_data.upper_bounds,
                  tol=1e-5)
    return cla

def test_claux(cla, input_data):
    np.testing.assert_equal(cla.covariance, input_data.covariance)
    np.testing.assert_equal(cla.mean, input_data.mean)
    np.testing.assert_equal(cla.lower_bounds, input_data.lower_bounds)
    np.testing.assert_equal(cla.upper_bounds, input_data.upper_bounds)
    assert cla.tol == 1e-5

def test_append(cla):
    weights = np.random.rand(10)
    weights = weights / np.sum(weights)

    assert np.all(weights <= cla.upper_bounds)
    assert np.all(weights >= cla.lower_bounds)

    tp = TurningPoint(
        weights=weights,
        free= np.full_like(cla.mean, fill_value=True, dtype=np.bool_), lamb=2.0
    )

    assert tp.lamb == 2.0

    cla.append(tp)

    assert cla.num_points == 1
    assert cla.turning_points[-1].lamb == 2.0
    # all variables are free
    assert np.all(cla.turning_points[-1].free)

def test_first_turning_point(cla):
    tp = cla.first_turning_point()
    np.testing.assert_almost_equal(tp.weights, np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]))

def test_minimum_variance(cla):
    x = cla.minimum_variance()
    np.testing.assert_almost_equal(x, np.array([0.03696858, 0.02690084, 0.09494243,
                                                0.12577595, 0.07674608, 0.21935567,
                                                0.0299871,  0.03596328, 0.06134984,
                                                0.29201023]))


def test_raise():
    cla = TestCla(
        covariance=np.eye(2),
        upper_bounds=np.ones(2),
        lower_bounds=np.zeros(2),
        mean=np.ones(2),
        tol=1e-5
    )

    with pytest.raises(AssertionError):
        tp = TurningPoint(
            weights=np.array([0.6, 0.6]),
            free= np.full_like(cla.mean, fill_value=True, dtype=np.bool_),
            lamb=2.0
        )
        cla.append(tp)

    with pytest.raises(AssertionError):
        tp = TurningPoint(
            weights=np.array([1.2, 0.6]),
            free= np.full_like(cla.mean, fill_value=True, dtype=np.bool_),
            lamb=2.0
        )
        cla.append(tp)

    with pytest.raises(AssertionError):
        tp = TurningPoint(
            weights=np.array([0.6, -0.6]),
            free= np.full_like(cla.mean, fill_value=True, dtype=np.bool_),
            lamb=2.0
        )
        cla.append(tp)

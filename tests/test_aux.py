from dataclasses import dataclass

import numpy as np
import pytest
from loguru import logger

from cvx.cla.claux import CLAUX
from cvx.cla.types import TurningPoint


@dataclass(frozen=True)
class Cla(CLAUX):
    pass


@pytest.fixture()
def cla(input_data):
    return Cla(
        covariance=input_data.covariance,
        mean=input_data.mean,
        lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds,
        tol=1e-5,
        A=np.ones((1, len(input_data.mean))),
        b=np.ones(1),
        logger=logger,
    )


def test_claux(cla, input_data):
    """
    Test that the CLAUX class is initialized correctly
    """
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
        free=np.full_like(cla.mean, fill_value=True, dtype=np.bool_),
        lamb=2.0,
    )

    assert tp.lamb == 2.0

    cla._append(tp)

    # assert cla.num_points == 1
    assert len(cla) == 1

    assert cla.turning_points[-1].lamb == 2.0
    # all variables are free
    assert np.all(cla.turning_points[-1].free)

    fig = cla.frontier.plot()
    assert fig is not None


def test_raise():
    cla = Cla(
        covariance=np.eye(2),
        upper_bounds=np.ones(2),
        lower_bounds=np.zeros(2),
        mean=np.ones(2),
        tol=1e-5,
        A=np.ones((1, 2)),
        b=np.ones(1),
    )

    with pytest.raises(AssertionError):
        tp = TurningPoint(
            weights=np.array([0.6, 0.6]),
            free=np.full_like(cla.mean, fill_value=True, dtype=np.bool_),
            lamb=2.0,
        )
        cla._append(tp)

    with pytest.raises(AssertionError):
        tp = TurningPoint(
            weights=np.array([1.2, 0.6]),
            free=np.full_like(cla.mean, fill_value=True, dtype=np.bool_),
            lamb=2.0,
        )
        cla._append(tp)

    with pytest.raises(AssertionError):
        tp = TurningPoint(
            weights=np.array([0.6, -0.6]),
            free=np.full_like(cla.mean, fill_value=True, dtype=np.bool_),
            lamb=2.0,
        )
        cla._append(tp)

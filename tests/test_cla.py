from __future__ import annotations

import numpy as np

from cvx.cla.cla import TurningPoint
from cvx.cla.frontier import Frontier


def test_cla(resource_dir):
    # 1) Path
    path = resource_dir / "CLA_Data.csv"
    # 2) Load data, set seed
    data = np.genfromtxt(path, delimiter=",", skip_header=1)  # load as numpy array
    mean = data[:1][0]
    lB = data[1:2][0]
    uB = data[2:3][0]
    covar = np.array(data[3:])

    turningPoints = TurningPoint.construct(
        mean=mean, lower_bounds=lB, upper_bounds=uB, covariance=covar
    )
    np.testing.assert_equal(
        np.array(turningPoints[5].free), np.array([1, 0, 3, 9, 7, 5])
    )
    np.testing.assert_almost_equal(turningPoints[4].lamb, 0.16458117494477612)
    np.testing.assert_array_almost_equal(
        turningPoints[3].weights,
        np.array(
            [
                4.339841e-01,
                2.312475e-01,
                0.000000e00,
                3.347684e-01,
                0.000000e00,
                0.000000e00,
                0.000000e00,
                0.000000e00,
                0.000000e00,
                -3.552714e-15,
            ]
        ),
    )

    np.testing.assert_almost_equal(turningPoints[5].gamma, -0.09312582327537038)


def test_cla_extrem():
    mean = np.array([-5.0, 0.1])
    lB = np.array([0.0, 0.0])
    uB = np.array([0.6, 0.6])
    covar = np.array([[2.0, 1.0], [1.0, 3.0]])
    turningPoints = TurningPoint.construct(
        mean=mean, lower_bounds=lB, upper_bounds=uB, covariance=covar
    )
    for point in turningPoints:
        print(point)
    f = Frontier.construct(
        mean=mean, lower_bounds=lB, upper_bounds=uB, covariance=covar
    )
    np.testing.assert_almost_equal(f.max_sharpe[0], -1.414890417529577)

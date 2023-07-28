from __future__ import annotations

import numpy as np

from cvx.cla.frontier import Frontier


def test_frontier(resource_dir):
    # 1) Path
    path = resource_dir / "CLA_Data.csv"
    # 2) Load data, set seed
    data = np.genfromtxt(path, delimiter=",", skip_header=1)  # load as numpy array
    mean = data[:1][0]
    lB = data[1:2][0]
    uB = data[2:3][0]
    covar = np.array(data[3:])

    f = Frontier.construct(
        mean=mean, lower_bounds=lB, upper_bounds=uB, covariance=covar
    )

    np.testing.assert_equal(f.covariance, covar)
    np.testing.assert_equal(f.num, 11)
    np.testing.assert_almost_equal(f.max_sharpe[0], 4.4535334766464025)

    np.testing.assert_almost_equal(f.mean, mean)
    np.testing.assert_almost_equal(
        f.returns,
        np.array(
            [
                1.19,
                1.19,
                1.1802595,
                1.1600565,
                1.1112623,
                1.1083602,
                1.0224839,
                1.0153059,
                0.9727204,
                0.9499368,
                0.8032154,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        f.variance,
        np.array(
            [
                0.9063047,
                0.9063047,
                0.2977414,
                0.1741023,
                0.0711394,
                0.070234,
                0.0527529,
                0.0519761,
                0.0482043,
                0.0466666,
                0.0421225,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        f.volatility,
        np.array(
            [
                0.9520004,
                0.9520004,
                0.5456569,
                0.4172557,
                0.2667196,
                0.265017,
                0.2296801,
                0.2279827,
                0.2195549,
                0.2160246,
                0.2052376,
            ]
        ),
    )

    f.interpolate(num=10)

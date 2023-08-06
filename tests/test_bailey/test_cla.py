import numpy as np
import pytest

from cvx.cla.bailey.cla import Schur, CLA



def test_cla_hist():
    mean = np.array([0.1, 0.2])

    lower_bounds = np.array([0.0, 0.0])
    upper_bounds = np.array([0.6, 0.7])
    covariance = np.array([[2.0, 1.0], [1.0, 3.0]])
    cla = CLA(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds, covariance=covariance)


def test_big(input_data, results):
    print(input_data.mean)
    cla = CLA(mean=input_data.mean, lower_bounds=input_data.lower_bounds,
              upper_bounds=input_data.upper_bounds, covariance=input_data.covariance)

    observed = [tp.lamb for tp in cla.turning_points[1:]]
    np.allclose(results.lamb, np.array(observed))

    observed = [tp.mean(input_data.mean) for tp in cla.turning_points[1:]]
    np.allclose(results.mean, np.array(observed))

    observed = [tp.variance(input_data.covariance) for tp in cla.turning_points[1:]]
    np.allclose(results.variance, np.array(observed))

    observed = [tp.weights for tp in cla.turning_points[1:]]
    np.allclose(results.weights, np.array(observed))


def test_remove():
    mean = np.array([0.1, 0.2])
    lB = np.array([0.0, 0.0])
    uB = np.array([0.6, 0.7])
    covar = np.array([[2.0, 1.0], [1.0, 3.0]])

    f = np.array([True, True])

    schur = Schur(
        covariance=covar,
        mean=mean,
        free=f,
        weights=np.array([0.3, 0.7]),
    )

    l_in = -np.inf
    for i in np.where(f)[0]:
        j = np.sum(f[:i])
        lamb, bi = schur.compute_lambda(index=j, bi=np.array([lB[i], uB[i]]))
        if lamb > l_in:
            l_in, i_in, bi_in = lamb, i, bi


    assert bi_in == pytest.approx(0.6)
    assert i_in == 0
    assert l_in == pytest.approx(2.0)

    schur = Schur(
        covariance=covar,
        mean=mean,
        weights=np.array([0.6, 0.7]),
        free=np.array([False, True]),
    )

    www = schur.update_weights(lamb=2.0)
    np.testing.assert_almost_equal(www, np.array([0.6, 0.4]))


def test_add():
    mean = np.array([0.1, 0.2])
    covar = np.array([[2.0, 1.0], [1.0, 3.0]])
    w = np.array([0.3, 0.7])

    f = np.array([False, True])

    l_out = -np.inf

    for i in np.where(~f)[0]:
        fff = np.copy(f)
        fff[i] = True

        schur = Schur(
            covariance=covar,
            mean=mean,
            free=fff,
            weights=w,
        )

        # count the number of entries that are True below the ith entry in fff
        j = np.sum(fff[:i])

        lamb, bi = schur.compute_lambda(
            # index i in fff corresponds to index j in mean_free
            index=j,
            bi=np.array([w[i]]),
        )

        if lamb > l_out:
            l_out, i_out = lamb, i


    #assert i_out == 0
    assert l_out == pytest.approx(11.0)

    schur = Schur(
        covariance=covar,
        mean=mean,
        weights=np.array([1.0, 1.0]),
        free=np.array([True, True]),
    )

    www = schur.update_weights(lamb=11.0)
    np.testing.assert_almost_equal(www, np.array([0.3, 0.7]))

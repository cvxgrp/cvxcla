import numpy as np
import pytest

from cvx.cla.first import init_algo
from cvx.cla.schur import Schur
from cvx.hist.cla import CLA


def test_cla_hist():
    mean = np.array([0.1, 0.2])

    lB = np.array([0.0, 0.0])
    uB = np.array([0.6, 0.7])
    covar = np.array([[2.0, 1.0], [1.0, 3.0]])
    cla = CLA(mean=mean, lB=lB, uB=uB, covar=covar)
    cla.solve()
    print(cla.w)
    print(cla.f)


# use pytest parameter
# https://docs.pytest.org/en/latest/parametrize.html#parametrize-basics


@pytest.mark.parametrize("n", [5, 20, 100, 1000])
def test_init_algo(n):
    mean = np.random.randn(n)
    lB = np.zeros(n)
    uB = np.random.rand(n)

    first = init_algo(mean=mean, lower_bounds=lB, upper_bounds=uB)

    assert np.sum(first.free) == 1
    assert np.sum(first.weights) == pytest.approx(1.0)


def test_lb_ub_mixed():
    uB = np.zeros(3)
    lB = np.ones(3)
    mean = np.ones(3)

    with pytest.raises(ValueError):
        init_algo(mean=mean, lower_bounds=lB, upper_bounds=uB)


def test_no_fully_invested():
    uB = 0.2 * np.ones(3)
    lB = np.zeros(3)
    mean = np.ones(3)

    with pytest.raises(ValueError):
        init_algo(mean=mean, lower_bounds=lB, upper_bounds=uB)




def test_remove():
    mean = np.array([0.1, 0.2])
    lB = np.array([0.0, 0.0])
    uB = np.array([0.6, 0.7])
    covar = np.array([[2.0, 1.0], [1.0, 3.0]])

    f = [0, 1]

    schur = Schur(
        covariance=covar,
        mean=mean,
        free=np.array([True, True]),
        weights=np.array([0.3, 0.7]),
    )

    l_in = -np.inf
    j = 0
    for i in f:
        lamb, bi = schur.compute_lambda(index=j, bi=np.array([lB[i], uB[i]]))
        if lamb > l_in:
            l_in, i_in, bi_in = lamb, i, bi
        j += 1

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

    f = [1]

    l_out = -np.inf

    b = [0]

    assert b == [0]

    i = 0
    schur = Schur(covariance=covar, mean=mean, weights=w, free=np.array([True, True]))

    lamb, _ = schur.compute_lambda(index=0, bi=np.array([w[i]]))

    if lamb > l_out:
        l_out, i_out = lamb, i

    assert i_out == 0
    assert l_out == pytest.approx(11.0)

    schur = Schur(
        covariance=covar,
        mean=mean,
        weights=np.array([1.0, 1.0]),
        free=np.array([True, True]),
    )

    www = schur.update_weights(lamb=11.0)
    np.testing.assert_almost_equal(www, np.array([0.3, 0.7]))

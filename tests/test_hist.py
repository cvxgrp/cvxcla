import numpy as np
import pytest

from cvx.cla._first import init_algo
from cvx.cla._cla import Schur
from cvx.cla._cla import CLA



def test_cla_hist():
    mean = np.array([0.1, 0.2])

    lB = np.array([0.0, 0.0])
    uB = np.array([0.6, 0.7])
    covar = np.array([[2.0, 1.0], [1.0, 3.0]])
    cla = CLA(mean=mean, lower_bounds=lB, upper_bounds=uB, covariance=covar)



# use pytest parameter
# https://docs.pytest.org/en/latest/parametrize.html#parametrize-basics

def test_big(resource_dir):

    # 1) Path
    path = resource_dir / "CLA_Data.csv"
    # 2) Load data, set seed
    data = np.genfromtxt(path, delimiter=",",
                         skip_header=1)  # load as numpy array
    mean = data[:1][0]
    lB = data[1:2][0]
    uB = data[2:3][0]
    covar = np.array(data[3:])
    cla = CLA(mean=mean, lower_bounds=lB, upper_bounds=uB, covariance=covar)
    for tp in cla.turning_points:
        print(tp.lamb)



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

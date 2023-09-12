import numpy as np
import pytest

from cvx.cla.linalg.algebra import bilinear

np.random.seed(42)


def test_last_diagonal():
    a = np.zeros((3, 3))
    a[2, 2] = 1.0

    a = np.zeros((3, 3))
    a[-1, -1] = 2.0

    assert a[-1, -1] == 2.0


def test_last_column():
    a = np.zeros((3, 3))
    a[:, 2] = 1.0

    a = np.zeros((3, 3))
    a[:, -1] = 2.0

    a[:-1, -1] = 3.0

    np.testing.assert_array_equal(a[:, -1], np.array([3.0, 3.0, 2.0]))


@pytest.mark.parametrize("n", [50, 100, 200])
def test_bilinear(n):
    A = np.random.rand(n, n)
    cov = A.T @ A

    x = bilinear(cov)
    assert np.isclose(x, np.sum(np.sum(cov, axis=0)))

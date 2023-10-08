import numpy as np

np.random.seed(42)


def test_last_diagonal():
    """test last diagonal element"""
    a = np.zeros((3, 3))
    a[2, 2] = 1.0

    a = np.zeros((3, 3))
    a[-1, -1] = 2.0

    assert a[-1, -1] == 2.0


def test_last_column():
    """test last column of a numpy array"""
    a = np.zeros((3, 3))
    a[:, 2] = 1.0

    a = np.zeros((3, 3))
    a[:, -1] = 2.0

    a[:-1, -1] = 3.0

    np.testing.assert_array_equal(a[:, -1], np.array([3.0, 3.0, 2.0]))

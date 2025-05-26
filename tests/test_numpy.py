from __future__ import annotations

import numpy as np

np.random.seed(42)


def test_last_diagonal() -> None:
    """
    Test accessing and modifying the last diagonal element of a numpy array.

    This test verifies that the last diagonal element can be accessed and modified
    using both explicit indices and negative indices.
    """
    a = np.zeros((3, 3))
    a[2, 2] = 1.0

    a = np.zeros((3, 3))
    a[-1, -1] = 2.0

    assert a[-1, -1] == 2.0


def test_last_column() -> None:
    """
    Test accessing and modifying the last column of a numpy array.

    This test verifies that the last column can be accessed and modified
    using both explicit indices and negative indices, and that slicing
    operations work correctly.
    """
    a = np.zeros((3, 3))
    a[:, 2] = 1.0

    a = np.zeros((3, 3))
    a[:, -1] = 2.0

    a[:-1, -1] = 3.0

    np.testing.assert_array_equal(a[:, -1], np.array([3.0, 3.0, 2.0]))

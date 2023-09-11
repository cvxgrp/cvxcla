import numpy as np
import pytest

from cvx.cla.linalg.algebra import Solver, bilinear, get_submatrix, replace_submatrix

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


def test_submatrix():
    A = np.random.rand(4, 4)
    A[:-1, :-1] = np.eye(3)
    np.testing.assert_array_equal(
        get_submatrix(A, rows=[0, 1, 2], columns=[0, 1, 2]), np.eye(3)
    )


def test_boolean():
    A = np.random.rand(4, 4)
    x = np.array([True, False, True, False])
    B = replace_submatrix(A, rows=x, columns=x, mat_replace=np.eye(2))
    np.testing.assert_array_equal(get_submatrix(B, rows=x, columns=x), np.eye(2))


def test_replace():
    A = np.random.rand(4, 4)
    x = np.array([True, False, True, False])
    B = replace_submatrix(
        A, rows=np.array([0, 2]), columns=np.array([0, 2]), mat_replace=np.eye(2)
    )
    np.testing.assert_array_equal(get_submatrix(B, rows=x, columns=x), np.eye(2))


def test_replace_list():
    A = np.random.rand(4, 4)
    x = np.array([True, False, True, False])
    B = replace_submatrix(A, rows=[0, 2], columns=[0, 2], mat_replace=np.eye(2))
    np.testing.assert_array_equal(get_submatrix(B, rows=x, columns=x), np.eye(2))


@pytest.mark.parametrize("n", [50, 100, 200])
def test_solver(n):
    A = np.random.rand(n, n)
    cov = A.T @ A

    b = np.random.rand(n + 1, 2)

    A = np.atleast_2d(np.ones(n))
    IN = np.zeros(n, dtype=bool)
    IN[-1] = True

    # create the solver
    s = Solver(C=cov, A=A, IN=IN)

    np.testing.assert_array_almost_equal(
        s.M, np.block([[cov, A.T], [A, np.zeros((1, 1))]])
    )

    # free a blocked variable
    for i in range(n):
        inv = s.free(new=i)
        np.testing.assert_array_almost_equal(inv, np.linalg.inv(s.sub_M), decimal=4)
        alpha, beta = s.solve(b=b)
        np.testing.assert_array_almost_equal(
            alpha[s.active], inv @ b[s.active, 0], decimal=4
        )
        np.testing.assert_array_almost_equal(
            beta[s.active], inv @ b[s.active, 1], decimal=4
        )

    # block a free variable
    for i in range(n - 2):
        inv = s.block(new=i)
        np.testing.assert_array_almost_equal(inv, np.linalg.inv(s.sub_M), decimal=4)
        alpha, beta = s.solve(b=b)
        np.testing.assert_array_almost_equal(
            alpha[s.active], inv @ b[s.active, 0], decimal=4
        )
        np.testing.assert_array_almost_equal(
            beta[s.active], inv @ b[s.active, 1], decimal=4
        )

    # block a blocked variable
    s.block(new=n - 4)


@pytest.mark.parametrize("n", [50, 100, 200])
def test_solver_speed(n):
    A = np.random.rand(n, n)
    cov = A.T @ A

    A = np.atleast_2d(np.ones(n))
    IN = np.zeros(n, dtype=bool)
    IN[-1] = True

    # create the solver
    s = Solver(C=cov, A=A, IN=IN)

    # free a blocked variable
    for i in range(n):
        s.free(new=i)

    # block a free variable
    for i in range(n - 2):
        s.block(new=i)


@pytest.mark.parametrize("n", [50, 100, 200])
def test_bilinear(n):
    A = np.random.rand(n, n)
    cov = A.T @ A

    x = bilinear(cov)
    assert np.isclose(x, np.sum(np.sum(cov, axis=0)))

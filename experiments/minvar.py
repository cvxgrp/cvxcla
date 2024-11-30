import cvxpy as cp
import numpy as np
from loguru import logger


def bilinear(mat, left=None, right=None):
    n, m = mat.shape
    assert n == m, "Matrix must be square"

    if left is None:
        left = np.ones(n)

    if right is None:
        right = np.ones(n)

    return left.T @ (mat @ right)


if __name__ == "__main__":
    n = 3
    A = np.random.randn(n, n)
    Sigma = A @ A.T

    # We minimize the variance of the portfolio
    # Var = min 1/2 * w'Sigma w
    # subject to the constraint that the portfolio is fully invested
    # and that all weights are non-negative.
    # sum w == 1
    # and w >= 0

    # We can _solve this problem using Lagrange multipliers

    # -gamma*(sum w - 1)
    # w - s^2 = 0
    # -theta_1*(w_1 - s_1^2)
    # -theta_2*(w_2 - s_2^2)

    # For now let's drop the constraint of long-only

    # from
    gamma = 1 / bilinear(np.linalg.inv(Sigma), left=np.ones(n), right=np.ones(n))
    w = gamma * np.linalg.solve(Sigma, np.ones(n))
    logger.info(f"Gamma: {gamma}")
    logger.info(f"Weights: {w}")
    logger.info(f"Sum of weights: {np.sum(w)}")
    logger.info(f"Variance: {w.T @ Sigma @ w}")

    # values are in ascending order
    # values, vectors = np.linalg.eigh(Sigma)
    # logger.info(f"Min eigenvalue: {np.min(values)}")

    # vector = vectors[:,0]
    # a = vector / np.sum(vector)
    # logger.info(f"Eigenvalues of Sigma: {values}")
    # logger.info(f"Scaled eigenvector: {a}")
    # print(a.T @ Sigma @ a)
    # print(np.sqrt(values))

    # print((Sigma @ a)/a)
    # MinVar without the non-negativity constraint
    # is equivalent to the inverse of the covariance matrix

    C = np.linalg.cholesky(Sigma)
    xxx = np.linalg.solve(C, np.ones(n))
    a = xxx / np.sum(xxx)

    print(a)
    print(a.T @ Sigma @ a)

    x = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(x, Sigma))
    parameters = [cp.sum(x) == 1]

    prob = cp.Problem(objective, parameters)
    prob.solve()

    a = x.value
    print(a.T @ Sigma @ a)

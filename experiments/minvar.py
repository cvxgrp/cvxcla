"""
Minimum Variance Portfolio Optimization Experiment

This script demonstrates different methods to solve the minimum variance portfolio optimization problem:
1. Analytical solution using Lagrange multipliers
2. Solution using Cholesky decomposition
3. Solution using convex optimization with CVXPY

The minimum variance portfolio problem is formulated as:
    minimize   (1/2) * w^T * Sigma * w
    subject to:
        sum(w) = 1  (fully invested constraint)

Where:
    w: Portfolio weights
    Sigma: Covariance matrix of asset returns
"""

import cvxpy as cp
import numpy as np
from loguru import logger


def bilinear(mat, left=None, right=None):
    """
    Compute the bilinear form: left^T * mat * right

    This function calculates a bilinear form with a matrix and two vectors.
    If left or right are not provided, they default to vectors of ones.

    Args:
        mat: A square matrix
        left: Left vector (default: vector of ones)
        right: Right vector (default: vector of ones)

    Returns:
        The scalar result of the bilinear form
    """
    n, m = mat.shape
    assert n == m, "Matrix must be square"

    if left is None:
        left = np.ones(n)

    if right is None:
        right = np.ones(n)

    return left.T @ (mat @ right)


if __name__ == "__main__":
    # Set up a small test problem with 3 assets
    n = 3
    # Generate a random matrix A
    A = np.random.randn(n, n)
    # Create a positive definite covariance matrix Sigma = A*A^T
    Sigma = A @ A.T

    # Problem formulation:
    # We minimize the variance of the portfolio
    # min (1/2) * w^T * Sigma * w
    # subject to the constraint that the portfolio is fully invested:
    # sum(w) = 1

    # Note: The non-negativity constraint (w >= 0) is not enforced in this example
    # to allow for analytical solutions.

    # Method 1: Analytical solution using Lagrange multipliers
    # The Lagrangian is:
    # L(w, gamma) = (1/2) * w^T * Sigma * w - gamma * (sum(w) - 1)
    # Taking derivatives and setting to zero:
    # dL/dw = Sigma * w - gamma * 1 = 0
    # dL/dgamma = sum(w) - 1 = 0

    # From the first equation: w = gamma * Sigma^(-1) * 1
    # Substituting into the second: gamma * sum(Sigma^(-1) * 1) = 1
    # Therefore: gamma = 1 / sum(Sigma^(-1) * 1)

    # Calculate gamma (the Lagrange multiplier)
    gamma = 1 / bilinear(np.linalg.inv(Sigma), left=np.ones(n), right=np.ones(n))
    # Calculate the optimal weights
    w = gamma * np.linalg.solve(Sigma, np.ones(n))

    # Log the results
    logger.info("Method 1: Analytical solution using Lagrange multipliers")
    logger.info(f"Gamma (Lagrange multiplier): {gamma}")
    logger.info(f"Optimal weights: {w}")
    logger.info(f"Sum of weights: {np.sum(w)}")
    logger.info(f"Portfolio variance: {w.T @ Sigma @ w}")

    # The following code is commented out as it explores an alternative approach
    # using eigenvalue decomposition, which is not used in the final solution
    #
    # # values are in ascending order
    # # values, vectors = np.linalg.eigh(Sigma)
    # # logger.info(f"Min eigenvalue: {np.min(values)}")
    # # vector = vectors[:,0]
    # # a = vector / np.sum(vector)
    # # logger.info(f"Eigenvalues of Sigma: {values}")
    # # logger.info(f"Scaled eigenvector: {a}")
    # # print(a.T @ Sigma @ a)
    # # print(np.sqrt(values))
    # # print((Sigma @ a)/a)

    # Method 2: Solution using Cholesky decomposition
    # The Cholesky decomposition gives Sigma = C*C^T
    # We can solve the system more efficiently using this decomposition
    logger.info("Method 2: Solution using Cholesky decomposition")
    # Compute the Cholesky decomposition of Sigma
    C = np.linalg.cholesky(Sigma)
    # Solve the system C*x = 1 (more efficient than solving Sigma*w = gamma*1)
    b = np.linalg.solve(C.T, np.linalg.solve(C, np.ones(n)))
    a = b / np.sum(b)

    logger.info(f"Optimal weights: {a}")
    logger.info(f"Portfolio variance: {a.T @ Sigma @ a}")

    # Method 3: Solution using convex optimization with CVXPY
    logger.info("Method 3: Solution using convex optimization (CVXPY)")
    # Define the variable representing portfolio weights
    x = cp.Variable(n)

    # Define the objective function: minimize w^T * Sigma * w
    objective = cp.Minimize(cp.quad_form(x, Sigma))
    # Define the constraint: sum(w) = 1
    parameters = [cp.sum(x) == 1]

    # Create and solve the optimization problem
    prob = cp.Problem(objective, parameters)
    prob.solve()

    # Get the optimal weights
    a = x.value
    logger.info(f"Optimal weights: {a}")
    logger.info(f"Portfolio variance: {a.T @ Sigma @ a}")

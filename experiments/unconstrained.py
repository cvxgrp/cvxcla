"""
Unconstrained Mean-Variance Portfolio Optimization Experiment

This script explores the analytical solution to the unconstrained mean-variance portfolio
optimization problem for different values of the risk aversion parameter (lambda).

The unconstrained problem is formulated as:
    maximize   lambda * (mu^T * w) - (1/2) * w^T * Sigma * w

Where:
    w: Portfolio weights (not constrained to sum to 1 or be non-negative)
    mu: Expected returns for each asset
    Sigma: Covariance matrix of asset returns
    lambda: Risk aversion parameter (controls the trade-off between return and risk)

Reference:
    https://www.cs.ubc.ca/~schmidtm/Courses/Notes/linearQuadraticGradients.pdf
"""

import numpy as np
from loguru import logger


def f(lamb):
    """
    Compute the optimal portfolio weights for a given risk aversion parameter (lambda).

    This function implements the analytical solution to the unconstrained mean-variance
    portfolio optimization problem. It calculates the optimal weights and reports
    various portfolio characteristics.

    The solution is derived by setting the gradient of the objective function to zero:
    lambda * mu - Sigma * w = 0

    Which gives: w = lambda * Sigma^(-1) * mu

    We then add a term to ensure the weights sum to 1:
    w = lambda * Sigma^(-1) * mu + gamma * Sigma^(-1) * 1

    Where gamma is chosen to ensure sum(w) = 1.

    Args:
        lamb: Risk aversion parameter (lambda)

    Global variables used:
        Sigma: Covariance matrix
        mu: Expected returns

    Prints:
        Lambda value, gamma value, optimal weights, sum of weights, and portfolio variance
    """
    # Calculate the denominator for gamma: sum(Sigma^(-1) * 1)
    denom = np.sum(np.sum(np.linalg.inv(Sigma), axis=1))
    # Calculate the numerator for gamma: 1 - lambda * sum(Sigma^(-1) * mu)
    numerator = 1 - lamb * np.sum(np.linalg.inv(Sigma) @ mu)

    # Calculate gamma to ensure the weights sum to 1
    gamma = numerator / denom

    # Calculate the optimal portfolio weights
    w = lamb * np.linalg.inv(Sigma) @ mu + gamma * np.sum(np.linalg.inv(Sigma), axis=1)

    # Log the results
    logger.info("********************************************************")
    logger.info(f"Lambda: {lamb}")
    logger.info(f"Gamma: {gamma}")
    logger.info(f"Weights: {w}")
    logger.info(f"Sum of weights: {np.sum(w)}")
    logger.info(f"Portfolio variance: {w.T @ Sigma @ w}")


if __name__ == "__main__":
    # Set up a small test problem with 3 assets
    n = 3
    # Generate a random matrix A
    A = np.random.randn(n, n)
    # Create a positive definite covariance matrix Sigma = A*A^T
    Sigma = A @ A.T

    # Generate random expected returns
    mu = np.random.randn(n)

    # Test the function with different lambda values
    # Lambda = 0: Minimum variance portfolio (no consideration for returns)
    f(lamb=+0.0)
    # Lambda > 0: Increasing weight on expected returns
    f(lamb=+0.1)
    f(lamb=+0.2)
    # Lambda < 0: Negative weight on expected returns (unusual in practice)
    f(lamb=-0.1)
    f(lamb=-0.2)

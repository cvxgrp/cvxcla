"""
Portfolio Optimization - Efficient Frontier Computation

This script implements a basic portfolio optimization model using MOSEK's Fusion API.
It computes points on the efficient frontier by solving a series of optimization problems
with different risk aversion parameters (alphas).

The optimization problem for each alpha is:
    maximize   expected return - alpha * variance
    subject to:
        sum(x) = w + sum(x0)  (budget constraint)
        x >= 0                (no short-selling)

Where:
    x: Portfolio weights
    w: Initial cash holding
    x0: Initial holdings
    alpha: Risk aversion parameter

Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
Original file: portfolio_2_frontier.py
"""

import numpy as np
from mosek.fusion import Domain, Expr, Model, ObjectiveSense


def EfficientFrontier(n, mu, GT, x0, w, alphas):
    """
    Compute points on the efficient frontier for a portfolio optimization problem.

    This function solves a series of portfolio optimization problems with different
    risk aversion parameters (alphas) to trace out the efficient frontier.

    Args:
        n: Number of assets in the portfolio
        mu: An n-dimensional vector of expected returns for each asset
        GT: A factor matrix such that (GT')*GT = covariance matrix
        x0: Initial holdings for each asset
        w: Initial cash holding
        alphas: List of risk aversion parameters to use

    Returns:
        A list of tuples (alpha, expected return, variance) representing points
        on the efficient frontier
    """
    # Create a MOSEK Fusion model for the efficient frontier computation
    with Model("Efficient frontier") as M:
        frontier = []

        # Define the portfolio weight variables
        # Shortselling is not allowed, so weights must be non-negative
        x = M.variable("x", n, Domain.greaterThan(0.0))  # Portfolio weights
        s = M.variable("s", 1, Domain.unbounded())  # Variable representing portfolio variance

        # Add the budget constraint: sum of weights equals initial wealth
        M.constraint("budget", Expr.sum(x), Domain.equalsTo(w + sum(x0)))

        # Add the variance computation constraint using a rotated quadratic cone
        # This efficiently represents: s >= x^T * Sigma * x, where Sigma = GT^T * GT
        M.constraint("variance", Expr.vstack(s, 0.5, Expr.mul(GT, x)), Domain.inRotatedQCone())

        # Define the objective function: maximize return - alpha * variance
        # Alpha is a parameter that will be varied to generate different points on the frontier
        alpha = M.parameter()
        M.objective(
            "obj",
            ObjectiveSense.Maximize,
            Expr.sub(Expr.dot(mu, x), Expr.mul(alpha, s)),
        )

        # Solve the optimization problem for each alpha value
        for a in alphas:
            # Set the current alpha value
            alpha.setValue(a)

            # Solve the optimization problem
            M.solve()

            # Record the solution point (alpha, expected return, variance)
            frontier.append((a, np.dot(mu, x.level()), s.level()[0]))

        return frontier


if __name__ == "__main__":
    # Define the portfolio problem parameters
    n = 8  # Number of assets
    w = 1.0  # Initial wealth

    # Expected returns for each asset
    mu = [0.07197, 0.15518, 0.17535, 0.08981, 0.42896, 0.39292, 0.32171, 0.18379]

    # Initial holdings (all zero in this example)
    x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Factor matrix GT such that Sigma = GT' * GT is the covariance matrix
    # This is a lower triangular matrix (Cholesky factor of the covariance)
    GT = [
        [0.30758, 0.12146, 0.11341, 0.11327, 0.17625, 0.11973, 0.10435, 0.10638],
        [0.0, 0.25042, 0.09946, 0.09164, 0.06692, 0.08706, 0.09173, 0.08506],
        [0.0, 0.0, 0.19914, 0.05867, 0.06453, 0.07367, 0.06468, 0.01914],
        [0.0, 0.0, 0.0, 0.20876, 0.04933, 0.03651, 0.09381, 0.07742],
        [0.0, 0.0, 0.0, 0.0, 0.36096, 0.12574, 0.10157, 0.0571],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.21552, 0.05663, 0.06187],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22514, 0.03327],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2202],
    ]

    # Define a range of risk aversion parameters (alphas) to trace the efficient frontier
    # Alpha = 0 corresponds to maximum return portfolio (risk-neutral)
    # Higher alpha values give more weight to risk reduction
    alphas = [
        0.0,  # Risk-neutral (maximum return)
        0.01,  # Very low risk aversion
        0.1,  # Low risk aversion
        0.25,
        0.30,
        0.35,
        0.4,
        0.45,
        0.5,  # Moderate risk aversion
        0.75,
        1.0,  # High risk aversion
        1.5,
        2.0,
        3.0,
        10.0,  # Very high risk aversion
    ]

    # Compute the efficient frontier
    frontier = EfficientFrontier(n, mu, GT, x0, w, alphas)

    # Print the results in a formatted table
    print("\n-----------------------------------------------------------------------------------")
    print("Efficient frontier")
    print("-----------------------------------------------------------------------------------\n")
    print("%-12s  %-12s  %-12s" % ("alpha", "return", "risk (std. dev.)"))

    # Print each point on the frontier with its alpha, expected return, and standard deviation
    for i in frontier:
        print("{:<12.4f}  {:<12.4e}  {:<12.4e}".format(i[0], i[1], np.sqrt(i[2])))

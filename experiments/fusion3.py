##
# Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
#
# File:      portfolio_2_frontier.py
#
#  Purpose :   Implements a basic portfolio optimization model.
#              Computes points on the efficient frontier.
#
##


import numpy as np
from mosek.fusion import Domain, Expr, Model, ObjectiveSense

"""
Purpose:
    Computes several portfolios on the optimal portfolios by

        for alpha in alphas:
            maximize   expected return - alpha * variance
            subject to the constraints

Input:
    n: Number of assets
    mu: An n dimensional vector of expected returns
    GT: A matrix with n columns so (GT')*GT  = covariance matrix
    x0: Initial holdings
    w: Initial cash holding
    alphas: List of the alphas

Output:
    The efficient frontier as list of tuples (alpha, expected return, variance)
"""


def EfficientFrontier(n, mu, GT, x0, w, alphas):
    with Model("Efficient frontier") as M:
        frontier = []

        # Defines the variables (holdings). Shortselling is not allowed.
        x = M.variable("x", n, Domain.greaterThan(0.0))  # Portfolio variables
        s = M.variable("s", 1, Domain.unbounded())  # Variance variable

        # Total budget constraint
        M.constraint("budget", Expr.sum(x), Domain.equalsTo(w + sum(x0)))

        # Computes the risk
        M.constraint("variance", Expr.vstack(s, 0.5, Expr.mul(GT, x)), Domain.inRotatedQCone())

        # Define objective as a weighted combination of return and variance
        alpha = M.parameter()
        M.objective(
            "obj",
            ObjectiveSense.Maximize,
            Expr.sub(Expr.dot(mu, x), Expr.mul(alpha, s)),
        )

        # Solve multiple instances by varying the parameter alpha
        for a in alphas:
            alpha.setValue(a)

            M._solve()

            frontier.append((a, np.dot(mu, x.level()), s.level()[0]))

        return frontier


if __name__ == "__main__":
    n = 8
    w = 1.0
    mu = [0.07197, 0.15518, 0.17535, 0.08981, 0.42896, 0.39292, 0.32171, 0.18379]
    x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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

    # Some predefined alphas are chosen
    alphas = [
        0.0,
        0.01,
        0.1,
        0.25,
        0.30,
        0.35,
        0.4,
        0.45,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        10.0,
    ]
    frontier = EfficientFrontier(n, mu, GT, x0, w, alphas)
    print("\n-----------------------------------------------------------------------------------")
    print("Efficient frontier")
    print("-----------------------------------------------------------------------------------\n")
    print("%-12s  %-12s  %-12s" % ("alpha", "return", "risk (std. dev.)"))
    for i in frontier:
        print("{:<12.4f}  {:<12.4e}  {:<12.4e}".format(i[0], i[1], np.sqrt(i[2])))

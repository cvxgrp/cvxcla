"""
Portfolio Optimization Example - Maximum Expected Return

This script demonstrates how to solve a simple portfolio optimization problem using MOSEK's Fusion API.
The problem maximizes the expected return of a portfolio subject to:
    - Lower and upper bounds on individual asset weights
    - Fully-invested constraint (sum of weights equals 1)

The input data is loaded from a BSON file containing:
    - mean: Expected returns for each asset
    - lower_bounds: Minimum allowed weight for each asset
    - upper_bounds: Maximum allowed weight for each asset
"""

from mosek.fusion import Domain, Expr, Model, ObjectiveSense

from cvx.bson import read_bson

if __name__ == "__main__":
    # Load portfolio data from BSON file
    data = read_bson(file="data/input_data.bson")

    # Create a model for the maximum expected return portfolio optimization problem
    with Model("max_expected_return") as M:
        # Create variable 'x' representing portfolio weights
        # The weights are constrained by the lower and upper bounds from the data
        x = M.variable(
            "x",
            len(data["mean"]),
            Domain.inRange(data["lower_bounds"], data["upper_bounds"]),
        )

        # Add the fully-invested constraint: sum of all weights must equal 1
        M.constraint("fully-invested", Expr.sum(x), Domain.equalsTo(1.0))

        # Set the objective function to maximize the expected return (mean^T * x)
        M.objective("expected_return", ObjectiveSense.Maximize, Expr.dot(data["mean"], x))

        # Solve the optimization problem
        M.solve()

        # Get the optimal portfolio weights
        sol = x.level()

        # Print the optimal portfolio weights
        print("Optimal portfolio weights:")
        print(sol)

"""
Basic Linear Optimization Example using MOSEK Fusion API

This script demonstrates how to solve a simple linear programming problem using MOSEK's Fusion API.
The problem is formulated as:
    maximize c^T * x
    subject to:
        x[1] <= 10.0
        A[0] * x == 30.0
        A[1] * x >= 15.0
        A[2] * x <= 25.0
        x >= 0
"""

from mosek.fusion import Domain, Expr, Model, ObjectiveSense

if __name__ == "__main__":
    # Define the coefficient matrix for constraints
    A = [[3.0, 1.0, 2.0, 0.0], [2.0, 1.0, 3.0, 1.0], [0.0, 2.0, 0.0, 3.0]]
    # Define the objective function coefficients
    c = [3.0, 1.0, 5.0, 1.0]

    # Create a model with the name 'lo1'
    with Model("lo1") as M:
        # Create variable 'x' of length 4
        x = M.variable("x", 4, Domain.greaterThan(0.0))

        # Create constraints
        # Constraint: x[1] <= 10.0 (upper bound on the second variable)
        M.constraint(x.index(1), Domain.lessThan(10.0))
        # Constraint: A[0] * x == 30.0 (equality constraint)
        M.constraint("c1", Expr.dot(A[0], x), Domain.equalsTo(30.0))
        # Constraint: A[1] * x >= 15.0 (lower bound constraint)
        M.constraint("c2", Expr.dot(A[1], x), Domain.greaterThan(15.0))
        # Constraint: A[2] * x <= 25.0 (upper bound constraint)
        M.constraint("c3", Expr.dot(A[2], x), Domain.lessThan(25.0))

        # Set the objective function to maximize (c^T * x)
        M.objective("obj", ObjectiveSense.Maximize, Expr.dot(c, x))

        # Solve the optimization problem
        M.solve()

        # Get the optimal solution values
        sol = x.level()
        # Print the solution values for each variable
        print("\n".join(["x[%d] = %f" % (i, sol[i]) for i in range(4)]))

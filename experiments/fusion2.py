from mosek.fusion import Domain, Expr, Model, ObjectiveSense

from cvx.bson import read_bson

if __name__ == "__main__":
    data = read_bson(file="data/input_data.bson")

    # Create a model with the name 'lo1'
    with Model("max_expected_return") as M:
        # Create variable 'x' of length 4
        x = M.variable(
            "x",
            len(data["mean"]),
            Domain.inRange(data["lower_bounds"], data["upper_bounds"]),
        )

        M.constraint("fully-invested", Expr.sum(x), Domain.equalsTo(1.0))

        M.objective("expected_return", ObjectiveSense.Maximize, Expr.dot(data["mean"], x))

        # Solve the problem
        M._solve()

        # Get the solution values
        sol = x.level()

        print(sol)

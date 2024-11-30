from mosek.fusion import Domain, Expr, Model, ObjectiveSense

if __name__ == "__main__":
    A = [[3.0, 1.0, 2.0, 0.0], [2.0, 1.0, 3.0, 1.0], [0.0, 2.0, 0.0, 3.0]]
    c = [3.0, 1.0, 5.0, 1.0]

    # Create a model with the name 'lo1'
    with Model("lo1") as M:
        # Create variable 'x' of length 4
        x = M.variable("x", 4, Domain.greaterThan(0.0))

        # Create constraints
        M.constraint(x.index(1), Domain.lessThan(10.0))
        M.constraint("c1", Expr.dot(A[0], x), Domain.equalsTo(30.0))
        M.constraint("c2", Expr.dot(A[1], x), Domain.greaterThan(15.0))
        M.constraint("c3", Expr.dot(A[2], x), Domain.lessThan(25.0))

        # Set the objective function to (c^t * x)
        M.objective("obj", ObjectiveSense.Maximize, Expr.dot(c, x))

        # Solve the problem
        M._solve()

        # Get the solution values
        sol = x.level()
        print("\n".join(["x[%d] = %f" % (i, sol[i]) for i in range(4)]))

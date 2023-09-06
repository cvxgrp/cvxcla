import numpy as np


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

    # We can solve this problem using Lagrange multipliers

    # -gamma*(sum w - 1)
    # w - s^2 = 0
    # -theta_1*(w_1 - s_1^2)
    # -theta_2*(w_2 - s_2^2)

    # For now let's drop the constraint of long-only

    # from

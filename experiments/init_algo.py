"""
Initial Algorithm Comparison Experiment

This script compares two different implementations of the algorithm for finding
the first turning point in the Critical Line Algorithm (CLA) for portfolio optimization:
1. init_algo: The standard implementation
2. init_algo_lp: An implementation using linear programming

Both algorithms compute the first turning point on the efficient frontier,
which is the portfolio with the highest expected return that satisfies the constraints.

The experiment uses a large-scale problem with 10,000 assets, all having the same
expected return and the same bounds, to test the performance and behavior of both implementations.
"""

import numpy as np
from loguru import logger

from cvx.cla.first import init_algo, init_algo_lp

if __name__ == "__main__":
    # Define a large-scale portfolio problem with 10,000 assets
    n = 10000
    # All assets have the same expected return
    mean = np.ones(n)
    # All assets have the same upper bound of 1.0
    upper_bound = np.ones(n)

    logger.info("Starting comparison of initial turning point algorithms")

    # Compute the first turning point using the standard implementation
    tp = init_algo(mean=mean, lower_bounds=np.zeros_like(upper_bound), upper_bounds=upper_bound)
    # Print the indices of free variables in the solution
    print("Free variables from init_algo:")
    print(np.where(tp.free)[0])

    # Compute the first turning point using the linear programming implementation
    tp = init_algo_lp(mean=mean, lower_bounds=np.zeros_like(upper_bound), upper_bounds=upper_bound)
    # Print the indices of free variables in the solution
    print("Free variables from init_algo_lp:")
    print(np.where(tp.free)[0])

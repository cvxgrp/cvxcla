import numpy as np

from cvx.cla import Frontier
from cvx.cla.plotting import plot_efficient_frontiers

if __name__ == "__main__":
    n = 20
    mean = np.random.randn(n)
    lower_bounds = np.zeros_like(mean)
    upper_bounds = np.ones_like(mean)

    factor = np.random.randn(n, n)
    covariance = factor @ factor.T

    f = Frontier.construct(
        mean=mean,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        covariance=covariance,
        name="Wurst",
    )

    fig = plot_efficient_frontiers([f])
    fig.show()

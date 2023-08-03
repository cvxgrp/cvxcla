import matplotlib.pyplot as plt
import numpy as np

from cvx.cla import Frontier
from cvx.cla.plotting import plot_efficient_frontiers

if __name__ == '__main__':
    #np.random.seed(42)

    plt.Figure()

    n = 20
    mean = np.random.randn(n)
    lower_bounds = np.zeros_like(mean)
    upper_bounds = np.ones_like(mean)

    factor = np.random.randn(n, n)
    covariance = factor @ factor.T

    f = Frontier.construct(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds, covariance=covariance, name="Wurst")


    plot_efficient_frontiers([f])
    print(f.variance[-1])
    print(f.returns[-1])

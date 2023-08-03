from typing import List

from matplotlib import pyplot as plt
from cvx.cla import Frontier


def plot_efficient_frontiers(frontiers: List[Frontier]) -> None:

    plt.figure()

    for frontier in frontiers:
        plt.plot(frontier.variance, frontier.returns, label=frontier.name)

    plt.title("Efficient Frontier")
    plt.xlabel("variance")
    plt.ylabel("mu")

    plt.legend()
    plt.show()

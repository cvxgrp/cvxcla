import numpy as np

from cvx.cla import Frontier
from cvx.cla.plotting import plot_efficient_frontiers
from cvx.cla.solver import Solver


def test_plot(input_data):
    f_markowitz = Frontier.build(
        Solver.MARKOWITZ,
        mean=input_data.mean,
        lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds,
        covariance=input_data.covariance,
        name="MARKOWITZ",
        A=np.ones((1, len(input_data.mean))),
        b=np.ones(1),
    )

    f_bailey = Frontier.build(
        Solver.BAILEY,
        mean=input_data.mean,
        lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds,
        covariance=input_data.covariance,
        name="BAILEY",
        A=np.ones((1, len(input_data.mean))),
        b=np.ones(1),
    )

    fig = plot_efficient_frontiers([f_markowitz, f_bailey])
    assert fig
    # fig.show()

import numpy as np

from cvx.cla.markowitz.cla import CLA as MARKOWITZ
from cvx.cla.plotting import plot_efficient_frontiers
from tests.bailey.cla import CLA as BAILEY


def test_plot(input_data):
    f_markowitz = MARKOWITZ(
        mean=input_data.mean,
        lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds,
        covariance=input_data.covariance,
        A=np.ones((1, len(input_data.mean))),
        b=np.ones(1),
    ).frontier(name="MARKOWITZ")

    f_bailey = BAILEY(
        mean=input_data.mean,
        lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds,
        covariance=input_data.covariance,
        A=np.ones((1, len(input_data.mean))),
        b=np.ones(1),
    ).frontier(name="BAILEY")

    fig = plot_efficient_frontiers([f_markowitz, f_bailey])
    assert fig
    # fig.show()

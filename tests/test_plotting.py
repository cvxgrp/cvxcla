from cvx.cla import Frontier
from cvx.cla.plotting import plot_efficient_frontiers

def test_plot(input_data):
    f = Frontier.construct(
        mean=input_data.mean, lower_bounds=input_data.lower_bounds,
        upper_bounds=input_data.upper_bounds, covariance=input_data.covariance, name="test"
    )

    fig = plot_efficient_frontiers([f])
    assert fig

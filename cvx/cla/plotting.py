from typing import List

from cvx.cla import Frontier
import plotly.graph_objects as go
def plot_efficient_frontiers(frontiers: List[Frontier]) -> None:
    # Create an empty list to store the traces
    traces = []

    # Loop through each set of y-values
    for frontier in frontiers:
        # Create a trace for each curve
        trace = go.Scatter(
            x=frontier.variance,
            y=frontier.returns,
            name=frontier.name,
        )
        # Add the trace to the list
        traces.append(trace)

    # Create the layout object
    layout = go.Layout(
        title='Efficient Frontier(s)',
        xaxis=dict(title='Expected Return'),
        yaxis=dict(title='Expected Variance'),
    )

    # Create the figure object
    fig = go.Figure(data=traces, layout=layout)

    # Open the figure
    fig.show()

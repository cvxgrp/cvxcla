#    Copyright 2023 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import List

import plotly.graph_objects as go

from cvx.cla.frontier import Frontier


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
        title="Efficient Frontier(s)",
        xaxis=dict(title="Expected Return"),
        yaxis=dict(title="Expected Variance"),
    )

    # Create the figure object
    return go.Figure(data=traces, layout=layout)

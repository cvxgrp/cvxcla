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
"""
Plotting utilities for the Critical Line Algorithm.

This module provides functions for visualizing the efficient frontier
and other results from the Critical Line Algorithm.
"""

import plotly.express as px
import plotly.graph_objects as go

from .types import Frontier


def plot_frontier(frontier: Frontier, volatility: bool = False, markers: bool = True) -> go.Figure:
    """
    Plot the efficient frontier.

    This function creates a line plot of the efficient frontier, with expected return
    on the y-axis and either variance or volatility on the x-axis.

    Args:
        frontier: A Frontier object containing the points to plot.
        volatility: If True, plot volatility (standard deviation) on the x-axis.
                   If False, plot variance on the x-axis.
        markers: If True, show markers at each point on the frontier.

    Returns:
        A plotly Figure object that can be displayed or saved.
    """
    if not volatility:
        fig = px.line(
            x=frontier.variance,
            y=frontier.returns,
            markers=markers,
            labels={"x": "Expected variance", "y": "Expected Return"},
        )
    else:
        fig = px.line(
            x=frontier.volatility,
            y=frontier.returns,
            markers=markers,
            labels={"x": "Expected volatility", "y": "Expected Return"},
        )
    return fig

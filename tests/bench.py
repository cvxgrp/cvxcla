import time

import numpy as np
import plotly.graph_objects as go

from cvx.cla.markowitz.cla import CLA as MARKOWITZ
from tests.bailey.cla import CLA as BAILEY


def f(solver, n):
    mean = np.random.randn(n)
    A = np.random.randn(n, n)
    sigma = 0.1

    t = time.time()
    solver(
        mean=mean,
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
        covariance=A @ A.T + sigma * np.eye(n),
        A=np.ones((1, len(mean))),
        b=np.ones(1),
    )
    return time.time() - t


if __name__ == "__main__":
    n = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])

    y1 = np.array([f(MARKOWITZ, k) for k in n])
    y2 = np.array([f(BAILEY, k) for k in n])

    print(y1)
    print(y2)

    # print(times)

    # y1 = np.array([f(k) for k in n])

    # f(k)
    # n = [2,5,10]
    # y1 = [3,8,20]
    # y2 = [4,9,21]

    fig = go.Figure()

    fig.add_scatter(x=n, y=y1, name="Schmelzer Schiele")
    fig.add_scatter(x=n, y=y2, name="Bailey LdP")

    fig.update_layout(
        title="Runtime",
        xaxis_title="n --- Number of Assets",
        yaxis_title="time[sec]",
        legend_title="Algorithm",
        font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
    )

    # fig.update_layout(labels={"x": "n", "y": "time"})
    # figlabels = {"x": "Expected variance", "y": "Expected Return"},
    fig.show()

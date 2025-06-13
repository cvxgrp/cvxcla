# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.13.15",
#     "numpy==2.3.0",
#     "cvxcla==1.1.7",
# ]
# ///
"""Little demo for the Critical Line Algorithm."""

import marimo

__generated_with = "0.13.15"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np

    import cvxcla as solver
    # from cvxcla import CLA


@app.cell
def _():
    mo.md(
        r"""
    # The Critical Line Algorithm
    We compute an efficient frontier using the critical line algorithm (cla).
    The method was introduced by Harry M Markowitz in 1956.
    """
    )
    return


@app.cell
def _():
    slider = mo.ui.slider(4, 100, step=1, value=10, label="Size of the problem")
    return (slider,)


@app.cell(hide_code=True)
def _(slider):
    n = slider.value
    mean = np.random.randn(n)
    lower_bounds = np.zeros(n)
    upper_bounds = np.ones(n)

    factor = np.random.randn(n, n)
    covariance = factor @ factor.T

    f1 = solver.CLA(
        mean=mean,
        covariance=covariance,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        A=np.ones((1, len(mean))),
        b=np.ones(1),
    ).frontier
    return (f1,)


@app.cell
def _(f1):
    f1.interpolate(10).plot(volatility=True, markers=True)
    return


@app.cell
def _(f1):
    f1.plot()
    return


if __name__ == "__main__":
    app.run()

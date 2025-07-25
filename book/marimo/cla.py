# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.14.13",
#     "numpy==2.3.0",
#     "cvxcla==1.3.2"
# ]
# ///
"""Little demo for the Critical Line Algorithm."""

import marimo

__generated_with = "0.14.13"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np

    from cvxcla import CLA


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
    # display the slider
    slider
    return (slider,)


@app.function(hide_code=True)
def cla(n):
    """Compute using the Critical Line Algorithm (CLA) an efficient frontier.

    Args:
        n (int): The dimension size of the mean vector, lower and upper bounds
            arrays, and covariance matrix used in the computation.

    Returns:
        numpy.ndarray: The efficient frontier generated by the CLA based on the
            provided parameters.
    """
    mean = np.random.randn(n)
    lower_bounds = np.zeros(n)
    upper_bounds = np.ones(n)

    factor = np.random.randn(n, n)
    covariance = factor @ factor.T

    f1 = CLA(
        mean=mean,
        covariance=covariance,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        a=np.ones((1, len(mean))),
        b=np.ones(1),
    ).frontier
    return f1


@app.cell
def _(slider):
    frontier = cla(slider.value)
    frontier.interpolate(2).plot(volatility=True, markers=True)
    frontier.plot()
    return


if __name__ == "__main__":
    app.run()

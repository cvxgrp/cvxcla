import marimo

__generated_with = "0.9.27"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(r"""# Critical Line Algorithm""")
    return


@app.cell
def __(mo):
    mo.md(r"""We compute an efficient frontier using the critical line algorithm (cla)""")
    return


@app.cell
def __():
    import numpy as np

    from cvx.cla.markowitz.cla import CLA as MARKOWITZ

    return MARKOWITZ, np


@app.cell
def __(MARKOWITZ, np):
    n = 10
    mean = np.random.randn(n)
    lower_bounds = np.zeros(n)
    upper_bounds = np.ones(n)

    factor = np.random.randn(n, n)
    covariance = factor @ factor.T

    f1 = MARKOWITZ(
        mean=mean,
        covariance=covariance,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        A=np.ones((1, len(mean))),
        b=np.ones(1),
    ).frontier
    return covariance, f1, factor, lower_bounds, mean, n, upper_bounds


@app.cell
def __(f1):
    f1.interpolate(10).plot(volatility=True, markers=False)
    return


@app.cell
def __(f1):
    f1.plot()
    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

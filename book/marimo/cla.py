import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""# Critical Line Algorithm""")
    return


@app.cell
def _(mo):
    mo.md(r"""We compute an efficient frontier using the critical line algorithm (cla)""")
    return


@app.cell
def _():
    import numpy as np

    from cvx.cla import CLA
    from cvx.cla.plot import plot_frontier

    return CLA, np, plot_frontier


@app.cell
def _(CLA, np):
    n = 10
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
        A=np.ones((1, len(mean))),
        b=np.ones(1),
    ).frontier
    return (f1,)


@app.cell
def _(f1, plot_frontier):
    plot_frontier(f1.interpolate(10), volatility=True, markers=False)
    return


@app.cell
def _(f1, plot_frontier):
    plot_frontier(f1)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

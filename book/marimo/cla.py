import marimo

__generated_with = "0.13.15"
app = marimo.App(layout_file="layouts/cla.slides.json")


@app.cell
def _(mo):
    mo.md(r"""# Critical Line Algorithm""")
    return


@app.cell
def _(mo):
    mo.md(r"""We compute an efficient frontier using the critical line algorithm (cla)""")
    return


@app.cell(hide_code=True)
async def _():
    # | hide_cell
    import sys

    IS_WASM = sys.platform == "emscripten"

    print(f"WASM notebook: {IS_WASM}")

    if IS_WASM:
        import micropip

        await micropip.install("cvxcla")
        await micropip.install("plotly")
        await micropip.install("pandas")

    return


@app.cell
def _():
    import numpy as np

    from cvx.cla import CLA

    return CLA, np


@app.cell
def _(mo):
    slider = mo.ui.slider(4, 100, step=1, value=10, label="Size of the problem")
    slider
    return (slider,)


@app.cell(hide_code=True)
def _(CLA, np, slider):
    n = slider.value
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
def _(f1):
    f1.interpolate(10).plot(volatility=True, markers=True)
    return


@app.cell
def _(f1):
    f1.plot()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

"""Little demo for the Critical Line Algorithm."""

import marimo

__generated_with = "0.13.15"
app = marimo.App()  # layout_file="layouts/cla.slides.json")

with app.setup:
    import marimo as mo
    import numpy as np
    
@app.cell
def _():
    mo.md(r"""# Critical Line Algorithm""")
    return


@app.cell
def _():
    mo.md(r"""We compute an efficient frontier using the critical line algorithm (cla).
    The method was introduced by Harry M Markowitz in 1956.""")
    return


@app.cell(hide_code=True)
async def _():
    # | hide_cell
    import sys

    IS_WASM = sys.platform == "emscripten"

    print(f"WASM notebook: {IS_WASM}")

    if IS_WASM:
        import micropip

        # install the cvxcla package from PyPI
        await micropip.install("cvxcla")

    return


@app.cell
def _():

    from cvx.cla import CLA

    return CLA


@app.cell
def _():
    slider = mo.ui.slider(4, 100, step=1, value=10, label="Size of the problem")
    return slider

@app.cell(hide_code=True)
def _(CLA, slider):
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


if __name__ == "__main__":
    app.run()

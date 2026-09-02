# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.14.13",
#     "numpy==2.3.0",
#     "plotly==6.7.0",
#     "polars==1.44.1",
#     "jquantstats==0.11.0",
#     "cvx-linalg>=0.9.3",
#     "cvxcla"
# ]
#
# [tool.uv.sources]
# cvxcla = { path = "../../..", editable=true }
#
# ///
"""Little demo for the Critical Line Algorithm."""

import marimo

__generated_with = "0.14.13"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import polars as pl
    from jquantstats import Data

    from cvxcla import CLA

    # Trading days in a year, and the length of the simulated history.
    PERIODS = 260
    HISTORY = 4 * PERIODS


@app.cell
def _():
    mo.md(
        r"""
    # The Critical Line Algorithm
    We compute an efficient frontier using the critical line algorithm (cla).
    The method was introduced by Harry M Markowitz in 1956.

    Rather than invent a mean vector and a covariance matrix out of thin air, we
    simulate a return history, estimate both from it, and trace the frontier of
    the estimated problem. That way every portfolio on the frontier has a
    *realised* return series too, which we hand to
    [jQuantStats](https://github.com/jebel-quant/jquantstats) at the bottom of
    this notebook.
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
def business_dates(periods):
    """Build a Monday-to-Friday date index of the given length."""
    calendar = pl.date_range(pl.date(2015, 1, 1), pl.date(2035, 1, 1), interval="1d", eager=True)
    return calendar.filter(calendar.dt.weekday() <= 5).head(periods)


@app.function(hide_code=True)
def simulate(n, seed=42):
    """Simulate a daily return history for n assets.

    The assets differ in their true drift and load on a handful of common
    factors, so the estimated problem below has a genuinely tilted frontier
    instead of one driven purely by estimation noise.

    Args:
        n (int): Number of assets.
        seed (int): Seed for the random generator, so the notebook is reproducible.

    Returns:
        polars.DataFrame: A frame with a ``date`` column and one return column
            per asset, with ``HISTORY`` rows.
    """
    rng = np.random.default_rng(seed)

    # True annual drifts spread across the assets, expressed per day.
    drift = np.linspace(0.02, 0.20, n) / PERIODS
    # A low-rank common factor structure plus idiosyncratic noise.
    k = max(2, n // 10)
    exposures = rng.standard_normal((n, k)) * 0.4
    factors = rng.standard_normal((HISTORY, k)) * 0.01
    idiosyncratic = rng.standard_normal((HISTORY, n)) * 0.01

    returns = drift + factors @ exposures.T + idiosyncratic
    columns = [f"asset_{i:03d}" for i in range(n)]
    return pl.DataFrame({"date": business_dates(HISTORY), **dict(zip(columns, returns.T, strict=True))})


@app.function(hide_code=True)
def cla(returns):
    """Compute using the Critical Line Algorithm (CLA) an efficient frontier.

    The mean vector and the covariance matrix are the sample estimates taken
    from the simulated return history. The portfolios are long-only, capped at
    100% per name, and fully invested.

    Args:
        returns (polars.DataFrame): Return history with a leading ``date`` column.

    Returns:
        cvxcla.types.Frontier: The efficient frontier of the estimated problem.
    """
    matrix = returns.drop("date").to_numpy()
    n = matrix.shape[1]

    return CLA(
        mean=matrix.mean(axis=0),
        covariance=np.cov(matrix, rowvar=False),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
        a=np.ones((1, n)),
        b=np.ones(1),
    ).frontier


@app.cell
def _(slider):
    returns = simulate(slider.value)
    frontier = cla(returns)
    mo.md(f"The frontier of the estimated problem has **{len(frontier)}** turning points.")
    return frontier, returns


@app.cell
def _(frontier):
    frontier.plot(volatility=True, markers=True)
    return


@app.cell
def _():
    mo.md(
        r"""
    ## From weights to a track record

    The frontier is a set of weight vectors. Applied to the return history they
    generated, each one becomes a return series, and a return series is what
    jQuantStats analyses. We look at three portfolios: the maximum-Sharpe point
    on the frontier, the minimum-variance point, and equal weight as a
    reference.

    These are *in-sample* numbers -- the same history produced the estimates the
    optimiser used -- so read the table as a description of the frontier, not as
    a backtest. `experiments/frontier_stats.py` runs the out-of-sample version
    on real S&P 500 data.
    """
    )
    return


@app.function(hide_code=True)
def track_records(frontier, returns):
    """Turn frontier portfolios into a jQuantStats `Data` object.

    Args:
        frontier (cvxcla.types.Frontier): The traced efficient frontier.
        returns (polars.DataFrame): The return history the frontier was estimated on.

    Returns:
        jquantstats.Data: The realised return series of the maximum-Sharpe, the
            minimum-variance and the equal-weight portfolio.
    """
    matrix = returns.drop("date").to_numpy()
    n = matrix.shape[1]

    _, max_sharpe = frontier.max_sharpe
    # Frontier order runs from maximum return towards minimum variance, but read
    # the minimiser off the variance vector rather than relying on that order.
    min_variance = frontier.weights[int(np.argmin(frontier.variance))]

    portfolios = {
        "max_sharpe": max_sharpe,
        "min_variance": min_variance,
        "equal_weight": np.full(n, 1.0 / n),
    }
    series = {name: matrix @ weights for name, weights in portfolios.items()}
    return Data.from_returns(pl.DataFrame({"date": returns["date"], **series}))


@app.cell
def _(frontier, returns):
    data = track_records(frontier, returns)
    return (data,)


@app.cell
def _(data):
    mo.ui.table(data.stats.summary(), selection=None)
    return


@app.cell
def _(data):
    data.plots.returns(title="Cumulative return of three frontier portfolios")
    return


if __name__ == "__main__":
    app.run()

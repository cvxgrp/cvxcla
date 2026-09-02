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
"""RMT-cleaned efficient frontier with the factor covariance backend."""

import marimo

__generated_with = "0.14.13"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import polars as pl
    from jquantstats import Data

    from cvxcla import CLA, FactorCovariance


@app.cell
def _():
    mo.md(
        r"""
    # From Marchenko-Pastur to Woodbury

    We trace the **exact** efficient frontier of an RMT-cleaned covariance
    without ever forming an $n \times n$ matrix.

    1. Simulate returns with a latent factor structure.
    2. Clip the sample eigenvalues at the Marchenko-Pastur edge.
    3. The cleaned covariance is diagonal-plus-low-rank,
       $\Sigma = \bar{d}\, I + V_k (\Lambda_k - \bar{d} I) V_k^\top$,
       which is exactly what `FactorCovariance` solves via the Woodbury
       identity in $O(nk)$ memory.
    4. Hand it to `CLA` and plot the frontier.
    5. Check the operator against the data: the volatility the Woodbury
       quadratic form predicts for a frontier portfolio has to equal the
       volatility its realised return series actually shows.
    """
    )
    return


@app.cell
def _():
    n_slider = mo.ui.slider(100, 1000, step=100, value=500, label="Number of assets")
    n_slider
    return (n_slider,)


@app.function(hide_code=True)
def simulate_returns(rng, t, n, k_true=10):
    """Simulate t observations of n asset returns with k_true latent factors."""
    exposures = rng.standard_normal((n, k_true)) * 0.3
    factor_returns = rng.standard_normal((t, k_true))
    idiosyncratic = rng.standard_normal((t, n))
    return factor_returns @ exposures.T + idiosyncratic


@app.function(hide_code=True)
def clip_covariance(returns):
    """Clean the sample covariance by Marchenko-Pastur eigenvalue clipping.

    Eigenvalues above the MP upper edge are kept; the remainder are replaced
    by their average, preserving the trace. The result is the
    diagonal-plus-low-rank model d * I + U @ diag(delta) @ U.T.
    """
    t, n = returns.shape
    sample = returns.T @ returns / t
    eigenvalues, eigenvectors = np.linalg.eigh(sample)

    # Marchenko-Pastur upper edge for variance sigma^2 and aspect ratio n/t
    noise_variance = np.median(eigenvalues) / (1 - np.sqrt(n / t)) ** 2
    edge = noise_variance * (1 + np.sqrt(n / t)) ** 2

    keep = eigenvalues > edge
    d_bar = float(eigenvalues[~keep].mean())

    u = eigenvectors[:, keep]
    delta = eigenvalues[keep] - d_bar
    return FactorCovariance(d=np.full(n, d_bar), u=u, delta=delta)


@app.cell
def _(n_slider):
    rng = np.random.default_rng(42)
    n = n_slider.value
    returns = simulate_returns(rng, t=2 * n, n=n)
    covariance = clip_covariance(returns)
    mo.md(f"Kept **{covariance.k}** factors out of {n} sample eigenvalues.")
    return covariance, n, returns, rng


@app.cell
def _(covariance, n, rng):
    frontier = CLA(
        mean=rng.uniform(0.0, 0.1, n),
        covariance=covariance,
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
        a=np.ones((1, n)),
        b=np.ones(1),
    ).frontier
    mo.md(f"The exact frontier has **{len(frontier)}** turning points.")
    return (frontier,)


@app.cell
def _(frontier):
    frontier.plot(volatility=True)
    return


@app.cell
def _():
    mo.md(
        r"""
    ## Does the operator agree with the data?

    `FactorCovariance` never forms the $n \times n$ matrix, so the volatilities
    plotted above come out of the Woodbury identity rather than a dense
    quadratic form. That is worth checking against the returns themselves.

    Below, each frontier portfolio is applied to the simulated history to give a
    realised return series, which
    [jQuantStats](https://github.com/jebel-quant/jquantstats) measures. The
    predicted column is $\sqrt{w^\top \Sigma w}$ from the operator; the
    realised column is the sample standard deviation of the series, and their
    ratio sits within a few percent of 1 across the whole slider range. Not
    exactly 1: clipping deliberately discards the part of the sample spectrum
    it calls noise, so the cleaned $\Sigma$ is *not* the sample covariance of
    this history. A few percent is that discarded noise. An order-of-magnitude
    gap, or one that widened with $n$, would instead point at the low-rank
    solve.

    Risk is all we ask of this table. The means fed to the CLA are a synthetic
    forecast unrelated to the simulation, and `simulate_returns` draws from a
    standard normal rather than at a realistic return magnitude, so the
    return-, Sharpe- and drawdown-based metrics jQuantStats also offers would
    be measuring the simulation's conventions rather than the frontier.
    `experiments/frontier_stats.py` reads those metrics off real S&P 500 data
    instead.
    """
    )
    return


@app.function(hide_code=True)
def score(frontier, returns):
    """Compare each portfolio's predicted volatility with its realised one.

    Args:
        frontier (cvxcla.types.Frontier): The traced efficient frontier.
        returns (numpy.ndarray): The t x n simulated return history.

    Returns:
        polars.DataFrame: One row per portfolio with the volatility the
            covariance operator predicts, the volatility jQuantStats measures on
            the realised series, and their ratio.
    """
    t, n = returns.shape
    # Frontier order runs from maximum return towards minimum variance, but read
    # the minimiser off the variance vector rather than relying on that order.
    min_variance = frontier.weights[int(np.argmin(frontier.variance))]
    portfolios = {
        "max_sharpe": frontier.max_sharpe[1],
        "min_variance": min_variance,
        "equal_weight": np.full(n, 1.0 / n),
    }

    calendar = pl.date_range(pl.date(2015, 1, 1), pl.date(2045, 1, 1), interval="1d", eager=True)
    dates = calendar.filter(calendar.dt.weekday() <= 5).head(t)
    series = {name: returns @ weights for name, weights in portfolios.items()}
    data = Data.from_returns(pl.DataFrame({"date": dates, **series}))

    # Per-observation volatility, so it is comparable with the frontier's own:
    # the simulation carries no annualisation convention.
    realised = data.stats.volatility(annualize=False)
    # The Woodbury operator's own quadratic form -- no dense matrix is formed.
    predicted = {name: float(np.sqrt(w @ frontier.covariance.matvec(w))) for name, w in portfolios.items()}
    return pl.DataFrame(
        {
            "portfolio": list(portfolios),
            "predicted volatility": [predicted[name] for name in portfolios],
            "realised volatility": [realised[name] for name in portfolios],
            "ratio": [realised[name] / predicted[name] for name in portfolios],
        }
    )


@app.cell
def _(frontier, returns):
    mo.ui.table(score(frontier, returns), selection=None)
    return


if __name__ == "__main__":
    app.run()

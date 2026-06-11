# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.14.13",
#     "numpy==2.3.0",
#     "plotly==6.7.0",
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
    return covariance, n, rng


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


if __name__ == "__main__":
    app.run()

"""Wall-clock benchmarks for the CLA covariance backends across problem shapes.

These measure the claim that the matrix-free backends are a *memory* play whose
*speed* depends entirely on the rank / observation count relative to ``n``:

* ``sample-covariance`` group -- ``DenseCovariance`` vs ``GramCovariance`` on the
  same sample-covariance problem, swept over ``T/n``. Both trace an identical
  frontier (the covariances are equal), so the only difference is the per-solve
  cost: dense slices a precomputed ``Sigma_FF`` while Gram re-forms it (or uses
  Woodbury in ``T``-space when ``T < n_free``). Expectation: dense wins for
  ``T >= n``; Gram catches up / wins only in the short-sample regime.

* ``factor-model`` group -- ``DenseCovariance`` vs ``FactorCovariance`` on a
  diagonal-plus-low-rank problem, swept over the factor count ``k``. Expectation:
  the factor backend wins for small ``k`` and *loses* as ``k`` approaches ``n``
  (a full-rank structured solve is more work than the plain dense matrix).

Run with ``make benchmark`` (these are excluded from the normal test run).
"""

from __future__ import annotations

import numpy as np
import pytest

from cvxcla import CLA, DenseCovariance, FactorCovariance, GramCovariance


def _long_only_problem(rng: np.random.Generator, n: int) -> dict:
    """A long-only, fully-invested CLA problem of size ``n`` (covariance added by caller)."""
    return {
        "mean": rng.uniform(0.0, 1.0, n),
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }


# (T, n): a long sample, a square one, and a short sample (T < n).
_SAMPLE_SHAPES = [(1500, 80), (120, 120), (30, 150)]


@pytest.mark.benchmark(group="sample-covariance")
@pytest.mark.parametrize(("t", "n"), _SAMPLE_SHAPES, ids=lambda v: f"{v}")
@pytest.mark.parametrize("backend", ["dense", "gram"])
def test_sample_covariance_backend(benchmark, backend: str, t: int, n: int) -> None:
    """Trace the frontier from a sample covariance via the dense vs the Gram backend.

    A ridge is added in the short-sample regime (``T < n``) so the otherwise
    rank-deficient covariance is positive definite for both backends; the dense
    reference uses exactly ``np.cov(returns) + ridge I``, which equals what
    ``GramCovariance`` represents, so both trace the same frontier.
    """
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((t, n))
    ridge = 0.0 if t >= n else 1e-2
    problem = _long_only_problem(rng, n)

    if backend == "dense":
        sigma = np.cov(returns, rowvar=False) + ridge * np.eye(n)
        covariance = DenseCovariance(sigma)
    else:
        covariance = GramCovariance(returns, ridge=ridge)

    result = benchmark(lambda: CLA(covariance=covariance, **problem))
    assert len(result) >= 2


# (n, k): low rank, mid rank, and (near) full rank.
_FACTOR_SHAPES = [(150, 3), (150, 40), (150, 150)]


@pytest.mark.benchmark(group="factor-model")
@pytest.mark.parametrize(("n", "k"), _FACTOR_SHAPES, ids=lambda v: f"{v}")
@pytest.mark.parametrize("backend", ["dense", "factor"])
def test_factor_model_backend(benchmark, backend: str, n: int, k: int) -> None:
    """Trace the frontier of a factor model via the dense vs the Woodbury factor backend.

    The dense reference materialises the same ``diag(d) + U Delta U^T`` matrix the
    factor backend represents, so both trace the same frontier and only the
    per-solve method differs.
    """
    rng = np.random.default_rng(0)
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 5.0, n)
    problem = _long_only_problem(rng, n)

    if backend == "dense":
        sigma = np.diag(d) + (u * delta) @ u.T
        covariance = DenseCovariance(sigma)
    else:
        covariance = FactorCovariance(d=d, u=u, delta=delta)

    result = benchmark(lambda: CLA(covariance=covariance, **problem))
    assert len(result) >= 2

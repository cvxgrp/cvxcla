"""Property-based tests for the Critical Line Algorithm.

Hypothesis generates random portfolio problems - including deliberately
degenerate ones with tied means, duplicated assets, and capped weights - and
checks invariants that must hold on every efficient frontier
(https://github.com/cvxgrp/cvxcla/issues/649):

- weights respect the bounds and sum to 1 at every turning point
- lambda is non-increasing along the frontier (within tol)
- expected return is non-increasing along the turning points
- the dense and FactorCovariance backends agree on the same problem
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cvxcla import CLA, FactorCovariance


def _spd_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    """Build a random symmetric positive definite covariance matrix."""
    l_matrix = rng.standard_normal((n, n))
    return l_matrix @ l_matrix.T + 0.05 * np.eye(n)


@st.composite
def problems(draw: st.DrawFn) -> dict:
    """Generate a random long-only fully-invested portfolio problem.

    The mean vector is drawn from one of three styles: generic random values,
    blocks of identical values (tied means), and values rounded to one decimal
    (many near-ties). Upper bounds are uniform and can be tight enough that
    several assets must sit at the cap.
    """
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    n = draw(st.integers(min_value=2, max_value=25))
    style = draw(st.sampled_from(["generic", "tied-blocks", "rounded"]))
    tight = draw(st.booleans())

    rng = np.random.default_rng(seed)
    covariance = _spd_matrix(rng, n)

    if style == "generic":
        mean = rng.standard_normal(n)
    elif style == "tied-blocks":
        block = max(1, n // 4)
        mean = np.resize(np.repeat(rng.standard_normal(block), 4), n)
    else:
        mean = np.round(rng.standard_normal(n), 1)

    upper = np.full(n, float(rng.uniform(2.0 / n, 1.0))) if tight and n >= 3 else np.ones(n)

    return {
        "mean": mean,
        "covariance": covariance,
        "lower_bounds": np.zeros(n),
        "upper_bounds": upper,
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }


@pytest.mark.property
class TestFrontierInvariants:
    """Invariants that every computed frontier must satisfy."""

    @given(problem=problems())
    @settings(max_examples=80, deadline=None)
    def test_turning_points_are_valid(self, problem):
        """Weights stay within bounds and sum to 1; lambda never increases beyond tol."""
        cla = CLA(**problem)
        tol = cla.tol

        weights = np.array([tp.weights for tp in cla.turning_points])
        assert np.all(weights >= problem["lower_bounds"] - tol)
        assert np.all(weights <= problem["upper_bounds"] + tol)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)

        lambdas = np.array([tp.lamb for tp in cla.turning_points])
        assert np.all(np.diff(lambdas) <= tol)

    @given(problem=problems())
    @settings(max_examples=80, deadline=None)
    def test_expected_return_is_non_increasing(self, problem):
        """The frontier is traced from the max-return portfolio downwards."""
        cla = CLA(**problem)
        returns = cla.frontier.returns
        assert np.all(np.diff(returns) <= 1e-6)


@pytest.mark.property
class TestBackendAgreement:
    """The dense and factor backends must produce the same frontier."""

    @given(
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        n=st.integers(min_value=2, max_value=20),
        k=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=40, deadline=None)
    def test_factor_matches_dense(self, seed, n, k):
        """A factor model solved via Woodbury matches its dense materialisation."""
        rng = np.random.default_rng(seed)
        d = rng.uniform(0.1, 1.0, n)
        u = rng.standard_normal((n, k))
        delta = rng.uniform(0.5, 2.0, k)

        factor = FactorCovariance(d=d, u=u, delta=delta)
        dense = np.diag(d) + u @ np.diag(delta) @ u.T

        problem = {
            "mean": rng.standard_normal(n),
            "lower_bounds": np.zeros(n),
            "upper_bounds": np.ones(n),
            "a": np.ones((1, n)),
            "b": np.ones(1),
        }
        cla_factor = CLA(covariance=factor, **problem)
        cla_dense = CLA(covariance=dense, **problem)

        assert len(cla_factor) == len(cla_dense)
        for tp_f, tp_d in zip(cla_factor.turning_points, cla_dense.turning_points, strict=True):
            np.testing.assert_allclose(tp_f.weights, tp_d.weights, atol=1e-6)

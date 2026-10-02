"""The CLA owns its array inputs: later changes to the caller's arrays do not reach it."""

import numpy as np
import pytest

from cvxcla import CLA, FactorCovariance


def _problem(n: int = 6, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((3 * n, n))
    return {
        "mean": rng.standard_normal(n),
        "covariance": x.T @ x / (3 * n),
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
        "g": np.eye(n)[:2],
        "h": np.array([0.5, 0.5]),
    }


@pytest.mark.parametrize("name", ["mean", "covariance", "lower_bounds", "upper_bounds", "a", "b", "g", "h"])
def test_mutating_an_input_after_construction_has_no_effect(name):
    """The frontier, returns and variances are unchanged when the caller edits an input."""
    data = _problem()
    cla = CLA(**data)
    returns, variance = cla.frontier.returns.copy(), cla.frontier.variance.copy()
    data[name][...] = 7.0
    np.testing.assert_array_equal(cla.frontier.returns, returns)
    np.testing.assert_array_equal(cla.frontier.variance, variance)


@pytest.mark.parametrize("name", ["mean", "covariance", "lower_bounds", "upper_bounds", "a", "b", "g", "h"])
def test_stored_inputs_are_read_only_copies(name):
    """Every stored array is a non-writeable float64 copy, never the caller's array."""
    data = _problem()
    cla = CLA(**data)
    stored = getattr(cla, name)
    assert stored is not data[name]
    assert not np.shares_memory(stored, data[name])
    assert stored.dtype == np.float64
    assert not stored.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        stored[...] = 0.0


def test_list_inputs_are_accepted_and_stored_as_arrays():
    """Plain Python sequences are converted, so the copy also normalises the type."""
    data = {k: v.tolist() for k, v in _problem().items()}
    cla = CLA(**data)
    assert isinstance(cla.mean, np.ndarray)
    assert not cla.mean.flags.writeable
    np.testing.assert_allclose(cla.frontier.weights.sum(axis=1), 1.0)


def test_quadratic_form_backend_is_passed_through():
    """A covariance backend is used as given, not copied."""
    rng = np.random.default_rng(1)
    n = 6
    factor = FactorCovariance(d=rng.uniform(0.1, 0.5, n), u=rng.standard_normal((n, 2)), delta=np.ones(2))
    cla = CLA(
        mean=rng.standard_normal(n),
        covariance=factor,
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
        a=np.ones((1, n)),
        b=np.ones(1),
    )
    assert cla.covariance is factor

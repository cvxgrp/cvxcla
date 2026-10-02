"""Non-finite data is refused at construction; infinite bounds remain allowed.

Before this check a NaN in ``mean`` traced a meaningless frontier without complaint,
an infinite ``mean`` entry collapsed the frontier to two points, and an infinite
covariance entry failed with an ``IndexError`` deep inside the first solve.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvxcla import CLA, DenseCovariance

N = 5


@pytest.fixture
def problem() -> dict:
    """A small long-only, fully invested problem."""
    rng = np.random.default_rng(0)
    f = rng.standard_normal((N, N))
    return {
        "mean": rng.uniform(0.0, 1.0, N),
        "covariance": f @ f.T + np.eye(N),
        "lower_bounds": np.zeros(N),
        "upper_bounds": np.ones(N),
        "a": np.ones((1, N)),
        "b": np.ones(1),
        "g": np.ones((1, N)),
        "h": np.array([2.0]),
    }


@pytest.mark.parametrize("name", ["mean", "a", "b", "g", "h"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_data_is_refused(problem, name, bad):
    """Any NaN or infinite entry in the returns or the constraints is a ValueError naming it."""
    value = np.array(problem[name], dtype=float)
    value.flat[0] = bad
    problem[name] = value
    with pytest.raises(ValueError, match=f"{name} must be finite"):
        CLA(**problem)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_covariance_is_refused(problem, bad):
    """A dense covariance with a NaN or infinite entry is refused when it is wrapped."""
    cov = problem["covariance"].copy()
    cov[0, 1] = cov[1, 0] = bad
    with pytest.raises(ValueError, match="Covariance must be finite"):
        DenseCovariance(cov)
    problem["covariance"] = cov
    with pytest.raises(ValueError, match="Covariance must be finite"):
        CLA(**problem)


@pytest.mark.parametrize("name", ["lower_bounds", "upper_bounds"])
def test_nan_bounds_are_refused(problem, name):
    """A NaN bound is refused; it is neither a bound nor the absence of one."""
    problem[name] = np.array(problem[name], dtype=float)
    problem[name][0] = np.nan
    with pytest.raises(ValueError, match=f"{name} must not contain NaN"):
        CLA(**problem)


def test_infinite_bounds_are_allowed(problem):
    """An infinite bound is an unbounded box and is traced as before."""
    problem["upper_bounds"] = np.full(N, np.inf)
    cla = CLA(**problem)
    assert len(cla) > 1
    assert all(np.isfinite(tp.weights).all() for tp in cla.turning_points)

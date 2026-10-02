"""Invariance of the trace under a change of units of the returns and the covariance.

Rescaling the expected returns by ``c > 0`` and the covariance by ``s > 0`` leaves
the efficient frontier unchanged: the optimiser of the rescaled problem at
``lambda`` is the original one at ``lambda * c / s``. The event tests must therefore
not depend on absolute magnitudes. Before the slope floors and the event-ordering
window were made relative to the problem's own scales, a small ``mu`` collapsed the
frontier to its endpoints, and a small ``s / c`` merged distinct events until the
tracer cycled into its iteration cap.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvxcla import CLA
from cvxcla._events import event_ratios, ineq_event_ratios
from cvxcla.pathtracer import select_next_event

SCALES = [1e-6, 1e-3, 1.0, 1e3, 1e6]


def _factor_problem(n: int = 30, k: int = 3, seed: int = 3) -> tuple[np.ndarray, np.ndarray, dict]:
    """A long-only, fully-invested problem on a small factor market, at realistic scale."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    cov = 1e-3 * (np.diag(rng.uniform(0.5, 2.0, n)) + (u * rng.uniform(0.5, 2.0, k) * n) @ u.T)
    mean = 1e-3 * rng.uniform(0.0, 1.0, n)
    bounds = {"lower_bounds": np.zeros(n), "upper_bounds": np.ones(n), "a": np.ones((1, n)), "b": np.ones(1)}
    return mean, cov, bounds


@pytest.fixture(scope="module")
def reference() -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
    """The unscaled problem and its traced turning points."""
    mean, cov, bounds = _factor_problem()
    cla = CLA(mean=mean, covariance=cov, **bounds)
    weights = np.array([tp.weights for tp in cla.turning_points])
    lambdas = np.array([tp.lamb for tp in cla.turning_points])
    return mean, cov, bounds, weights, lambdas


@pytest.mark.parametrize("c", SCALES)
@pytest.mark.parametrize("s", SCALES)
def test_turning_points_invariant_under_rescaling(reference, c, s):
    """Rescaling mu by c and Sigma by s keeps every turning point, with lambda scaled by s / c."""
    mean, cov, bounds, weights, lambdas = reference
    cla = CLA(mean=c * mean, covariance=s * cov, **bounds)
    scaled_weights = np.array([tp.weights for tp in cla.turning_points])
    scaled_lambdas = np.array([tp.lamb for tp in cla.turning_points])

    assert scaled_weights.shape == weights.shape
    np.testing.assert_allclose(scaled_weights, weights, atol=1e-10)
    finite = np.isfinite(lambdas) & (lambdas > 0)
    np.testing.assert_allclose(scaled_lambdas[finite] * c / s, lambdas[finite], rtol=1e-9)


def test_lambda_scale_is_homogeneous(reference):
    """The lambda scale moves with s / c, which is what makes the tests unit-free."""
    mean, cov, bounds, _, _ = reference
    base = CLA(mean=mean, covariance=cov, **bounds).lambda_scale
    scaled = CLA(mean=1e3 * mean, covariance=1e-2 * cov, **bounds).lambda_scale
    assert scaled == pytest.approx(base * 1e-2 / 1e3)


def test_event_ratios_honour_custom_floors():
    """A slope above the default floor is ignored when the floor is raised, and vice versa."""
    slope = 1e-7  # above sqrt(eps) ~ 1.5e-8
    common = {
        "r_alpha": np.array([0.5]),
        "gamma": np.array([1.0]),
        "free_in": np.array([True]),
        "at_upper": np.array([False]),
        "at_lower": np.array([False]),
        "lower": np.zeros(1),
        "upper": np.ones(1),
    }
    default = event_ratios(r_beta=np.array([-slope]), delta=np.zeros(1), **common)
    raised = event_ratios(r_beta=np.array([-slope]), delta=np.zeros(1), beta_floor=1e-6, **common)
    assert np.isfinite(default[0, 0])
    assert raised[0, 0] == -np.inf

    blocked = common | {"free_in": np.array([False]), "at_lower": np.array([True])}
    tiny = 1e-10  # below sqrt(eps), above a lowered floor
    assert event_ratios(r_beta=np.zeros(1), delta=np.array([tiny]), **blocked)[0, 3] == -np.inf
    lowered = event_ratios(r_beta=np.zeros(1), delta=np.array([tiny]), delta_floor=1e-12, **blocked)
    assert np.isfinite(lowered[0, 3])


def test_ineq_event_ratios_honour_custom_floors():
    """The row events use their own slack and multiplier floors."""
    g, h = np.ones((1, 2)), np.array([1.0])
    r_alpha, r_beta = np.array([0.2, 0.2]), np.array([-5e-11, -5e-11])  # slack slope -1e-10
    inactive = np.array([False])
    assert ineq_event_ratios(r_alpha, r_beta, np.zeros(1), np.zeros(1), inactive, g, h)[0, 0] == -np.inf
    entered = ineq_event_ratios(r_alpha, r_beta, np.zeros(1), np.zeros(1), inactive, g, h, slack_floor=1e-12)
    assert np.isfinite(entered[0, 0])

    active = np.array([True])
    eta_alpha, eta_beta = np.array([1e-10]), np.array([1e-10])
    assert ineq_event_ratios(r_alpha, r_beta, eta_alpha, eta_beta, active, g, h)[0, 1] == -np.inf
    released = ineq_event_ratios(r_alpha, r_beta, eta_alpha, eta_beta, active, g, h, eta_floor=1e-12)
    assert np.isfinite(released[0, 1])


def test_select_next_event_window_follows_scale():
    """Near lambda = 0 the tie window is relative to the lambda scale, not an absolute 1."""
    l_mat = np.full((2, 1), -np.inf)
    l_mat[0, 0] = 1.0e-9
    l_mat[1, 0] = 1.0e-9 - 5e-11  # distinct at the scale 1e-8, tied under an absolute floor
    # With the absolute floor of 1 the window is 1e-10, so both count as tied and the
    # lower index wins even though row 1 is not the largest ratio; at scale 1e-8 the
    # window is 1e-18 and only the true maximum remains.
    assert select_next_event(l_mat, lam=1.0, tol=1e-5)[0] == 0
    l_mat[0, 0], l_mat[1, 0] = 1.0e-9 - 5e-11, 1.0e-9
    assert select_next_event(l_mat, lam=1.0, tol=1e-5)[0] == 0  # merged: lowest index
    assert select_next_event(l_mat, lam=1.0, tol=1e-5, scale=1e-8)[0] == 1  # resolved

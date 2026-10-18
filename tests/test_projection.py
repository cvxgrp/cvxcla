"""Tests for the feasibility projection's convergence test and failure policy."""

from __future__ import annotations

import numpy as np
import pytest

import cvxcla
from cvxcla._projection import project_alternating, project_capped_simplex
from cvxcla.errors import ProjectionError


def test_capped_simplex_with_infinite_upper_bounds():
    """Uncapped long-only weights a hair below zero come back finite and feasible (issue #949)."""
    weights = np.array([-2.8e-17, 0.3, 0.7 + 2.8e-17])
    projected = project_capped_simplex(weights, np.zeros(3), np.full(3, np.inf), 1.0)
    assert np.all(np.isfinite(projected))
    assert np.all(projected >= 0.0)
    assert projected.sum() == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(projected, weights, atol=1e-14)


def test_capped_simplex_with_unbounded_box_and_a_distant_candidate():
    """Infinite bounds on both sides, and a candidate far off the budget: the bracket widens."""
    weights = np.array([50.0, -20.0, 10.0])
    projected = project_capped_simplex(weights, np.array([-np.inf, 0.0, -np.inf]), np.full(3, np.inf), 1.0)
    assert np.all(np.isfinite(projected))
    assert projected[1] >= 0.0
    assert projected.sum() == pytest.approx(1.0, abs=1e-12)


def test_alternating_projection_returns_a_feasible_point():
    """A round-off-sized box violation is cleared and the equality is kept."""
    lower, upper = np.zeros(3), np.ones(3)
    c, d = np.array([[1.0, 1.0, 1.0]]), np.array([1.0])
    weights = np.array([-1e-15, 0.5, 0.5 + 1e-15])  # on the budget, a hair outside the box
    projected = project_alternating(weights, lower, upper, c, d)
    assert np.all(projected >= lower)
    assert np.all(projected <= upper)
    np.testing.assert_allclose(c @ projected, d, atol=1e-14)
    np.testing.assert_allclose(projected, weights, atol=1e-14)


def test_alternating_projection_raises_when_it_cannot_converge():
    """An empty intersection of the box and the affine set is reported, not hidden."""
    lower, upper = np.zeros(2), np.ones(2)
    c, d = np.array([[1.0, 1.0]]), np.array([3.0])  # sum = 3 is unreachable inside [0, 1]^2
    with pytest.raises(ProjectionError, match="did not converge") as info:
        project_alternating(np.array([1.5, 1.5]), lower, upper, c, d)
    assert "box violation" in str(info.value)
    assert "equality residual" in str(info.value)


def test_projection_error_is_public_and_a_runtime_error():
    """Callers can catch it by name, and code that catches RuntimeError still does."""
    assert cvxcla.ProjectionError is ProjectionError
    assert issubclass(ProjectionError, RuntimeError)

"""Tests for the LASSO path traced by the generic parametric active-set tracer.

The path is validated two independent ways: against the LASSO subgradient (KKT)
optimality conditions, and against a from-scratch coordinate-descent solver. Both
the enter and the leave event families are exercised.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from cvxcla.lasso import Breakpoint, Lasso


def _coordinate_descent(x: np.ndarray, y: np.ndarray, lam: float, iters: int = 5000) -> np.ndarray:
    """An independent LASSO solver (cyclic coordinate descent with soft-threshold)."""
    n = x.shape[1]
    beta = np.zeros(n)
    col_sq = np.sum(x * x, axis=0)
    r = y - x @ beta
    for _ in range(iters):
        max_change = 0.0
        for j in range(n):
            if col_sq[j] == 0.0:
                continue
            rho = x[:, j] @ r + col_sq[j] * beta[j]
            new = np.sign(rho) * max(abs(rho) - lam, 0.0) / col_sq[j]
            if new != beta[j]:
                r += x[:, j] * (beta[j] - new)
                max_change = max(max_change, abs(new - beta[j]))
                beta[j] = new
        if max_change < 1e-12:
            break
    return beta


def _kkt_violation(x: np.ndarray, y: np.ndarray, beta: np.ndarray, lam: float) -> float:
    """Max subgradient-optimality violation of ``beta`` for LASSO at ``lam``."""
    c = x.T @ (y - x @ beta)
    viol = 0.0
    for j in range(len(beta)):
        if abs(beta[j]) > 1e-9:
            viol = max(viol, abs(c[j] - lam * np.sign(beta[j])))
        else:
            viol = max(viol, max(0.0, abs(c[j]) - lam))
    return float(viol)


def _uncorrelated_problem():
    """A generic design whose path only grows the support (enter events only)."""
    rng = np.random.default_rng(0)
    m, n = 40, 12
    x = rng.standard_normal((m, n))
    true_beta = np.zeros(n)
    true_beta[[1, 4, 7]] = [2.0, -3.0, 1.5]
    y = x @ true_beta + 0.1 * rng.standard_normal(m)
    return x, y


def _correlated_problem():
    """A correlated design whose support oscillates (enter and leave events)."""
    rng = np.random.default_rng(0)
    m, n = 30, 8
    f = rng.standard_normal((m, 2))
    loadings = rng.standard_normal((2, n))
    x = f @ loadings + 0.3 * rng.standard_normal((m, n))
    true_beta = np.zeros(n)
    true_beta[[0, 3]] = [3.0, -2.0]
    y = x @ true_beta + 0.1 * rng.standard_normal(m)
    return x, y


class TestLassoPath:
    """End-to-end correctness of the traced LASSO path."""

    def test_endpoints(self):
        """The path starts at (lam_max, 0) and ends at (0, OLS-on-support)."""
        x, y = _uncorrelated_problem()
        lasso = Lasso(x, y)
        first, last = lasso.path[0], lasso.path[-1]
        assert np.isclose(first.lam, lasso.lam_max)
        np.testing.assert_allclose(first.beta, 0.0)
        assert last.lam == 0.0

    def test_path_matches_kkt_and_coordinate_descent(self):
        """Sampled along the path, beta satisfies KKT and matches coordinate descent."""
        x, y = _uncorrelated_problem()
        lasso = Lasso(x, y)
        for frac in (0.9, 0.7, 0.5, 0.3, 0.15, 0.05):
            lam = frac * lasso.lam_max
            beta = lasso.solution(lam)
            assert _kkt_violation(x, y, beta, lam) < 1e-7
            np.testing.assert_allclose(beta, _coordinate_descent(x, y, lam), atol=1e-5)

    def test_leave_events_fire_and_path_stays_exact(self):
        """A correlated design forces drop (leave) events; the path remains exact."""
        x, y = _correlated_problem()
        lasso = Lasso(x, y)
        sizes = [int(bp.active.sum()) for bp in lasso.path]
        assert any(b < a for a, b in pairwise(sizes)), "expected at least one drop event"
        for frac in (0.9, 0.6, 0.35, 0.1):
            lam = frac * lasso.lam_max
            beta = lasso.solution(lam)
            assert _kkt_violation(x, y, beta, lam) < 1e-7
            np.testing.assert_allclose(beta, _coordinate_descent(x, y, lam), atol=1e-5)

    def test_solution_clamps_outside_range(self):
        """Querying beyond lam_max returns 0; at or below 0 returns the final fit."""
        x, y = _uncorrelated_problem()
        lasso = Lasso(x, y)
        np.testing.assert_allclose(lasso.solution(2.0 * lasso.lam_max), 0.0)
        np.testing.assert_allclose(lasso.solution(-1.0), lasso.path[-1].beta)

    def test_solution_at_breakpoint(self):
        """Evaluating exactly at a stored breakpoint reproduces its coefficients."""
        x, y = _uncorrelated_problem()
        lasso = Lasso(x, y)
        mid = lasso.path[len(lasso.path) // 2]
        np.testing.assert_allclose(lasso.solution(mid.lam), mid.beta, atol=1e-12)

    def test_lambda_non_increasing(self):
        """Breakpoint lambdas are recorded in non-increasing order."""
        x, y = _uncorrelated_problem()
        lams = [bp.lam for bp in Lasso(x, y).path]
        assert all(a >= b for a, b in pairwise(lams))


class TestLassoValidation:
    """Input validation on construction."""

    def test_rejects_non_2d_design(self):
        """A 1d design matrix is rejected."""
        with pytest.raises(ValueError, match="2d design matrix"):
            Lasso(np.ones(5), np.ones(5))

    def test_rejects_mismatched_response_length(self):
        """A response whose length does not match the rows of x is rejected."""
        with pytest.raises(ValueError, match="y must have shape"):
            Lasso(np.ones((5, 3)), np.ones(4))


def test_breakpoint_is_frozen():
    """Breakpoint is an immutable record."""
    import dataclasses

    bp = Breakpoint(lam=1.0, beta=np.zeros(2), active=np.zeros(2, dtype=bool))
    with pytest.raises(dataclasses.FrozenInstanceError):
        bp.lam = 2.0

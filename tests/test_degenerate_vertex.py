"""Degenerate maximum-return vertices resolved from the optimal dual set (issue #918).

At a degenerate vertex the optimal duals of the maximum-return linear program are
not unique, and the dual solution HiGHS reports need not show which free set works.
The partition is therefore chosen at a vertex of the optimal dual set
(:func:`cvxcla.first._dual_vertex`), not from the reported reduced costs. These
tests cover the three reproducers of the issue, check each traced frontier against
an exact active-set enumeration, and pin down the cases that must still be declined.
"""

import itertools

import numpy as np
import pytest

from cvxcla import CLA
from cvxcla.errors import DegenerateProblemError
from cvxcla.first import _dual_step, _dual_vertex, _vertex_partition, classify_vertex

MU = np.array([3.0, 2.0, 1.0])
SIGMA = np.array([[1.0, 0.2, 0.1], [0.2, 0.8, 0.1], [0.1, 0.1, 0.5]])


def _exact_solution(mean, cov, lower, upper, a, b, g, h, lam):
    """The minimiser of ``w' cov w / 2 - lam mean' w`` by enumerating every active set.

    Exact for the tiny problems here: each candidate active set gives an equality
    constrained QP, and the best feasible candidate is the optimum.
    """
    n = mean.shape[0]
    best, best_w = np.inf, None
    for state in itertools.product((0, 1, 2), repeat=n):
        for active in itertools.product((0, 1), repeat=g.shape[0]):
            rows, rhs = [a], [b]
            for i, s in enumerate(state):
                if s:
                    rows.append(np.eye(n)[[i]])
                    rhs.append([lower[i] if s == 1 else upper[i]])
            for j, s in enumerate(active):
                if s:
                    rows.append(g[[j]])
                    rhs.append(h[[j]])
            c, d = np.vstack(rows), np.concatenate([np.ravel(r) for r in rhs])
            kkt = np.block([[cov, c.T], [c, np.zeros((d.size, d.size))]])
            w = np.linalg.lstsq(kkt, np.r_[lam * mean, d], rcond=None)[0][:n]
            feasible = (
                np.allclose(c @ w, d, atol=1e-9)
                and np.all(w >= lower - 1e-9)
                and np.all(w <= upper + 1e-9)
                and np.all(g @ w <= h + 1e-9)
            )
            value = 0.5 * w @ cov @ w - lam * mean @ w
            if feasible and value < best:
                best, best_w = value, w
    return best_w


def _weights_at(cla, lam):
    """The frontier weights at ``lam``, interpolated between the turning points."""
    tps = cla.turning_points
    for left, right in itertools.pairwise(tps):
        if right.lamb <= lam <= left.lamb:
            if np.isinf(left.lamb):
                return left.weights
            t = (lam - right.lamb) / (left.lamb - right.lamb)
            return t * left.weights + (1 - t) * right.weights
    return tps[0].weights


def _assert_frontier_exact(cla, mean, cov, lower, upper, a, b, g, h):
    """Every turning point is feasible, and the path matches the exact optimum."""
    for tp in cla.turning_points:
        assert np.allclose(a @ tp.weights, b, atol=1e-9)
        assert np.all(tp.weights >= lower - 1e-9)
        assert np.all(tp.weights <= upper + 1e-9)
        assert np.all(g @ tp.weights <= h + 1e-9)
    finite = [tp.lamb for tp in cla.turning_points if 0 < tp.lamb < np.inf]
    for lam in np.r_[np.linspace(1e-3, 3 * max(finite), 13), 0.5 * np.array(finite)]:
        w = _weights_at(cla, lam)
        np.testing.assert_allclose(w, _exact_solution(mean, cov, lower, upper, a, b, g, h, lam), atol=1e-8)


class TestIssueReproducers:
    """The three degenerate vertices of #918 trace, and their frontiers are exact."""

    def test_equality_rows_only(self):
        """Budget plus ``w_1 = 0``: the reported dual ``y = (3, 0)`` frees one asset only.

        ``y = (3, 0)`` sits inside an edge of the optimal dual set; at either end of it
        (``(1, 1)`` or ``(3, -1)``) two reduced costs vanish and the free block is
        nonsingular.
        """
        a, b = np.array([[1.0, 1, 1], [0, 1, 0]]), np.array([1.0, 0])
        g, h = np.zeros((0, 3)), np.zeros(0)
        lower, upper = np.zeros(3), np.ones(3)
        cla = CLA(mean=MU, covariance=SIGMA, lower_bounds=lower, upper_bounds=upper, a=a, b=b)
        first = cla.turning_points[0]
        np.testing.assert_allclose(first.weights, [1.0, 0.0, 0.0], atol=1e-12)
        assert np.count_nonzero(first.free) == 2
        _assert_frontier_exact(cla, MU, SIGMA, lower, upper, a, b, g, h)

    def test_cap_with_zero_multiplier(self):
        """A cap ``w_0 + w_1 <= 1`` tight with a zero multiplier stays inactive."""
        a, b = np.ones((1, 3)), np.ones(1)
        g, h = np.array([[1.0, 1, 0]]), np.array([1.0])
        lower, upper = np.zeros(3), np.ones(3)
        cla = CLA(mean=MU, covariance=SIGMA, lower_bounds=lower, upper_bounds=upper, a=a, b=b, g=g, h=h)
        assert not cla.turning_points[0].active_ineq.any()
        _assert_frontier_exact(cla, MU, SIGMA, lower, upper, a, b, g, h)

    def test_cap_duplicating_a_bound(self):
        """A cap ``w_0 <= 0.5`` on top of the bound ``u_0 = 0.5``, with HiGHS's multiplier on the bound."""
        a, b = np.ones((1, 3)), np.ones(1)
        g, h = np.array([[1.0, 0, 0]]), np.array([0.5])
        lower, upper = np.zeros(3), np.array([0.5, 1.0, 1.0])
        cla = CLA(mean=MU, covariance=SIGMA, lower_bounds=lower, upper_bounds=upper, a=a, b=b, g=g, h=h)
        np.testing.assert_allclose(cla.turning_points[0].weights, [0.5, 0.5, 0.0], atol=1e-12)
        _assert_frontier_exact(cla, MU, SIGMA, lower, upper, a, b, g, h)


class TestStillDeclined:
    """A degenerate vertex with no valid partition is still reported."""

    def test_row_on_fixed_assets_only(self):
        """A row that only touches fixed assets has a multiplier nothing determines.

        ``w_0 + w_1 = 0.5`` with ``w_0`` and ``w_1`` fixed at ``0.25``: no asset that can
        leave its bound enters the row, so no free set spans it and the optimal
        duals have no vertex.
        """
        a = np.array([[1.0, 1, 1], [1.0, 1, 0]])
        b = np.array([1.0, 0.5])
        lower, upper = np.array([0.25, 0.25, 0.0]), np.array([0.25, 0.25, 1.0])
        with pytest.raises(DegenerateProblemError, match="optimal duals have no vertex"):
            CLA(mean=MU, covariance=SIGMA, lower_bounds=lower, upper_bounds=upper, a=a, b=b)

    def test_classify_without_duals_declines(self):
        """Without duals (the netted leverage vertex) a degenerate vertex is declined as before."""
        with pytest.raises(DegenerateProblemError, match="maximum-return vertex is degenerate"):
            classify_vertex(
                np.array([1.0, 0.0, 0.0]),
                np.zeros(3),
                np.ones(3),
                np.array([[1.0, 1, 1], [0, 1, 0]]),
                np.zeros((0, 3)),
                np.zeros(0),
                1e-9,
            )


class TestDualVertexHelpers:
    """The dual-vertex walk and the partition read off it, on hand-made inputs."""

    def test_walk_reaches_a_vertex(self):
        """From inside the edge ``1 <= y_0 <= 3`` the walk stops at an end of it."""
        rows = np.array([[1.0, 1, 1], [0, 1, 0]])
        sign = np.array([1.0, -1.0, -1.0])
        y = _dual_vertex(rows, MU, np.zeros(3, dtype=bool), sign, 2, np.array([3.0, 0.0]), 1e-9)
        gap = MU - rows.T @ y
        assert np.all(sign * gap >= -1e-12)
        assert np.count_nonzero(np.abs(gap) <= 1e-9) == 2

    def test_step_tries_the_opposite_direction(self):
        """When ``+d`` meets no constraint the step is taken along ``-d``."""
        rows = np.array([[1.0]])
        step = _dual_step(rows, np.array([-1.0]), np.zeros(1), np.array([-1.0]), np.zeros(1, dtype=bool), np.ones(1))
        np.testing.assert_allclose(step, [-1.0])

    def test_step_none_without_blocking_constraint(self):
        """No constraint along either direction: ``None``."""
        rows = np.array([[1.0]])
        step = _dual_step(rows, np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1, dtype=bool), np.ones(1))
        assert step is None

    def test_step_stops_on_a_row_multiplier(self):
        """An inequality multiplier reaching zero ends the move."""
        rows = np.array([[1.0, 0.0], [0.0, 1.0]])
        step = _dual_step(
            rows, np.zeros(2), np.array([0.0, 2.0]), np.zeros(2), np.array([False, True]), np.array([0.0, 1.0])
        )
        np.testing.assert_allclose(step, [0.0, -2.0])

    def test_partition_prefers_releasing_rows(self):
        """A zero-multiplier row is released before a zero-reduced-cost asset is freed."""
        rows = np.array([[1.0, 1, 1], [1.0, 1, 0]])
        freed, released = _vertex_partition(
            rows,
            np.array([True, False, False]),
            np.array([False, True, True]),
            np.array([False, True]),
            -MU,
        )
        np.testing.assert_array_equal(released, [False, True])
        np.testing.assert_array_equal(freed, [False, False, False])

"""Tests for the generic parametric active-set path tracer.

These cover the problem-independent pieces directly: the Bland event selection
(``select_next_event``) and the ``trace`` control loop, including its safety cap.
A tiny in-test ``ParametricProblem`` exercises the loop without any portfolio or
regression machinery.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cvxcla.pathtracer import ParametricProblem, select_next_event, trace


class TestSelectNextEvent:
    """Unit tests for the Bland min-ratio event selector."""

    def test_picks_largest_valid_ratio(self):
        """The event with the largest ratio at or below the current lam wins."""
        l_mat = np.full((3, 2), -np.inf)
        l_mat[0, 0] = 0.2
        l_mat[2, 1] = 0.9
        assert select_next_event(l_mat, lam=np.inf, tol=1e-9) == (2, 1, 0.9)

    def test_filters_ratios_above_current_lam(self):
        """Ratios above the current lam window are spurious and discarded."""
        l_mat = np.full((2, 2), -np.inf)
        l_mat[0, 0] = 5.0  # above the window
        l_mat[1, 1] = 0.4
        assert select_next_event(l_mat, lam=1.0, tol=1e-9) == (1, 1, 0.4)

    def test_bland_lowest_index_tiebreak(self):
        """Among ratios tied within tol, the lowest (row, col) index wins."""
        tol = 1e-5
        l_mat = np.full((4, 4), -np.inf)
        l_mat[3, 1] = 0.5
        l_mat[1, 0] = 0.5 + tol / 2  # tied with [3, 1] to within tol
        assert select_next_event(l_mat, lam=np.inf, tol=tol)[:2] == (1, 0)

    def test_returns_none_when_no_valid_event(self):
        """When every ratio lies above the lam window, the trace stops."""
        l_mat = np.full((2, 2), -np.inf)
        l_mat[0, 0] = 5.0
        assert select_next_event(l_mat, lam=1.0, tol=1e-9) is None

    def test_returns_none_when_all_neg_inf(self):
        """An all -inf matrix means no candidate events remain."""
        assert select_next_event(np.full((3, 2), -np.inf), lam=np.inf, tol=1e-9) is None

    def test_does_not_mutate_input(self):
        """Selection works on a copy, leaving the caller's matrix untouched."""
        l_mat = np.full((2, 2), -np.inf)
        l_mat[0, 0] = 5.0
        before = l_mat.copy()
        select_next_event(l_mat, lam=1.0, tol=1e-9)
        np.testing.assert_array_equal(l_mat, before)


class _CountdownProblem:
    """A trivial 1-coordinate problem: step lam down a fixed schedule, then stop.

    It records the lambdas the tracer drives it through. Used only to exercise the
    generic loop end to end without any real linear algebra.
    """

    def __init__(self, schedule: list[float], tol: float = 1e-9) -> None:
        self._schedule = schedule
        self._i = 0
        self.tol = tol
        self.dimension = 1
        self.visited: list[float] = []
        self.finished_at: float | None = None

    def begin(self) -> tuple[float, Any]:
        """Start above the schedule with an empty state."""
        self.visited.append(np.inf)
        return np.inf, None

    def segment(self, state: Any) -> Any:
        """No segment data is needed for this toy problem."""
        return None

    def event_matrix(self, state: Any, segment: Any) -> np.ndarray:
        """Emit the next scheduled lambda as the sole candidate event, else stop."""
        if self._i < len(self._schedule):
            return np.array([[self._schedule[self._i]]])
        return np.array([[-np.inf]])

    def step(self, state: Any, segment: Any, sec: int, direction: int, lam: float) -> Any:
        """Advance the schedule and record the visited lambda."""
        self.visited.append(lam)
        self._i += 1
        return None

    def finish(self, state: Any, segment: Any) -> None:
        """Record where the trace terminated."""
        self.finished_at = 0.0


class _NeverConvergesProblem:
    """A problem that always offers the same event, so the loop never terminates."""

    def __init__(self, n: int) -> None:
        self.tol = 1e-9
        self.dimension = n

    def begin(self) -> tuple[float, Any]:
        """Start at +inf."""
        return np.inf, None

    def segment(self, state: Any) -> Any:
        """No segment data needed."""
        return None

    def event_matrix(self, state: Any, segment: Any) -> np.ndarray:
        """Always return a fixed, in-window event so the loop cannot finish."""
        return np.array([[1.0]])

    def step(self, state: Any, segment: Any, sec: int, direction: int, lam: float) -> Any:
        """No state change, guaranteeing non-termination."""
        return None

    def finish(self, state: Any, segment: Any) -> None:  # pragma: no cover - never reached
        """Never called: the loop raises before converging."""


def test_trace_visits_schedule_and_finishes():
    """Trace drives the problem down its event schedule, then calls finish."""
    problem = _CountdownProblem([0.8, 0.5, 0.2])
    trace(problem)
    assert problem.visited == [np.inf, 0.8, 0.5, 0.2]
    assert problem.finished_at == 0.0


def test_trace_raises_when_loop_does_not_terminate():
    """The safety cap turns non-termination into a clear RuntimeError."""
    with pytest.raises(RuntimeError, match="failed to converge"):
        trace(_NeverConvergesProblem(n=2))


def test_problems_are_recognised_as_parametric_problems():
    """The toy problems structurally satisfy the runtime-checkable protocol."""
    assert isinstance(_CountdownProblem([0.5]), ParametricProblem)
    assert isinstance(_NeverConvergesProblem(n=1), ParametricProblem)

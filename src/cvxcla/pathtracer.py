"""Generic parametric active-set path tracer.

The Critical Line Algorithm is one instance of a broader scheme: the optimiser of
a convex quadratic program is followed as a single scalar parameter ``lambda``
varies, the solution path is piecewise linear, and it is traced exactly by
stepping from one breakpoint to the next while maintaining a *coordinate* active
set. The same scheme drives least angle regression (LARS) and the LASSO homotopy.

This module factors out the part that is identical across those problems:

* ``select_next_event`` -- the simplex-style "smallest positive ratio to the next
  event" rule with a Bland lowest-index tie-break for anti-cycling. It works on an
  ``(n, k)`` matrix of candidate critical ``lambda`` values, where the row is the
  coordinate and the column is the (problem-defined) event type.
* ``trace`` -- the control loop: build the first vertex, solve the current affine
  segment, scan events, pick the next one, flip one coordinate's activity, repeat.

Everything problem-specific (the segment solve, what an event *means*, how a vertex
is recorded) lives behind the ``ParametricProblem`` protocol. ``CLA`` (portfolio
frontier) and ``Lasso`` (regression path) are two implementations; both are driven
by the same ``trace`` here.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class ParametricProblem(Protocol):
    """The problem-specific contract the generic ``trace`` loop drives.

    A ``State`` describes the current vertex (which coordinates are free/blocked
    and their values); a ``Segment`` describes the affine piece valid until the
    next event. Both are opaque to the tracer: it only passes them back to the
    problem. Implementations record vertices as they are discovered (the tracer
    returns nothing).
    """

    @property
    def tol(self) -> float:
        """Tolerance for the event-selection tie-break and validity window."""
        ...  # pragma: no cover

    @property
    def dimension(self) -> int:
        """Number of coordinates ``n`` (sets the iteration safety cap)."""
        ...  # pragma: no cover

    def begin(self) -> tuple[float, Any]:
        """Record the first vertex and return ``(lambda_start, initial_state)``."""
        ...  # pragma: no cover

    def segment(self, state: Any) -> Any:
        """Solve the affine segment valid at ``state`` (the reduced KKT system)."""
        ...  # pragma: no cover

    def event_matrix(self, state: Any, segment: Any) -> NDArray[np.float64]:
        """Return the ``(n, k)`` matrix of candidate critical ``lambda`` values.

        Entry ``(i, j)`` is the ``lambda`` at which coordinate ``i`` would fire
        event type ``j`` along ``segment``; ``-inf`` means "no such event".
        """
        ...  # pragma: no cover

    def step(self, state: Any, segment: Any, sec: int, direction: int, lam: float) -> Any:
        """Record the vertex at ``lam`` after flipping coordinate ``sec``'s activity.

        ``direction`` is the winning event-matrix column; the implementation maps
        it to the new activity (and, where relevant, the sign) of ``sec``. Returns
        the next ``State``.
        """
        ...  # pragma: no cover

    def finish(self, state: Any, segment: Any) -> None:
        """Record the final ``lambda = 0`` vertex (the segment's intercept)."""
        ...  # pragma: no cover


def select_next_event(l_mat: NDArray[np.float64], lam: float, tol: float) -> tuple[int, int, float] | None:
    """Pick the next breakpoint from the event matrix, or ``None`` to stop.

    The current segment is valid only for ``lambda`` at or below the current value
    (the path is traced with non-increasing ``lambda``), so ratios above it are
    spurious crossings outside the segment and are discarded; if none remain the
    trace is complete. Among events tied for the largest valid ratio, a Bland-style
    lowest-``(coordinate, event type)`` rule makes the choice deterministic and
    prevents cycling on degenerate problems.

    Two events are "the same" breakpoint only when their ``lambda`` values agree to
    floating-point scale, so the validity window and the tie-break use a tolerance
    *relative* to the ``lambda`` magnitude. A fixed absolute tolerance mis-ranks the
    densely spaced breakpoints of large paths: their spacing shrinks with the problem
    size, so an absolute window eventually merges genuinely distinct events, picks the
    wrong one by the tie-break, and lets ``lambda`` step backwards. The caller's
    ``tol`` is treated as an upper bound on this relative rate; event ordering needs a
    roundoff-scale window regardless of the coarser ``tol`` used elsewhere (e.g. to
    classify a weight as sitting on a bound). Finally, the chosen ``lambda`` is clamped
    below the current value, so roundoff in the segment solve can never make a
    breakpoint appear above it.

    Args:
        l_mat: The ``(n, k)`` matrix of candidate critical ``lambda`` values.
        lam: The current (upper) ``lambda`` bound on valid events.
        tol: Upper bound on the relative event-ordering tolerance.

    Returns:
        ``(sec, direction, lam_next)`` for the chosen event, or ``None`` if no
        valid event remains.
    """
    l_mat = l_mat.copy()  # do not mutate the caller's matrix
    rate = min(tol, 1e-10)  # roundoff-scale relative window; tol is only an upper bound
    l_mat[l_mat > lam + rate * max(1.0, abs(lam))] = -np.inf  # pragma: no mutate

    lam_max = np.max(l_mat)
    if lam_max < 0:  # pragma: no mutate
        return None

    tied = np.argwhere(l_mat >= lam_max - rate * max(1.0, abs(lam_max)))  # pragma: no mutate
    sec, direction = tied[0]
    # lambda is non-increasing along the path: clamp away any roundoff overshoot.
    return int(sec), int(direction), float(min(lam, l_mat[sec, direction]))


def trace(problem: ParametricProblem) -> None:
    """Trace the full piecewise-linear solution path of ``problem``.

    Mirrors the turning-point loop of the Critical Line Algorithm, but driven
    entirely through the ``ParametricProblem`` interface so the same control flow
    serves any coordinate-active-set parametric quadratic program.

    Args:
        problem: The problem to trace; its ``begin``/``segment``/``event_matrix``/
            ``step``/``finish`` hooks record the discovered vertices.

    Raises:
        RuntimeError: If the event loop fails to terminate within the safety cap
            (each step fixes the activity of at least one coordinate, so a correct
            trace runs ``O(n)`` times; vastly exceeding this signals cycling).
    """
    lam, state = problem.begin()

    # Safety bound: far above any valid path length; exceeding it means the event
    # loop failed to terminate, so we fail loudly rather than hang. Each step fixes
    # the activity of one coordinate -- a box bound (n of them) or an inequality row
    # (p of them) -- so the bound scales with the total n + p, not n alone; a problem
    # that exposes only ``dimension`` (no inequality rows) falls back to n.
    path_coords = getattr(problem, "event_dimension", problem.dimension)  # pragma: no mutate
    max_iterations = 100 * (path_coords + 1)  # pragma: no mutate
    iterations = 0  # pragma: no mutate

    while True:  # pragma: no mutate
        iterations += 1  # pragma: no mutate
        if iterations > max_iterations:  # pragma: no mutate
            msg = "path tracer failed to converge: too many iterations"  # pragma: no mutate
            raise RuntimeError(msg)  # pragma: no mutate

        segment = problem.segment(state)
        event = select_next_event(problem.event_matrix(state, segment), lam, problem.tol)
        if event is None:
            problem.finish(state, segment)
            return

        sec, direction, lam = event
        state = problem.step(state, segment, sec, direction, lam)

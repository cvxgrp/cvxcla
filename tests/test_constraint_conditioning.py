"""Nearly dependent constraint rows with a well-conditioned covariance.

The rows ``1^T w = 1`` and ``(1 + eps v)^T w = 1`` describe the same feasible set as
``1^T w = 1`` and ``v^T w = 0`` for every ``eps > 0``, so the frontier does not depend
on ``eps``. Only the conditioning of the constraint system does, which isolates the
reduced (Schur-complement) solve from the covariance: the free blocks are the same
well-conditioned ones throughout.
"""

from __future__ import annotations

import numpy as np
import pytest
from cvx.linalg import DenseOperator
from cvx.linalg import bordered_solve as plain_bordered_solve

from cvxcla import CLA, DegenerateProblemError
from cvxcla.operators import bordered_solve, orthonormal_rows

N = 40


@pytest.fixture(scope="module")
def problem() -> dict:
    """A well-conditioned covariance, capped long-only bounds and the hidden row ``v``."""
    rng = np.random.default_rng(7)
    f = rng.standard_normal((N, N))
    v = rng.standard_normal(N)
    return {
        "mean": rng.uniform(0.0, 1.0, N),
        "covariance": f @ f.T / N + 0.5 * np.eye(N),
        "lower_bounds": np.zeros(N),
        "upper_bounds": np.full(N, 0.1),
        "v": v - v.mean(),
    }


def _trace(problem: dict, eps: float | None) -> CLA:
    """Trace with the hidden row stated directly (``eps=None``) or folded into ``1 + eps v``."""
    ones, v = np.ones(N), problem["v"]
    a = np.vstack([ones, v]) if eps is None else np.vstack([ones, ones + eps * v])
    b = np.array([1.0, 0.0]) if eps is None else np.ones(2)
    kwargs = {k: problem[k] for k in ("mean", "covariance", "lower_bounds", "upper_bounds")}
    return CLA(a=a, b=b, **kwargs)


def _weights(cla: CLA) -> np.ndarray:
    """Turning-point weights stacked row by row."""
    return np.array([tp.weights for tp in cla.turning_points])


@pytest.mark.parametrize(("eps", "atol"), [(1e-3, 1e-11), (1e-5, 1e-9), (1e-8, 1e-6)])
def test_nearly_dependent_rows_trace_the_same_frontier(problem, eps, atol):
    """Accuracy degrades like cond(C) times round-off, not cond(C) squared.

    Solving the Schur complement in the stated rows lost about 1e-6 at eps = 1e-5,
    and the linear program enforced the hidden row only to its tolerance / eps, so
    from eps ~ 1e-7 the first vertex was misread as degenerate.
    """
    reference = _weights(_trace(problem, None))
    traced = _weights(_trace(problem, eps))
    assert traced.shape == reference.shape
    np.testing.assert_allclose(traced, reference, atol=atol)


def test_rows_dependent_below_the_guard_are_declined(problem):
    """Rows dependent on the free set to below the 1e-12 floor are refused, not traced."""
    with pytest.raises(DegenerateProblemError, match="linearly dependent on the free set"):
        _trace(problem, 1e-13)


def test_orthonormal_rows_span_the_same_constraints():
    """``Q^T w = R^{-T} b`` holds exactly where ``A w = b`` does."""
    rng = np.random.default_rng(0)
    a, w = rng.standard_normal((3, 8)), rng.standard_normal(8)
    q, d = orthonormal_rows(a, a @ w)
    np.testing.assert_allclose(q @ q.T, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(q @ w, d, atol=1e-12)


@pytest.mark.parametrize(
    "a",
    [np.zeros((0, 4)), np.ones((5, 4)), np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])],
    ids=["no rows", "more rows than columns", "dependent rows"],
)
def test_orthonormal_rows_passes_degenerate_systems_through(a):
    """Rows that cannot be orthonormalised are left for the linear program to judge."""
    b = np.ones(a.shape[0])
    out_a, out_b = orthonormal_rows(a, b)
    assert out_a is a
    assert out_b is b


def test_bordered_solve_matches_the_plain_range_space_solve():
    """The orthonormal-basis solve returns the same weights and multipliers."""
    rng = np.random.default_rng(1)
    m = rng.standard_normal((6, 6))
    op = DenseOperator(m @ m.T + np.eye(6))
    free = np.array([True, True, False, True, True, True])
    c = rng.standard_normal((2, 5))
    rhs, d = rng.standard_normal((5, 2)), rng.standard_normal((2, 2))
    x_c, x_s, nu_c, nu_s = bordered_solve(op, free, c, rhs[:, 0], rhs[:, 1], d[:, 0], d[:, 1])
    x, nu = plain_bordered_solve(op, np.flatnonzero(free), c, rhs, d)
    np.testing.assert_allclose(np.column_stack([x_c, x_s]), x, atol=1e-12)
    np.testing.assert_allclose(np.column_stack([nu_c, nu_s]), nu, atol=1e-12)


def test_bordered_solve_with_more_rows_than_free_coordinates_uses_the_plain_solve():
    """With more rows than free coordinates there is no basis to switch to."""
    op = DenseOperator(np.diag([1.0, 2.0, 3.0]))
    free = np.array([True, False, False])
    c = np.array([[1.0], [2.0]])
    rhs, d = np.zeros((1, 2)), np.ones((2, 2))
    with pytest.raises(np.linalg.LinAlgError):
        plain_bordered_solve(op, np.flatnonzero(free), c, rhs, d)
    with pytest.raises(np.linalg.LinAlgError):
        bordered_solve(op, free, c, rhs[:, 0], rhs[:, 1], d[:, 0], d[:, 1])

"""Supported inputs, and the exception each unsupported input raises.

Each test pins one row of the table of supported and unsupported cases: what
the input is, whether it is traced, and which :mod:`cvxcla.errors` class reports
it otherwise. Inputs that used to be declined but are mathematically harmless
(redundant equality rows, a duplicated or never-binding inequality row) are
checked to give exactly the frontier of the equivalent reduced problem.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import cvxcla
import cvxcla._projection as projection
from cvxcla import (
    CLA,
    CLAError,
    DegenerateProblemError,
    FeasibilityError,
    InfeasibleProblemError,
    NumericalError,
    ProjectionError,
)
from cvxcla._kkt import guard_active_rows

N = 12


@pytest.fixture(scope="module")
def market() -> dict:
    """A long-only, fully-invested problem on a small, well-conditioned market."""
    rng = np.random.default_rng(0)
    u = rng.standard_normal((N, 3))
    return {
        "mean": rng.uniform(0.0, 1.0, N),
        "covariance": np.diag(rng.uniform(0.5, 2.0, N)) + u @ u.T,
        "lower_bounds": np.zeros(N),
        "upper_bounds": np.ones(N),
        "a": np.ones((1, N)),
        "b": np.ones(1),
    }


def _weights(cla: CLA) -> np.ndarray:
    """Turning-point weights with repeated consecutive points removed."""
    w = np.array([tp.weights for tp in cla.turning_points])
    return w[np.r_[True, np.any(np.abs(np.diff(w, axis=0)) > 1e-12, axis=1)]]


SECTOR = (np.arange(N) % 3 == 0).astype(float)


class TestHierarchy:
    """Every class is a CLAError and still the builtin it replaced."""

    @pytest.mark.parametrize(
        ("cls", "builtin"),
        [
            (InfeasibleProblemError, ValueError),
            (DegenerateProblemError, ValueError),
            (NumericalError, RuntimeError),
            (FeasibilityError, ValueError),
            (FeasibilityError, RuntimeError),
            (ProjectionError, RuntimeError),
        ],
    )
    def test_subclasses(self, cls, builtin):
        """Each error is catchable as CLAError and as its historical builtin."""
        assert issubclass(cls, CLAError)
        assert issubclass(cls, builtin)

    def test_public(self):
        """The classes are exported from the package."""
        for name in ("CLAError", "InfeasibleProblemError", "DegenerateProblemError", "NumericalError"):
            assert name in cvxcla.__all__
        assert issubclass(ProjectionError, NumericalError)


class TestInfeasible:
    """Constraints that admit no portfolio raise InfeasibleProblemError."""

    def test_crossed_bounds(self, market):
        """A lower bound above its upper bound."""
        with pytest.raises(InfeasibleProblemError, match="Lower bounds"):
            CLA(**market | {"lower_bounds": np.r_[2.0, np.zeros(N - 1)]})

    def test_unreachable_budget(self, market):
        """Caps that cannot sum to the budget."""
        with pytest.raises(InfeasibleProblemError, match="fully invested"):
            CLA(**market | {"upper_bounds": np.full(N, 0.05)})

    def test_infeasible_equality(self, market):
        """A second equality row the box cannot meet (the LP proves infeasibility)."""
        with pytest.raises(InfeasibleProblemError, match="infeasible"):
            CLA(**market | {"a": np.vstack([np.ones(N), SECTOR]), "b": np.array([1.0, 2.0])})

    def test_infeasible_inequality(self, market):
        """An inequality row no long-only portfolio satisfies."""
        with pytest.raises(InfeasibleProblemError, match="infeasible"):
            CLA(**market | {"g": SECTOR[None, :], "h": np.array([-1.0])})

    def test_inconsistent_dependent_equality(self, market):
        """A dependent equality row whose right-hand side contradicts the others."""
        with pytest.raises(InfeasibleProblemError, match="inconsistent"):
            CLA(**market | {"a": np.vstack([np.ones(N), 2.0 * np.ones(N)]), "b": np.array([1.0, 3.0])})


class TestRedundantRowsAreHarmless:
    """Redundant constraints are reduced away and leave the frontier unchanged."""

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (np.vstack([np.ones(N), np.ones(N)]), np.ones(2)),
            (np.vstack([np.ones(N), 2.0 * np.ones(N)]), np.array([1.0, 2.0])),
        ],
    )
    def test_dependent_equality_rows(self, market, a, b):
        """A repeated or scaled budget row is dropped; the frontier is the budget's."""
        reference = _weights(CLA(**market))
        cla = CLA(**market | {"a": a, "b": b})
        assert cla.a.shape == (1, N)
        np.testing.assert_allclose(_weights(cla), reference, atol=1e-10)

    def test_never_binding_inequality(self, market):
        """A cap that never binds gives the budget-only frontier.

        It routes the first vertex through the linear program, whose solution puts
        everything in one asset at its cap; that degenerate vertex is resolved.
        """
        reference = _weights(CLA(**market))
        cla = CLA(**market | {"g": SECTOR[None, :], "h": np.array([10.0])})
        np.testing.assert_allclose(_weights(cla), reference, atol=1e-10)

    def test_duplicated_inequality_row(self, market):
        """A cap given twice traces exactly as the cap given once."""
        once = _weights(CLA(**market | {"g": SECTOR[None, :], "h": np.array([0.3])}))
        twice = CLA(**market | {"g": np.vstack([SECTOR, SECTOR]), "h": np.array([0.3, 0.3])})
        np.testing.assert_allclose(_weights(twice), once, atol=1e-10)


class TestDegenerate:
    """Feasible problems outside the supported domain raise DegenerateProblemError."""

    def test_singular_covariance(self, market):
        """A rank-3 covariance makes a free block singular during the trace."""
        rng = np.random.default_rng(0)
        u = rng.standard_normal((N, 3))
        with pytest.raises(DegenerateProblemError, match="numerically singular"):
            CLA(**market | {"covariance": u @ u.T})

    def test_unbounded_linear_program(self):
        """Without bounds the maximum-return linear program is unbounded."""
        with pytest.raises(DegenerateProblemError, match="unbounded"):
            CLA(
                mean=np.array([1.0, 2.0]),
                covariance=np.eye(2),
                lower_bounds=np.full(2, -np.inf),
                upper_bounds=np.full(2, np.inf),
                a=np.array([[1.0, 1.0]]),
                b=np.ones(1),
            )

    def test_dependent_active_rows_on_the_free_set(self):
        """Two active rows that coincide on the free set are refused."""
        c = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        with pytest.raises(DegenerateProblemError, match="linearly dependent"):
            guard_active_rows(c, np.array([True, True, False]), 0.5)

    def test_more_active_rows_than_free_assets(self):
        """Two active rows cannot be spanned by one free asset."""
        with pytest.raises(DegenerateProblemError, match="2 active rows, 1 free"):
            guard_active_rows(np.eye(2), np.array([True, False]), 0.5)

    def test_independent_active_rows_pass(self):
        """A well-posed partition passes the guard silently."""
        guard_active_rows(np.ones((1, 3)), np.array([True, True, False]), 0.5)
        guard_active_rows(np.zeros((0, 3)), np.array([True, False, False]), 0.5)


class TestNumerical:
    """Numerical breakdowns raise NumericalError subclasses."""

    def test_linear_program_failure(self, market, monkeypatch):
        """A linear program that stops for another reason is a numerical failure."""
        failed = SimpleNamespace(success=False, status=4, message="numerical difficulties")
        monkeypatch.setattr("cvxcla.first.linprog", lambda **_kwargs: failed)
        with pytest.raises(NumericalError, match="numerical difficulties"):
            CLA(**market | {"g": SECTOR[None, :], "h": np.array([0.3])})

    def test_projection_stall_within_round_off_is_accepted(self, monkeypatch):
        """Iterations that stall short of the target but within sqrt(eps) return the clip."""
        monkeypatch.setattr(projection, "_PROJECTION_TOL", -1.0)  # never converge early
        lower, upper = np.zeros(3), np.ones(3)
        c, d = np.ones((1, 3)), np.ones(1)
        out = projection.project_alternating(np.array([-1e-15, 0.5, 0.5 + 1e-15]), lower, upper, c, d)
        assert np.all(out >= lower)
        assert np.all(out <= upper)

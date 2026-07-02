"""Tests for the fluent ``ProblemBuilder`` convenience layer.

The builder is sugar over the explicit ``CLA(...)`` constructor, so the central
guarantee is equivalence: a problem assembled through the builder must produce
exactly the same frontier as the same problem passed to the constructor
directly. The rest of the suite covers the ergonomic helpers and the
actionable validation errors.
"""

import numpy as np
import pytest

from cvxcla import CLA, FactorCovariance, ProblemBuilder


@pytest.fixture
def problem():
    """A small, well-conditioned long-only problem (mean, covariance)."""
    rng = np.random.default_rng(0)
    n = 6
    mean = rng.uniform(0.0, 1.0, n)
    factors = rng.standard_normal((n, n))
    covariance = factors @ factors.T + n * np.eye(n)
    return mean, covariance


def _same_frontier(a: CLA, b: CLA) -> bool:
    """Whether two traces have identical turning-point weights."""
    if len(a) != len(b):
        return False
    return all(np.allclose(p.weights, q.weights) for p, q in zip(a.turning_points, b.turning_points, strict=True))


class TestEquivalence:
    """The builder must reproduce the explicit constructor exactly."""

    def test_long_only_budget_matches_constructor(self, problem):
        """long_only().budget() equals the canonical all-ones budget construction."""
        mean, covariance = problem
        n = len(mean)
        built = CLA.problem(mean, covariance).long_only().budget().trace()
        explicit = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=np.zeros(n),
            upper_bounds=np.ones(n),
            a=np.ones((1, n)),
            b=np.ones(1),
        )
        assert _same_frontier(built, explicit)

    def test_inequality_matches_constructor(self, problem):
        """An inequality row through the builder matches the explicit g/h arguments."""
        mean, covariance = problem
        n = len(mean)
        # Per-asset cap of 0.3 forces the max-return vertex to spread across
        # several assets, so the first vertex is non-degenerate and the row
        # binds along the trace.
        g = np.zeros((1, n))
        g[0, :3] = 1.0  # cap the first three assets' combined weight
        h = np.array([0.5])
        built = CLA.problem(mean, covariance).long_only(0.3).budget().inequality(g[0], 0.5).trace()
        explicit = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=np.zeros(n),
            upper_bounds=np.full(n, 0.3),
            a=np.ones((1, n)),
            b=np.ones(1),
            g=g,
            h=h,
        )
        assert _same_frontier(built, explicit)

    def test_factor_backend_passes_through(self, problem):
        """A QuadraticForm backend reaches CLA unchanged (structured solve preserved)."""
        mean, _ = problem
        n = len(mean)
        rng = np.random.default_rng(1)
        backend = FactorCovariance(d=np.full(n, 1.0), u=rng.standard_normal((n, 2)), delta=np.array([1.0, 2.0]))
        built = CLA.problem(mean, backend).long_only().budget().trace()
        explicit = CLA(
            mean=mean,
            covariance=backend,
            lower_bounds=np.zeros(n),
            upper_bounds=np.ones(n),
            a=np.ones((1, n)),
            b=np.ones(1),
        )
        assert built.covariance is backend
        assert _same_frontier(built, explicit)


class TestErgonomics:
    """The named helpers behave as documented."""

    def test_problem_returns_builder(self, problem):
        """CLA.problem yields a ProblemBuilder fixed to the given dimension."""
        mean, covariance = problem
        builder = CLA.problem(mean, covariance)
        assert isinstance(builder, ProblemBuilder)
        assert builder._n == len(mean)

    def test_bounds_scalar_and_array(self, problem):
        """The bounds setter accepts both scalars (broadcast) and length-n arrays."""
        mean, covariance = problem
        n = len(mean)
        upper = np.full(n, 0.5)
        scalar = CLA.problem(mean, covariance).bounds(0.0, 0.5).budget().trace()
        array = CLA.problem(mean, covariance).bounds(np.zeros(n), upper).budget().trace()
        assert _same_frontier(scalar, array)
        assert np.all(scalar.upper_bounds == 0.5)

    def test_long_only_custom_upper(self, problem):
        """long_only(upper) sets the upper cap and keeps the zero floor."""
        mean, covariance = problem
        cla = CLA.problem(mean, covariance).long_only(0.4).budget().trace()
        assert np.all(cla.lower_bounds == 0.0)
        assert np.all(cla.upper_bounds == 0.4)

    def test_budget_total(self, problem):
        """budget(total) sets the right-hand side of the all-ones equality."""
        mean, covariance = problem
        cla = CLA.problem(mean, covariance).long_only(2.0).budget(2.0).trace()
        assert cla.b == pytest.approx([2.0])
        assert np.allclose(cla.a, 1.0)

    def test_general_equality_matrix_and_accumulation(self, problem):
        """The equality method accepts a matrix block and accumulates across calls."""
        mean, covariance = problem
        n = len(mean)
        extra = np.zeros(n)
        extra[0] = 1.0  # pin the first asset's weight
        cla = CLA.problem(mean, covariance).bounds(-1.0, 1.0).budget().equality(extra, 0.1).trace()
        assert cla.a.shape == (2, n)  # budget row plus the pin
        assert cla.b == pytest.approx([1.0, 0.1])


class TestValidation:
    """Mistakes are rejected with actionable messages."""

    def test_trace_without_bounds_raises(self, problem):
        """Tracing before setting bounds names the fix."""
        mean, covariance = problem
        with pytest.raises(ValueError, match="set box bounds"):
            CLA.problem(mean, covariance).budget().trace()

    def test_trace_without_equality_raises(self, problem):
        """Tracing without any equality constraint names the fix."""
        mean, covariance = problem
        with pytest.raises(ValueError, match="needs an equality constraint"):
            CLA.problem(mean, covariance).long_only().trace()

    def test_bounds_wrong_length_raises(self, problem):
        """A bounds vector of the wrong length is rejected."""
        mean, covariance = problem
        with pytest.raises(ValueError, match="length-6 vector"):
            CLA.problem(mean, covariance).bounds(0.0, np.ones(3))

    def test_equality_wrong_width_raises(self, problem):
        """An equality row with the wrong number of columns is rejected."""
        mean, covariance = problem
        with pytest.raises(ValueError, match="must have 6 columns"):
            CLA.problem(mean, covariance).equality(np.ones(3), 1.0)

    def test_inequality_rhs_mismatch_raises(self, problem):
        """An inequality block whose rhs length disagrees with its rows is rejected."""
        mean, covariance = problem
        n = len(mean)
        with pytest.raises(ValueError, match="to match the rows"):
            CLA.problem(mean, covariance).inequality(np.ones((2, n)), np.ones(3))

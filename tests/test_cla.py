"""Tests for the CLA module.

This module tests the main Critical Line Algorithm implementation,
including initialization, turning point computation, and frontier generation.
"""

from unittest.mock import patch

import numpy as np
import pytest

from cvxcla import CLA, FactorCovariance
from cvxcla.pathtracer import select_next_event
from cvxcla.types import TurningPoint


class TestCLA:
    """Tests for the CLA class."""

    @pytest.fixture
    def simple_problem(self):
        """Create a simple portfolio optimization problem."""
        n = 3
        np.random.seed(42)
        mean = np.array([0.1, 0.15, 0.2])
        # Create a positive definite covariance matrix
        l_matrix = np.random.randn(n, n)
        covariance = l_matrix @ l_matrix.T
        lower_bounds = np.zeros(n)
        upper_bounds = np.ones(n)
        a = np.ones((1, n))  # Fully invested constraint
        b = np.ones(1)
        return {
            "mean": mean,
            "covariance": covariance,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "a": a,
            "b": b,
        }

    def test_iteration_cap_raises(self, simple_problem):
        """The safety cap raises if the event loop fails to terminate.

        ``_append`` is patched to keep only the first turning point, so the last
        turning point never changes: lambda is recomputed to the same positive
        value every iteration and the loop runs until the ``max_iterations``
        guard fires (the ``RuntimeError`` in ``__post_init__``).
        """

        def append_only_first(self, tp, tol=None):
            if not self.turning_points:
                self.turning_points.append(tp)

        with patch.object(CLA, "_append", append_only_first), pytest.raises(RuntimeError, match="failed to converge"):
            CLA(**simple_problem)

    def test_cla_initialization(self, simple_problem):
        """Test that CLA can be initialized with valid inputs."""
        cla = CLA(**simple_problem)
        assert cla.mean.shape == (3,)
        assert cla.covariance.shape == (3, 3)
        assert len(cla.turning_points) > 0

    def test_turning_points_generated(self, simple_problem):
        """Test that multiple turning points are generated."""
        cla = CLA(**simple_problem)
        # Should have at least 2 turning points (first at lambda=inf and last at lambda=0)
        assert len(cla) >= 2
        # First turning point should have lambda = inf
        assert cla.turning_points[0].lamb == np.inf
        # Last turning point should have lambda = 0
        assert cla.turning_points[-1].lamb == 0.0

    def test_weights_sum_to_one(self, simple_problem):
        """Test that all turning point weights sum to 1."""
        cla = CLA(**simple_problem)
        for tp in cla.turning_points:
            assert np.isclose(np.sum(tp.weights), 1.0)

    def test_weights_respect_bounds(self, simple_problem):
        """Test that all weights respect lower and upper bounds."""
        cla = CLA(**simple_problem)
        for tp in cla.turning_points:
            assert np.all(tp.weights >= simple_problem["lower_bounds"] - cla.tol)
            assert np.all(tp.weights <= simple_problem["upper_bounds"] + cla.tol)

    def test_lambda_decreasing(self, simple_problem):
        """Test that lambda values are decreasing along the frontier."""
        cla = CLA(**simple_problem)
        lambdas = [tp.lamb for tp in cla.turning_points]
        # Lambda should be monotonically decreasing
        for i in range(len(lambdas) - 1):
            assert lambdas[i] >= lambdas[i + 1]

    def test_frontier_property(self, simple_problem):
        """Test that the frontier property returns a valid Frontier object."""
        cla = CLA(**simple_problem)
        frontier = cla.frontier
        assert len(frontier) == len(cla.turning_points)
        assert np.array_equal(frontier.mean, cla.mean)
        assert np.array_equal(frontier.covariance, cla.covariance)

    def test_with_tight_bounds(self):
        """Test CLA with tight bounds on weights."""
        n = 4
        mean = np.array([0.08, 0.10, 0.12, 0.15])
        # Simple diagonal covariance
        covariance = np.diag([0.04, 0.09, 0.16, 0.25])
        lower_bounds = np.array([0.1, 0.1, 0.1, 0.1])
        upper_bounds = np.array([0.4, 0.4, 0.3, 0.3])
        a = np.ones((1, n))
        b = np.ones(1)

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            a=a,
            b=b,
        )

        assert len(cla) > 0
        for tp in cla.turning_points:
            assert np.all(tp.weights >= lower_bounds - cla.tol)
            assert np.all(tp.weights <= upper_bounds + cla.tol)

    def test_max_sharpe_ratio(self, simple_problem):
        """Test that maximum Sharpe ratio can be computed."""
        cla = CLA(**simple_problem)
        max_sr, max_weights = cla.frontier.max_sharpe

        assert isinstance(max_sr, float)
        assert max_weights.shape == (3,)
        assert np.isclose(np.sum(max_weights), 1.0)
        assert max_sr > 0  # Sharpe ratio should be positive with positive returns

    def test_with_different_tolerance(self, simple_problem):
        """Test CLA with different tolerance values."""
        cla1 = CLA(**simple_problem, tol=1e-5)
        cla2 = CLA(**simple_problem, tol=1e-8)

        # Both should produce valid frontiers
        assert len(cla1) > 0
        assert len(cla2) > 0

    def test_two_asset_problem(self):
        """Test CLA with a simple two-asset problem."""
        mean = np.array([0.1, 0.15])
        covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        lower_bounds = np.array([0.0, 0.0])
        upper_bounds = np.array([1.0, 1.0])
        a = np.ones((1, 2))
        b = np.ones(1)

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            a=a,
            b=b,
        )

        # Should generate a valid frontier
        assert len(cla) >= 2
        # First point should have all weight on higher return asset (if allowed)
        first_tp = cla.turning_points[0]
        assert first_tp.weights[1] >= first_tp.weights[0]  # Asset 1 has higher return


class TestCLAEdgeCases:
    """Test edge cases and special scenarios for CLA."""

    def test_single_asset(self):
        """Test with a single asset."""
        mean = np.array([0.1])
        covariance = np.array([[0.04]])
        lower_bounds = np.array([1.0])  # Must invest fully
        upper_bounds = np.array([1.0])
        a = np.ones((1, 1))
        b = np.ones(1)

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            a=a,
            b=b,
        )

        # Should have at least one turning point
        assert len(cla) >= 1
        # All weights should be 1.0
        for tp in cla.turning_points:
            assert np.isclose(tp.weights[0], 1.0)

    def test_many_assets(self):
        """Test with a larger number of assets."""
        n = 10
        np.random.seed(123)
        mean = np.random.rand(n) * 0.2
        l_matrix = np.random.randn(n, n)
        covariance = l_matrix @ l_matrix.T
        lower_bounds = np.zeros(n)
        upper_bounds = np.ones(n)
        a = np.ones((1, n))
        b = np.ones(1)

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            a=a,
            b=b,
        )

        assert len(cla) > 0
        # Verify all constraints are satisfied
        for tp in cla.turning_points:
            assert np.isclose(np.sum(tp.weights), 1.0)
            assert np.all(tp.weights >= -cla.tol)
            assert np.all(tp.weights <= 1.0 + cla.tol)

    def test_rank_deficient_covariance_raises_degeneracy(self):
        """A severely rank-deficient covariance yields a clear degeneracy diagnosis.

        When the free set grows past the covariance rank the free-asset block is
        numerically singular and its solve is unreliable. ``_emit`` detects this
        from the block's reciprocal condition number (a deterministic, portable
        signal, unlike the magnitude of the resulting box violation, which is the
        residual of a singular solve and varies with the BLAS/LAPACK build), so it
        raises an actionable degeneracy diagnosis (naming the remedy) rather than
        silently returning a possibly-suboptimal frontier. Contrast
        ``test_near_degenerate_trace_completes_via_projection``, where a merely
        near-degenerate (full-rank) problem completes cleanly.
        """
        rng = np.random.default_rng(2)
        factors = rng.standard_normal((20, 8))
        covariance = factors @ factors.T  # rank 8 < 20 assets
        n = 20
        with pytest.raises(ValueError, match="numerically singular"):
            CLA(
                mean=rng.uniform(0.0, 1.0, n),
                covariance=covariance,
                lower_bounds=np.zeros(n),
                upper_bounds=np.ones(n),
                a=np.ones((1, n)),
                b=np.ones(1),
            )

    def test_near_degenerate_trace_completes_via_projection(self):
        """A near-degenerate but adequately-ranked trace completes cleanly (issue #686).

        This case used to abort: accumulated round-off at a degenerate vertex put
        a free weight a hair (order 1e-5) outside its box. The round-off lies in
        the covariance's near-flat directions, so the candidate is optimal but not
        exactly feasible; ``_emit`` projects it back onto the feasible box and
        restores the budget, so the trace now completes. Here the covariance is
        even full-rank and well conditioned (the violation is pure round-off, not
        rank deficiency). We assert it completes with every turning point exactly
        feasible and a non-increasing lambda.
        """
        rng = np.random.default_rng(0)
        n, t_days = 80, 30  # T < n -> near-degenerate sample covariance
        returns = rng.standard_normal((t_days, n)) * 0.01 + rng.uniform(0.0, 1e-3, n)
        sample = np.cov(returns, rowvar=False)
        intensity = 0.3  # shrink to a full-rank, well-conditioned matrix
        covariance = (1.0 - intensity) * sample + intensity * (np.trace(sample) / n) * np.eye(n)
        assert np.linalg.matrix_rank(covariance) == n
        assert np.linalg.cond(covariance) < 100.0

        cla = CLA(
            mean=returns.mean(axis=0),
            covariance=covariance,
            lower_bounds=np.zeros(n),
            upper_bounds=np.ones(n),
            a=np.ones((1, n)),
            b=np.ones(1),
        )

        assert len(cla) > 2  # it traces a real frontier rather than aborting
        for tp in cla.turning_points:
            assert np.all(tp.weights >= -1e-9)  # exactly feasible after projection
            assert np.all(tp.weights <= 1.0 + 1e-9)
            assert abs(float(tp.weights.sum()) - 1.0) < 1e-9  # budget restored

    def test_factor_covariance_resolves_degeneracy(self):
        """The FactorCovariance backend is PD by construction and traces cleanly.

        The same low-rank risk structure that breaks the dense backend is handled
        when wrapped as a FactorCovariance with a positive idiosyncratic floor --
        the documented remedy for the degeneracy above.
        """
        rng = np.random.default_rng(2)
        n, k = 20, 8
        factors = rng.standard_normal((n, k))
        cov = factors @ factors.T
        evals, evecs = np.linalg.eigh(cov)
        top = np.argsort(evals)[::-1][:k]
        backend = FactorCovariance(
            d=np.full(n, 0.5 * np.trace(cov) / n),
            u=evecs[:, top],
            delta=np.clip(evals[top], 1e-8, None),
        )
        cla = CLA(
            mean=rng.uniform(0.0, 1.0, n),
            covariance=backend,
            lower_bounds=np.zeros(n),
            upper_bounds=np.ones(n),
            a=np.ones((1, n)),
            b=np.ones(1),
        )
        assert len(cla) >= 2

    def test_equal_returns(self):
        """Test with assets having equal expected returns."""
        mean = np.array([0.1, 0.1, 0.1])
        covariance = np.eye(3) * 0.04
        lower_bounds = np.zeros(3)
        upper_bounds = np.ones(3)
        a = np.ones((1, 3))
        b = np.ones(1)

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            a=a,
            b=b,
        )

        # Should still produce a valid frontier
        assert len(cla) > 0

    def test_no_short_selling(self):
        """Test no short selling constraint (lower bounds = 0)."""
        n = 5
        np.random.seed(456)
        mean = np.random.rand(n) * 0.15
        l_matrix = np.random.randn(n, n)
        covariance = l_matrix @ l_matrix.T
        lower_bounds = np.zeros(n)  # No short selling
        upper_bounds = np.ones(n)
        a = np.ones((1, n))
        b = np.ones(1)

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            a=a,
            b=b,
        )

        # All weights should be non-negative
        for tp in cla.turning_points:
            assert np.all(tp.weights >= -cla.tol)


class TestCLAErrors:
    """Test error conditions in CLA."""

    def test_all_variables_blocked_raises_runtime_error(self):
        """Test that RuntimeError is raised when all variables are blocked."""
        mean = np.array([0.1, 0.15, 0.2])
        covariance = np.eye(3) * 0.04
        lower_bounds = np.zeros(3)
        upper_bounds = np.ones(3)
        a = np.ones((1, 3))
        b = np.ones(1)

        # A TurningPoint with all free=False triggers the error in the main loop
        bad_tp = TurningPoint(
            weights=np.array([1.0, 0.0, 0.0]),
            free=np.array([False, False, False]),
        )

        with (
            patch.object(CLA, "_first_turning_point", return_value=bad_tp),
            pytest.raises(RuntimeError, match=r"^All variables cannot be blocked$"),
        ):
            CLA(
                mean=mean,
                covariance=covariance,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                a=a,
                b=b,
            )

    def test_append_weights_below_lower_bounds(self):
        """Test that _append raises ValueError when weights are below lower bounds."""
        mean = np.array([0.1, 0.15, 0.2])
        covariance = np.eye(3) * 0.04
        lower_bounds = np.zeros(3)
        upper_bounds = np.ones(3)
        a = np.ones((1, 3))
        b = np.ones(1)

        cla = CLA(mean=mean, covariance=covariance, lower_bounds=lower_bounds, upper_bounds=upper_bounds, a=a, b=b)

        # -0.5 is below lower_bounds[0]=0 - tol
        tp = TurningPoint(weights=np.array([-0.5, 0.75, 0.75]), free=np.array([True, False, False]))
        with pytest.raises(ValueError, match=r"^Weights below lower bounds$"):
            cla._append(tp)

    def test_append_weights_above_upper_bounds(self):
        """Test that _append raises ValueError when weights are above upper bounds."""
        mean = np.array([0.1, 0.15, 0.2])
        covariance = np.diag([0.04, 0.04, 0.04])
        lower_bounds = np.zeros(3)
        upper_bounds = np.array([0.8, 0.8, 0.8])  # tight upper bounds
        a = np.ones((1, 3))
        b = np.ones(1)

        cla = CLA(mean=mean, covariance=covariance, lower_bounds=lower_bounds, upper_bounds=upper_bounds, a=a, b=b)

        # weight 1.0 > upper_bounds[2]=0.8 + tol
        tp = TurningPoint(weights=np.array([0.0, 0.0, 1.0]), free=np.array([True, False, False]))
        with pytest.raises(ValueError, match=r"^Weights above upper bounds$"):
            cla._append(tp)

    def test_append_violates_equality_constraint(self):
        """_append raises when weights violate the equality constraint A w = b."""
        mean = np.array([0.1, 0.15, 0.2])
        covariance = np.eye(3) * 0.04
        lower_bounds = np.zeros(3)
        upper_bounds = np.ones(3)
        a = np.ones((1, 3))
        b = np.ones(1)

        cla = CLA(mean=mean, covariance=covariance, lower_bounds=lower_bounds, upper_bounds=upper_bounds, a=a, b=b)

        class FakeTP:
            weights = np.array([0.2, 0.2, 0.2])  # A w = 0.6, not 1

        with pytest.raises(ValueError, match=r"^Weights violate the equality constraint A w = b$"):
            cla._append(FakeTP())


class TestDegenerateProblems:
    """Regression tests for degenerate problems (https://github.com/cvxgrp/cvxcla/issues/648).

    Each case below raised ValueError ("Weights below lower bounds") before the
    event logic learned to (a) classify bounds with the configurable tol,
    (b) discard spurious event ratios above the current lambda, (c) keep
    slow bound crossings whose slope falls below the old tol filter, and
    (d) tolerate float error in the fully-invested check of init_algo.
    """

    def _assert_valid_frontier(self, cla, upper_bound):
        """Check lambda monotonicity (within tol) and bounds on all turning points."""
        lambdas = np.array([tp.lamb for tp in cla.turning_points])
        assert np.all(np.diff(lambdas) <= cla.tol)
        weights = np.array([tp.weights for tp in cla.turning_points])
        assert np.all(weights >= -cla.tol)
        assert np.all(weights <= upper_bound + cla.tol)

    @pytest.mark.parametrize("n", [20, 50, 100])
    def test_tied_mean_blocks_with_capped_weights(self, n):
        """Blocks of identical means with capped upper bounds produce tied events.

        Spurious ratios above the current lambda used to win the argmax and
        push weights far outside their bounds.
        """
        rng = np.random.default_rng(0)
        _ = rng.standard_normal((10, 10))  # keep historical draw order of the report
        _ = rng.standard_normal((6, 6))
        _ = rng.standard_normal(6)
        l_matrix = rng.standard_normal((n, n))
        covariance = l_matrix @ l_matrix.T + 0.1 * np.eye(n)
        mean = np.repeat(rng.standard_normal(n // 5), 5)

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=np.zeros(n),
            upper_bounds=np.full(n, 0.4),
            a=np.ones((1, n)),
            b=np.ones(1),
        )
        self._assert_valid_frontier(cla, 0.4)

    def test_slow_bound_crossing_is_not_missed(self):
        """A free weight with slope just below tol still crosses its bound.

        Over a long lambda range even a slope of ~7e-6 walks a weight out of
        bounds; the old event filter (|slope| > tol) silently dropped the
        crossing.
        """
        rng = np.random.default_rng(63)
        n = int(rng.integers(3, 40))
        l_matrix = rng.standard_normal((n, n))
        covariance = l_matrix @ l_matrix.T + 0.05 * np.eye(n)
        mean = rng.standard_normal(n)
        upper = np.full(n, float(rng.uniform(2.0 / n, 1.0)))

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=np.zeros(n),
            upper_bounds=upper,
            a=np.ones((1, n)),
            b=np.ones(1),
        )
        self._assert_valid_frontier(cla, upper[0])

    def test_first_turning_point_full_within_float_error(self):
        """The fully-invested check of init_algo needs a float tolerance.

        When the partial fill brings the sum to 1 only up to float error, the
        next asset (weight ~0, on its bound) used to be marked free while the
        interior asset stayed blocked, breaking the first segment.
        """
        rng = np.random.default_rng(411)
        n = int(rng.integers(3, 40))
        l_matrix = rng.standard_normal((n, n))
        covariance = l_matrix @ l_matrix.T + 0.05 * np.eye(n)
        mean = rng.standard_normal(n)
        upper = np.full(n, float(rng.uniform(2.0 / n, 1.0)))

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=np.zeros(n),
            upper_bounds=upper,
            a=np.ones((1, n)),
            b=np.ones(1),
        )
        # the first turning point must have its free asset strictly inside the bounds
        first = cla.turning_points[0]
        assert first.weights[first.free].item() > cla.tol
        self._assert_valid_frontier(cla, upper[0])

    def test_tie_heavy_equicorrelation_completes(self):
        """A large, highly tied problem traces to completion (issues #648, #708).

        Equicorrelated covariance with means tied in blocks and a tight cap forces
        many assets to their bound at once. Round-off then nudges several free
        weights a hair past the box at well-conditioned vertices. The old
        clip-then-rescale projection re-violated the cap (rescaling to restore the
        budget pushed the capped weights back over their bound) and aborted with a
        spurious "Weights above upper bounds". The capped-simplex projection
        respects box and budget together, so the trace now completes cleanly.
        """
        n, rho = 200, 0.99
        rng = np.random.default_rng(1)
        covariance = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
        mean = np.repeat(rng.standard_normal(n // 10), 10)
        cap = 2.0 / n

        cla = CLA(
            mean=mean,
            covariance=covariance,
            lower_bounds=np.zeros(n),
            upper_bounds=np.full(n, cap),
            a=np.ones((1, n)),
            b=np.ones(1),
        )
        assert len(cla) > 2  # a real frontier, not an immediate stop
        self._assert_valid_frontier(cla, cap)

    def test_project_feasible_respects_box_under_heavy_capping(self):
        """The projection never returns a point outside the box (issue #708).

        Constructs a budget-feasible (sum == 1) but box-violating candidate of the
        kind a degenerate vertex produces: several weights a hair over a tight cap,
        the rest compensating below. A plain clip-then-rescale would re-inflate the
        capped weights past the cap; the capped-simplex projection must not.
        """
        n = 10
        cap = 0.15
        cla = CLA(
            mean=np.linspace(0.1, 0.2, n),
            covariance=np.eye(n) * 0.04,
            lower_bounds=np.zeros(n),
            upper_bounds=np.full(n, cap),
            a=np.ones((1, n)),
            b=np.ones(1),
        )

        over = cap + 1e-4
        weights = np.full(n, over)
        weights[6:] = (1.0 - 6 * over) / (n - 6)  # restore sum == 1 below the cap
        assert np.isclose(weights.sum(), 1.0)
        assert weights.max() > cap  # genuinely box-infeasible

        projected = cla._project_feasible(weights)
        assert np.all(projected >= -1e-12)
        assert np.all(projected <= cap + 1e-12)
        assert np.isclose(projected.sum(), 1.0)

        # already-feasible candidates pass through untouched (exact no-op)
        feasible = np.full(n, 0.1)
        assert np.array_equal(cla._project_feasible(feasible), feasible)


class TestAppendTolerance:
    """Tests for the tol argument of _append (https://github.com/cvxgrp/cvxcla/issues/651)."""

    @pytest.fixture
    def cla(self):
        """A solved 3-asset problem to call _append on."""
        return CLA(
            mean=np.array([0.1, 0.15, 0.2]),
            covariance=np.eye(3) * 0.04,
            lower_bounds=np.zeros(3),
            upper_bounds=np.ones(3),
            a=np.ones((1, 3)),
            b=np.ones(1),
        )

    def test_explicit_zero_tol_is_honored(self, cla):
        """tol=0 must validate exactly instead of falling back to self.tol."""
        # violates the lower bound by less than self.tol but more than 0
        tp = TurningPoint(weights=np.array([-1e-7, 0.5, 0.5 + 1e-7]), free=np.array([True, False, False]))
        with pytest.raises(ValueError, match="Weights below lower bounds"):
            cla._append(tp, tol=0)

    def test_default_tol_accepts_tiny_violation(self, cla):
        """The same point passes under the default tolerance."""
        tp = TurningPoint(weights=np.array([-1e-7, 0.5, 0.5 + 1e-7]), free=np.array([True, False, False]))
        before = len(cla)
        cla._append(tp)
        assert len(cla) == before + 1


class TestCLAMutationHardening:
    """Targeted tests pinning behaviour that mutation testing flagged as unguarded."""

    @pytest.fixture
    def small_problem(self):
        """A small, well-conditioned problem with a fully-invested constraint."""
        np.random.seed(7)
        n = 4
        mean = np.array([0.10, 0.12, 0.15, 0.20])
        m = np.random.randn(n, n)
        covariance = m @ m.T
        return {
            "mean": mean,
            "covariance": covariance,
            "a": np.ones((1, n)),
            "b": np.ones(1),
        }

    def test_cla_is_frozen(self, small_problem):
        """CLA is an immutable (frozen) dataclass."""
        import dataclasses

        cla = CLA(lower_bounds=np.zeros(4), upper_bounds=np.ones(4), **small_problem)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cla.tol = 1e-3

    def test_cla_logger_is_a_logger(self, small_problem):
        """The default logger is a real ``logging.Logger`` instance, not None."""
        import logging

        cla = CLA(lower_bounds=np.zeros(4), upper_bounds=np.ones(4), **small_problem)
        assert isinstance(cla.logger, logging.Logger)

    def test_nonzero_lower_bounds_respected(self, small_problem):
        """Every turning point honours non-zero lower bounds.

        With non-zero lower bounds, the at-lower-bound test must subtract (not
        add) the bounds; otherwise blocked assets are misclassified and the
        frontier drifts below its lower bounds.
        """
        n = 4
        lower = np.full(n, 0.1)
        upper = np.ones(n)
        cla = CLA(lower_bounds=lower, upper_bounds=upper, **small_problem)

        assert len(cla.turning_points) >= 2
        for tp in cla.turning_points:
            assert np.all(tp.weights >= lower - cla.tol)
            assert np.all(tp.weights <= upper + cla.tol)
            assert np.isclose(np.sum(tp.weights), 1.0)


class TestCLAInternals:
    """Unit tests for the extracted per-iteration helpers of the trace.

    The turning-point loop (``cvxcla.pathtracer.trace``) delegates to three
    focused CLA helpers (``_active_set``, ``_solve_kkt``, ``_event_ratios``) plus
    the generic ``select_next_event``. These tests exercise each in isolation on a
    small, well-conditioned problem, asserting the contract the loop relies on.
    """

    @pytest.fixture
    def cla(self):
        """A small, well-conditioned long-only CLA whose helpers we probe."""
        n = 4
        rng = np.random.default_rng(7)
        m = rng.standard_normal((n, n))
        return CLA(
            mean=np.array([0.10, 0.12, 0.15, 0.20]),
            covariance=m @ m.T + n * np.eye(n),
            lower_bounds=np.zeros(n),
            upper_bounds=np.ones(n),
            a=np.ones((1, n)),
            b=np.ones(1),
        )

    def test_active_set_partitions_and_pins_bounds(self, cla):
        """at_upper/at_lower/free_in partition the assets; fixed weights sit on bounds."""
        last = cla.turning_points[0]
        at_upper, at_lower, free_in, fixed = cla._active_set(last)

        # A blocked asset is either at its upper or lower bound, never both;
        # free_in is exactly the complement of the union.
        assert not np.any(at_upper & at_lower)
        np.testing.assert_array_equal(free_in, ~(at_upper | at_lower))
        np.testing.assert_allclose(fixed[at_upper], cla.upper_bounds[at_upper])
        np.testing.assert_allclose(fixed[at_lower], cla.lower_bounds[at_lower])
        assert np.all(fixed[free_in] == 0.0)  # in-set entries are not pinned

    def test_active_set_all_blocked_raises(self, cla):
        """An all-blocked active set makes the reduced system singular and is rejected."""
        weights = np.zeros(cla.mean.size)
        weights[0] = 1.0  # sum to 1 so TurningPoint validation passes
        all_blocked = TurningPoint(lamb=0.0, weights=weights, free=np.zeros(cla.mean.size, dtype=bool))
        with pytest.raises(RuntimeError, match="All variables cannot be blocked"):
            cla._active_set(all_blocked)

    def test_solve_kkt_is_feasible_and_optimal_segment(self, cla):
        """The KKT solve returns an affine segment satisfying the equality constraints.

        ``r_alpha`` must satisfy ``A r_alpha = b`` and ``r_beta`` must lie in the
        null space of ``A`` (``A r_beta = 0``), so every point ``r_alpha + lam
        r_beta`` along the segment stays budget-feasible.
        """
        _, _, free_in, fixed = cla._active_set(cla.turning_points[0])
        r_alpha, r_beta, gamma, delta, eta_alpha, eta_beta = cla._solve_kkt(free_in, fixed)

        np.testing.assert_allclose(cla.a @ r_alpha, cla.b, atol=1e-10)
        np.testing.assert_allclose(cla.a @ r_beta, np.zeros(cla.a.shape[0]), atol=1e-10)
        assert gamma.shape == cla.mean.shape
        assert delta.shape == cla.mean.shape
        # No inequality rows on this fixture, so the multipliers are empty.
        assert eta_alpha.shape == (0,)
        assert eta_beta.shape == (0,)

    def test_event_ratios_shape_and_inactive_entries(self, cla):
        """The event matrix is (n, 4) and respects which events each asset can fire."""
        at_upper, at_lower, free_in, fixed = cla._active_set(cla.turning_points[0])
        r_alpha, r_beta, gamma, delta, _, _ = cla._solve_kkt(free_in, fixed)
        l_mat = cla._event_ratios(r_alpha, r_beta, gamma, delta, free_in, at_upper, at_lower)

        assert l_mat.shape == (cla.mean.size, 4)
        # A free asset has no "leave a bound" event (columns 2, 3); a blocked
        # asset has no "move to a bound" event (columns 0, 1).
        assert np.all(l_mat[free_in][:, 2:] == -np.inf)
        assert np.all(l_mat[~free_in][:, :2] == -np.inf)

    def test_select_next_event_bland_lowest_index_tiebreak(self, cla):
        """Among ratios tied within tol, the lowest (asset, event) index wins."""
        l_mat = np.full((cla.mean.size, 4), -np.inf)
        l_mat[3, 1] = 0.5
        l_mat[1, 0] = 0.5 + cla.tol / 2  # tied with [3, 1] to within tol
        event = select_next_event(l_mat, lam=np.inf, tol=cla.tol)
        assert event is not None
        secchg, dirchg, _ = event
        assert (secchg, dirchg) == (1, 0)

    def test_select_next_event_no_valid_event_returns_none(self, cla):
        """When every ratio lies above the current lam window, the trace stops."""
        l_mat = np.full((cla.mean.size, 4), -np.inf)
        l_mat[0, 0] = 5.0  # above the window -> filtered out
        assert select_next_event(l_mat, lam=1.0, tol=cla.tol) is None

    def test_select_next_event_does_not_mutate_input(self, cla):
        """Selection works on a copy, leaving the caller's matrix untouched."""
        l_mat = np.full((cla.mean.size, 4), -np.inf)
        l_mat[0, 0] = 5.0
        before = l_mat.copy()
        select_next_event(l_mat, lam=1.0, tol=cla.tol)
        np.testing.assert_array_equal(l_mat, before)


class TestGeneralEqualityConstraints:
    """General ``A w = b``: leverage, dollar-neutral, weighted, and multi-row.

    The turning-point KKT solve is general in ``A``; these tests exercise the
    general first vertex (the greedy fill for an all-ones row, the HiGHS LP for a
    weighted or multi-row ``A``), the ``A w = b`` validation in ``_append``, and
    the general feasibility projection.
    """

    @staticmethod
    def _problem(n, seed):
        """Return a random mean vector and a positive-definite covariance."""
        rng = np.random.default_rng(seed)
        lm = rng.standard_normal((n, n))
        return rng.standard_normal(n), lm @ lm.T + 0.1 * np.eye(n)

    def _assert_feasible_monotone(self, cla, a, b, lower, upper):
        """Assert every turning point is box/equality feasible and lambda decreases."""
        weights = np.array([tp.weights for tp in cla.turning_points])
        lambdas = np.array([tp.lamb for tp in cla.turning_points])
        assert np.all(weights >= lower - cla.tol)
        assert np.all(weights <= upper + cla.tol)
        assert np.allclose(weights @ a.T, b, atol=1e-6)
        assert np.all(np.diff(lambdas) <= cla.tol)

    def _assert_optimal(self, cla, mean, cov, a, b, lower, upper):
        """At sampled interior lambdas, the CLA point is no worse than a QP solve."""
        from scipy.optimize import minimize

        tps = cla.turning_points
        lambdas = [tp.lamb for tp in tps]
        weights = np.array([tp.weights for tp in tps])
        x0 = tps[0].weights  # the max-return vertex is feasible
        for frac in (0.3, 0.6, 0.85):
            i = int(frac * (len(tps) - 1))
            lam = lambdas[i]
            if not np.isfinite(lam) or lam <= 0:
                continue
            res = minimize(
                lambda w, lam=lam: 0.5 * w @ cov @ w - lam * mean @ w,
                x0=x0,
                jac=lambda w, lam=lam: cov @ w - lam * mean,
                method="SLSQP",
                bounds=list(zip(lower, upper, strict=True)),
                constraints=[{"type": "eq", "fun": lambda w: a @ w - b, "jac": lambda w: a}],
                options={"ftol": 1e-12, "maxiter": 500},
            )

            def obj(w, lam=lam):
                return 0.5 * w @ cov @ w - lam * mean @ w

            assert obj(weights[i]) <= obj(res.x) + 1e-5

    def test_leverage_all_ones_total_two(self):
        """All-ones row with b != 1 (leveraged total) traces via the greedy."""
        n = 12
        mean, cov = self._problem(n, 0)
        lower, upper = np.zeros(n), np.full(n, 0.5)
        a, b = np.ones((1, n)), np.array([2.0])
        cla = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b)
        self._assert_feasible_monotone(cla, a, b, lower, upper)
        self._assert_optimal(cla, mean, cov, a, b, lower, upper)

    def test_dollar_neutral_long_short(self):
        """All-ones row with b = 0 and short positions (negative lower bounds)."""
        n = 12
        mean, cov = self._problem(n, 1)
        lower, upper = np.full(n, -0.3), np.full(n, 0.3)
        a, b = np.ones((1, n)), np.array([0.0])
        cla = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b)
        self._assert_feasible_monotone(cla, a, b, lower, upper)
        self._assert_optimal(cla, mean, cov, a, b, lower, upper)

    def test_weighted_single_equality(self):
        """A single non-all-ones equality row uses the HiGHS LP first vertex."""
        n = 12
        mean, cov = self._problem(n, 2)
        lower, upper = np.zeros(n), np.ones(n)
        a = np.random.default_rng(99).uniform(0.5, 2.0, (1, n))
        b = np.array([1.0])
        cla = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b)
        self._assert_feasible_monotone(cla, a, b, lower, upper)
        self._assert_optimal(cla, mean, cov, a, b, lower, upper)

    def test_budget_plus_sector_neutral(self):
        """Two equality rows: a budget plus a sector-exposure target."""
        n = 12
        mean, cov = self._problem(n, 3)
        lower, upper = np.zeros(n), np.ones(n)
        a = np.vstack([np.ones(n), np.r_[np.ones(n // 2), np.zeros(n - n // 2)]])
        b = np.array([1.0, 0.5])
        cla = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b)
        self._assert_feasible_monotone(cla, a, b, lower, upper)
        self._assert_optimal(cla, mean, cov, a, b, lower, upper)

    def test_budget_plus_factor_neutral(self):
        """Three equality rows: a budget plus two factor-neutrality constraints."""
        n = 14
        mean, cov = self._problem(n, 4)
        lower, upper = np.full(n, -0.5), np.full(n, 0.5)
        a = np.vstack([np.ones(n), np.random.default_rng(7).standard_normal((2, n))])
        b = np.array([1.0, 0.0, 0.0])
        cla = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b)
        self._assert_feasible_monotone(cla, a, b, lower, upper)
        self._assert_optimal(cla, mean, cov, a, b, lower, upper)

    def test_infeasible_general_constraint_raises(self):
        """An unsatisfiable equality system is reported, not silently mis-solved."""
        n = 5
        mean, cov = self._problem(n, 5)
        # sum(w) = 10 is infeasible under 0 <= w <= 1 (max achievable is 5)
        with pytest.raises(ValueError, match="maximum-return vertex"):
            CLA(
                mean=mean,
                covariance=cov,
                lower_bounds=np.zeros(n),
                upper_bounds=np.ones(n),
                a=np.vstack([np.ones(n), np.eye(n)[0]]),
                b=np.array([10.0, 0.5]),
            )

    def test_degenerate_first_vertex_declined_with_diagnosis(self):
        """A degenerate general-A max-return vertex is declined with a clear error.

        With a = [budget; e_0] (so w_0 is pinned by the second row), the
        maximum-return vertex pins a basic asset on a box bound: the free set does
        not span the two equalities and the reduced KKT system is singular. This
        must be declined at the first vertex with an actionable message, not left
        to surface as an opaque "Singular matrix" error later in the trace.
        """
        n = 8
        mean, cov = self._problem(n, 6)
        a = np.vstack([np.ones(n), np.eye(n)[0]])
        b = np.array([1.0, 0.2])
        with pytest.raises(ValueError, match="maximum-return vertex is degenerate"):
            CLA(
                mean=mean,
                covariance=cov,
                lower_bounds=np.zeros(n),
                upper_bounds=np.full(n, 0.4),
                a=a,
                b=b,
            )

    def test_project_feasible_general_constraint(self):
        """The general (multi-row) projection restores box feasibility on A w = b.

        A candidate that satisfies A w = b but violates the box must be projected
        back inside the box while staying on the equality set (the alternating
        box / affine projection used for a general A).
        """
        n = 12
        mean, cov = self._problem(n, 3)
        half = np.r_[np.ones(n // 2), np.zeros(n - n // 2)]
        a = np.vstack([np.ones(n), half])
        b = np.array([1.0, 0.5])  # sum(w) = 1 and sum(first half) = 0.5
        cla = CLA(mean=mean, covariance=cov, lower_bounds=np.zeros(n), upper_bounds=np.ones(n), a=a, b=b)

        # start from the uniform feasible point, then push one weight below 0 while
        # preserving both equalities (offset within the first half)
        w = np.full(n, 1.0 / n)
        w[0] += 0.15
        w[1] -= 0.15
        assert np.allclose(a @ w, b, atol=1e-12)
        assert w.min() < 0.0  # box-infeasible

        projected = cla._project_feasible(w)
        assert np.all(projected >= -1e-9)
        assert np.all(projected <= 1.0 + 1e-9)
        assert np.allclose(a @ projected, b, atol=1e-7)


class TestInequalityConstraints:
    """General inequalities ``G w <= h`` (e.g. group- or sector-exposure caps).

    An active inequality row enters the reduced KKT solve as an extra equality
    row (the bordered system), and the events are the row analogue of the box
    events: an inactive row's slack reaching zero activates it, an active row's
    multiplier reaching zero releases it. These tests cover the activate and
    release transitions, exactness against a reference QP solver (with the
    inequality), multi-row caps, ``>=`` via negation, validation, and that the
    equality-only problem is recovered exactly when no inequality is supplied.
    """

    @staticmethod
    def _problem(n, seed):
        """Return a random mean vector and a positive-definite covariance."""
        rng = np.random.default_rng(seed)
        lm = rng.standard_normal((n, n))
        return rng.standard_normal(n), lm @ lm.T + 0.1 * np.eye(n)

    @staticmethod
    def _group_cap(n, cap):
        """A single inequality row capping the first half of the assets at ``cap``."""
        g = np.zeros((1, n))
        g[0, : n // 2] = 1.0
        return g, np.array([cap])

    def _assert_feasible_monotone(self, cla, a, b, g, h, lower, upper):
        """Every turning point is box/equality/inequality feasible; lambda decreases."""
        weights = np.array([tp.weights for tp in cla.turning_points])
        lambdas = np.array([tp.lamb for tp in cla.turning_points])
        assert np.all(weights >= lower - cla.tol)
        assert np.all(weights <= upper + cla.tol)
        assert np.allclose(weights @ a.T, b, atol=1e-6)
        assert np.all(weights @ g.T <= h + cla.tol)
        assert np.all(np.diff(lambdas) <= cla.tol)

    def _assert_optimal(self, cla, mean, cov, a, b, g, h, lower, upper):
        """At sampled interior lambdas, the CLA point is no worse than a QP solve."""
        from scipy.optimize import minimize

        tps = cla.turning_points
        lambdas = [tp.lamb for tp in tps]
        weights = np.array([tp.weights for tp in tps])
        x0 = tps[0].weights  # the max-return vertex is feasible
        for frac in (0.2, 0.4, 0.6, 0.8):
            i = int(frac * (len(tps) - 1))
            lam = lambdas[i]
            if not np.isfinite(lam) or lam <= 0:
                continue
            res = minimize(
                lambda w, lam=lam: 0.5 * w @ cov @ w - lam * mean @ w,
                x0=x0,
                jac=lambda w, lam=lam: cov @ w - lam * mean,
                method="SLSQP",
                bounds=list(zip(lower, upper, strict=True)),
                constraints=[
                    {"type": "eq", "fun": lambda w: a @ w - b, "jac": lambda w: a},
                    {"type": "ineq", "fun": lambda w: h - g @ w, "jac": lambda w: -g},
                ],
                options={"ftol": 1e-12, "maxiter": 1000},
            )

            def obj(w, lam=lam):
                return 0.5 * w @ cov @ w - lam * mean @ w

            assert obj(weights[i]) <= obj(res.x) + 1e-5

    def test_cap_binds_throughout(self):
        """A tight group cap that binds along the whole frontier (active every step)."""
        n = 10
        mean, cov = self._problem(n, 1)
        lower, upper = np.zeros(n), np.full(n, 0.34)
        g, h = self._group_cap(n, 0.5)
        cla = CLA(
            mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=np.ones((1, n)), b=np.ones(1), g=g, h=h
        )
        self._assert_feasible_monotone(cla, np.ones((1, n)), np.ones(1), g, h, lower, upper)
        self._assert_optimal(cla, mean, cov, np.ones((1, n)), np.ones(1), g, h, lower, upper)
        assert all(tp.active_ineq[0] for tp in cla.turning_points)

    def test_enter_event(self):
        """A cap that starts inactive at the max-return vertex and activates mid-trace."""
        n = 10
        mean, cov = self._problem(n, 3)
        lower, upper = np.zeros(n), np.full(n, 0.34)
        g, h = self._group_cap(n, 0.5)
        a, b = np.ones((1, n)), np.ones(1)
        cla = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b, g=g, h=h)
        self._assert_feasible_monotone(cla, a, b, g, h, lower, upper)
        self._assert_optimal(cla, mean, cov, a, b, g, h, lower, upper)
        active = [bool(tp.active_ineq[0]) for tp in cla.turning_points]
        assert not active[0]  # inactive at the max-return vertex
        assert any(active)  # an enter event activates it later
        # the cap is tight exactly where it is recorded active
        for tp in cla.turning_points:
            if tp.active_ineq[0]:
                assert np.isclose(g @ tp.weights, h, atol=1e-6)

    def test_release_event(self):
        """A cap that starts active at the max-return vertex and releases mid-trace."""
        n = 8
        mean, cov = self._problem(n, 0)
        lower, upper = np.zeros(n), np.ones(n)
        g = np.zeros((1, n))
        g[0, :4] = 1.0
        h = np.array([0.7])
        a, b = np.ones((1, n)), np.ones(1)
        cla = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b, g=g, h=h)
        self._assert_feasible_monotone(cla, a, b, g, h, lower, upper)
        self._assert_optimal(cla, mean, cov, a, b, g, h, lower, upper)
        active = [bool(tp.active_ineq[0]) for tp in cla.turning_points]
        assert active[0]  # active at the max-return vertex
        assert not active[-1]  # a release event frees it before the end

    def test_redundant_row_never_binds(self):
        """A loose cap that is never tight leaves the frontier identical to no cap."""
        n = 10
        mean, cov = self._problem(n, 2)
        lower, upper = np.zeros(n), np.full(n, 0.34)
        a, b = np.ones((1, n)), np.ones(1)
        # The first half can sum to at most the budget (1); a cap of 1.5 never binds.
        g, h = self._group_cap(n, 1.5)
        with_cap = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b, g=g, h=h)
        without = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b)
        assert not any(tp.active_ineq[0] for tp in with_cap.turning_points)
        w_cap = np.array([tp.weights for tp in with_cap.turning_points])
        w_no = np.array([tp.weights for tp in without.turning_points])
        assert w_cap.shape == w_no.shape
        np.testing.assert_allclose(w_cap, w_no, atol=1e-9)

    def test_multi_row_caps(self):
        """Two simultaneous group caps (first half and second half)."""
        n = 10
        mean, cov = self._problem(n, 5)
        lower, upper = np.zeros(n), np.full(n, 0.34)
        a, b = np.ones((1, n)), np.ones(1)
        g = np.zeros((2, n))
        g[0, : n // 2] = 1.0
        g[1, n // 2 :] = 1.0
        h = np.array([0.6, 0.6])
        cla = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b, g=g, h=h)
        self._assert_feasible_monotone(cla, a, b, g, h, lower, upper)
        self._assert_optimal(cla, mean, cov, a, b, g, h, lower, upper)

    def test_ge_constraint_via_negation(self):
        """A ``>=`` floor is expressed by negating both sides into ``G w <= h``."""
        n = 10
        mean, cov = self._problem(n, 7)
        lower, upper = np.zeros(n), np.full(n, 0.34)
        a, b = np.ones((1, n)), np.ones(1)
        # require the first half to hold at least 0.5: -sum(first half) <= -0.5
        g = np.zeros((1, n))
        g[0, : n // 2] = -1.0
        h = np.array([-0.5])
        cla = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b, g=g, h=h)
        self._assert_feasible_monotone(cla, a, b, g, h, lower, upper)
        self._assert_optimal(cla, mean, cov, a, b, g, h, lower, upper)
        # the floor is met everywhere
        for tp in cla.turning_points:
            assert tp.weights[: n // 2].sum() >= 0.5 - cla.tol

    def test_no_inequality_recovers_equality_problem(self):
        """``g=None`` (the default) reproduces the equality-only frontier exactly."""
        n = 8
        mean, cov = self._problem(n, 9)
        lower, upper = np.zeros(n), np.ones(n)
        a, b = np.ones((1, n)), np.ones(1)
        explicit_none = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b, g=None, h=None)
        default = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b)
        w1 = np.array([tp.weights for tp in explicit_none.turning_points])
        w2 = np.array([tp.weights for tp in default.turning_points])
        assert w1.shape == w2.shape
        np.testing.assert_array_equal(w1, w2)
        assert all(tp.active_ineq.shape == (0,) for tp in default.turning_points)

    def test_g_column_mismatch_raises(self):
        """An inequality matrix with the wrong number of columns is rejected."""
        n = 5
        mean, cov = self._problem(n, 0)
        with pytest.raises(ValueError, match="g must have 5 columns"):
            CLA(
                mean=mean,
                covariance=cov,
                lower_bounds=np.zeros(n),
                upper_bounds=np.ones(n),
                a=np.ones((1, n)),
                b=np.ones(1),
                g=np.ones((1, n + 1)),
                h=np.ones(1),
            )

    def test_h_length_mismatch_raises(self):
        """An inequality right-hand side whose length disagrees with ``g`` is rejected."""
        n = 5
        mean, cov = self._problem(n, 0)
        with pytest.raises(ValueError, match="h must have 2 entries"):
            CLA(
                mean=mean,
                covariance=cov,
                lower_bounds=np.zeros(n),
                upper_bounds=np.ones(n),
                a=np.ones((1, n)),
                b=np.ones(1),
                g=np.ones((2, n)),
                h=np.ones(3),
            )

    def test_append_rejects_inequality_violation(self):
        """``_append`` rejects a turning point that violates ``G w <= h``."""
        n = 4
        mean, cov = self._problem(n, 0)
        g = np.zeros((1, n))
        g[0, :2] = 1.0
        cla = CLA(
            mean=mean,
            covariance=cov,
            lower_bounds=np.zeros(n),
            upper_bounds=np.full(n, 0.4),
            a=np.ones((1, n)),
            b=np.ones(1),
            g=g,
            h=np.array([0.5]),
        )
        # a budget-feasible point whose first two weights breach the 0.5 cap
        bad = TurningPoint(weights=np.array([0.4, 0.4, 0.1, 0.1]), free=np.ones(n, dtype=bool))
        with pytest.raises(ValueError, match="inequality constraint G w <= h"):
            cla._append(bad, tol=0.0)

    def test_project_feasible_with_active_row(self):
        """The projection restores box feasibility while holding an active row at equality."""
        n = 8
        mean, cov = self._problem(n, 3)
        g, h = self._group_cap(n, 0.5)
        a, b = np.ones((1, n)), np.ones(1)
        cla = CLA(
            mean=mean, covariance=cov, lower_bounds=np.zeros(n), upper_bounds=np.full(n, 0.34), a=a, b=b, g=g, h=h
        )

        # Build a point on {sum(w)=1, sum(first half)=0.5} that breaches the box,
        # then project with the cap row active: it must come back inside the box
        # while preserving both the budget and the (active) cap held at equality.
        w = np.full(n, 1.0 / n)
        w[: n // 2] = 0.5 / (n // 2)  # first half sums to the cap
        w[0] += 0.3
        w[1] -= 0.3  # still on both affine sets, but now box-infeasible
        assert w.min() < 0.0
        active = np.array([True])
        projected = cla._project_feasible(w, active)
        assert np.all(projected >= -1e-9)
        assert np.all(projected <= 0.34 + 1e-9)
        assert np.isclose(projected.sum(), 1.0, atol=1e-7)
        assert np.isclose(g @ projected, h, atol=1e-7)

    def test_degenerate_inequality_first_vertex_declined(self):
        """A degenerate max-return vertex with an inequality present is declined.

        With a loose box and a single best asset that the cap does not touch, the
        max-return LP puts the whole budget on that one asset: the free set is
        empty, so it cannot span the budget row and the bordered KKT system is
        singular. The inequality machinery must not mask this; it is declined at
        the first vertex with the actionable diagnosis rather than surfacing as an
        opaque singular-matrix error later in the trace.
        """
        n = 8
        # asset n-1 has the strictly highest return and lies in the uncapped
        # second half, so the LP loads the entire budget onto it.
        mean = np.arange(n, dtype=float)
        _, cov = self._problem(n, 6)
        g, h = self._group_cap(n, 0.5)
        with pytest.raises(ValueError, match="maximum-return vertex is degenerate"):
            CLA(
                mean=mean,
                covariance=cov,
                lower_bounds=np.zeros(n),
                upper_bounds=np.ones(n),
                a=np.ones((1, n)),
                b=np.ones(1),
                g=g,
                h=h,
            )

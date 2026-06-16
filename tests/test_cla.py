"""Tests for the CLA module.

This module tests the main Critical Line Algorithm implementation,
including initialization, turning point computation, and frontier generation.
"""

from unittest.mock import patch

import numpy as np
import pytest

from cvxcla import CLA, FactorCovariance
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

    def test_append_weights_do_not_sum_to_one(self):
        """Test that _append raises ValueError when weights don't sum to 1."""
        mean = np.array([0.1, 0.15, 0.2])
        covariance = np.eye(3) * 0.04
        lower_bounds = np.zeros(3)
        upper_bounds = np.ones(3)
        a = np.ones((1, 3))
        b = np.ones(1)

        cla = CLA(mean=mean, covariance=covariance, lower_bounds=lower_bounds, upper_bounds=upper_bounds, a=a, b=b)

        # Use a duck-typed object to bypass TurningPoint's own sum-to-1 validation
        class FakeTP:
            weights = np.array([0.2, 0.2, 0.2])  # sums to 0.6, not 1

        with pytest.raises(ValueError, match=r"^Weights do not sum to 1$"):
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

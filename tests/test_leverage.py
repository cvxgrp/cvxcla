"""Tests for the gross-exposure (leverage) constraint ``||w||_1 <= c``.

The CLA traces the cap exactly by splitting every asset whose box straddles zero
into a long and a short leg (see :mod:`cvxcla._leverage`). The central guarantees
checked here are that every turning point, and every point between two of them,
is optimal for the leverage-constrained QP at its ``lambda``; that a cap which
never binds reproduces the plain frontier; and that the lift's building blocks
(the signed operator, the leg map, the event mask, the first-vertex netting)
behave as documented.
"""

import itertools
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import minimize

from cvxcla import CLA, DenseCovariance, FactorCovariance
from cvxcla._leverage import LeverageLift, SignedLift, mask_leg_events
from cvxcla.first import first_vertex_lp
from cvxcla.types import TurningPoint


def _problem(n, seed):
    """A random well-conditioned (mean, covariance) pair."""
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((n, n))
    return rng.standard_normal(n), factors @ factors.T / n + 0.05 * np.eye(n)


def _reference(mean, cov, lower, upper, a, b, c, lam):
    """Optimal objective of the leverage-constrained QP at ``lam``, via SLSQP on the split legs."""
    n = len(mean)

    def objective(x):
        w = x[:n] - x[n:]
        return 0.5 * w @ cov @ w - lam * mean @ w

    def gradient(x):
        g = cov @ (x[:n] - x[n:]) - lam * mean
        return np.concatenate([g, -g])

    split = np.hstack([np.eye(n), -np.eye(n)])
    x0 = np.concatenate([np.clip((lower + upper) / 2, 0, None), np.clip(-(lower + upper) / 2, 0, None)])
    res = minimize(
        objective,
        x0=x0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, max(u, 0.0)) for u in upper] + [(0.0, max(-lo, 0.0)) for lo in lower],
        constraints=[
            {"type": "eq", "fun": lambda x: a @ split @ x - b, "jac": lambda x: a @ split},
            {"type": "ineq", "fun": lambda x: c - np.sum(x), "jac": lambda x: -np.ones(2 * n)},
            {"type": "ineq", "fun": lambda x: split @ x - lower, "jac": lambda x: split},
            {"type": "ineq", "fun": lambda x: upper - split @ x, "jac": lambda x: -split},
        ],
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    return res.fun


def _assert_optimal_path(cla, mean, cov, lower, upper, a, b, c):
    """Turning points and segment midpoints are optimal and feasible under the cap."""
    tps = cla.turning_points
    for prev, tp in itertools.pairwise(tps):
        if np.isfinite(prev.lamb):
            lam, w = 0.5 * (prev.lamb + tp.lamb), 0.5 * (prev.weights + tp.weights)
        else:
            lam, w = tp.lamb, tp.weights
        assert np.abs(w).sum() <= c + 1e-8
        assert np.allclose(a @ w, b)
        assert 0.5 * w @ cov @ w - lam * mean @ w <= _reference(mean, cov, lower, upper, a, b, c, lam) + 1e-7


def _plain_weights_at(cla, lam):
    """Weights of a traced frontier at ``lam``, interpolated between its turning points."""
    lams = np.array([tp.lamb for tp in cla.turning_points])
    weights = np.array([tp.weights for tp in cla.turning_points])
    i = int(np.searchsorted(-lams, -lam))  # lams are non-increasing
    if lams[i] == lam or not np.isfinite(lams[i - 1]):
        return weights[i]
    t = (lams[i - 1] - lam) / (lams[i - 1] - lams[i])
    return (1 - t) * weights[i - 1] + t * weights[i]


class TestLeverageFrontier:
    """End-to-end traces under a gross-exposure cap."""

    @pytest.mark.parametrize("seed", range(6))
    def test_long_short_130_30(self, seed):
        """A 130/30 book: fully invested, gross exposure at most 1.3, optimal along the path."""
        n = 8
        mean, cov = _problem(n, seed)
        lower, upper = np.full(n, -0.3), np.full(n, 0.6)
        a, b = np.ones((1, n)), np.ones(1)
        cla = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b, leverage=1.3)
        _assert_optimal_path(cla, mean, cov, lower, upper, a, b, 1.3)
        gross = [np.abs(tp.weights).sum() for tp in cla.turning_points]
        assert np.isclose(max(gross), 1.3)  # the cap binds somewhere along the path
        assert np.all(np.diff([tp.lamb for tp in cla.turning_points]) <= cla.tol)

    def test_dollar_neutral(self):
        """A dollar-neutral book whose gross exposure is capped at 1."""
        n = 7
        mean, cov = _problem(n, 11)
        lower, upper = np.full(n, -0.4), np.full(n, 0.4)
        a, b = np.ones((1, n)), np.zeros(1)
        cla = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b, leverage=1.0)
        _assert_optimal_path(cla, mean, cov, lower, upper, a, b, 1.0)
        lams = [tp.lamb for tp in cla.turning_points]
        assert np.all(np.diff(lams) < 0)  # no repeated endpoint at lambda = 0
        assert lams[-1] == 0.0
        assert np.allclose(cla.turning_points[-1].weights, 0.0)

    def test_homogeneous_lasso_endpoint_is_recorded_once(self):
        """Under Sigma = X^T X, mu = X^T y and no budget, the path ends once at the origin."""
        rng = np.random.default_rng(0)
        n = 8
        x = rng.standard_normal((40, n))
        y = x @ rng.standard_normal(n) + 0.3 * rng.standard_normal(40)
        cla = CLA(
            mean=x.T @ y,
            covariance=x.T @ x,
            lower_bounds=np.full(n, -50.0),
            upper_bounds=np.full(n, 50.0),
            a=np.zeros((0, n)),
            b=np.zeros(0),
            leverage=5.0,
        )
        lams = [tp.lamb for tp in cla.turning_points]
        assert np.all(np.diff(lams) < 0)
        assert lams.count(0.0) == 1

    def test_cap_at_budget_is_the_long_only_frontier(self):
        """With a fully-invested budget, leverage = 1 forbids shorts: the long-only frontier."""
        n = 7
        mean, cov = _problem(n, 11)
        kwargs = {"mean": mean, "covariance": cov, "a": np.ones((1, n)), "b": np.ones(1)}
        capped = CLA(**kwargs, lower_bounds=np.full(n, -0.4), upper_bounds=np.full(n, 0.4), leverage=1.0)
        long_only = CLA(**kwargs, lower_bounds=np.zeros(n), upper_bounds=np.full(n, 0.4))
        assert len(capped) == len(long_only)
        for p, q in zip(capped.turning_points, long_only.turning_points, strict=True):
            assert np.allclose(p.weights, q.weights)

    def test_mixed_long_only_short_only_and_two_sided_assets(self):
        """Long-only, short-only and two-sided assets together, plus a group cap."""
        n = 6
        mean, cov = _problem(n, 5)
        lower = np.array([0.0, 0.0, -0.53, -0.47, -0.41, -0.29])
        upper = np.array([0.61, 0.57, 0.52, 0.49, -0.05, 0.43])
        a, b = np.ones((1, n)), np.ones(1)
        g, h = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]), np.array([0.83])
        cla = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b, g=g, h=h, leverage=1.93)
        _assert_optimal_path(cla, mean, cov, lower, upper, a, b, 1.93)
        for tp in cla.turning_points:
            assert tp.active_ineq.shape == (1,)  # the cap row is not reported
            assert tp.weights[4] <= -0.05 + cla.tol
            assert g @ tp.weights <= h + cla.tol

    def test_loose_cap_reproduces_plain_frontier(self):
        """A cap that never binds traces the same frontier as no cap at all.

        The lift adds collinear turning points where a weight crosses zero, so the
        comparison is made at the lifted trace's lambdas on the plain path.
        """
        n = 8
        mean, cov = _problem(n, 3)
        lower, upper = np.full(n, -0.5), np.full(n, 0.8)
        kwargs = {"mean": mean, "covariance": cov, "lower_bounds": lower, "upper_bounds": upper}
        kwargs |= {"a": np.ones((1, n)), "b": np.ones(1)}
        plain = CLA(**kwargs)
        capped = CLA(**kwargs, leverage=100.0)
        for tp in capped.turning_points:
            assert np.allclose(tp.weights, _plain_weights_at(plain, tp.lamb), atol=1e-8)

    def test_long_only_is_not_enlarged(self):
        """With only long-only assets the cap equals the budget row and changes nothing."""
        n = 6
        mean, cov = _problem(n, 7)
        kwargs = {"mean": mean, "covariance": cov, "lower_bounds": np.zeros(n), "upper_bounds": np.full(n, 0.37)}
        kwargs |= {"a": np.ones((1, n)), "b": np.ones(1)}
        plain = CLA(**kwargs)
        capped = CLA(**kwargs, leverage=1.5)
        assert len(plain) == len(capped)
        for p, q in zip(plain.turning_points, capped.turning_points, strict=True):
            assert np.allclose(p.weights, q.weights)

    def test_factor_backend(self):
        """A structured ``FactorCovariance`` backend traces the same capped frontier as dense."""
        n, k = 10, 3
        rng = np.random.default_rng(4)
        loadings, d = rng.standard_normal((n, k)), rng.uniform(0.05, 0.2, n)
        cov = loadings @ loadings.T + np.diag(d)
        mean = rng.standard_normal(n)
        lower, upper = np.full(n, -0.3), np.full(n, 0.5)
        kwargs = {"mean": mean, "lower_bounds": lower, "upper_bounds": upper}
        kwargs |= {"a": np.ones((1, n)), "b": np.ones(1), "leverage": 1.57}
        dense = CLA(covariance=cov, **kwargs)
        factor = CLA(covariance=FactorCovariance(d, loadings, np.ones(k)), **kwargs)
        assert len(dense) == len(factor)
        for p, q in zip(dense.turning_points, factor.turning_points, strict=True):
            assert np.allclose(p.weights, q.weights)

    def test_frontier_uses_asset_weights(self):
        """The ``frontier`` is built on the original covariance over asset weights."""
        n = 5
        mean, cov = _problem(n, 1)
        cla = CLA(
            mean=mean,
            covariance=cov,
            lower_bounds=np.full(n, -0.5),
            upper_bounds=np.ones(n),
            a=np.ones((1, n)),
            b=np.ones(1),
            leverage=1.4,
        )
        frontier = cla.frontier
        assert frontier.weights.shape == (len(cla), n)
        assert np.allclose(frontier.variance, [w @ cov @ w for w in frontier.weights])

    def test_rank_deficient_covariance_raises_degeneracy(self):
        """The degeneracy guard still fires through the lift when the free set outgrows the rank."""
        rng = np.random.default_rng(2)
        n = 20
        factors = rng.standard_normal((n, 8))
        with pytest.raises(ValueError, match="numerically singular"):
            CLA(
                mean=rng.uniform(0.0, 1.0, n),
                covariance=factors @ factors.T,
                lower_bounds=np.full(n, -0.2),
                upper_bounds=np.ones(n),
                a=np.ones((1, n)),
                b=np.ones(1),
                leverage=1.5,
            )


class TestFirstVertexNetting:
    """A max-return vertex with both legs of one asset positive is netted before tracing."""

    def test_overlapping_lp_vertex_is_netted(self):
        """HiGHS returns asset 3 with both legs positive here; the trace still is optimal.

        At this vertex the cap is tight but carries a zero multiplier (the box
        binds), so the LP optimum is not unique and HiGHS picks the overlapping one.
        """
        rng = np.random.default_rng(57)
        n = int(rng.integers(4, 20))
        factors = rng.standard_normal((n, n))
        cov = factors @ factors.T / n + 0.05 * np.eye(n)
        mean = rng.standard_normal(n)
        lower, upper = -rng.uniform(0, 1, n), rng.uniform(0, 1, n)
        lower[rng.random(n) < 0.3] = 0.0
        c = float(rng.uniform(1.2, 2.5))
        a, b = np.ones((1, n)), np.zeros(1)

        net = LeverageLift.net
        netted = []

        def spy(self, x):
            y = net(self, x)
            netted.append(not np.array_equal(x, y))
            return y

        with patch.object(LeverageLift, "net", spy):
            cla = CLA(mean=mean, covariance=cov, lower_bounds=lower, upper_bounds=upper, a=a, b=b, leverage=c)
        assert netted == [True]
        _assert_optimal_path(cla, mean, cov, lower, upper, a, b, c)

    def test_injected_overlap_restores_the_trace(self):
        """Adding the same amount to both legs of a free asset is undone by the netting."""
        n = 6
        mean, cov = _problem(n, 2)
        kwargs = {"mean": mean, "covariance": cov, "lower_bounds": np.full(n, -0.5), "upper_bounds": np.full(n, 0.9)}
        kwargs |= {"a": np.ones((1, n)), "b": np.ones(1), "leverage": 2.7}
        clean = CLA(**kwargs)

        def overlapping(*args, **kw):
            tp = first_vertex_lp(*args, **kw)
            weights = tp.weights.copy()
            weights[:2] += 0.1  # both legs of asset 0 (legs 0 and 1)
            return TurningPoint(weights=weights, free=tp.free, active_ineq=tp.active_ineq)

        with patch("cvxcla.first.first_vertex_lp", overlapping):
            perturbed = CLA(**kwargs)
        assert len(clean) == len(perturbed)
        for p, q in zip(clean.turning_points, perturbed.turning_points, strict=True):
            assert np.allclose(p.weights, q.weights)


class TestLeverageValidation:
    """Invalid caps and infeasible problems are declined with a clear message."""

    @pytest.mark.parametrize("leverage", [0.0, -1.0, np.inf, np.nan])
    def test_invalid_leverage_raises(self, leverage):
        """A non-positive or non-finite cap is rejected."""
        n = 3
        mean, cov = _problem(n, 0)
        with pytest.raises(ValueError, match="leverage must be a positive finite number"):
            CLA(
                mean=mean,
                covariance=cov,
                lower_bounds=np.full(n, -1.0),
                upper_bounds=np.ones(n),
                a=np.ones((1, n)),
                b=np.ones(1),
                leverage=leverage,
            )

    def test_infeasible_cap_raises(self):
        """A cap below the budget admits no portfolio."""
        n = 3
        mean, cov = _problem(n, 0)
        with pytest.raises(ValueError, match="Could not find a maximum-return vertex"):
            CLA(
                mean=mean,
                covariance=cov,
                lower_bounds=np.full(n, -1.0),
                upper_bounds=np.ones(n),
                a=np.ones((1, n)),
                b=np.ones(1),
                leverage=0.5,
            )

    def test_infeasible_budget_raises_through_the_first_vertex(self):
        """Constraints infeasible before any cap still fail with the first-vertex message."""
        n = 3
        mean, cov = _problem(n, 0)
        with pytest.raises(ValueError, match="Could not find a maximum-return vertex"):
            CLA(
                mean=mean,
                covariance=cov,
                lower_bounds=np.full(n, -1.0),
                upper_bounds=np.ones(n),
                a=np.ones((1, n)),
                b=np.array([10.0]),  # more than the boxes allow
                leverage=20.0,
            )

    def test_append_rejects_leverage_violation(self):
        """``_append`` refuses a turning point whose gross exposure exceeds the cap."""
        n = 3
        mean, cov = _problem(n, 0)
        cla = CLA(
            mean=mean,
            covariance=cov,
            lower_bounds=np.full(n, -1.0),
            upper_bounds=np.ones(n),
            a=np.ones((1, n)),
            b=np.ones(1),
            leverage=1.2,
        )
        tp = TurningPoint(weights=np.array([1.0, 0.5, -0.5]), free=np.ones(n, dtype=bool))
        with pytest.raises(ValueError, match=r"^Weights violate the leverage constraint"):
            cla._append(tp)


class TestLeverageLift:
    """The leg map built from the asset box."""

    def test_legs_from_bounds(self):
        """Long-only, short-only and two-sided assets get the documented legs."""
        lift = LeverageLift.from_bounds(np.array([0.1, -0.5, -0.3]), np.array([0.6, -0.2, 0.4]))
        assert lift.asset.tolist() == [0, 1, 2, 2]
        assert lift.sign.tolist() == [1.0, -1.0, 1.0, -1.0]
        assert lift.partner.tolist() == [-1, -1, 3, 2]
        assert lift.lower.tolist() == [0.1, 0.2, 0.0, 0.0]
        assert lift.upper.tolist() == [0.6, 0.5, 0.4, 0.3]

    def test_maps_and_net(self):
        """``columns``/``to_assets``/``any_leg`` apply ``P``; ``net`` keeps ``P x`` fixed."""
        lift = LeverageLift.from_bounds(np.array([0.0, -1.0]), np.array([1.0, 1.0]))
        x = np.array([0.5, 0.4, 0.1])
        assert np.allclose(lift.to_assets(x, 2), [0.5, 0.3])
        assert np.allclose(lift.columns(np.array([[1.0, 2.0]])), [[1.0, 2.0, -2.0]])
        assert lift.any_leg(np.array([False, False, True]), 2).tolist() == [False, True]
        netted = lift.net(x)
        assert np.allclose(netted, [0.5, 0.3, 0.0])
        assert np.allclose(lift.to_assets(netted, 2), lift.to_assets(x, 2))


class TestSignedLift:
    """The lifted quadratic form ``P.T Sigma P`` accessed without forming it."""

    @pytest.fixture
    def lifted(self):
        """A 3-asset covariance lifted onto legs (asset 1 two-sided)."""
        _, cov = _problem(3, 9)
        lift = LeverageLift.from_bounds(np.array([0.0, -1.0, -1.0]), np.array([1.0, 1.0, -0.1]))
        p = np.zeros((3, lift.asset.shape[0]))
        p[lift.asset, np.arange(lift.asset.shape[0])] = lift.sign
        return SignedLift(DenseCovariance(cov), lift.asset, lift.sign), p.T @ cov @ p

    def test_products_match_dense(self, lifted):
        """``matvec`` and ``block_matvec`` (with duplicate assets) match the dense lift."""
        op, dense = lifted
        rng = np.random.default_rng(0)
        x = rng.standard_normal(op.n)
        assert op.n == 4
        assert np.allclose(op.matvec(x), dense @ x)
        rows, cols = np.array([0, 3]), np.array([1, 2, 3])
        assert np.allclose(op.block_matvec(rows, cols, x[cols]), dense[np.ix_(rows, cols)] @ x[cols])
        block = rng.standard_normal((3, 2))
        assert np.allclose(op.block_matvec(rows, cols, block), dense[np.ix_(rows, cols)] @ block)

    def test_solve_and_rcond_on_one_leg_per_asset(self, lifted):
        """With one leg per asset the free block is invertible and matches the dense solve."""
        op, dense = lifted
        free = np.array([0, 2, 3])
        rhs = np.arange(6.0).reshape(3, 2)
        assert np.allclose(op.solve_free(free, rhs), np.linalg.solve(dense[np.ix_(free, free)], rhs))
        assert np.isclose(op.rcond_free(free), 1.0 / np.linalg.cond(dense[np.ix_(free, free)]))

    def test_both_legs_free_is_singular(self, lifted):
        """Both legs of one asset free: rcond is zero and the solve refuses."""
        op, _ = lifted
        both = np.array([1, 2])
        assert op.rcond_free(both) == 0.0
        with pytest.raises(np.linalg.LinAlgError, match="both legs"):
            op.solve_free(both, np.ones(2))


def test_mask_leg_events():
    """A leg's leave events are dropped exactly when its partner is off its lower bound."""
    box = np.arange(12.0).reshape(3, 4)
    partner = np.array([1, 0, -1])
    at_lower = np.array([True, False, False])
    masked = mask_leg_events(box, partner, at_lower)
    assert masked[0, 2:].tolist() == [-np.inf, -np.inf]  # partner (leg 1) is off its lower bound
    assert masked[1].tolist() == box[1].tolist()  # partner (leg 0) is at its lower bound
    assert masked[2].tolist() == box[2].tolist()  # no partner
    assert masked[0, :2].tolist() == box[0, :2].tolist()
    assert box[0, 2] == 2.0  # the input is not mutated

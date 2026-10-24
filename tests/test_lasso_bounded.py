"""Tests for the LASSO under coefficient bounds and equality rows ``A beta = b``.

Per-coefficient bounds ``lower <= beta <= upper`` (caps, and signs on some of the
coefficients) and nonhomogeneous rows are traced through the CLA in the penalty
form; homogeneous rows with signs on some of the coefficients through the leverage
CLA. The central checks are that each path is feasible and optimal against an
certificate of its optimality conditions at and between its breakpoints, and that without binding
constraints it reproduces the homotopy's breakpoints.
"""

import itertools

import numpy as np
import pytest
from scipy.optimize import linprog

import cvxcla._lasso_bounded as bounded
from cvxcla import Lasso
from cvxcla._lasso_bounded import kkt_violation
from cvxcla.errors import DegenerateProblemError, InfeasibleProblemError

N = 6


def _data(m, n, seed):
    """A random design and response."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m, n))
    return x, x @ rng.standard_normal(n) + 0.3 * rng.standard_normal(m)


def _kkt_violation(x, y, lam, beta, a, b, lower, upper, g=None, h=None):
    """Largest violation of the LASSO's optimality conditions, relative to ``||X^T y||_inf``.

    The problem is convex, so these conditions certify optimality on their own. With
    ``r = X^T X beta - X^T y + A^T nu + G_R^T eta`` over the active rows ``R`` of
    ``G beta <= h``, a nonzero coefficient strictly inside its bounds needs
    ``r_j = -lam sign(beta_j)``, one at a cap ``u_j > 0`` needs ``r_j <= -lam`` and one
    at a floor ``l_j < 0`` needs ``r_j >= lam``; a zero one needs
    ``-lam <= r_j <= lam``, less the side a sign bound takes away; and ``eta >= 0``.
    The multipliers ``nu`` and ``eta`` are solved for on the interior coefficients.
    """
    gram, xty = x.T @ x, x.T @ y
    g = np.zeros((0, x.shape[1])) if g is None else g
    h = np.zeros(0) if h is None else h
    tol = 1e-9 * max(1.0, np.abs(beta).max())
    zero = np.abs(beta) <= tol
    at_up, at_lo = ~zero & (beta >= upper - tol), ~zero & (beta <= lower + tol)
    inner = ~zero & ~at_up & ~at_lo
    rows = np.vstack([a, g[g @ beta >= h - tol * max(1.0, np.abs(h).max(initial=1.0))]])
    r = gram @ beta - xty
    eta = np.zeros(0)
    if rows.shape[0] and inner.any():
        mult = np.linalg.lstsq(rows[:, inner].T, -lam * np.sign(beta[inner]) - r[inner], rcond=None)[0]
        r = r + rows.T @ mult
        eta = mult[a.shape[0] :]
    elif a.shape[0]:
        # At beta = 0 nothing pins nu down; take the one that violates the conditions least.
        m = a.shape[0]
        rows = np.vstack([a.T[lower < 0], -a.T[upper > 0]])
        rhs = np.concatenate([lam - r[lower < 0], lam + r[upper > 0]])
        lp = linprog(
            np.append(np.zeros(m), 1.0),
            A_ub=np.hstack([rows, -np.ones((len(rhs), 1))]),
            b_ub=rhs,
            bounds=[(None, None)] * m + [(0.0, None)],
            method="highs",
        )
        r = r + a.T @ lp.x[:m]
    viol = np.concatenate(
        [
            np.abs(r[inner] + lam * np.sign(beta[inner])),
            np.maximum(r[at_up] + lam, 0),
            np.maximum(lam - r[at_lo], 0),
            np.maximum(r[zero & (lower < 0)] - lam, 0),
            np.maximum(-r[zero & (upper > 0)] - lam, 0),
            np.abs(a @ beta - b),
            np.maximum(beta - upper, 0),
            np.maximum(lower - beta, 0),
            np.maximum(g @ beta - h, 0),
            np.maximum(-eta, 0),
        ]
    )
    return float(viol.max(initial=0.0) / np.abs(xty).max())


def _assert_optimal(lasso, x, y, a, b, lower, upper, g=None, h=None):
    """The path is optimal at its breakpoints and at penalties strictly between them."""
    lams = [bp.lam for bp in lasso.path]
    for lam in [*lams, *np.linspace(0.0, lams[0], 9)[1:-1]]:
        assert _kkt_violation(x, y, lam, lasso.solution(lam), a, b, lower, upper, g, h) < 1e-10, lam


def _knots(lasso):
    """The breakpoints as ``(lam, beta)`` arrays."""
    return np.array([bp.lam for bp in lasso.path]), np.array([bp.beta for bp in lasso.path])


def _box(x, y):
    """A cap of half the largest least-squares coefficient, so that it binds."""
    cap = 0.5 * np.abs(np.linalg.lstsq(x, y, rcond=None)[0]).max()
    return np.full(x.shape[1], -cap), np.full(x.shape[1], cap)


SIGNS_LOWER = np.array([0.0, -np.inf, -np.inf, 0.0, -np.inf, -np.inf])
SIGNS_UPPER = np.array([np.inf, np.inf, 0.0, np.inf, np.inf, np.inf])


class TestAgainstTheHomotopy:
    """Without binding constraints, the CLA route returns the homotopy's breakpoints."""

    @pytest.mark.parametrize("seed", range(3))
    def test_infinite_bounds_match_the_plain_path(self, seed):
        """Bounds of +/- inf are the plain LASSO."""
        x, y = _data(40, N, seed)
        lam, beta = _knots(Lasso(x, y, lower=np.full(N, -np.inf), upper=np.full(N, np.inf)))
        ref_lam, ref_beta = _knots(Lasso(x, y))
        assert np.allclose(lam, ref_lam, atol=1e-9)
        assert np.allclose(beta, ref_beta, atol=1e-9)

    @pytest.mark.parametrize("seed", range(3))
    def test_zero_lower_bounds_match_nonneg(self, seed):
        """``lower = 0`` is the non-negative LASSO, and so is ``nonneg=True`` with bounds."""
        x, y = _data(40, N, seed)
        ref_lam, ref_beta = _knots(Lasso(x, y, nonneg=True))
        for lasso in (Lasso(x, y, lower=np.zeros(N)), Lasso(x, y, nonneg=True, upper=np.full(N, np.inf))):
            lam, beta = _knots(lasso)
            assert np.allclose(lam, ref_lam, atol=1e-9)
            assert np.allclose(beta, ref_beta, atol=1e-9)

    def test_lam_max_is_the_first_breakpoint(self):
        """On the CLA route the threshold is read off the traced path."""
        x, y = _data(40, N, 0)
        lasso = Lasso(x, y, upper=np.full(N, np.inf))
        assert lasso.lam_max == lasso.path[0].lam == pytest.approx(np.abs(x.T @ y).max())


class TestBoundedPaths:
    """Paths under bounds and nonhomogeneous rows, certified by their optimality conditions."""

    @pytest.mark.parametrize("seed", range(3))
    def test_caps(self, seed):
        """A cap on every coefficient binds, and the path respects it."""
        x, y = _data(40, N, seed)
        lower, upper = _box(x, y)
        lasso = Lasso(x, y, lower=lower, upper=upper)
        assert np.isclose(np.abs(lasso.path[-1].beta).max(), upper[0])
        _assert_optimal(lasso, x, y, np.zeros((0, N)), np.zeros(0), lower, upper)

    @pytest.mark.parametrize("seed", range(3))
    def test_signs_on_some_coefficients(self, seed):
        """Signs on some of the coefficients only, without rows."""
        x, y = _data(40, N, seed)
        lasso = Lasso(x, y, lower=SIGNS_LOWER, upper=SIGNS_UPPER)
        _assert_optimal(lasso, x, y, np.zeros((0, N)), np.zeros(0), SIGNS_LOWER, SIGNS_UPPER)

    @pytest.mark.parametrize("seed", range(3))
    def test_nonzero_right_hand_side(self, seed):
        """A fit pinned at one observation, ``x_0^T beta = y_0``."""
        x, y = _data(40, N, seed)
        a, b = x[:1], y[:1]
        lasso = Lasso(x, y, a=a, b=b)
        assert np.allclose(a @ lasso.path[0].beta, b)
        inf = np.full(N, np.inf)
        _assert_optimal(lasso, x, y, a, b, -inf, inf)

    @pytest.mark.parametrize("seed", range(3))
    def test_caps_signs_and_a_nonzero_right_hand_side(self, seed):
        """All three at once, through the same penalty form."""
        x, y = _data(40, N, seed)
        lower, upper = _box(x, y)
        lower, upper = np.maximum(lower, SIGNS_LOWER), np.minimum(upper, SIGNS_UPPER)
        a, b = x[:1], 0.5 * y[:1]
        lasso = Lasso(x, y, a=a, b=b, lower=lower, upper=upper)
        _assert_optimal(lasso, x, y, a, b, lower, upper)

    @pytest.mark.parametrize("seed", range(3))
    def test_inequality_rows_with_signs_and_caps(self, seed):
        """A cap on a group's net effect, ``|sum_{j<3} beta_j| <= h``, with signs and caps."""
        x, y = _data(40, N, seed)
        lower, upper = _box(x, y)
        lower, upper = np.maximum(lower, SIGNS_LOWER), np.minimum(upper, SIGNS_UPPER)
        g = np.vstack([np.r_[np.ones(3), np.zeros(N - 3)], -np.r_[np.ones(3), np.zeros(N - 3)]])
        # Half the group's net effect at the end of the path without the row, so that it binds.
        free_end = Lasso(x, y, lower=lower, upper=upper).path[-1].beta
        h = np.full(2, 0.5 * abs(g[0] @ free_end) + 1e-3)
        lasso = Lasso(x, y, g=g, h=h, lower=lower, upper=upper)
        assert np.isclose(np.abs(g[0] @ lasso.path[-1].beta), h[0])  # the row binds
        _assert_optimal(lasso, x, y, np.zeros((0, N)), np.zeros(0), lower, upper, g, h)

    @pytest.mark.parametrize("seed", range(3))
    def test_inequality_rows_with_a_nonzero_right_hand_side(self, seed):
        """Rows ``G beta <= h`` together with a fit pinned at one observation."""
        x, y = _data(40, N, seed)
        a, b = x[:1], y[:1]
        g, h = np.eye(N)[:2], np.full(2, 0.5)
        inf = np.full(N, np.inf)
        lasso = Lasso(x, y, a=a, b=b, g=g, h=h, lower=-inf)
        _assert_optimal(lasso, x, y, a, b, -inf, inf, g, h)

    @pytest.mark.parametrize("seed", range(3))
    def test_zero_sum_with_signs_on_some_coefficients(self, seed):
        """``A beta = 0`` with signs goes through the leverage CLA."""
        x, y = _data(40, N, seed)
        a = np.ones((1, N))
        lasso = Lasso(x, y, a=a, lower=SIGNS_LOWER, upper=SIGNS_UPPER)
        _assert_optimal(lasso, x, y, a, np.zeros(1), SIGNS_LOWER, SIGNS_UPPER)

    def test_active_masks_follow_the_support(self):
        """Each breakpoint's mask is the support of the segment that leaves it."""
        x, y = _data(40, N, 1)
        lasso = Lasso(x, y, lower=SIGNS_LOWER, upper=SIGNS_UPPER)
        for hi, lo in zip(lasso.path, lasso.path[1:], strict=False):
            mid = lasso.solution(0.5 * (hi.lam + lo.lam))
            assert np.array_equal(hi.active, np.abs(mid) > 1e-9)
        assert np.array_equal(lasso.path[-1].active, np.abs(lasso.path[-1].beta) > 1e-9)


class TestSinglePoints:
    """Problems whose path is one point."""

    def test_every_coefficient_fixed(self):
        """Bounds of zero on every coefficient leave only ``beta = 0``."""
        x, y = _data(30, N, 0)
        lasso = Lasso(x, y, lower=np.zeros(N), upper=np.zeros(N))
        assert len(lasso.path) == 1
        assert not lasso.path[0].beta.any()

    def test_no_coefficient_can_enter(self):
        """Signs against every correlation keep ``beta = 0`` optimal at every penalty."""
        x, y = _data(30, N, 0)
        c = x.T @ y
        lasso = Lasso(x, y, lower=np.where(c < 0, 0.0, -np.inf), upper=np.where(c > 0, 0.0, np.inf))
        assert len(lasso.path) == 1
        assert lasso.path[0].lam == 0.0
        assert not lasso.path[0].beta.any()


class TestRejected:
    """Inputs the routes cannot take, and why."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"lower": np.full(N, 0.1)}, "admit beta_j = 0"),
            ({"upper": np.full(N, -0.1)}, "admit beta_j = 0"),
            ({"lower": np.zeros(N - 1)}, "must have shape"),
            ({"a": np.ones((1, N)), "b": np.ones(2)}, "b must have shape"),
        ],
    )
    def test_invalid_bounds(self, kwargs, match):
        """Bounds that exclude zero or have the wrong shape."""
        x, y = _data(30, N, 0)
        with pytest.raises(ValueError, match=match):
            Lasso(x, y, **kwargs)

    def test_b_needs_rows(self):
        """A right-hand side without equality rows has nothing to constrain."""
        x, y = _data(30, N, 0)
        with pytest.raises(ValueError, match="needs equality rows"):
            Lasso(x, y, b=np.ones(1))

    def test_contradictory_rows_are_infeasible(self):
        """``beta_0 = 1`` against ``beta_0 <= 0.5``: the CLA's classification comes through."""
        x, y = _data(30, N, 0)
        e = np.eye(N)[:1]
        with pytest.raises(InfeasibleProblemError, match="could not be traced"):
            Lasso(x, y, a=e, b=np.ones(1), g=e, h=np.array([0.5]))

    def test_fixed_coefficients_against_a_nonzero_right_hand_side(self):
        """``beta = 0`` cannot meet ``A beta = b`` with ``b != 0``."""
        x, y = _data(30, N, 0)
        with pytest.raises(InfeasibleProblemError):
            Lasso(x, y, a=np.ones((1, N)), b=np.ones(1), lower=np.zeros(N), upper=np.zeros(N))


class TestBuilder:
    """The fluent builder maps onto the same constructor arguments."""

    def test_bounds_and_a_right_hand_side(self):
        """``.equality(a, b).bounds(lower, upper)`` traces the constructor's path."""
        x, y = _data(40, N, 0)
        lower, upper = _box(x, y)
        built = Lasso.problem(x, y).equality(x[0], 0.5 * y[0]).bounds(lower, upper).trace()
        direct = Lasso(x, y, a=x[:1], b=0.5 * y[:1], lower=lower, upper=upper)
        for got, want in zip(_knots(built), _knots(direct), strict=True):
            assert np.array_equal(got, want)

    def test_homogeneous_rows_keep_a_zero_right_hand_side(self):
        """Two rows with the default ``b = 0`` and signs go through the leverage CLA."""
        x, y = _data(40, N, 0)
        a = np.vstack([np.ones(N), np.arange(N, dtype=float) - 2.5])
        built = Lasso.problem(x, y).equality(a).bounds(SIGNS_LOWER, SIGNS_UPPER).trace()
        _assert_optimal(built, x, y, a, np.zeros(2), SIGNS_LOWER, SIGNS_UPPER)


def _monotone(seed=7, n=100):
    """A Gaussian design with an ordered truth, and the rows ``beta_j - beta_{j+1} <= 0``."""
    rng = np.random.default_rng(seed)
    truth = np.array([-2.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 3.0])
    x = rng.standard_normal((n, len(truth)))
    y = x @ truth + 0.5 * rng.standard_normal(n)
    d = np.eye(len(truth) - 1, len(truth)) - np.eye(len(truth) - 1, len(truth), k=1)
    return x, y - y.mean(), d


def _certified(lasso, x, y, a, b, g, h, lower, upper):
    """The largest optimality violation over the knots and the segment midpoints, by the LP certificate."""
    gram, xty = x.T @ x, x.T @ y
    lams = [bp.lam for bp in lasso.path]
    points = [*lams, *(0.5 * (hi + lo) for hi, lo in itertools.pairwise(lams))]
    return max(kkt_violation(gram, xty, lam, lasso.solution(lam), a, b, g, h, lower, upper) for lam in points)


class TestCertificate:
    """The LP certificate that guards the homogeneous route, checked on its own."""

    def test_agrees_with_the_direct_check_where_both_apply(self):
        """On a path with interior multipliers it certifies what the direct check certifies."""
        x, y = _data(40, N, 0)
        lower, upper = _box(x, y)
        lasso = Lasso(x, y, lower=lower, upper=upper)
        empty = np.zeros((0, N))
        assert _certified(lasso, x, y, empty, np.zeros(0), empty, np.zeros(0), lower, upper) < 1e-12
        _assert_optimal(lasso, x, y, empty, np.zeros(0), lower, upper)

    def test_flags_a_point_off_the_path(self):
        """A coefficient vector that is not optimal at its penalty is flagged."""
        x, y = _data(40, N, 0)
        lasso = Lasso(x, y)
        lam = 0.5 * lasso.lam_max
        beta = lasso.solution(lam) * 1.01
        inf, empty = np.full(N, np.inf), np.zeros((0, N))
        assert kkt_violation(x.T @ x, x.T @ y, lam, beta, empty, np.zeros(0), empty, np.zeros(0), -inf, inf) > 1e-4

    def test_flags_an_infeasible_point(self):
        """A point that breaks a row is flagged by the feasibility part alone."""
        x, y = _data(40, N, 0)
        g, h = np.eye(N)[:1], np.zeros(1)
        beta = np.r_[1.0, np.zeros(N - 1)]
        inf, empty = np.full(N, np.inf), np.zeros((0, N))
        assert kkt_violation(x.T @ x, x.T @ y, 0.0, beta, empty, np.zeros(0), g, h, -inf, inf) > 1e-4


class TestHomogeneousRows:
    """Rows ``G beta <= 0``, resolved by two relaxed traces and an extrapolation (issue #965)."""

    def test_monotone_fit(self):
        """The monotone fit traces at ``h = 0``, ordered at every knot and certified."""
        x, y, d = _monotone()
        lasso = Lasso(x, y, g=d, h=np.zeros(d.shape[0]))
        inf = np.full(x.shape[1], np.inf)
        assert all(np.all(d @ bp.beta <= 1e-12) for bp in lasso.path)
        assert (
            _certified(lasso, x, y, np.zeros((0, x.shape[1])), np.zeros(0), d, np.zeros(d.shape[0]), -inf, inf) < 1e-12
        )

    def test_relaxed_paths_converge_to_it(self):
        """Independently of the certificate: the path at ``h = eps`` lies about ``eps`` away."""
        x, y, d = _monotone()
        exact = Lasso(x, y, g=d, h=np.zeros(d.shape[0]))
        for eps in (1e-3, 1e-5):
            relaxed = Lasso(x, y, g=d, h=np.full(d.shape[0], eps))
            lams = np.linspace(0.0, exact.lam_max, 23)[1:-1]
            gap = max(np.abs(exact.solution(lam) - relaxed.solution(lam)).max() for lam in lams)
            assert 0.1 * eps < gap < 10 * eps

    def test_mixed_with_positive_rows_and_bounds(self):
        """Homogeneous rows alongside a positive row, signs and caps, and a pinned fit."""
        x, y, d = _monotone()
        p = x.shape[1]
        g = np.vstack([d, np.ones(p)])
        h = np.r_[np.zeros(d.shape[0]), 2.0]
        lower, upper = np.full(p, -2.5), np.r_[np.full(p - 1, 2.5), np.inf]
        a, b = x[:1], y[:1]
        lasso = Lasso(x, y, a=a, b=b, g=g, h=h, lower=lower, upper=upper)
        assert _certified(lasso, x, y, a, b, g, h, lower, upper) < 1e-12

    def test_builder(self):
        """``.inequality(G, 0)`` reaches the same route."""
        x, y, d = _monotone()
        built = Lasso.problem(x, y).inequality(d, np.zeros(d.shape[0])).trace()
        direct = Lasso(x, y, g=d, h=np.zeros(d.shape[0]))
        assert np.array_equal(_knots(built)[1], _knots(direct)[1])

    def test_declined_when_the_relaxations_disagree(self, monkeypatch):
        """Two relaxed traces with different numbers of knots cannot be extrapolated."""
        x, y, d = _monotone()
        real = bounded._trace
        calls = []

        def trace(*args):
            calls.append(1)
            knots = real(*args)
            return knots if len(calls) == 1 else knots[:-1]

        monkeypatch.setattr(bounded, "_trace", trace)
        with pytest.raises(DegenerateProblemError, match="sequence of active sets depends on the relaxation"):
            Lasso(x, y, g=d, h=np.zeros(d.shape[0]))

    def test_declined_when_the_certificate_fails(self, monkeypatch):
        """An extrapolated path that does not certify is not returned."""
        x, y, d = _monotone()
        monkeypatch.setattr(bounded, "kkt_violation", lambda *args: 1.0)
        with pytest.raises(DegenerateProblemError, match="violates the optimality conditions"):
            Lasso(x, y, g=d, h=np.zeros(d.shape[0]))

    def test_zero_correlation(self):
        """With ``X^T y = 0`` the rows ``beta <= 0, -beta <= 0`` give the knot ``(0, 0)`` (issue #982)."""
        inf = np.full(1, np.inf)
        lasso = Lasso(np.eye(1), np.zeros(1), lower=-inf, upper=inf, g=np.array([[1.0], [-1.0]]), h=np.zeros(2))
        assert [(bp.lam, *bp.beta) for bp in lasso.path] == [(0.0, 0.0)]


class TestFlatStart:
    """A whole face of smallest-l1 starting points is resolved by the quadratic term."""

    def test_sum_to_one(self):
        """Sum-to-one ties every positive leg at the start; the traced path is certified."""
        x, y = _data(30, N, 0)
        a, b = np.ones((1, N)), np.ones(1)
        lasso = Lasso(x, y, a=a, b=b)
        free = np.full(N, np.inf)
        assert _certified(lasso, x, y, a, b, np.zeros((0, N)), np.zeros(0), -free, free) < 1e-12


class TestZeroSum:
    """``A beta = 0`` with finite bounds or inequality rows, relaxed and extrapolated (issue #969)."""

    @staticmethod
    def _compositional():
        """The reproduction of issue #969: centred data and a sparse zero-sum truth."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal((60, 10))
        x -= x.mean(0)
        y = x @ np.r_[1.0, -1.0, np.zeros(8)] + 0.3 * rng.standard_normal(60)
        return x, y - y.mean()

    def test_capped_group(self):
        """The issue's example: a zero sum with ``beta_0 <= 0.5`` is traced and certified."""
        x, y = self._compositional()
        a, b = np.ones((1, 10)), np.zeros(1)
        g, h = np.eye(10)[:1], np.array([0.5])
        lasso = Lasso(x, y, a=a, b=b, g=g, h=h)
        free = np.full(10, np.inf)
        assert _certified(lasso, x, y, a, b, g, h, -free, free) < 1e-12
        assert max(abs(bp.beta.sum()) for bp in lasso.path) < 1e-12
        assert max(bp.beta[0] for bp in lasso.path) == pytest.approx(0.5)

    def test_finite_bounds(self):
        """A zero sum inside a box ``|beta_j| <= 0.6`` is certified."""
        x, y = self._compositional()
        a, b = np.ones((1, 10)), np.zeros(1)
        lower, upper = np.full(10, -0.6), np.full(10, 0.6)
        lasso = Lasso(x, y, a=a, b=b, lower=lower, upper=upper)
        assert _certified(lasso, x, y, a, b, np.zeros((0, 10)), np.zeros(0), lower, upper) < 1e-12

    def test_slack_bounds_agree_with_the_leverage_route(self):
        """Bounds that never bind go through the relaxation and give the signs-only path."""
        x, y = self._compositional()
        a = np.ones((1, 10))
        relaxed = Lasso(x, y, a=a, b=np.zeros(1), lower=np.full(10, -100.0), upper=np.full(10, 100.0)).path
        leverage = Lasso(x, y, a=a, b=np.zeros(1)).path
        assert len(relaxed) == len(leverage)
        for got, want in zip(relaxed, leverage, strict=True):
            assert got.lam == pytest.approx(want.lam, rel=1e-9, abs=1e-12)
            np.testing.assert_allclose(got.beta, want.beta, atol=1e-10)

    def test_with_homogeneous_rows(self):
        """A zero sum, a sign and the ordering ``beta_0 <= beta_1``: both kinds relaxed together."""
        x, y = _data(40, N, 3)
        a, b = np.ones((1, N)), np.zeros(1)
        g, h = np.diff(np.eye(N), axis=0)[:2], np.array([0.0, 0.3])
        lower, upper = np.full(N, -np.inf), np.full(N, np.inf)
        lower[0] = 0.0
        lasso = Lasso(x, y, a=a, b=b, g=g, h=h, lower=lower, upper=upper)
        assert _certified(lasso, x, y, a, b, g, h, lower, upper) < 1e-12

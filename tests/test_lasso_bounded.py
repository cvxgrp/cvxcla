"""Tests for the LASSO under coefficient bounds and equality rows ``A beta = b``.

Per-coefficient bounds ``lower <= beta <= upper`` (caps, and signs on some of the
coefficients) and nonhomogeneous rows are traced through the CLA in the penalty
form; homogeneous rows with signs on some of the coefficients through the leverage
CLA. The central checks are that each path is feasible and optimal against an
certificate of its optimality conditions at and between its breakpoints, and that without binding
constraints it reproduces the homotopy's breakpoints.
"""

import numpy as np
import pytest
from scipy.optimize import linprog

from cvxcla import Lasso
from cvxcla.errors import DegenerateProblemError, InfeasibleProblemError

N = 6


def _data(m, n, seed):
    """A random design and response."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m, n))
    return x, x @ rng.standard_normal(n) + 0.3 * rng.standard_normal(m)


def _kkt_violation(x, y, lam, beta, a, b, lower, upper):
    """Largest violation of the LASSO's optimality conditions, relative to ``||X^T y||_inf``.

    The problem is convex, so these conditions certify optimality on their own. With
    ``r = X^T X beta - X^T y + A^T nu``, a nonzero coefficient strictly inside its
    bounds needs ``r_j = -lam sign(beta_j)``, one at a cap ``u_j > 0`` needs
    ``r_j <= -lam`` and one at a floor ``l_j < 0`` needs ``r_j >= lam``; a zero one
    needs ``-lam <= r_j <= lam``, less the side a sign bound takes away. The
    multipliers ``nu`` are solved for on the interior coefficients.
    """
    gram, xty = x.T @ x, x.T @ y
    tol = 1e-9 * max(1.0, np.abs(beta).max())
    zero = np.abs(beta) <= tol
    at_up, at_lo = ~zero & (beta >= upper - tol), ~zero & (beta <= lower + tol)
    inner = ~zero & ~at_up & ~at_lo
    r = gram @ beta - xty
    if a.shape[0] and inner.any():
        nu = np.linalg.lstsq(a[:, inner].T, -lam * np.sign(beta[inner]) - r[inner], rcond=None)[0]
        r = r + a.T @ nu
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
        ]
    )
    return float(viol.max(initial=0.0) / np.abs(xty).max())


def _assert_optimal(lasso, x, y, a, b, lower, upper):
    """The path is optimal at its breakpoints and at penalties strictly between them."""
    lams = [bp.lam for bp in lasso.path]
    for lam in [*lams, *np.linspace(0.0, lams[0], 9)[1:-1]]:
        assert _kkt_violation(x, y, lam, lasso.solution(lam), a, b, lower, upper) < 1e-10, lam


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
            ({"a": np.ones((1, N)), "upper": np.ones(N)}, "no finite bound"),
        ],
    )
    def test_invalid_bounds(self, kwargs, match):
        """Bounds that exclude zero or have the wrong shape, and A beta = 0 with a cap."""
        x, y = _data(30, N, 0)
        with pytest.raises(ValueError, match=match):
            Lasso(x, y, **kwargs)

    def test_b_needs_rows(self):
        """A right-hand side without equality rows has nothing to constrain."""
        x, y = _data(30, N, 0)
        with pytest.raises(ValueError, match="needs equality rows"):
            Lasso(x, y, b=np.ones(1))

    def test_bounds_with_inequality_rows(self):
        """Bounds are not combined with ``G beta <= h``."""
        x, y = _data(30, N, 0)
        with pytest.raises(ValueError, match="cannot be combined with inequality rows"):
            Lasso(x, y, g=np.eye(N), h=np.ones(N), upper=np.ones(N))

    def test_fixed_coefficients_against_a_nonzero_right_hand_side(self):
        """``beta = 0`` cannot meet ``A beta = b`` with ``b != 0``."""
        x, y = _data(30, N, 0)
        with pytest.raises(InfeasibleProblemError):
            Lasso(x, y, a=np.ones((1, N)), b=np.ones(1), lower=np.zeros(N), upper=np.zeros(N))

    def test_a_flat_start_is_declined(self):
        """Sum-to-one has a whole face of smallest-l1 points to start from, not one vertex."""
        x, y = _data(30, N, 0)
        with pytest.raises(DegenerateProblemError, match="could not be traced"):
            Lasso(x, y, a=np.ones((1, N)), b=np.ones(1))


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

"""Tests for the LASSO path traced by the generic parametric active-set tracer.

The path is validated two independent ways: against the LASSO subgradient (KKT)
optimality conditions, and against a from-scratch coordinate-descent solver. Both
the enter and the leave event families are exercised.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from cvxcla.lasso import Breakpoint, Lasso


def _coordinate_descent(x: np.ndarray, y: np.ndarray, lam: float, iters: int = 5000) -> np.ndarray:
    """An independent LASSO solver (cyclic coordinate descent with soft-threshold)."""
    n = x.shape[1]
    beta = np.zeros(n)
    col_sq = np.sum(x * x, axis=0)
    r = y - x @ beta
    for _ in range(iters):
        max_change = 0.0
        for j in range(n):
            if col_sq[j] == 0.0:
                continue
            rho = x[:, j] @ r + col_sq[j] * beta[j]
            new = np.sign(rho) * max(abs(rho) - lam, 0.0) / col_sq[j]
            if new != beta[j]:
                r += x[:, j] * (beta[j] - new)
                max_change = max(max_change, abs(new - beta[j]))
                beta[j] = new
        if max_change < 1e-12:
            break
    return beta


def _kkt_violation(x: np.ndarray, y: np.ndarray, beta: np.ndarray, lam: float) -> float:
    """Max subgradient-optimality violation of ``beta`` for LASSO at ``lam``."""
    c = x.T @ (y - x @ beta)
    viol = 0.0
    for j in range(len(beta)):
        if abs(beta[j]) > 1e-9:
            viol = max(viol, abs(c[j] - lam * np.sign(beta[j])))
        else:
            viol = max(viol, max(0.0, abs(c[j]) - lam))
    return float(viol)


def _uncorrelated_problem():
    """A generic design whose path only grows the support (enter events only)."""
    rng = np.random.default_rng(0)
    m, n = 40, 12
    x = rng.standard_normal((m, n))
    true_beta = np.zeros(n)
    true_beta[[1, 4, 7]] = [2.0, -3.0, 1.5]
    y = x @ true_beta + 0.1 * rng.standard_normal(m)
    return x, y


def _correlated_problem():
    """A correlated design whose support oscillates (enter and leave events)."""
    rng = np.random.default_rng(0)
    m, n = 30, 8
    f = rng.standard_normal((m, 2))
    loadings = rng.standard_normal((2, n))
    x = f @ loadings + 0.3 * rng.standard_normal((m, n))
    true_beta = np.zeros(n)
    true_beta[[0, 3]] = [3.0, -2.0]
    y = x @ true_beta + 0.1 * rng.standard_normal(m)
    return x, y


class TestLassoPath:
    """End-to-end correctness of the traced LASSO path."""

    def test_endpoints(self):
        """The path starts at (lam_max, 0) and ends at (0, OLS-on-support)."""
        x, y = _uncorrelated_problem()
        lasso = Lasso(x, y)
        first, last = lasso.path[0], lasso.path[-1]
        assert np.isclose(first.lam, lasso.lam_max)
        np.testing.assert_allclose(first.beta, 0.0)
        assert last.lam == 0.0

    def test_path_matches_kkt_and_coordinate_descent(self):
        """Sampled along the path, beta satisfies KKT and matches coordinate descent."""
        x, y = _uncorrelated_problem()
        lasso = Lasso(x, y)
        for frac in (0.9, 0.7, 0.5, 0.3, 0.15, 0.05):
            lam = frac * lasso.lam_max
            beta = lasso.solution(lam)
            assert _kkt_violation(x, y, beta, lam) < 1e-7
            np.testing.assert_allclose(beta, _coordinate_descent(x, y, lam), atol=1e-5)

    def test_leave_events_fire_and_path_stays_exact(self):
        """A correlated design forces drop (leave) events; the path remains exact."""
        x, y = _correlated_problem()
        lasso = Lasso(x, y)
        sizes = [int(bp.active.sum()) for bp in lasso.path]
        assert any(b < a for a, b in pairwise(sizes)), "expected at least one drop event"
        for frac in (0.9, 0.6, 0.35, 0.1):
            lam = frac * lasso.lam_max
            beta = lasso.solution(lam)
            assert _kkt_violation(x, y, beta, lam) < 1e-7
            np.testing.assert_allclose(beta, _coordinate_descent(x, y, lam), atol=1e-5)

    def test_solution_clamps_outside_range(self):
        """Querying beyond lam_max returns 0; at or below 0 returns the final fit."""
        x, y = _uncorrelated_problem()
        lasso = Lasso(x, y)
        np.testing.assert_allclose(lasso.solution(2.0 * lasso.lam_max), 0.0)
        np.testing.assert_allclose(lasso.solution(-1.0), lasso.path[-1].beta)

    def test_solution_at_breakpoint(self):
        """Evaluating exactly at a stored breakpoint reproduces its coefficients."""
        x, y = _uncorrelated_problem()
        lasso = Lasso(x, y)
        mid = lasso.path[len(lasso.path) // 2]
        np.testing.assert_allclose(lasso.solution(mid.lam), mid.beta, atol=1e-12)

    def test_lambda_non_increasing(self):
        """Breakpoint lambdas are recorded in non-increasing order."""
        x, y = _uncorrelated_problem()
        lams = [bp.lam for bp in Lasso(x, y).path]
        assert all(a >= b for a, b in pairwise(lams))


class TestLassoValidation:
    """Input validation on construction."""

    def test_rejects_non_2d_design(self):
        """A 1d design matrix is rejected."""
        with pytest.raises(ValueError, match="2d design matrix"):
            Lasso(np.ones(5), np.ones(5))

    def test_rejects_mismatched_response_length(self):
        """A response whose length does not match the rows of x is rejected."""
        with pytest.raises(ValueError, match="y must have shape"):
            Lasso(np.ones((5, 3)), np.ones(4))


class TestConstrainedLasso:
    """The LASSO path under linear inequality constraints ``G beta <= h``."""

    @staticmethod
    def _problem_with_caps():
        """A standardised regression with group-sum caps that bind along the path."""
        rng = np.random.default_rng(1)
        m, n = 50, 10
        x = rng.standard_normal((m, n))
        x = (x - x.mean(0)) / x.std(0)
        beta = rng.standard_normal(n)
        y = x @ beta + 0.1 * rng.standard_normal(m)
        y = y - y.mean()
        group = np.arange(n) % 3
        g = np.array([(group == j).astype(float) for j in range(3)])
        beta_ols = np.linalg.lstsq(x, y, rcond=None)[0]
        h = np.maximum(np.abs(g @ beta_ols) * 0.4, 0.1)
        return x, y, g, h

    def test_path_is_feasible(self):
        """Every breakpoint satisfies the inequality constraints."""
        x, y, g, h = self._problem_with_caps()
        lasso = Lasso(x=x, y=y, g=g, h=h)
        for bp in lasso.path:
            assert np.all(g @ bp.beta <= h + 1e-7)

    def test_cap_binds(self):
        """At least one breakpoint holds a constraint row active (the cap bites)."""
        x, y, g, h = self._problem_with_caps()
        lasso = Lasso(x=x, y=y, g=g, h=h)
        assert any(np.any(np.abs(g @ bp.beta - h) <= 1e-7) for bp in lasso.path), "expected a binding cap"

    def test_loose_caps_match_unconstrained(self):
        """With caps too loose to bind, the constrained path equals the plain LASSO."""
        x, y, g, _ = self._problem_with_caps()
        constrained = Lasso(x=x, y=y, g=g, h=np.full(g.shape[0], 1e6))
        plain = Lasso(x=x, y=y)
        for frac in (0.8, 0.5, 0.2):
            lam = frac * plain.lam_max
            np.testing.assert_allclose(constrained.solution(lam), plain.solution(lam), atol=1e-7)

    def test_rejects_nonpositive_h(self):
        """A non-positive ``h`` entry (``beta = 0`` infeasible) is rejected."""
        x, y, g, h = self._problem_with_caps()
        h[0] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            Lasso(x=x, y=y, g=g, h=h)

    def test_rejects_g_without_h(self):
        """Providing ``g`` without ``h`` is rejected."""
        x, y, g, _ = self._problem_with_caps()
        with pytest.raises(ValueError, match="together"):
            Lasso(x=x, y=y, g=g)

    def test_rejects_bad_g_shape(self):
        """A ``g`` whose column count is not ``n`` is rejected."""
        x, y, g, h = self._problem_with_caps()
        with pytest.raises(ValueError, match="g must have shape"):
            Lasso(x=x, y=y, g=g[:, :-1], h=h)


def _nonneg_coordinate_descent(x: np.ndarray, y: np.ndarray, lam: float, iters: int = 5000) -> np.ndarray:
    """An independent non-negative LASSO solver (coordinate descent, clamped at 0)."""
    n = x.shape[1]
    beta = np.zeros(n)
    col_sq = np.sum(x * x, axis=0)
    r = y - x @ beta
    for _ in range(iters):
        max_change = 0.0
        for j in range(n):
            if col_sq[j] == 0.0:
                continue
            rho = x[:, j] @ r + col_sq[j] * beta[j]
            new = max(rho - lam, 0.0) / col_sq[j]  # soft-threshold then clamp at 0
            if new != beta[j]:
                r += x[:, j] * (beta[j] - new)
                max_change = max(max_change, abs(new - beta[j]))
                beta[j] = new
        if max_change < 1e-12:
            break
    return beta


class TestNonNegativeLasso:
    """The LASSO path under the sign restriction ``beta >= 0``."""

    def test_path_is_non_negative(self):
        """Every breakpoint has non-negative coefficients."""
        x, y = _uncorrelated_problem()
        lasso = Lasso(x=x, y=y, nonneg=True)
        for bp in lasso.path:
            assert np.all(bp.beta >= -1e-9)

    def test_matches_nonneg_coordinate_descent(self):
        """Sampled along the path, beta matches an independent non-negative solver."""
        x, y = _correlated_problem()
        lasso = Lasso(x=x, y=y, nonneg=True)
        for frac in (0.8, 0.5, 0.25, 0.05):
            lam = frac * float(np.max(x.T @ y))
            np.testing.assert_allclose(lasso.solution(lam), _nonneg_coordinate_descent(x, y, lam), atol=1e-5)

    def test_suppresses_negative_coefficients(self):
        """A coefficient the plain LASSO drives negative is held at zero here."""
        x, y = _correlated_problem()
        plain = Lasso(x=x, y=y)
        nonneg = Lasso(x=x, y=y, nonneg=True)
        lam = 0.2 * plain.lam_max
        assert np.min(plain.solution(lam)) < -1e-6, "expected the plain path to go negative"
        assert np.all(nonneg.solution(lam) >= -1e-9)

    def test_builder_non_negative_matches_constructor(self):
        """``.non_negative()`` on the builder equals ``nonneg=True`` directly."""
        x, y = _uncorrelated_problem()
        built = Lasso.problem(x, y).non_negative().trace()
        direct = Lasso(x=x, y=y, nonneg=True)
        assert len(built.path) == len(direct.path)
        for frac in (0.8, 0.4, 0.1):
            lam = frac * direct.path[0].lam
            np.testing.assert_allclose(built.solution(lam), direct.solution(lam), atol=1e-12)

    def test_all_nonpositive_correlations_give_zero_path(self):
        """With no positive correlation, the non-negative path is identically zero."""
        rng = np.random.default_rng(3)
        x = np.abs(rng.standard_normal((30, 5))) + 0.1  # all-positive design
        y = -x.sum(axis=1)  # so X^T y < 0 componentwise: nothing can enter
        lasso = Lasso(x=x, y=y, nonneg=True)
        for bp in lasso.path:
            np.testing.assert_allclose(bp.beta, 0.0, atol=1e-9)
        np.testing.assert_allclose(lasso.solution(0.0), 0.0, atol=1e-9)


class TestLassoBuilder:
    """The fluent ``Lasso.problem(...).trace()`` builder."""

    def test_plain_builder_matches_constructor(self):
        """``Lasso.problem(x, y).trace()`` equals the direct ``Lasso(x, y)``."""
        x, y = _uncorrelated_problem()
        built = Lasso.problem(x, y).trace()
        direct = Lasso(x=x, y=y)
        assert len(built.path) == len(direct.path)
        for frac in (0.8, 0.4, 0.1):
            lam = frac * direct.lam_max
            np.testing.assert_allclose(built.solution(lam), direct.solution(lam), atol=1e-12)

    def test_inequality_builder_matches_constructor(self):
        """The builder's ``.inequality(g, h)`` equals passing ``g, h`` directly."""
        x, y, g, h = TestConstrainedLasso._problem_with_caps()
        built = Lasso.problem(x, y).inequality(g, h).trace()
        direct = Lasso(x=x, y=y, g=g, h=h)
        assert len(built.path) == len(direct.path)
        for bp in built.path:
            assert np.all(g @ bp.beta <= h + 1e-7)

    def test_inequality_accumulates_rows(self):
        """Repeated ``.inequality`` calls stack rows, like the CLA builder."""
        x, y, g, h = TestConstrainedLasso._problem_with_caps()
        stacked = Lasso.problem(x, y).inequality(g[:1], h[:1]).inequality(g[1:], h[1:]).trace()
        direct = Lasso(x=x, y=y, g=g, h=h)
        assert len(stacked.path) == len(direct.path)

    def test_builder_rejects_bad_inequality_shape(self):
        """A row vector of the wrong width is rejected at ``.inequality``."""
        x, y, g, h = TestConstrainedLasso._problem_with_caps()
        with pytest.raises(ValueError, match=r"must have .* columns"):
            Lasso.problem(x, y).inequality(g[:, :-1], h)

    def test_builder_rejects_bad_inequality_h_length(self):
        """An ``h`` whose length does not match the rows of ``g`` is rejected."""
        x, y, g, h = TestConstrainedLasso._problem_with_caps()
        with pytest.raises(ValueError, match="h must have"):
            Lasso.problem(x, y).inequality(g, h[:-1])


def test_gram_backend_matches_dense_high_dimensional():
    """gram=True (Woodbury in observation space) matches the dense path when p > n."""
    rng = np.random.default_rng(4)
    m, n = 40, 120  # p = n > m: the high-dimensional regime
    x = rng.standard_normal((m, n))
    x = x - x.mean(0)  # centred design (the gram route's convention)
    beta = np.zeros(n)
    beta[rng.choice(n, 4, replace=False)] = rng.standard_normal(4)
    y = x @ beta + 0.05 * rng.standard_normal(m)
    y = y - y.mean()
    dense = Lasso(x=x, y=y)
    gram = Lasso(x=x, y=y, gram=True)
    assert len(dense.path) == len(gram.path)
    for bp in dense.path:
        if bp.lam > 1e-9:
            np.testing.assert_allclose(gram.solution(bp.lam), dense.solution(bp.lam), atol=1e-7)


def test_constraint_accessors_empty_without_constraints():
    """The g/h accessors return empty arrays when no inequality is supplied."""
    x, y = _uncorrelated_problem()
    lasso = Lasso(x=x, y=y)
    assert lasso.h_vector.shape == (0,)
    assert lasso.g_matrix.shape == (0, x.shape[1])


def test_breakpoint_is_frozen():
    """Breakpoint is an immutable record."""
    import dataclasses

    bp = Breakpoint(lam=1.0, beta=np.zeros(2), active=np.zeros(2, dtype=bool))
    with pytest.raises(dataclasses.FrozenInstanceError):
        bp.lam = 2.0


def _factor_gram(seed: int = 0, n: int = 12, k: int = 3):
    """A diagonal-plus-low-rank Gram H = diag(d) + U diag(delta) U^T and a linear term."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    sigma = np.diag(d) + (u * delta) @ u.T
    beta_true = np.zeros(n)
    beta_true[:4] = [2.0, -1.5, 1.1, 0.7]
    xty = sigma @ beta_true + 0.05 * rng.standard_normal(n)
    return d, u, delta, sigma, xty


def test_from_operator_factor_matches_dense():
    """The factor (Woodbury) operator traces the identical path as the dense Gram."""
    from cvxcla.operators import DenseCovariance, FactorCovariance

    d, u, delta, sigma, xty = _factor_gram()
    factor = Lasso.from_operator(FactorCovariance(d=d, u=u, delta=delta), xty)
    dense = Lasso.from_operator(DenseCovariance(sigma), xty)
    assert len(factor.path) == len(dense.path)
    for bp_f, bp_d in zip(factor.path, dense.path, strict=True):
        np.testing.assert_allclose(bp_f.beta, bp_d.beta, atol=1e-10)


def test_from_operator_matches_design_matrix():
    """from_operator(H = X^T X, X^T y) reproduces the design-matrix LASSO path."""
    from cvxcla.operators import DenseCovariance

    _d, _u, _delta, sigma, xty = _factor_gram(seed=1)
    chol = np.linalg.cholesky(sigma)
    x = chol.T  # X^T X = sigma exactly
    y = x @ np.linalg.solve(sigma, xty)  # so that X^T y = xty
    op = Lasso.from_operator(DenseCovariance(sigma), xty)
    design = Lasso(x=x, y=y)
    assert len(op.path) == len(design.path)
    for bp_o, bp_x in zip(op.path, design.path, strict=True):
        np.testing.assert_allclose(bp_o.beta, bp_x.beta, atol=1e-8)


def test_from_operator_with_inequality():
    """A structured operator traces a path under a genuine inequality constraint."""
    from cvxcla.operators import FactorCovariance

    d, u, delta, _sigma, xty = _factor_gram(seed=2)
    n = xty.shape[0]
    lasso = Lasso.from_operator(FactorCovariance(d=d, u=u, delta=delta), xty, g=np.ones((1, n)), h=np.array([3.0]))
    assert len(lasso.path) >= 1
    assert float(np.ones(n) @ lasso.path[-1].beta) <= 3.0 + 1e-6


def test_operator_mode_rejects_design_and_operator_together():
    """Supplying both (x, y) and (quad_form, linear) is rejected."""
    from cvxcla.operators import DenseCovariance

    _d, _u, _delta, sigma, xty = _factor_gram(seed=3)
    with pytest.raises(ValueError, match="either"):
        Lasso(x=np.eye(xty.shape[0]), y=np.zeros(xty.shape[0]), quad_form=DenseCovariance(sigma), linear=xty)


def test_operator_mode_requires_both_quad_form_and_linear():
    """In operator mode, quad_form and linear (X^T y) must be supplied together."""
    from cvxcla.operators import DenseCovariance

    _d, _u, _delta, sigma, xty = _factor_gram(seed=4)
    with pytest.raises(ValueError, match="together"):
        Lasso(quad_form=DenseCovariance(sigma))
    with pytest.raises(ValueError, match="together"):
        Lasso(linear=xty)


def test_operator_mode_rejects_non_1d_linear():
    """The linear term X^T y must be a 1d vector."""
    from cvxcla.operators import DenseCovariance

    _d, _u, _delta, sigma, xty = _factor_gram(seed=5)
    with pytest.raises(ValueError, match="1d vector"):
        Lasso(quad_form=DenseCovariance(sigma), linear=xty[:, None])


def test_requires_design_or_operator():
    """Constructing a Lasso with neither a design nor an operator is rejected."""
    with pytest.raises(ValueError, match="provide a design"):
        Lasso()

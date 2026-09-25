"""Tests for the equality-constrained LASSO ``A beta = 0`` traced through the leverage CLA.

Under ``Sigma = X^T X`` and ``mu = X^T y`` one leverage-capped CLA trace, rescaled,
is the constrained LASSO path (Schmelzer and Hastie, arXiv:2609.25704, Theorem 1
and Corollary 2). The central checks are that the path is optimal against an
independent solver at penalties between the breakpoints, that it reproduces the
plain LASSO path when there are no rows, and that it ends at the constrained
least-squares fit.
"""

import itertools

import numpy as np
import pytest
from scipy.optimize import minimize

from cvxcla import CLA, DenseCovariance, Lasso


def _data(m, n, seed):
    """A random design and response."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m, n))
    return x, x @ rng.standard_normal(n) + 0.3 * rng.standard_normal(m)


def _objective(x, y, beta, lam):
    """The LASSO objective ``1/2 ||y - X beta||^2 + lam ||beta||_1``."""
    return 0.5 * np.sum((y - x @ beta) ** 2) + lam * np.abs(beta).sum()


def _reference(x, y, a, lam, nonneg=False):
    """Optimal LASSO objective under ``A beta = 0``, via SLSQP on the split legs."""
    n = x.shape[1]
    xtx, xty = x.T @ x, x.T @ y
    split = np.hstack([np.eye(n), -np.eye(n)])

    def objective(z):
        beta = split @ z
        return 0.5 * np.sum((y - x @ beta) ** 2) + lam * z.sum()

    def gradient(z):
        g = xtx @ (split @ z) - xty
        return np.concatenate([g + lam, -g + lam])

    res = minimize(
        objective,
        np.zeros(2 * n),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, None)] * n + [(0.0, 0.0 if nonneg else None)] * n,
        constraints=[{"type": "eq", "fun": lambda z: a @ split @ z, "jac": lambda z: a @ split}],
        options={"ftol": 1e-15, "maxiter": 5000},
    )
    return res.fun


def _assert_optimal(lasso, x, y, a, nonneg=False):
    """The path is feasible and optimal at penalties strictly between its breakpoints."""
    lams = [bp.lam for bp in lasso.path]
    for lam in np.linspace(0.0, lams[0], 11)[1:-1]:
        beta = lasso.solution(lam)
        assert np.allclose(a @ beta, 0.0, atol=1e-10)
        if nonneg:
            assert np.all(beta >= -1e-12)
        assert _objective(x, y, beta, lam) <= _reference(x, y, a, lam, nonneg) + 1e-7


def _constrained_least_squares(x, y, a):
    """The fit ``argmin ||y - X beta||`` subject to ``A beta = 0``."""
    n, m = x.shape[1], a.shape[0]
    kkt = np.block([[x.T @ x, a.T], [a, np.zeros((m, m))]])
    return np.linalg.solve(kkt, np.concatenate([x.T @ y, np.zeros(m)]))[:n]


class TestEqualityPath:
    """End-to-end paths under ``A beta = 0``."""

    @pytest.mark.parametrize("seed", range(4))
    def test_sum_to_zero(self, seed):
        """The sum-to-zero LASSO is optimal along the path and ends at the constrained fit."""
        x, y = _data(40, 8, seed)
        a = np.ones((1, 8))
        lasso = Lasso(x=x, y=y, a=a)
        _assert_optimal(lasso, x, y, a)
        lams = [bp.lam for bp in lasso.path]
        assert np.all(np.diff(lams) < 0)
        assert np.allclose(lasso.path[0].beta, 0.0)
        assert lasso.path[-1].lam == pytest.approx(0.0, abs=1e-9)
        assert np.allclose(lasso.path[-1].beta, _constrained_least_squares(x, y, a))

    def test_several_dense_rows(self):
        """Two dense equality rows are carried together."""
        x, y = _data(40, 8, 7)
        a = np.random.default_rng(7).standard_normal((2, 8))
        lasso = Lasso(x=x, y=y, a=a)
        _assert_optimal(lasso, x, y, a)
        assert np.allclose(lasso.path[-1].beta, _constrained_least_squares(x, y, a))

    def test_non_negative_contrast(self):
        """``beta >= 0`` with a contrast row (the first half balances the second)."""
        x, y = _data(40, 8, 3)
        a = np.concatenate([np.ones(4), -np.ones(4)])[None, :]
        lasso = Lasso(x=x, y=y, a=a, nonneg=True)
        _assert_optimal(lasso, x, y, a, nonneg=True)

    def test_no_rows_reproduces_the_plain_path(self):
        """With zero equality rows the CLA route is the ordinary LASSO path (Theorem 1)."""
        x, y = _data(40, 8, 1)
        routed = Lasso(x=x, y=y, a=np.zeros((0, 8)))
        plain = Lasso(x=x, y=y)
        assert routed.lam_max == pytest.approx(plain.lam_max)
        for lam in np.linspace(0.0, plain.lam_max, 13):
            assert np.allclose(routed.solution(lam), plain.solution(lam), atol=1e-10)

    def test_lam_max_is_where_the_path_leaves_zero(self):
        """``lam_max`` comes from the path: zero above it, nonzero just below."""
        x, y = _data(40, 8, 2)
        lasso = Lasso(x=x, y=y, a=np.ones((1, 8)))
        assert lasso.lam_max == lasso.path[0].lam
        assert lasso.lam_max < np.max(np.abs(x.T @ y))  # the row removes part of X^T y
        assert np.allclose(lasso.solution(1.01 * lasso.lam_max), 0.0)
        assert np.abs(lasso.solution(0.99 * lasso.lam_max)).sum() > 0

    def test_zero_correlation_is_a_single_point(self):
        """With ``y = 0`` no coordinate can enter, so the path is ``beta = 0``."""
        x, _ = _data(20, 5, 0)
        lasso = Lasso(x=x, y=np.zeros(20), a=np.ones((1, 5)))
        assert len(lasso.path) == 1
        assert lasso.path[0].lam == 0.0
        assert np.allclose(lasso.path[0].beta, 0.0)

    def test_operator_mode(self):
        """``from_operator`` with ``a`` traces the same path as the design."""
        x, y = _data(40, 6, 4)
        a = np.ones((1, 6))
        design = Lasso(x=x, y=y, a=a)
        operator = Lasso.from_operator(DenseCovariance(x.T @ x), x.T @ y, a=a)
        assert len(design.path) == len(operator.path)
        for p, q in zip(design.path, operator.path, strict=True):
            assert p.lam == pytest.approx(q.lam)
            assert np.allclose(p.beta, q.beta)


class TestBuilder:
    """``LassoBuilder.equality`` maps onto the ``a`` argument."""

    def test_builder_matches_constructor(self):
        """Repeated ``.equality`` calls stack their rows."""
        x, y = _data(40, 8, 5)
        rows = np.random.default_rng(5).standard_normal((2, 8))
        built = Lasso.problem(x, y).equality(rows[0]).equality(rows[1]).trace()
        explicit = Lasso(x=x, y=y, a=rows)
        assert len(built.path) == len(explicit.path)
        for p, q in zip(built.path, explicit.path, strict=True):
            assert np.allclose(p.beta, q.beta)

    def test_builder_rejects_wrong_columns(self):
        """A row with the wrong length is refused at once."""
        x, y = _data(20, 5, 0)
        with pytest.raises(ValueError, match="must have 5 columns"):
            Lasso.problem(x, y).equality(np.ones(4))


class TestValidation:
    """Unsupported inputs are declined with a clear message."""

    def test_wrong_shape(self):
        """``a`` must be a matrix with one column per feature."""
        x, y = _data(20, 5, 0)
        with pytest.raises(ValueError, match=r"a must have shape \(m, 5\)"):
            Lasso(x=x, y=y, a=np.ones((1, 4)))

    def test_cannot_combine_with_inequalities(self):
        """Equality and inequality rows together are refused."""
        x, y = _data(20, 5, 0)
        with pytest.raises(ValueError, match="cannot be combined"):
            Lasso(x=x, y=y, a=np.ones((1, 5)), g=np.eye(5)[:1], h=np.ones(1))

    def test_more_features_than_observations(self):
        """A singular Gram leaves the least-squares end non-unique, so it is refused."""
        x, y = _data(10, 20, 0)
        with pytest.raises(ValueError, match="positive-definite Gram"):
            Lasso(x=x, y=y, a=np.ones((1, 20)))

    def test_degenerate_first_vertex_is_reported(self):
        """Block-structured rows (two groups, each summing to zero) make the first vertex degenerate."""
        x, y = _data(40, 8, 0)
        groups = np.kron(np.eye(2), np.ones((1, 4)))
        with pytest.raises(ValueError, match="could not be traced through the leverage CLA"):
            Lasso(x=x, y=y, a=groups)


class TestLongOnlyFrontierIsNonnegativeLasso:
    """Proposition 3: the long-only, fully invested frontier is the non-negative LASSO path.

    With ``Sigma = L L^T``, the substitution ``X = L^T``, ``y = L^{-1} mu`` gives
    ``X^T X = Sigma`` and ``X^T y = mu``. Every vertex ``v`` of the non-negative LASSO
    path on ``(X, y)``, divided by its own sum, is a turning point of the frontier at
    tilt ``1 / sum(v)``. The LASSO path ends at the NNLS fit, so it covers only the
    frontier's tilts at or above ``1 / sum(v_nnls)``. Below that, the budget row keeps
    binding with a negative multiplier, which a penalty cannot express.
    """

    @staticmethod
    def _frontier_at(cla, lam):
        """The CLA weights at tilt ``lam``, linear in ``lam`` between turning points."""
        tps = cla.turning_points
        for hi, lo in itertools.pairwise(tps):
            if lo.lamb <= lam <= hi.lamb:
                if np.isinf(hi.lamb):
                    return hi.weights
                t = (hi.lamb - lam) / (hi.lamb - lo.lamb)
                return (1 - t) * hi.weights + t * lo.weights
        msg = f"lambda {lam} outside the traced frontier"  # pragma: no cover
        raise AssertionError(msg)  # pragma: no cover

    @pytest.mark.parametrize("seed", range(4))
    def test_rescaled_vertices_are_turning_points(self, seed):
        """Each rescaled LASSO vertex is on the frontier, and the corners match one-to-one."""
        rng = np.random.default_rng(seed)
        n = 12
        factors = rng.standard_normal((n, n))
        sigma = factors @ factors.T / n + 0.05 * np.eye(n)
        mu = rng.standard_normal(n) + 0.5

        chol = np.linalg.cholesky(sigma)
        x, y = chol.T, np.linalg.solve(chol, mu)
        assert np.allclose(x.T @ x, sigma)
        assert np.allclose(x.T @ y, mu)

        cla = CLA(
            mean=mu,
            covariance=sigma,
            lower_bounds=np.zeros(n),
            upper_bounds=np.ones(n),
            a=np.ones((1, n)),
            b=np.ones(1),
        )
        lasso = Lasso(x=x, y=y, nonneg=True)
        vertices = [bp.beta for bp in lasso.path if bp.beta.sum() > 0]
        tilts = [1.0 / v.sum() for v in vertices]
        for v, lam in zip(vertices, tilts, strict=True):
            assert np.allclose(v / v.sum(), self._frontier_at(cla, lam), atol=1e-9)

        # One-to-one: the frontier's corners above the path's end are exactly its
        # interior breakpoints, in the same order. The end itself (the NNLS fit at
        # penalty 0) is not a corner: the budget row stays active there and only its
        # multiplier passes through zero, so the CLA's active set does not change.
        assert lasso.path[-1].lam == 0.0
        end = tilts[-1]
        corners = [tp.lamb for tp in cla.turning_points if np.isfinite(tp.lamb) and tp.lamb > end * (1 + 1e-9)]
        assert corners == pytest.approx(tilts[:-1], rel=1e-8)
        assert any(tp.lamb < end for tp in cla.turning_points)  # the frontier continues past it

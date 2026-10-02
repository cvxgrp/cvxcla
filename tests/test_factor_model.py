"""The admissible factor model, and the accuracy of its Woodbury solve.

``FactorCovariance(d, u, delta)`` represents ``Sigma = diag(d) + U Delta U.T``.
``Delta`` must be symmetric positive semidefinite; it is folded into the loadings
(``U' = U V_+ Lambda_+^{1/2}``), so the Woodbury solve never inverts it and a
singular factor covariance is admissible. ``d > 0`` is checked by the operator
itself. With ``d > 0`` the capacitance matrix ``I + U_F'.T D_F^{-1} U_F'`` is
positive definite, and the solve stays as accurate as the dense one even for an
ill-conditioned ``Delta`` or nearly collinear loadings -- which is why no separate
guard on it is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvxcla import CLA, FactorCovariance

N, K = 60, 6


@pytest.fixture(scope="module")
def loadings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Idiosyncratic variances, loadings and expected returns for a small factor market."""
    rng = np.random.default_rng(1)
    return rng.uniform(0.5, 2.0, N), rng.standard_normal((N, K)) / np.sqrt(N), rng.uniform(0.0, 1.0, N)


class TestAdmissibleDelta:
    """``delta`` must be a symmetric positive-semidefinite factor covariance."""

    def test_asymmetric_delta_is_refused(self, loadings):
        """A non-symmetric delta would make Sigma non-symmetric."""
        d, u, _ = loadings
        delta = np.eye(K)
        delta[0, 1] = 0.5
        with pytest.raises(ValueError, match="symmetric"):
            FactorCovariance(d=d, u=u, delta=delta)

    @pytest.mark.parametrize(
        "delta",
        [np.r_[1.0, -0.5, np.ones(K - 2)], np.diag(np.r_[1.0, -1e-3, np.ones(K - 2)])],
        ids=["negative variance", "indefinite matrix"],
    )
    def test_indefinite_delta_is_refused(self, loadings, delta):
        """A negative factor variance is refused, with the dense matrix as the way out."""
        d, u, _ = loadings
        with pytest.raises(ValueError, match="positive semidefinite"):
            FactorCovariance(d=d, u=u, delta=delta)

    @pytest.mark.parametrize(
        ("delta", "rank"),
        [(np.r_[1.0, np.zeros(K - 1)], 1), (np.ones((K, K)), 1), (np.zeros(K), 1), (np.r_[np.ones(K - 1), 0.0], K - 1)],
        ids=["zero variances", "rank-one matrix", "zero", "one zero variance"],
    )
    def test_singular_delta_is_folded_into_the_loadings(self, loadings, delta, rank):
        """A singular delta keeps only the factor directions it spans, and Sigma is unchanged."""
        d, u, mean = loadings
        op = FactorCovariance(d=d, u=u, delta=delta)
        inner = np.diag(delta) if delta.ndim == 1 else delta
        dense = np.diag(d) + u @ inner @ u.T
        assert op.k == rank
        x = np.random.default_rng(3).standard_normal(N)
        np.testing.assert_allclose(op.matvec(x), dense @ x, atol=1e-12)
        np.testing.assert_allclose(_trace_weights(op, mean), _trace_weights(dense, mean), atol=1e-10)

    def test_tiny_positive_variance_is_accepted(self, loadings):
        """A factor with a tiny but positive variance is admissible."""
        d, u, _ = loadings
        FactorCovariance(d=d, u=u, delta=np.r_[1.0, np.full(K - 1, 1e-14)])


def _trace_weights(covariance: object, mean: np.ndarray) -> np.ndarray:
    """Turning-point weights of the long-only, fully-invested frontier."""
    cla = CLA(
        mean=mean,
        covariance=covariance,
        lower_bounds=np.zeros(N),
        upper_bounds=np.ones(N),
        a=np.ones((1, N)),
        b=np.ones(1),
    )
    return np.array([tp.weights for tp in cla.turning_points])


@pytest.mark.parametrize(
    "case",
    ["ill-conditioned delta", "nearly collinear loadings", "full rank"],
)
def test_factor_trace_matches_dense(loadings, case):
    """The Woodbury trace equals the dense trace of the same Sigma on hard factor models."""
    d, u, mean = loadings
    rng = np.random.default_rng(2)
    if case == "ill-conditioned delta":
        delta = np.geomspace(float(N), N * 1e-14, K)  # condition number 1e14
    elif case == "nearly collinear loadings":
        u = u.copy()
        u[:, 1] = u[:, 0] + 1e-10 * rng.standard_normal(N)
        delta = np.full(K, float(N))
    else:
        u = rng.standard_normal((N, N)) / np.sqrt(N)  # K = n: no low-rank saving, still exact
        delta = np.full(N, float(N))
    dense = np.diag(d) + (u * delta) @ u.T
    factor_w = _trace_weights(FactorCovariance(d=d, u=u, delta=delta), mean)
    dense_w = _trace_weights(dense, mean)
    assert factor_w.shape == dense_w.shape
    np.testing.assert_allclose(factor_w, dense_w, atol=1e-10)

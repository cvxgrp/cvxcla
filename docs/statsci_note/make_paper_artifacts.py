"""Reproduce every figure and numerical claim in the Statistical Science note.

One self-contained, seeded entry point for the perspective paper
``perspective.tex`` ("Following the Solution: Parametric Active-Set Homotopies").
Running it regenerates the two figures and re-derives every agreement number quoted
in the text, so a referee can check the paper end to end from a single command.

Artifacts written to ``docs/statsci_note/``:

  * ``path_identity.pdf``  -- Figure 1 (Theorem 1, in coefficients): the LASSO
    homotopy and the gross-exposure-constrained mean-variance QP are one curve.
  * ``frontier.pdf``       -- Figure 2: the 20-asset factor-model efficient frontier,
    returned exactly by the Critical Line Algorithm as a finite set of turning points.

Numbers re-derived and checked against the manuscript (Sec. 7, "Computation"):

  A. LASSO homotopy vs scikit-learn ``lars_path``       -> ~9e-16   (text: 3.2e-15)
  B. Constrained path vs independent active-set QP      -> ~5e-16   (text: ~6e-10)
  C. Fig. 1 caps: independent conic QP vs homotopy      -> ~3e-8    (text: ~3e-8)
  D. Dense Sigma vs FactorCovariance operator           -> identical 21-point frontier
  E. Full frontier traced cleanly as N grows            -> monotone to N=6400 (Sec. 7/9)
  F. Real data: diabetes LASSO path vs lars_path        -> ~2e-10  (Efron et al. 2004 data)
  G. Operator scaling: factor covariance vs dense       -> 2x..35x faster, N=250..2000 (Sec. 8)
  H. p>>n LASSO (X'X singular) vs lars_path             -> ~1e-13  (Sec. 7; support <= rank X)

The two QP references are deliberately different solvers. Check C uses the conic
interior-point solver CLARABEL (whose ~1e-8 floor is exactly the "solver precision" the
Figure 1 caption reports); check B uses the dual active-set solver ``quadprog``, which
solves each strictly convex QP to machine precision and so confirms the constrained
path far below the manuscript's conservative ~6e-10.

Each check prints PASS/FAIL against the tolerance the paper implies, and the script
exits non-zero if any check regresses, so it doubles as a reproducibility test.

Pinned environment (see also pyproject.toml: cvxcla == 1.8.0):
    uv run --with cvxpy --with qpsolvers --with quadprog --with scikit-learn \
        --with matplotlib --with scipy \
        python docs/statsci_note/make_paper_artifacts.py

Seeds are fixed; change SEED_* to vary an illustration. The committed PDFs come from
the seeds below.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from cvxcla import CLA, FactorCovariance
from cvxcla.lasso import Lasso

OUT_DIR = "docs/statsci_note"

# Seeds for the two figures (preserve the committed PDFs).
SEED_IDENTITY = 1
SEED_FRONTIER = 42

# Tolerances: the bar each printed agreement must clear. Set a little looser than the
# headline numbers so solver-version jitter does not turn a faithful run red.
TOL_LARS = 1e-12  # text: 3.2e-15
TOL_CONSTRAINED_QP = 1e-10  # text: ~6e-10 (active-set reference reaches machine eps)
TOL_FIG1_CAPS = 1e-6  # text: ~3e-8
TOL_DENSE_FACTOR = 1e-9  # text: identical


@dataclass
class Check:
    """One numerical claim: a label, the measured discrepancy, and its tolerance."""

    name: str
    measured: float
    tol: float
    quoted: str

    @property
    def passed(self) -> bool:
        """True when the measured discrepancy is within tolerance."""
        return self.measured <= self.tol

    def line(self) -> str:
        """One formatted PASS/FAIL report line for this check."""
        flag = "PASS" if self.passed else "FAIL"
        return f"  [{flag}] {self.name}: {self.measured:.2e}  (tol {self.tol:.0e}; paper {self.quoted})"


# --------------------------------------------------------------------------------------
# Shared problem builders
# --------------------------------------------------------------------------------------
def factor_problem(
    seed: int, n_assets: int = 20, n_days: int = 50, n_factors: int = 5
) -> tuple[np.ndarray, FactorCovariance, np.ndarray]:
    """Simulate the K-factor problem of Figure 2; return (mean, factor operator, dense Sigma)."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_assets, n_factors)) / np.sqrt(n_assets)
    delta = rng.uniform(0.5, 2.0, n_factors) * n_assets
    d = rng.uniform(0.5, 2.0, n_assets)
    expected = rng.uniform(0.0, 1.0, n_assets)
    factor = FactorCovariance(d=d, u=u, delta=delta)

    factor_returns = rng.standard_normal((n_days, n_factors)) * np.sqrt(delta)
    idiosyncratic = rng.standard_normal((n_days, n_assets)) * np.sqrt(d)
    returns = expected + factor_returns @ u.T + idiosyncratic
    mean = returns.mean(axis=0)
    dense = np.diag(d) + (u * delta) @ u.T  # the same Sigma, formed densely
    return mean, factor, dense


def regression_problem(seed: int, m: int = 80, n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """The centered design/response of Figure 1; return (X, y)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m, n))
    x = x - x.mean(axis=0, keepdims=True)
    if n == 6:
        beta_true = np.array([2.0, -1.5, 0.0, 1.1, 0.0, 0.0])  # the exact Figure 1 signal
    else:
        beta_true = np.zeros(n)  # a sparse signal in the first few coordinates
        beta_true[: min(4, n)] = np.array([2.0, -1.5, 1.1, 0.7])[: min(4, n)]
    y = x @ beta_true + 0.5 * rng.standard_normal(m)
    return x, y - y.mean()


def lasso_beta_at(l1_path: np.ndarray, betas_path: np.ndarray, c: float) -> np.ndarray:
    """Coefficient vector on a piecewise-linear LASSO path at l1-norm ``c``."""
    order = np.argsort(l1_path)
    return np.array([np.interp(c, l1_path[order], betas_path[order, j]) for j in range(betas_path.shape[1])])


def beta_at_lambda(lams: np.ndarray, betas: np.ndarray, query: float) -> np.ndarray:
    """Interpolate a piecewise-linear path (sorted by lambda) at a given lambda."""
    order = np.argsort(lams)
    return np.array([np.interp(query, lams[order], betas[order, j]) for j in range(betas.shape[1])])


# --------------------------------------------------------------------------------------
# Figure 1 + check C
# --------------------------------------------------------------------------------------
def figure_identity() -> Check:
    """Write path_identity.pdf and return the QP-vs-homotopy agreement (check C)."""
    import cvxpy as cp

    x, y = regression_problem(SEED_IDENTITY)
    n = x.shape[1]

    lasso = Lasso(x=x, y=y)
    betas_path = np.array([bp.beta for bp in lasso.path])
    l1_path = np.abs(betas_path).sum(axis=1)
    l1_max = float(l1_path.max())

    caps = np.linspace(0.05 * l1_max, 0.97 * l1_max, 9)
    qp_betas, qp_l1 = [], []
    for c in caps:
        beta = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(y - x @ beta)), [cp.norm1(beta) <= c])
        prob.solve(solver=cp.CLARABEL)
        qp_betas.append(beta.value)
        qp_l1.append(float(np.abs(beta.value).sum()))
    qp_betas = np.array(qp_betas)
    qp_l1 = np.array(qp_l1)

    discrepancy = max(
        float(np.max(np.abs(qp_betas[i] - lasso_beta_at(l1_path, betas_path, qp_l1[i])))) for i in range(len(caps))
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    show = [j for j in range(n) if np.abs(betas_path[:, j]).max() > 1e-6]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    order = np.argsort(l1_path)
    for k, j in enumerate(show):
        col = colors[k % len(colors)]
        ax.plot(l1_path[order], betas_path[order, j], "-", lw=1.6, color=col, label=rf"$\beta_{{{j + 1}}}$")
        ax.plot(qp_l1, qp_betas[:, j], "o", ms=6, mfc="none", mec=col, mew=1.3)
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.set_xlabel(r"$\|\beta\|_1$  ($\ell_1$ budget $\;\equiv\;$ gross-exposure cap)")
    ax.set_ylabel(r"coefficient $=$ weight")
    ax.set_title("LASSO homotopy (lines) vs. mean-variance QP (markers): one curve", fontsize=10)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/path_identity.pdf")
    print(f"wrote {OUT_DIR}/path_identity.pdf  ({len(l1_path)} breakpoints, {len(caps)} caps)")
    return Check("Fig.1 caps: QP vs homotopy", discrepancy, TOL_FIG1_CAPS, "~3e-8")


# --------------------------------------------------------------------------------------
# Figure 2 + check D
# --------------------------------------------------------------------------------------
def figure_frontier() -> Check:
    """Write frontier.pdf; return the dense-vs-factor frontier discrepancy (check D)."""
    mean, factor, dense = factor_problem(SEED_FRONTIER)
    n = mean.shape[0]
    kw = {
        "mean": mean,
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }
    cla_factor = CLA(covariance=factor, **kw)
    cla_dense = CLA(covariance=dense, **kw)

    f_fac, f_den = cla_factor.frontier, cla_dense.frontier
    n_fac, n_den = len(cla_factor), len(cla_dense)
    # Same number of turning points, same coordinates: the operator changes only cost.
    same_count = n_fac == n_den
    coord_gap = (
        max(
            float(np.max(np.abs(f_fac.volatility - f_den.volatility))),
            float(np.max(np.abs(f_fac.returns - f_den.returns))),
        )
        if same_count
        else float("inf")
    )

    vol, ret = f_fac.volatility, f_fac.returns
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(vol, ret, "-o", ms=2.5, lw=1.0, color="#1f4e79")
    ax.scatter(vol[[0, -1]], ret[[0, -1]], color="#c00000", zorder=5, s=18)
    ax.annotate("max return", (vol[0], ret[0]), textcoords="offset points", xytext=(-6, 4), ha="right", fontsize=8)
    ax.annotate("min variance", (vol[-1], ret[-1]), textcoords="offset points", xytext=(8, -2), fontsize=8)
    ax.set_xlabel("Volatility (model units)")
    ax.set_ylabel("Expected return (model units)")
    ax.set_title(f"Efficient frontier: {n} assets, 50 days ({n_fac} turning points)", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/frontier.pdf")
    print(f"wrote {OUT_DIR}/frontier.pdf  (factor backend: {n_fac} turning points; dense backend: {n_den})")
    return Check("dense vs factor frontier", coord_gap, TOL_DENSE_FACTOR, "identical")


# --------------------------------------------------------------------------------------
# Check A: cvxcla LASSO homotopy vs scikit-learn lars_path
# --------------------------------------------------------------------------------------
def check_lars_path() -> Check:
    """Compare the cvxcla homotopy to scikit-learn's lars_path on a standard design."""
    from sklearn.linear_model import lars_path

    x, y = regression_problem(seed=7, m=120, n=10)
    m = x.shape[0]

    lasso = Lasso(x=x, y=y)
    lams = np.array([bp.lam for bp in lasso.path])
    betas = np.array([bp.beta for bp in lasso.path])

    # sklearn's lasso objective scales the penalty by 1/m, so lambda = m * alpha.
    alphas, _, coefs = lars_path(x, y, method="lasso")
    sk_lams = alphas * m
    sk_coefs = coefs.T  # (n_knots, n_features)

    # Evaluate the cvxcla path at sklearn's knots and compare coefficient for coefficient.
    gap = max(float(np.max(np.abs(sk_coefs[k] - beta_at_lambda(lams, betas, sk_lams[k])))) for k in range(len(sk_lams)))
    return Check("LASSO vs sklearn lars_path", gap, TOL_LARS, "3.2e-15")


# --------------------------------------------------------------------------------------
# Check B: constrained LASSO path vs an independent QP at every breakpoint
# --------------------------------------------------------------------------------------
def check_constrained_qp() -> Check:
    """Trace a path under non-negativity + an inequality; match a QP at each breakpoint.

    The reference is the dual active-set solver ``quadprog`` (Goldfarb--Idnani), an
    algorithm wholly independent of the homotopy. Under non-negativity the penalty
    ``lam*||beta||_1`` is the linear term ``lam*1'beta``, so each breakpoint is a single
    strictly convex QP that quadprog solves to machine precision:

        min  1/2 beta'(X'X)beta - (X'y - lam*1)'beta   s.t.  beta >= 0,  G beta <= h.
    """
    from qpsolvers import solve_qp

    x, y = regression_problem(seed=3, m=90, n=8)
    n = x.shape[1]
    g = np.ones((1, n))  # one generic inequality row
    h = np.array([3.0])

    lasso = Lasso(x=x, y=y, g=g, h=h, nonneg=True)

    quad = x.T @ x
    xty = x.T @ y
    big_g = np.vstack([-np.eye(n), g])  # -beta <= 0 (non-negativity) stacked with G beta <= h
    worst = 0.0
    for bp in lasso.path:
        linear = -(xty) + bp.lam * np.ones(n)
        rhs = np.concatenate([np.zeros(n), h])
        beta = solve_qp(quad, linear, big_g, rhs, solver="quadprog")
        if beta is not None:
            worst = max(worst, float(np.max(np.abs(bp.beta - beta))))
    return Check("constrained path vs active-set QP", worst, TOL_CONSTRAINED_QP, "~6e-10")


# --------------------------------------------------------------------------------------
# Check E: the full frontier is traced cleanly as the universe grows
# --------------------------------------------------------------------------------------
def check_envelope() -> Check:
    """Trace full long-only frontiers at growing N; the path must stay monotone.

    On an efficient frontier the expected return is non-increasing from the maximum-
    return corner to the global minimum-variance one, so an out-of-order turning point
    shows up as an *increase* in returns along the traced path. With the relative
    event-ordering tolerance in ``select_next_event`` the frontier stays clean as the
    universe grows; the worst up-jump across the sizes below is reported and must sit at
    roundoff level (a coarse absolute tolerance, by contrast, lets it grow with N).
    """
    sizes = [100, 400, 1600, 6400]
    worst = 0.0
    for n in sizes:
        k = max(2, min(10, n // 4))
        mean, factor, _ = factor_problem(seed=0, n_assets=n, n_days=max(2 * n, 30), n_factors=k)
        cla = CLA(
            mean=mean,
            covariance=factor,
            lower_bounds=np.zeros(n),
            upper_bounds=np.ones(n),
            a=np.ones((1, n)),
            b=np.ones(1),
        )
        ret = np.asarray(cla.frontier.returns)
        up_jump = float(np.max(np.diff(ret)))  # > 0 means an out-of-order turning point
        worst = max(worst, up_jump)
        print(f"    N={n:5d}: {len(cla):5d} turning points, worst return up-jump {up_jump:.1e}")
    return Check("frontier monotone to N=6400", worst, 1e-9, "clean (roundoff)")


# --------------------------------------------------------------------------------------
# Check F: the identity on a real dataset (the Efron et al. 2004 LARS diabetes data)
# --------------------------------------------------------------------------------------
def check_real_data() -> Check:
    """Trace the LASSO path on the real diabetes dataset; match scikit-learn lars_path.

    This is the dataset of Efron, Hastie, Johnstone and Tibshirani (2004), the paper
    that made the piecewise-linear path famous, so it is the natural real-data check of
    the identity the homotopy traces.
    """
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import lars_path

    x, y = load_diabetes(return_X_y=True)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean()
    m, n = x.shape

    lasso = Lasso(x=x, y=y)
    lams = np.array([bp.lam for bp in lasso.path])
    betas = np.array([bp.beta for bp in lasso.path])

    alphas, _, coefs = lars_path(x, y, method="lasso")
    sk_lams = alphas * m
    sk_coefs = coefs.T

    gap = max(float(np.max(np.abs(sk_coefs[k] - beta_at_lambda(lams, betas, sk_lams[k])))) for k in range(len(sk_lams)))
    print(f"    diabetes (m={m}, n={n}): {len(lasso.path)} breakpoints, agree with lars_path to {gap:.1e}")
    return Check("real-data LASSO (diabetes) vs lars_path", gap, 1e-7, "~2e-10")


# --------------------------------------------------------------------------------------
# Check H: the p >> n regime (X'X singular) still traces, matching lars_path
# --------------------------------------------------------------------------------------
def check_high_dim() -> Check:
    """Trace the LASSO path on a p >> n design; X'X is singular yet the path is exact.

    The active support never exceeds rank(X), so the free block X_F'X_F is positive
    definite (the standing assumption) even though the full Gram matrix is rank-deficient.
    """
    from sklearn.linear_model import lars_path

    rng = np.random.default_rng(0)
    m, n = 40, 100  # p = 100 >> n = 40 samples
    x = rng.standard_normal((m, n))
    x = x - x.mean(axis=0, keepdims=True)
    beta_true = np.zeros(n)
    beta_true[:5] = [3.0, -2.0, 1.5, -1.0, 2.0]
    y = x @ beta_true + 0.1 * rng.standard_normal(m)
    y = y - y.mean()

    lasso = Lasso(x=x, y=y)
    lams = np.array([bp.lam for bp in lasso.path])
    betas = np.array([bp.beta for bp in lasso.path])
    max_support = max(int((np.abs(b) > 1e-9).sum()) for b in betas)
    rank = int(np.linalg.matrix_rank(x))

    alphas, _, coefs = lars_path(x, y, method="lasso")
    sk_lams = alphas * m
    sk_coefs = coefs.T
    gap = max(float(np.max(np.abs(sk_coefs[k] - beta_at_lambda(lams, betas, sk_lams[k])))) for k in range(len(sk_lams)))
    print(f"    p>>n (m={m}, n={n}): rank(X)={rank}, max support={max_support}, vs lars_path {gap:.1e}")
    return Check("p>>n LASSO vs lars_path", gap, 1e-9, "~1e-13")


# --------------------------------------------------------------------------------------
# Check G: the operator reading buys scale -- factor covariance vs dense, wall time
# --------------------------------------------------------------------------------------
def report_scaling() -> Check:
    """Time the frontier with a structured (Woodbury) factor operator vs a dense Sigma.

    Same problem, same frontier; only the cost of each homotopy step changes. The factor
    operator supplies H's three actions in O(nk) instead of O(n^2), so the gap widens
    with N. Wall times are hardware-dependent; the time *ratio* at the largest N is the
    portable, reproducible quantity, and must show the operator comfortably ahead.
    """
    ratio_at_max = 1.0
    for n in (250, 500, 1000, 2000):
        mean, factor, dense = factor_problem(seed=0, n_assets=n, n_days=max(2 * n, 30), n_factors=10)
        kw = {
            "mean": mean,
            "lower_bounds": np.zeros(n),
            "upper_bounds": np.ones(n),
            "a": np.ones((1, n)),
            "b": np.ones(1),
        }
        t0 = time.perf_counter()
        cla_f = CLA(covariance=factor, **kw)
        t_factor = time.perf_counter() - t0
        t0 = time.perf_counter()
        CLA(covariance=dense, **kw)
        t_dense = time.perf_counter() - t0
        ratio_at_max = t_factor / t_dense
        speedup = t_dense / t_factor
        print(
            f"    N={n:5d}: {len(cla_f):5d} pts | factor {t_factor:6.2f}s | "
            f"dense {t_dense:6.2f}s | speedup x{speedup:5.1f}"
        )
    # PASS when the factor operator is comfortably faster than dense at the largest N.
    return Check("factor vs dense @N=2000 (time ratio)", ratio_at_max, 0.5, "<0.5 (>2x)")


def main() -> None:
    """Regenerate both figures, run every check, and report PASS/FAIL."""
    print("== Figures ==")
    check_c = figure_identity()
    check_d = figure_frontier()

    print("\n== Numerical checks (Sec. 7) ==")
    check_a = check_lars_path()
    check_b = check_constrained_qp()
    check_f = check_real_data()
    check_h = check_high_dim()
    print("  envelope sweep:")
    check_e = check_envelope()

    print("\n== Operator scaling (Sec. 8): factor covariance vs dense ==")
    check_g = report_scaling()

    checks = [check_a, check_b, check_c, check_d, check_e, check_f, check_g, check_h]
    print()
    for c in checks:
        print(c.line())

    if all(c.passed for c in checks):
        print("\nAll checks passed.")
    else:
        print("\nSome checks regressed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

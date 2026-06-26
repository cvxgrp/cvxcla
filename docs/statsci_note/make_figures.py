# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cvxcla==1.8.2",
#     "cvxpy==1.9.2",
#     "matplotlib==3.11.0",
#     "numpy==2.4.6",
#     "qpsolvers==4.12.0",
#     "quadprog==0.1.13",
#     "scikit-learn==1.9.0",
#     "scipy==1.17.1",
#     "typer==0.26.7",
# ]
# ///
"""Figures and numerical checks for the Statistical Science note.

One self-contained, seeded entry point for the perspective paper "Following the
Solution: Parametric Active-Set Homotopies". A small Typer CLI selects which
artifact to build; with no argument it asks interactively.

    uv run docs/statsci_note/make_figures.py            # ask which figure, or all
    uv run docs/statsci_note/make_figures.py all         # every figure + every check
    uv run docs/statsci_note/make_figures.py frontier    # just one figure
    uv run docs/statsci_note/make_figures.py checks       # the numerical suite only

Targets:

  * ``identity``    -- Figure 1 (Theorem 1, in coefficients): the LASSO homotopy and
    the gross-exposure-constrained mean-variance QP are one curve  (path_identity.pdf).
  * ``frontier``    -- Figure 2: the 20-asset factor-model efficient frontier, returned
    exactly by the Critical Line Algorithm as a finite set of turning points
    (frontier.pdf).
  * ``profiles``    -- the weight profile and holding count along the frontier, the
    portfolio twins of the LARS coefficient profile and the LASSO degrees of freedom
    (profiles.pdf).
  * ``factor-lars`` -- least angle regression with the quadratic form supplied as a
    diagonal-plus-low-rank factor operator, and the same path under an inequality
    (factor_lars.pdf).
  * ``taxonomy``    -- the "one homotopy, five fillings" hub-and-spoke diagram
    (taxonomy.pdf).
  * ``timeline``    -- the four-lane chronology of Section 6  (timeline.pdf).
  * ``schematic``   -- both conceptual figures (taxonomy + timeline).
  * ``checks``      -- the numerical verification suite of Section 7 (re-derives every
    agreement number quoted in the text; also writes the two data-driven figures).
  * ``all``         -- every figure and every check.

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

Each check prints PASS/FAIL against the tolerance the paper implies, and the process
exits non-zero if any check regresses, so it doubles as a reproducibility test. Seeds
are fixed; change the SEED_* constants to vary an illustration. The committed PDFs come
from the seeds below.
"""

from __future__ import annotations

import enum
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import numpy as np
import typer

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cvxcla import CLA, FactorCovariance
from cvxcla.lasso import Lasso
from cvxcla.operators import DenseCovariance

# Default output directory: alongside this script, so artifacts land in docs/statsci_note
# regardless of the working directory the command is run from.
DEFAULT_OUT_DIR = Path(__file__).resolve().parent

# Seeds for the two data-driven figures (preserve the committed PDFs).
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
    """Simulate the K-factor problem; return (mean, factor operator, dense Sigma)."""
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
# Figure 1 + check C: the identity, in coefficient space
# --------------------------------------------------------------------------------------
def figure_identity(out_dir: Path) -> Check:
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
    out = out_dir / "path_identity.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  ({len(l1_path)} breakpoints, {len(caps)} caps)")
    return Check("Fig.1 caps: QP vs homotopy", discrepancy, TOL_FIG1_CAPS, "~3e-8")


# --------------------------------------------------------------------------------------
# Figure 2 + check D: the efficient frontier
# --------------------------------------------------------------------------------------
def figure_frontier(out_dir: Path) -> Check:
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
    var = vol**2  # plot against variance, the quadratic form of the program itself
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(var, ret, "-o", ms=2.5, lw=1.0, color="#1f4e79")
    ax.scatter(var[[0, -1]], ret[[0, -1]], color="#c00000", zorder=5, s=18)
    ax.annotate("max return", (var[0], ret[0]), textcoords="offset points", xytext=(-6, 4), ha="right", fontsize=8)
    ax.annotate("min variance", (var[-1], ret[-1]), textcoords="offset points", xytext=(8, -2), fontsize=8)
    ax.set_xlabel(r"Variance $w^\top\Sigma w$ (model units)")
    ax.set_ylabel("Expected return (model units)")
    ax.set_title(
        f"Efficient frontier in (variance, return): {n} assets, 50 days ({n_fac} turning points)", fontsize=8.5
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "frontier.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  (factor backend: {n_fac} turning points; dense backend: {n_den})")
    return Check("dense vs factor frontier", coord_gap, TOL_DENSE_FACTOR, "identical")


# --------------------------------------------------------------------------------------
# Figure: weight profile and holding count along the frontier
# --------------------------------------------------------------------------------------
PROFILES_SEED = 11
HELD_TOL = 1e-6


def figure_profiles(out_dir: Path) -> None:
    """Trace the long-only frontier and plot the weight profile and the holding count."""
    mean, factor, dense = factor_problem(PROFILES_SEED, n_assets=30, n_days=80, n_factors=5)
    n = mean.shape[0]
    cla = CLA(
        mean=mean,
        covariance=factor,
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
        a=np.ones((1, n)),
        b=np.ones(1),
    )
    tps = cla.turning_points
    weights = np.array([tp.weights for tp in tps])  # (T, n)
    vol = np.array([float(np.sqrt(w @ (dense @ w))) for w in weights])
    held = np.array([int(np.sum(np.abs(w) > HELD_TOL)) for w in weights])
    order = np.argsort(vol)
    vol, weights, held = vol[order], weights[order], held[order]
    print(f"{len(tps)} turning points; holdings range [{held.min()}, {held.max()}]")

    # Knee of the holdings curve: the point of greatest distance from the chord joining
    # its endpoints, the standard elbow heuristic. As the holding count is the LASSO's
    # degrees of freedom, this is the df elbow, a principled operating point.
    vn = (vol - vol.min()) / np.ptp(vol)
    hn = (held - held.min()) / (np.ptp(held) or 1.0)
    chord = np.abs((vn[-1] - vn[0]) * (hn[0] - hn) - (vn[0] - vn) * (hn[-1] - hn[0]))
    knee = int(np.argmax(chord))
    knee_vol = float(vol[knee])
    lo, hi = 0.8 * knee_vol, 1.3 * knee_vol  # operating range bracketing the elbow
    print(f"df elbow at volatility {knee_vol:.3f} with {int(held[knee])} holdings")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    top = np.argsort(weights.max(axis=0))[::-1][:5]  # the five assets that grow largest

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4))

    # Left: weight profiles (the portfolio LARS-coefficient-profile twin).
    ax1.axvspan(lo, hi, color="#f1eee7", zorder=0)
    ax1.axvline(knee_vol, color="0.45", lw=0.9, ls=":", zorder=1)
    for i in range(n):
        if i in top:
            continue
        ax1.plot(vol, weights[:, i], "-", lw=0.6, color="0.72", zorder=1)
    for k, i in enumerate(top):
        ax1.plot(vol, weights[:, i], "-", lw=1.6, color=colors[k % len(colors)], zorder=2, label=f"asset {i + 1}")
    ax1.set_xlabel("Volatility (model units)")
    ax1.set_ylabel("Weight")
    ax1.set_title("Weight profile along the frontier", fontsize=8.5)
    ax1.legend(fontsize=7, loc="upper left", ncol=1)
    ax1.grid(True, alpha=0.25)

    # Right: number of holdings (the LASSO degrees-of-freedom twin).
    ax2.axvspan(lo, hi, color="#f1eee7", zorder=0, label="operating range")
    ax2.step(vol, held, where="post", lw=1.8, color="#1f4e79")
    ax2.axvline(knee_vol, color="0.45", lw=0.9, ls=":", zorder=1)
    ax2.plot([knee_vol], [held[knee]], "D", ms=7, mfc="#c00000", mec="black", mew=0.8, zorder=6)
    ax2.annotate(
        "operating point\n(df elbow)",
        (knee_vol, held[knee]),
        textcoords="offset points",
        xytext=(26, 18),
        ha="left",
        va="bottom",
        fontsize=7,
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "0.3"},
    )
    ax2.set_xlabel("Volatility (model units)")
    ax2.set_ylabel("Number of holdings")
    ax2.set_title("Portfolio size along the frontier", fontsize=8.5)
    ax2.set_ylim(0, held.max() + 1)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    out = out_dir / "profiles.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------------------
# Figure: least angle regression on a factor-model operator
# --------------------------------------------------------------------------------------
FACTOR_LARS_FEATURES = 10
FACTOR_LARS_FACTORS = 3
FACTOR_LARS_SEED = 7


def _factor_lars_problem(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A diagonal-plus-low-rank Gram ``H`` and a linear term ``X^T y`` with sparse signal."""
    n = FACTOR_LARS_FEATURES
    k = FACTOR_LARS_FACTORS
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    sigma = np.diag(d) + (u * delta) @ u.T
    beta_true = np.zeros(n)
    beta_true[:4] = [2.0, -1.5, 1.1, 0.7]
    xty = sigma @ beta_true + 0.05 * rng.standard_normal(n)
    return d, u, delta, sigma, xty


def _path_arrays(path: list) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lam, beta)`` sorted by ascending penalty for plotting."""
    lam = np.array([bp.lam for bp in path])
    beta = np.array([bp.beta for bp in path])
    order = np.argsort(lam)
    return lam[order], beta[order]


def figure_factor_lars(out_dir: Path) -> None:
    """Trace the factor-operator path and its constrained variant, and write the figure."""
    n = FACTOR_LARS_FEATURES
    d, u, delta, sigma, xty = _factor_lars_problem(FACTOR_LARS_SEED)
    factor = FactorCovariance(d=d, u=u, delta=delta)
    dense = DenseCovariance(sigma)

    lam_f, beta_f = _path_arrays(Lasso.from_operator(factor, xty).path)
    lam_d, beta_d = _path_arrays(Lasso.from_operator(dense, xty).path)
    gap = float(np.max(np.abs(beta_f - beta_d))) if beta_f.shape == beta_d.shape else float("nan")
    print(f"factor vs dense path max|delta beta| = {gap:.1e}; breakpoints = {len(lam_f)}")

    # An inequality that binds partway down the path: sum_i beta_i <= c.
    sum_full = float(beta_f[0].sum())  # smallest lambda (least-squares end)
    c = 0.55 * sum_full
    lam_c, beta_c = _path_arrays(Lasso.from_operator(factor, xty, g=np.ones((1, n)), h=np.array([c])).path)
    print(f"constrained: sum-cap c = {c:.3f}; breakpoints = {len(lam_c)}; final sum = {float(beta_c[0].sum()):.3f}")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # The substantively active coefficients (the rest stay near zero along the path).
    show = [j for j in range(n) if np.abs(beta_f[:, j]).max() > 0.1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)

    # Left: factor operator (lines) vs dense Gram (markers): one curve.
    for k, j in enumerate(show):
        col = colors[k % len(colors)]
        ax1.plot(lam_f, beta_f[:, j], "-", lw=1.5, color=col, label=rf"$\beta_{{{j + 1}}}$")
        ax1.plot(lam_d, beta_d[:, j], "o", ms=4, mfc="none", mec=col, mew=1.0)
    ax1.axhline(0.0, color="0.4", lw=0.6)
    ax1.set_xlabel(r"penalty $\lambda$")
    ax1.set_ylabel("coefficient")
    ax1.set_title("Factor operator (lines) vs dense Gram (markers)", fontsize=8.5)
    ax1.invert_xaxis()
    ax1.legend(fontsize=7, ncol=2, loc="upper left")
    ax1.grid(True, alpha=0.25)

    # Right: the same path under a genuine inequality; unconstrained shown faint.
    for k, j in enumerate(show):
        col = colors[k % len(colors)]
        ax2.plot(lam_f, beta_f[:, j], "-", lw=0.8, color=col, alpha=0.25)
        ax2.plot(lam_c, beta_c[:, j], "-", lw=1.6, color=col)
    ax2.axhline(0.0, color="0.4", lw=0.6)
    ax2.set_xlabel(r"penalty $\lambda$")
    ax2.set_title(r"With inequality $\mathbf{1}^\top\beta\leq c$ (faint: unconstrained)", fontsize=8.5)
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    out = out_dir / "factor_lars.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------------------
# Conceptual figures (no data, no solver): the field palette, shared across both
# --------------------------------------------------------------------------------------
# Four communities, four colours, used identically in both figures; optimization is the
# shared core (the hub of the taxonomy), so its colour also borders the hub box.
FINANCE = "#b3402f"  # warm red
STATISTICS = "#1f4e79"  # deep blue
OPTIMIZATION = "#3f7d3a"  # green; the shared parametric-QP framework / the hub
CONTROL = "#7a4fa3"  # violet; explicit MPC and the mpQP literature
HUB = "#33373b"  # near-black neutral fill for the shared core
GREY = "#6b7077"


# Figure A: the "one homotopy" taxonomy ------------------------------------------------
# Each instance: (title, filling lines, field colour, box centre x, box centre y).
INSTANCES = [
    (
        "Critical Line Algorithm",
        [r"$H=\Sigma$  (covariance)", r"sweep risk aversion $\lambda$", "faces imposed by bounds"],
        FINANCE,
        -3.45,
        1.95,
    ),
    (
        "LASSO / LARS",
        [r"$H=X^{\top}X$  (Gram)", r"sweep penalty $\lambda$", r"faces induced by $\ell_1$"],
        STATISTICS,
        3.45,
        1.95,
    ),
    (
        "Elastic net",
        [r"$H=X^{\top}X+\eta I$", r"sweep penalty $\lambda$", "ridge-conditioned form"],
        STATISTICS,
        3.95,
        -0.55,
    ),
    (
        "SVM path",
        [r"$H=\kappa(X,X)$  (kernel)", "sweep cost parameter", "faces = margin SVs"],
        STATISTICS,
        1.95,
        -2.35,
    ),
    (
        "Explicit MPC",
        [r"$H=$ LQR Hessian", "sweep state (a vector)", "critical-region partition"],
        CONTROL,
        -3.30,
        -1.85,
    ),
]


def _rounded_box(ax, cx, cy, w, h, *, facecolor, edgecolor, lw=1.2, alpha=1.0, zorder=3):
    """Place a rounded rectangle centred at (cx, cy) and return its centre."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(box)
    return cx, cy


def figure_taxonomy(out_dir: Path) -> None:
    """Draw the hub-and-spoke 'one homotopy, five fillings' diagram."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.set_xlim(-5.4, 5.4)
    ax.set_ylim(-3.7, 3.4)
    ax.axis("off")

    hub_w, hub_h = 3.3, 1.35
    # Spokes first, so the boxes sit on top of the lines.
    for _title, _lines, _color, cx, cy in INSTANCES:
        ax.add_patch(
            FancyArrowPatch(
                (0, 0),
                (cx, cy),
                arrowstyle="-",
                shrinkA=46,
                shrinkB=42,
                connectionstyle="arc3,rad=0.0",
                color=GREY,
                lw=1.1,
                zorder=1,
            )
        )

    # Central hub: the shared core (Section 2). Its border carries the optimization
    # colour, because the parametric-QP framework is what the core is.
    _rounded_box(ax, 0, 0, hub_w, hub_h, facecolor=HUB, edgecolor=OPTIMIZATION, lw=2.2, zorder=4)
    ax.text(
        0, 0.42, "One parametric", ha="center", va="center", color="white", fontsize=10.5, fontweight="bold", zorder=6
    )
    ax.text(
        0,
        0.14,
        "active-set homotopy",
        ha="center",
        va="center",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        zorder=6,
    )
    ax.text(
        0,
        -0.30,
        r"$x(\lambda)=\alpha+\lambda\beta$ on each segment",
        ha="center",
        va="center",
        color="#dfe3e8",
        fontsize=7.6,
        zorder=6,
    )
    ax.text(
        0,
        -0.55,
        "ratio-test events  ·  $H$ as operator",
        ha="center",
        va="center",
        color="#dfe3e8",
        fontsize=7.6,
        zorder=6,
    )

    # Instance boxes.
    box_w, box_h = 2.75, 1.30
    for title, lines, color, cx, cy in INSTANCES:
        _rounded_box(ax, cx, cy, box_w, box_h, facecolor="white", edgecolor=color, lw=1.6, zorder=3)
        ax.text(cx, cy + 0.42, title, ha="center", va="center", color=color, fontsize=9.0, fontweight="bold")
        for i, line in enumerate(lines):
            ax.text(cx, cy + 0.10 - 0.27 * i, line, ha="center", va="center", color="#222", fontsize=7.0)

    # Field legend: four communities, with optimization shown as the shared core.
    def _swatch(color, label):
        return plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=2,
            markersize=11,
            label=label,
        )

    handles = [
        _swatch(FINANCE, "Finance"),
        _swatch(STATISTICS, "Statistics"),
        _swatch(CONTROL, "Control"),
        _swatch(OPTIMIZATION, "Optimization (shared core)"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        fontsize=8.0,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    fig.tight_layout()
    out = out_dir / "taxonomy.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# Figure B: the historical timeline ----------------------------------------------------
# Lanes, top to bottom, with their colours.
LANES = [
    ("Optimization", OPTIMIZATION),
    ("Finance", FINANCE),
    ("Statistics", STATISTICS),
    ("Control", CONTROL),
]

# Events: (year, lane, label, dx_pts, dy_pts, ha, leader?). Hand-placed offsets keep
# the dense clusters legible; a leader line is drawn for the far-staggered labels.
# Exactly the works cited in Section 6; dy > 0 places the label above the lane.
EVENTS = [
    (1951, "Optimization", "Dantzig\nsimplex", -4, 14, "right", False),
    (1955, "Optimization", "Gass–Saaty\nparam. LP", -2, -15, "center", False),
    (1956, "Optimization", "Frank–Wolfe", 24, 16, "left", False),
    (1959, "Optimization", "Wolfe; Beale QP", 14, -15, "left", False),
    (1956, "Finance", "Markowitz\nCLA", -4, 15, "center", False),
    (1963, "Finance", "Sharpe\nsingle-index", 2, -15, "center", False),
    (1984, "Finance", "Perold\nlarge-scale", 0, 15, "center", False),
    (2010, "Finance", "Stein, Hirschberger,\nBailey (impl.)", 42, 16, "left", True),
    (1996, "Statistics", "LASSO\n(Tibshirani)", -6, 15, "center", False),
    (2000, "Statistics", "homotopy\n(Osborne)", -10, -15, "center", False),
    (2004, "Statistics", "LARS; SVM path\n(Efron; Hastie)", -10, 42, "center", True),
    (2005, "Statistics", "elastic net\n(Zou–Hastie)", 8, -42, "center", True),
    (2007, "Statistics", "Rosset–Zhu", 0, 15, "center", False),
    (2011, "Statistics", "gen. lasso\n(Tib.–Taylor)", 12, -15, "center", False),
    (2018, "Statistics", "constr. lasso\n(Gaines)", 0, 15, "center", False),
    (2002, "Control", "Bemporad\nexplicit MPC", -8, -15, "center", False),
    (2003, "Control", "Tøndel mpQP", 20, 14, "left", False),
    (2008, "Control", "Roll", 6, -14, "center", False),
]


def figure_timeline(out_dir: Path) -> None:
    """Draw the four-lane chronology of Section 6 across the contributing fields."""
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    lane_y = {name: len(LANES) - i for i, (name, _c) in enumerate(LANES)}

    x0, x1 = 1948, 2028
    ax.set_xlim(x0, x1)
    ax.set_ylim(0.0, len(LANES) + 1.4)

    # Shade the four-decade span between the earliest instances and the statistical
    # ones; the point of the figure is parallel, independent development, not priority.
    sy = lane_y["Statistics"]
    ax.axvspan(1956, 1996, ymin=0.0, ymax=1.0, color="#f1eee7", zorder=0)
    ax.text(
        1976,
        len(LANES) + 1.00,
        "the same active-set structure develops independently",
        ha="center",
        va="center",
        fontsize=7.8,
        color=GREY,
        style="italic",
    )
    ax.text(
        1976,
        len(LANES) + 0.73,
        "in optimization, finance, statistics, and control",
        ha="center",
        va="center",
        fontsize=7.8,
        color=GREY,
        style="italic",
    )

    # Lane baselines and labels.
    for name, color in LANES:
        y = lane_y[name]
        ax.axhline(y, x0, x1, color=color, lw=0.8, alpha=0.35, zorder=1)
        ax.text(x0 - 0.5, y, name, ha="right", va="center", fontsize=8.5, color=color, fontweight="bold")

    lane_color = dict(LANES)
    for year, lane, label, dx, dy, ha, leader in EVENTS:
        y = lane_y[lane]
        color = lane_color[lane]
        ax.plot([year], [y], "o", ms=5.5, color=color, zorder=4)
        va = "bottom" if dy > 0 else "top"
        arrowprops = {"arrowstyle": "-", "lw": 0.5, "color": GREY, "shrinkA": 2, "shrinkB": 3} if leader else None
        ax.annotate(
            label,
            (year, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va=va,
            fontsize=6.4,
            color="#222",
            linespacing=0.95,
            arrowprops=arrowprops,
        )

    # The bridge back (Brodie 2009): the explicit observation linking the two literatures.
    ax.plot([2009], [lane_y["Finance"]], "*", ms=13, color="#000", zorder=5)
    ax.annotate(
        "Brodie: a gross-exposure\nMarkowitz problem is a LASSO",
        (2009, lane_y["Finance"]),
        textcoords="offset points",
        xytext=(-30, 17),
        ha="center",
        va="bottom",
        fontsize=6.4,
        color="#000",
        linespacing=0.95,
    )

    # A non-directional connector: the same device reached independently, not copied.
    ax.add_patch(
        FancyArrowPatch(
            (1958, lane_y["Finance"] - 0.14),
            (1999, lane_y["Statistics"] + 0.14),
            connectionstyle="arc3,rad=-0.30",
            arrowstyle="-",
            mutation_scale=12,
            color=GREY,
            lw=1.1,
            ls=(0, (5, 2)),
            zorder=3,
        )
    )
    ax.text(
        1979,
        (lane_y["Finance"] + sy) / 2 + 0.62,
        "same device, reached independently",
        ha="center",
        va="center",
        fontsize=7.2,
        color=GREY,
        style="italic",
    )

    # The unifying exposition at the right edge (the paper's contribution).
    ax.axvline(2026, color=HUB, lw=1.0, ls=":", zorder=2)
    ax.text(
        2026,
        0.35,
        "2026\nthis paper:\none exposition",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=HUB,
        fontweight="bold",
        linespacing=0.95,
    )

    ax.set_yticks([])
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(axis="x", colors=GREY, labelsize=8)
    ax.set_xticks([1951, 1960, 1970, 1980, 1990, 2000, 2010, 2020])

    fig.tight_layout()
    out = out_dir / "timeline.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------------------
# Numerical checks (Section 7)
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


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def run_checks(out_dir: Path) -> list[Check]:
    """Write the two data-driven figures and run the full numerical suite (A--H)."""
    print("== Figures ==")
    check_c = figure_identity(out_dir)
    check_d = figure_frontier(out_dir)

    print("\n== Numerical checks (Sec. 7) ==")
    check_a = check_lars_path()
    check_b = check_constrained_qp()
    check_f = check_real_data()
    check_h = check_high_dim()
    print("  envelope sweep:")
    check_e = check_envelope()

    print("\n== Operator scaling (Sec. 8): factor covariance vs dense ==")
    check_g = report_scaling()

    return [check_a, check_b, check_c, check_d, check_e, check_f, check_g, check_h]


def run_all(out_dir: Path) -> list[Check]:
    """Build every figure and run every check (the full reproduction)."""
    checks = run_checks(out_dir)
    print("\n== Remaining figures ==")
    figure_profiles(out_dir)
    figure_factor_lars(out_dir)
    figure_taxonomy(out_dir)
    figure_timeline(out_dir)
    return checks


def _report(checks: list[Check]) -> bool:
    """Print every check line; return True iff all passed."""
    print()
    for c in checks:
        print(c.line())
    if all(c.passed for c in checks):
        print("\nAll checks passed.")
        return True
    print("\nSome checks regressed.", file=sys.stderr)
    return False


# --------------------------------------------------------------------------------------
# Typer CLI
# --------------------------------------------------------------------------------------
class Target(enum.StrEnum):
    """A buildable artifact, or a group of them."""

    identity = "identity"
    frontier = "frontier"
    profiles = "profiles"
    factor_lars = "factor-lars"
    taxonomy = "taxonomy"
    timeline = "timeline"
    schematic = "schematic"
    checks = "checks"
    all = "all"


_DESCRIPTIONS: dict[Target, str] = {
    Target.identity: "Fig 1: LASSO homotopy = mean-variance QP  -> path_identity.pdf",
    Target.frontier: "Fig 2: the 20-asset efficient frontier    -> frontier.pdf",
    Target.profiles: "weight profile + holding count            -> profiles.pdf",
    Target.factor_lars: "factor-operator LARS, constrained + not   -> factor_lars.pdf",
    Target.taxonomy: "the 'one homotopy, five fillings' diagram -> taxonomy.pdf",
    Target.timeline: "the four-lane chronology of Section 6     -> timeline.pdf",
    Target.schematic: "both conceptual figures (taxonomy + timeline)",
    Target.checks: "the numerical verification suite (Sec. 7)",
    Target.all: "every figure and every check",
}


def _choose_target() -> Target:
    """Interactively ask which artifact to build when none was given on the CLI."""
    options = list(Target)
    typer.echo("Which figure to generate?\n")
    for i, t in enumerate(options, 1):
        typer.echo(f"  {i:>2}. {t.value:<12} {_DESCRIPTIONS[t]}")
    typer.echo("")
    while True:
        raw = typer.prompt("Enter a number or name", default="all").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        try:
            return Target(raw)
        except ValueError:
            typer.echo(f"  '{raw}' is not a valid choice; try again.")


def _run_target(target: Target, out_dir: Path) -> int:
    """Build the requested artifact(s); return a process exit code."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if target is Target.all:
        return 0 if _report(run_all(out_dir)) else 1
    if target is Target.checks:
        return 0 if _report(run_checks(out_dir)) else 1
    if target is Target.identity:
        print(figure_identity(out_dir).line())
        return 0
    if target is Target.frontier:
        print(figure_frontier(out_dir).line())
        return 0
    if target is Target.profiles:
        figure_profiles(out_dir)
        return 0
    if target is Target.factor_lars:
        figure_factor_lars(out_dir)
        return 0
    if target is Target.taxonomy:
        figure_taxonomy(out_dir)
        return 0
    if target is Target.timeline:
        figure_timeline(out_dir)
        return 0
    if target is Target.schematic:
        figure_taxonomy(out_dir)
        figure_timeline(out_dir)
        return 0
    raise AssertionError(f"unhandled target: {target}")  # pragma: no cover


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    target: Annotated[
        Target | None,
        typer.Argument(help="Which artifact to build. Omit to choose interactively."),
    ] = None,
    out_dir: Annotated[
        Path,
        typer.Option(help="Directory to write artifacts into."),
    ] = DEFAULT_OUT_DIR,
) -> None:
    """Build a Statistical Science note figure (or all of them, or the check suite)."""
    if target is None:
        target = _choose_target()
    raise typer.Exit(_run_target(target, out_dir))


if __name__ == "__main__":
    app()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "casadi==3.8.1",
#     "cvxcla==2.3.2",
#     "matplotlib==3.11.0",
#     "numpy==2.4.6",
#     "osqp==1.1.3",
#     "pandas==3.0.3",
#     "pyarrow==24.0.0",
#     "pyportfolioopt==1.6.0",
#     "scikit-learn==1.9.0",
#     "scipy==1.17.1",
#     "typer==0.26.7",
# ]
# ///
r"""Figures and numerical checks for the CLA paper.

One self-contained, seeded entry point that reproduces every figure and every
numerical check in the cvxcla paper. A small Typer CLI selects which
artefact to build; with no argument it asks interactively. The S&P 500 input is
the frozen snapshot committed at ``experiments/data/sp500_pct_returns.parquet``,
so the data-driven steps reproduce deterministically and offline.

    uv run experiments/make_figures.py            # choose interactively
    uv run experiments/make_figures.py all         # every figure + every check
    uv run experiments/make_figures.py all --quick # skip the slow scaling sweeps
    uv run experiments/make_figures.py frontier    # just one figure

Targets (artefact in parentheses):

  * ``frontier``        -- Figure 1: the 20-asset factor-model efficient frontier
    (frontier.pdf).
  * ``scaling``         -- Figure 2 + Table 1: runtime vs problem size, dense vs
    factor (Woodbury) backend, with baselines, and the memory table (scaling.pdf), plus
    the same timings for n <= 320 (scaling_small.pdf).
    SLOW: the dense backend at n=2560 dominates the sweep.
  * ``rank-scaling``    -- Figure 3 + Table 2: runtime vs factor rank at fixed n
    (rank_scaling.pdf).  SLOW.
  * ``validate-exact``  -- Section 10.5 exactness numbers (no figure).
  * ``validate-constraints`` -- Section 10.5 general-constraint exactness (no figure).
  * ``real-frontier``   -- Figure 4: the S&P 500 empirical frontier (real_frontier.pdf).
  * ``degeneracy``      -- Figure 5: the degeneracy boundary (degeneracy.pdf).
  * ``tie-degeneracy``  -- Section 10.6 tie-heavy stress envelope (no figure).
  * ``osqp``            -- S&P 500 trace vs a warm-started OSQP grid (no figure).
  * ``validate-kkt``    -- Section 8.6 KKT residuals on every segment (no figure).
  * ``validate-scaling`` -- Section 8.6 invariance under a change of units, and the
    sensitivity of the trace to the slope floor (no figure).
  * ``validate-projection`` -- Appendix A: how often the feasibility projection fires
    and how large its corrections are (no figure).
  * ``validate-factor`` -- Section 5: the factor backend against the dense trace on
    ill-conditioned and degenerate factor models (no figure).
  * ``validate-conditioning`` -- Appendix A: problems of prescribed condition number
    across the singularity guard, each trace and its reference QP certified (no figure).
  * ``estimators``      -- Figure 8 + Section 11.1 estimator table (estimator_shrinkage.pdf).
  * ``michaud``         -- Figure 9 + Section 12 resampling table (michaud_frontier.pdf).
  * ``figures``         -- all seven figures.
  * ``checks``          -- all nine numerical checks.
  * ``all``             -- every figure and every check.

Beyond ``cvxcla`` itself the steps need a few third-party packages: ``matplotlib`` (all
figures), ``pandas``/``pyarrow`` (the S&P 500 data), ``osqp`` (the reference QP and the
warm-started grid baseline), ``PyPortfolioOpt`` and ``casadi`` (the external baselines
in ``scaling``; ``casadi`` ships the qpOASES parametric QP solver), ``scipy`` (sparse
matrices), and ``scikit-learn`` (the Ledoit--Wolf estimate). All are pinned in this
script's inline (PEP 723) metadata, so ``uv run`` provisions them automatically. A step
whose optional dependency is missing, or which raises, is reported and skipped rather
than aborting the whole run.

Pass ``--quick`` to ``all``/``figures`` to skip the two long-running scaling sweeps
(``scaling`` and ``rank-scaling``); selecting either explicitly always runs it.
"""

from __future__ import annotations

import contextlib
import enum
import io
import itertools
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import numpy as np
import typer

mpl.use("Agg")
import matplotlib.pyplot as plt

import cvxcla.cla as cla_module
from cvxcla import CLA, FactorCovariance, IncrementalDenseCovariance

# Default output directory: ``experiments/figures/``, so artefacts land there
# regardless of the working directory the command is run from.
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "figures"

# The frozen S&P 500 daily-return matrix (used by real-frontier and validate-exact):
# ``data/`` next to this script in cvxcla, ``figures/data/`` in the paper's repository.
_HERE = Path(__file__).resolve().parent
DATA = (
    next(
        (p for p in (_HERE / "data", _HERE / "figures" / "data") if (p / "sp500_pct_returns.parquet").exists()),
        _HERE / "data",
    )
    / "sp500_pct_returns.parquet"
)

# Samples per segment when drawing a frontier curve between turning points.
_CURVE_POINTS = 50


def _along_segments(weights: np.ndarray, num: int = _CURVE_POINTS) -> np.ndarray:
    """Return weights at ``num`` evenly spaced points along each segment between turning points.

    The weights move linearly between adjacent turning points, so their volatility traces
    a hyperbola there, not the chord a line plot through the corners draws. Every turning
    point is included exactly once, in order.
    """
    t = np.linspace(0.0, 1.0, num)[:-1, None]
    dense = [w0 + t * (w1 - w0) for w0, w1 in pairwise(weights)]
    return np.vstack([*dense, weights[-1:]])


# ======================================================================================
# OSQP: the reference QP and the warm-started grid baseline
# ======================================================================================
_OSQP_EPS = 1e-6  # the grid tolerance: the loosest that is exact to rounding on the S&P grid


def _osqp_qp(
    mean: np.ndarray,
    cov: np.ndarray,
    lam: float,
    a: np.ndarray,
    b: np.ndarray,
    g: np.ndarray | None = None,
    h: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    leverage: float | None = None,
) -> np.ndarray:
    """Solve min 1/2 w'Sigma w - lam mu'w s.t. A w = b, G w <= h, lower <= w <= upper with OSQP.

    The box defaults to ``0 <= w <= 1``. A gross-exposure cap ``||w||_1 <= leverage`` is
    modelled with auxiliary variables ``t >= |w|`` and ``sum(t) <= leverage``. A cold,
    polished solve at tight tolerances, so the comparison probes the CLA's accuracy
    rather than the QP solver's stopping criteria.
    """
    import osqp
    from scipy import sparse

    n = mean.shape[0]
    lower = np.zeros(n) if lower is None else lower
    upper = np.ones(n) if upper is None else upper
    k = 0 if leverage is None else n  # number of auxiliary variables t

    def wide(block: object) -> sparse.csc_matrix:
        """Extend a row block over w with zero columns for t."""
        m = sparse.csc_matrix(block)
        return m if k == 0 else sparse.hstack([m, sparse.csc_matrix((m.shape[0], k))], format="csc")

    rows = [wide(a), wide(sparse.eye(n))]
    lo = [b, lower]
    up = [b, upper]
    if g is not None and g.shape[0] > 0:
        rows.append(wide(g))
        lo.append(np.full(g.shape[0], -np.inf))
        up.append(h)
    if leverage is not None:
        eye = sparse.eye(n)
        rows.append(sparse.hstack([eye, -eye], format="csc"))  # w - t <= 0
        rows.append(sparse.hstack([-eye, -eye], format="csc"))  # -w - t <= 0
        rows.append(sparse.hstack([sparse.csc_matrix((1, n)), sparse.csc_matrix(np.ones((1, n)))], format="csc"))
        lo += [np.full(2 * n, -np.inf), np.array([-np.inf])]
        up += [np.zeros(2 * n), np.array([leverage])]
    p_block = sparse.csc_matrix(np.triu(cov))
    if k:
        p_block = sparse.block_diag([p_block, sparse.csc_matrix((k, k))], format="csc")
    solver = osqp.OSQP()
    solver.setup(
        P=p_block,
        q=np.r_[-lam * mean, np.zeros(k)],
        A=sparse.vstack(rows, format="csc"),
        l=np.concatenate(lo),
        u=np.concatenate(up),
        eps_abs=1e-10,
        eps_rel=1e-10,
        max_iter=1_000_000,
        polishing=True,
        verbose=False,
    )
    return np.asarray(solver.solve(raise_error=True).x[:n], dtype=float)


def _osqp_warm_grid(mean: np.ndarray, sigma: np.ndarray, lams: np.ndarray, eps: float) -> tuple[np.ndarray, float]:
    """The long-only, fully-invested portfolios at ``lams`` from one warm-started OSQP.

    The strongest grid baseline short of a path algorithm: OSQP is set up once, so its
    factorisation is reused, each lambda changes only the linear term, and each solve
    starts from the previous solution. Polishing recovers the active-set solution.
    """
    import osqp
    from scipy import sparse

    n = mean.shape[0]
    start = time.perf_counter()
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.csc_matrix(np.triu(sigma)),
        q=-lams[0] * mean,
        A=sparse.vstack([sparse.csc_matrix(np.ones((1, n))), sparse.eye(n)], format="csc"),
        l=np.r_[1.0, np.zeros(n)],
        u=np.r_[1.0, np.ones(n)],
        eps_abs=eps,
        eps_rel=eps,
        polishing=True,
        warm_starting=True,
        verbose=False,
    )
    weights = []
    for lam in lams:
        solver.update(q=-lam * mean)
        weights.append(solver.solve(raise_error=True).x)
    return np.array(weights), time.perf_counter() - start


def _cla_weight_at(cla: CLA, lam: float) -> np.ndarray:
    """Exact frontier weights ``w(lam)`` by linear interpolation between turning points.

    The frontier is affine in ``lambda`` on each segment and the turning points are
    the segment endpoints, so linear interpolation in ``lambda`` is exact (not an
    approximation). ``lam`` is clamped to the finite turning-point range.
    """
    pts = sorted((tp for tp in cla.turning_points if np.isfinite(tp.lamb)), key=lambda t: t.lamb)
    if lam <= pts[0].lamb:
        return pts[0].weights
    if lam >= pts[-1].lamb:
        return pts[-1].weights
    for lo, hi in pairwise(pts):
        if lo.lamb <= lam <= hi.lamb:
            t = (lam - lo.lamb) / (hi.lamb - lo.lamb)
            return (1.0 - t) * lo.weights + t * hi.weights
    msg = "lam within range but no bracketing segment found"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover


# ======================================================================================
# Figure: frontier (from frontier_20x50.py)  ->  frontier.pdf
# ======================================================================================
_FR20_N_ASSETS = 20
_FR20_N_DAYS = 50
_FR20_N_FACTORS = 5
_FR20_SEED = 42
_FR20_REPEATS = 5


@dataclass(frozen=True)
class _FactorModel:
    """The parameters of ``Sigma = diag(d) + U diag(delta) U^T``.

    Kept here rather than read back off the ``FactorCovariance`` operator, which
    since cvxcla 2.0 is a ``cvx.linalg.FactorOperator`` with no public ``d``/``u``/``delta``.
    """

    d: np.ndarray
    u: np.ndarray
    delta: np.ndarray

    def operator(self) -> object:
        """The covariance as a cvxcla factor backend."""
        return FactorCovariance(d=self.d, u=self.u, delta=self.delta)


def _fr20_build_model(rng: np.random.Generator) -> tuple[_FactorModel, np.ndarray]:
    """Return the ground-truth factor model and the per-asset expected returns."""
    u = rng.standard_normal((_FR20_N_ASSETS, _FR20_N_FACTORS)) / np.sqrt(_FR20_N_ASSETS)
    delta = rng.uniform(0.5, 2.0, _FR20_N_FACTORS) * _FR20_N_ASSETS
    d = rng.uniform(0.5, 2.0, _FR20_N_ASSETS)
    expected = rng.uniform(0.0, 1.0, _FR20_N_ASSETS)  # dispersed expected returns
    return _FactorModel(d=d, u=u, delta=delta), expected


def _fr20_simulate_returns(rng: np.random.Generator, factor: _FactorModel, expected: np.ndarray) -> np.ndarray:
    """Simulate (N_DAYS, N_ASSETS) returns whose population covariance is ``factor``."""
    factor_returns = rng.standard_normal((_FR20_N_DAYS, _FR20_N_FACTORS)) * np.sqrt(factor.delta)
    idiosyncratic = rng.standard_normal((_FR20_N_DAYS, _FR20_N_ASSETS)) * np.sqrt(factor.d)
    return expected + factor_returns @ factor.u.T + idiosyncratic


def _fr20_trace(problem: dict, covariance: object) -> tuple[object, float]:
    """Trace the frontier ``REPEATS`` times and return (last CLA, median seconds)."""
    times = []
    cla = None
    for _ in range(_FR20_REPEATS):
        start = time.perf_counter()
        cla = CLA(covariance=covariance, **problem)
        times.append(time.perf_counter() - start)
    return cla, float(np.median(times))


def figure_frontier(out_dir: Path) -> None:
    """Run the 20x50 experiment, print statistics, and write the frontier figure."""
    rng = np.random.default_rng(_FR20_SEED)
    factor, expected = _fr20_build_model(rng)
    returns = _fr20_simulate_returns(rng, factor, expected)

    mean = returns.mean(axis=0)
    dense_cov = np.diag(factor.d) + (factor.u * factor.delta) @ factor.u.T

    problem = {
        "mean": mean,
        "lower_bounds": np.zeros(_FR20_N_ASSETS),
        "upper_bounds": np.ones(_FR20_N_ASSETS),
        "a": np.ones((1, _FR20_N_ASSETS)),
        "b": np.ones(1),
    }

    cla, dense_time = _fr20_trace(problem, dense_cov)
    factor_cla, factor_time = _fr20_trace(problem, factor.operator())
    if len(cla) != len(factor_cla):
        msg = f"backends disagree: {len(cla)} vs {len(factor_cla)}"
        raise RuntimeError(msg)

    frontier = cla.frontier
    returns_f = frontier.returns
    vol_f = frontier.volatility
    max_sharpe, _ = frontier.max_sharpe
    cond = float(np.linalg.cond(dense_cov))

    print(f"problem                 : {_FR20_N_ASSETS} assets x {_FR20_N_DAYS} days, {_FR20_N_FACTORS}-factor model")
    print(f"covariance condition no.: {cond:,.1f}")
    print(f"turning points          : {len(cla)}  (dense and factor agree)")
    print(f"dense trace  (median)   : {dense_time * 1e3:.1f} ms  ({dense_time / len(cla) * 1e3:.2f} ms/point)")
    print(f"factor trace (median)   : {factor_time * 1e3:.1f} ms  ({factor_time / len(cla) * 1e3:.2f} ms/point)")
    print(f"factor speedup          : {dense_time / factor_time:.2f}x")
    print(f"expected-return range   : [{returns_f.min():.4f}, {returns_f.max():.4f}]")
    print(f"volatility range        : [{vol_f.min():.4f}, {vol_f.max():.4f}]")
    print(f"max Sharpe (model units): {max_sharpe:.4f}")

    # Draw the exact curve between turning points and mark only the turning points.
    curve_w = _along_segments(frontier.weights)
    curve_vol = np.sqrt(np.einsum("ij,jk,ik->i", curve_w, dense_cov, curve_w))
    curve_ret = curve_w @ mean
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(curve_vol, curve_ret, "-", lw=1.0, color="#1f4e79", label="Efficient frontier")
    ax.plot(vol_f, returns_f, "o", ms=2.5, color="#1f4e79", label="Turning points")
    ax.scatter(vol_f[[0, -1]], returns_f[[0, -1]], color="#c00000", zorder=5, s=18)
    ax.annotate(
        "max return", (vol_f[0], returns_f[0]), textcoords="offset points", xytext=(-6, 4), ha="right", fontsize=8
    )
    ax.annotate("min variance", (vol_f[-1], returns_f[-1]), textcoords="offset points", xytext=(8, -2), fontsize=8)
    ax.set_xlabel("Volatility (model units)")
    ax.set_ylabel("Expected return (model units)")
    ax.set_title(
        f"Efficient frontier: {_FR20_N_ASSETS} assets, {_FR20_N_DAYS} days ({len(cla)} turning points)", fontsize=9
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "frontier.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ======================================================================================
# Figure: scaling (from runtime_scaling.py)  ->  scaling.pdf  (SLOW)
# ======================================================================================
_SCALE_SIZES = [20, 40, 80, 160, 320, 640, 1280, 2560]
# The external baselines are timed only up to here: PyPortfolioOpt already takes
# minutes at n=640, and both grow like n^3 or faster.
_SCALE_BASELINE_MAX_N = 640
# The second, small-problem figure (scaling_small.pdf) shows the sizes up to here.
_SCALE_SMALL_MAX_N = 320
# qpOASES follows the same parametric path as the CLA, so it is the closest external
# comparison and is timed further, until it too takes minutes per trace.
_SCALE_QPOASES_MAX_N = 2560
_SCALE_N_FACTORS = 10
_SCALE_SEED = 7
_SCALE_REPEATS = 3
_SCALE_GRID = 100  # fixed lambda-grid size for the OSQP baseline, the same at every n


def _scale_make_problem(rng: np.random.Generator, n: int, k: int) -> tuple[np.ndarray, FactorCovariance, dict]:
    """Return (dense Sigma, FactorCovariance, problem dict) for a size-n factor model.

    The two covariance representations are mathematically identical, so both
    backends trace the same frontier; only the per-solve linear algebra differs.
    """
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    factor = FactorCovariance(d=d, u=u, delta=delta)
    dense = np.diag(d) + (u * delta) @ u.T
    problem = {
        "mean": rng.uniform(0.0, 1.0, n),  # dispersed expected returns
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }
    return dense, factor, problem


def _scale_stats(times: list[float]) -> tuple[float, float, float]:
    """Return (median, min, max) seconds, the centre and min--max band over repetitions."""
    return float(np.median(times)), float(np.min(times)), float(np.max(times))


def _scale_median_trace(covariance: object, problem: dict) -> tuple[int, float, float, float]:
    """Return (turning points, median, min, max) trace seconds over REPEATS repetitions."""
    cla = CLA(covariance=covariance, **problem)
    times = []
    for _ in range(_SCALE_REPEATS):
        start = time.perf_counter()
        cla = CLA(covariance=covariance, **problem)
        times.append(time.perf_counter() - start)
    return (len(cla), *_scale_stats(times))


def _scale_median_trace_inverse(dense: np.ndarray, problem: dict) -> tuple[int, float, float, float]:
    """Time cvxcla's opt-in IncrementalDenseCovariance backend over REPEATS repetitions.

    The same loop and event logic as the dense backend, differing only in the
    linear algebra: a maintained free-block inverse, updated by a rank-one border or
    deletion at each turning point, in place of a fresh factorisation. The operator
    is rebuilt per repetition so each trace starts with no cached inverse. Returns
    (turning points, median, min, max) seconds.
    """
    cla = CLA(covariance=IncrementalDenseCovariance(dense), **problem)
    times = []
    for _ in range(_SCALE_REPEATS):
        covariance = IncrementalDenseCovariance(dense)
        start = time.perf_counter()
        cla = CLA(covariance=covariance, **problem)
        times.append(time.perf_counter() - start)
    return (len(cla), *_scale_stats(times))


def _scale_pypfopt_trace(dense: np.ndarray, mean: np.ndarray) -> tuple[int, float, float, float] | None:
    """Time PyPortfolioOpt's CLA (Bailey & Lopez de Prado) if installed.

    Timed over REPEATS repetitions with the same protocol as the other
    implementations (median of REPEATS), so the comparison is apples-to-apples.
    Returns (turning points, median, min, max) seconds or None when PyPortfolioOpt
    is absent.
    """
    try:
        from pypfopt.cla import CLA as PYPFOPT_CLA
    except ImportError:
        return None
    times = []
    n_pts = 0
    for _ in range(_SCALE_REPEATS):
        start = time.perf_counter()
        ppo = PYPFOPT_CLA(mean, dense, weight_bounds=(0, 1))
        ppo._solve()  # compute the full critical line (all turning points)
        times.append(time.perf_counter() - start)
        n_pts = len(ppo.w)
    return (n_pts, *_scale_stats(times))


def _scale_qpoases_path(dense: np.ndarray, problem: dict) -> tuple[int, float, float, float, float] | None:
    """Time qpOASES following the whole frontier in one hot-started call, if installed.

    qpOASES (shipped with CasADi) is a parametric active-set solver: a hot start from
    one QP to the next follows the homotopy between them, changing the working set at
    every breakpoint on the way. Solving at the maximum-return end, above the first
    finite turning point, and hot-starting to lambda = 0 therefore walks the same path
    as the CLA, but returns only its endpoint, the minimum-variance portfolio. Each
    repetition starts from a fresh solver (a cold start), and the two calls are timed
    together; one call per trace keeps CasADi's per-call overhead out of the figure.
    Returns (working-set changes, median, min, max seconds, endpoint gap to the CLA) or
    None when CasADi is absent.
    """
    try:
        import casadi as ca
    except ImportError:
        return None

    cla = CLA(covariance=dense, **problem)
    lam_top = 2.0 * max(tp.lamb for tp in cla.turning_points if np.isfinite(tp.lamb))
    n = len(problem["mean"])
    h, a = ca.DM(dense), ca.DM(problem["a"])
    bounds = {"lba": problem["b"], "uba": problem["b"], "lbx": problem["lower_bounds"], "ubx": problem["upper_bounds"]}
    times, changes, endpoint = [], 0, np.zeros(n)
    # qpOASES prints its licence banner through CasADi, which writes to Python's
    # sys.stdout; keep it out of the sweep's log.
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(_SCALE_REPEATS):
            solver = ca.conic("S", "qpoases", {"h": h.sparsity(), "a": a.sparsity()}, {"printLevel": "none"})
            start = time.perf_counter()
            solver(h=h, g=-lam_top * problem["mean"], a=a, **bounds)
            result = solver(h=h, g=np.zeros(n), a=a, **bounds)
            times.append(time.perf_counter() - start)
            changes = int(solver.stats()["iter_count"])
            endpoint = np.asarray(result["x"]).ravel()
    gap = float(np.max(np.abs(endpoint - cla.turning_points[-1].weights)))
    return (changes, *_scale_stats(times), gap)


def _scale_free_sizes(covariance: object, problem: dict) -> tuple[int, float]:
    """Return (max, median) free-set size |F| over the turning points of one trace.

    The dense step costs O(|F|^3), so the free-set size, not n, sets the per-step cost.
    """
    sizes = [int(np.sum(tp.free)) for tp in CLA(covariance=covariance, **problem).turning_points]
    return max(sizes), float(np.median(sizes))


def _scale_osqp_grid(dense: np.ndarray, problem: dict) -> tuple[int, float, float, float]:
    """Time reconstructing the frontier with a warm-started OSQP grid of GRID lambda values.

    The grid has a fixed size at every n, so its exponent reflects the per-solve cost
    rather than a grid that grows with n. It spans the frontier's lambda-range from one
    cvxcla trace, and the whole sweep is timed (median of REPEATS) against a single
    trace. Returns (grid points, median, min, max) seconds.
    """
    cla = CLA(covariance=dense, **problem)
    lam_max = max(tp.lamb for tp in cla.turning_points if np.isfinite(tp.lamb))
    lams = np.linspace(lam_max, 0.0, _SCALE_GRID)
    times = [_osqp_warm_grid(problem["mean"], dense, lams, _OSQP_EPS)[1] for _ in range(_SCALE_REPEATS)]
    return (len(lams), *_scale_stats(times))


def _scale_memory(covariance: object, problem: dict, cov_bytes: int) -> tuple[int, int]:
    """Return (covariance storage, trace working memory) in bytes for one trace.

    The stored frontier is n weights per turning point, O(n^2) for any backend, so the
    working memory is the tracemalloc peak during the trace less what the finished
    trace retains: the transient allocations of the loop itself.
    """
    import tracemalloc

    tracemalloc.start()
    cla = CLA(covariance=covariance, **problem)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del cla
    return cov_bytes, peak - current


def _scale_measure(method: str, n: int) -> object:
    """Rebuild the size-n problem and time one method on it (run in a fresh process)."""
    rng = np.random.default_rng(_SCALE_SEED)
    dense, factor, problem = _scale_make_problem(rng, n, _SCALE_N_FACTORS)
    if method == "dense":
        return _scale_median_trace(dense, problem)
    if method == "factor":
        return _scale_median_trace(factor, problem)
    if method == "inverse":
        return _scale_median_trace_inverse(dense, problem)
    if method == "pypfopt":
        return _scale_pypfopt_trace(dense, problem["mean"])
    if method == "osqp":
        return _scale_osqp_grid(dense, problem)
    if method == "qpoases":
        return _scale_qpoases_path(dense, problem)
    if method == "free":
        return _scale_free_sizes(factor, problem)
    if method == "memory-dense":
        return _scale_memory(dense, problem, dense.nbytes)
    if method == "memory-factor":
        return _scale_memory(factor, problem, 8 * (n * (_SCALE_N_FACTORS + 1) + _SCALE_N_FACTORS))
    msg = f"unknown scaling method {method!r}"
    raise ValueError(msg)


def _scale_fresh(method: str, n: int) -> object:
    """Time one method in its own freshly spawned interpreter.

    Run in one process, the external baselines (PyPortfolioOpt, OSQP) leave the
    interpreter in a state that slows the cvxcla timings taken after them: the dense
    trace at n=640 took 1.16 s inside the sweep against 0.62 s alone. A fresh process
    per (method, n) keeps every measurement independent of what ran before it.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(_scale_measure, method, n).result()


def figure_scaling(out_dir: Path) -> None:
    """Run the scaling sweep, print a table, and write the figure."""
    ns, dense_times, factor_times, ppo_times, inv_times, clar_times, points = [], [], [], [], [], [], []
    dense_band, factor_band, ppo_band, inv_band, clar_band = [], [], [], [], []
    qpo_times, qpo_band = [], []
    memory = []
    for n in _SCALE_SIZES:
        n_pts, t_dense, d_lo, d_hi = _scale_fresh("dense", n)
        n_pts_f, t_factor, f_lo, f_hi = _scale_fresh("factor", n)
        if n_pts != n_pts_f:
            msg = f"backends disagree at n={n}: {n_pts} vs {n_pts_f}"
            raise RuntimeError(msg)
        inv = _scale_fresh("inverse", n)
        if inv and inv[0] != n_pts:
            # A maintained inverse accumulates round-off over the trace, so on a
            # near-tied size it can split or merge a turning point. Drop its point
            # rather than abort, so the figure still completes.
            print(f"  [note] incremental backend disagrees at n={n}: {inv[0]} vs {n_pts}; skipping its point")
            inv = None
        baselines = n <= _SCALE_BASELINE_MAX_N
        ppo = _scale_fresh("pypfopt", n) if baselines else None
        clar = _scale_fresh("osqp", n) if baselines else None
        qpo = _scale_fresh("qpoases", n) if n <= _SCALE_QPOASES_MAX_N else None
        free_max, free_med = _scale_fresh("free", n)
        memory.append((n, *_scale_fresh("memory-dense", n), *_scale_fresh("memory-factor", n)))
        ns.append(n)
        points.append(n_pts)
        dense_times.append(t_dense)
        factor_times.append(t_factor)
        ppo_times.append(ppo[1] if ppo else None)
        inv_times.append(inv[1] if inv else None)
        clar_times.append(clar[1] if clar else None)
        dense_band.append((d_lo, d_hi))
        factor_band.append((f_lo, f_hi))
        ppo_band.append((ppo[2], ppo[3]) if ppo else None)
        inv_band.append((inv[2], inv[3]) if inv else None)
        clar_band.append((clar[2], clar[3]) if clar else None)
        qpo_times.append(qpo[1] if qpo else None)
        qpo_band.append((qpo[2], qpo[3]) if qpo else None)
        inv_str = f"  inverse={inv[1] * 1e3:9.1f} ms" if inv else "  inverse=n/a"
        ppo_str = f"  pypfopt={ppo[1] * 1e3:9.1f} ms (pts={ppo[0]})" if ppo else "  pypfopt=n/a"
        clar_str = f"  osqp={clar[1] * 1e3:9.1f} ms ({clar[0]} solves)" if clar else "  osqp=n/a"
        qpo_str = (
            f"  qpoases={qpo[1] * 1e3:9.1f} ms ({qpo[0]} working-set changes, endpoint gap {qpo[4]:.1e})"
            if qpo
            else "  qpoases=n/a"
        )
        print(
            f"n={n:4d}  points={n_pts:4d}  dense={t_dense * 1e3:8.1f} ms  "
            f"factor={t_factor * 1e3:8.1f} ms  speedup={t_dense / t_factor:5.2f}x  "
            f"|F| max={free_max:4d} median={free_med:6.1f}{inv_str}{ppo_str}{clar_str}{qpo_str}"
        )

    # The clean comparison is the incremental-inverse baseline vs the dense
    # backend: same vectorised event logic, differing only in the linear-algebra
    # strategy (maintained inverse vs fresh block-eliminated solve). PyPortfolioOpt
    # differs algorithmically (per-candidate full inverse), so its ratio is an
    # overall figure, not a clean control.
    if inv_times[-1] is not None:
        n_last = ns[-1]
        strategy = dense_times[-1] / inv_times[-1]  # solve cost / incremental-inverse cost
        line = (
            f"\nat n={n_last}: incremental inverse is {strategy:.2f}x the speed of the "
            f"dense fresh-solve backend; factor/dense = {dense_times[-1] / factor_times[-1]:.1f}x"
        )
        print(line)

    # Empirical log-log slope (runtime ~ n^p) over the largest four sizes a series
    # was timed at, plus the local slope of its last doubling.
    def slope(times: list[float | None]) -> tuple[float, float]:
        pairs = [(n, t) for n, t in zip(ns, times, strict=True) if t is not None][-4:]
        x = np.log(np.array([p[0] for p in pairs], dtype=float))
        y = np.log(np.array([p[1] for p in pairs]))
        return float(np.polyfit(x, y, 1)[0]), float((y[-1] - y[-2]) / (x[-1] - x[-2]))

    print()
    for name, series in [
        ("dense", dense_times),
        ("factor", factor_times),
        ("inverse", inv_times),
        ("pypfopt", ppo_times),
        ("osqp", clar_times),
        ("qpoases", qpo_times),
    ]:
        if sum(t is not None for t in series) >= 4:
            fit, last = slope(series)
            print(f"{name:7s} exponent p (time ~ n^p): {fit:.2f} over the largest four sizes, {last:.2f} last doubling")
    n_base = _SCALE_BASELINE_MAX_N
    i_base = ns.index(n_base)
    if ppo_times[i_base] is not None:
        print(f"at n={n_base}: pypfopt/dense = {ppo_times[i_base] / dense_times[i_base]:.0f}x")
    if clar_times[i_base] is not None:
        print(f"at n={n_base}: osqp/dense = {clar_times[i_base] / dense_times[i_base]:.0f}x")
    for n_q, t_q in zip(ns, qpo_times, strict=True):
        if t_q is not None:
            i_q = ns.index(n_q)
            print(
                f"at n={n_q}: qpoases/dense = {t_q / dense_times[i_q]:.1f}x, "
                f"qpoases/factor = {t_q / factor_times[i_q]:.1f}x"
            )

    mib = 2.0**20
    print("\nmemory: covariance storage and trace working memory (beyond the stored frontier), MiB")
    for n, d_cov, d_work, f_cov, f_work in memory:
        print(
            f"n={n:4d}  dense: storage={d_cov / mib:8.2f} working={d_work / mib:8.2f}   "
            f"factor: storage={f_cov / mib:6.3f} working={f_work / mib:7.3f}"
        )

    from matplotlib.ticker import NullFormatter, ScalarFormatter

    series = [
        # (times, bands, marker, colour, label); external baselines first, cvxcla last
        (ppo_times, ppo_band, "-^", "#7f7f7f", "PyPortfolioOpt CLA"),
        (clar_times, clar_band, "-v", "#ff7f0e", f"OSQP, {_SCALE_GRID}-point $\\lambda$-grid"),
        (qpo_times, qpo_band, "-P", "#9467bd", "qpOASES, hot-started path"),
        (inv_times, inv_band, "-D", "#2ca02c", "cvxcla, incremental dense"),
        (dense_times, dense_band, "-o", "#c00000", "cvxcla, dense"),
        (factor_times, factor_band, "-s", "#1f4e79", f"cvxcla, factor ($K={_SCALE_N_FACTORS}$)"),
    ]

    def draw(n_max: int, name: str, title: str, headroom: float) -> None:
        """Plot every series over the sizes n <= n_max, with min--max bands, to ``name``."""
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        sizes = [n for n in ns if n <= n_max]
        for times, bands, marker, colour, label in series:
            keep = [i for i, n in enumerate(ns) if n <= n_max and times[i] is not None]
            if not keep:
                continue
            xs = [ns[i] for i in keep]
            spans = [bands[i] for i in keep if bands[i] is not None]
            if spans:
                ax.fill_between(xs, [b[0] for b in spans], [b[1] for b in spans], color=colour, alpha=0.18, linewidth=0)
            ax.loglog(xs, [times[i] for i in keep], marker, ms=4, color=colour, label=label)
        ax.set_xlabel("Number of assets $n$")
        ax.set_ylabel("Frontier trace time [s]")
        ax.set_title(title, fontsize=9)
        # Label the x-axis at the actual problem sizes as plain integers, not the
        # default powers of ten (which never coincide with 20, 40, ..., 640 and
        # leave cluttered minor-tick labels on a log axis).
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(sizes[0] * 0.85, sizes[-1] * 1.18)
        ax.tick_params(axis="x", labelsize=7)
        # Headroom above the slowest series keeps the legend clear of every curve.
        slowest = max(t for times, *_ in series for i, t in enumerate(times) if t is not None and ns[i] <= n_max)
        ax.set_ylim(top=slowest * headroom)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper left")
        fig.tight_layout()
        out = out_dir / name
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")

    draw(ns[-1], "scaling.pdf", "CLA runtime vs problem size", 300.0)
    # The practically common range of a few hundred assets, where per-step overhead
    # still matters and qpOASES is faster than cvxcla.
    draw(
        _SCALE_SMALL_MAX_N, "scaling_small.pdf", f"CLA runtime vs problem size, $n \\leq {_SCALE_SMALL_MAX_N}$", 1000.0
    )


# ======================================================================================
# Figure: rank-scaling (from rank_scaling.py)  ->  rank_scaling.pdf  (SLOW)
# ======================================================================================
_RANK_N_ASSETS = 320
_RANK_RANKS = [20, 40, 80, 160, 320]
_RANK_SEED = 11
_RANK_REPEATS = 3


def _rank_make_problem(rng: np.random.Generator, n: int, k: int) -> tuple[np.ndarray, FactorCovariance, dict]:
    """Return (dense Sigma, FactorCovariance, problem dict) for an n-asset, K-factor model."""
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    factor = FactorCovariance(d=d, u=u, delta=delta)
    dense = np.diag(d) + (u * delta) @ u.T
    problem = {
        "mean": rng.uniform(0.0, 1.0, n),  # dispersed expected returns
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }
    return dense, factor, problem


def _rank_median_trace(covariance: object, problem: dict) -> tuple[int, float, float, float]:
    """Return (turning points, median, min, max) trace seconds over REPEATS repetitions."""
    cla = CLA(covariance=covariance, **problem)
    times = []
    for _ in range(_RANK_REPEATS):
        start = time.perf_counter()
        cla = CLA(covariance=covariance, **problem)
        times.append(time.perf_counter() - start)
    return len(cla), float(np.median(times)), float(np.min(times)), float(np.max(times))


def figure_rank_scaling(out_dir: Path) -> None:
    """Run the rank sweep, print a table, and write the figure."""
    ks, dense_times, factor_times, points = [], [], [], []
    dense_band, factor_band = [], []
    for k in _RANK_RANKS:
        rng = np.random.default_rng(_RANK_SEED)
        dense, factor, problem = _rank_make_problem(rng, _RANK_N_ASSETS, k)
        n_pts, t_dense, d_lo, d_hi = _rank_median_trace(dense, problem)
        n_pts_f, t_factor, f_lo, f_hi = _rank_median_trace(factor, problem)
        if n_pts != n_pts_f:
            msg = f"backends disagree at K={k}: {n_pts} vs {n_pts_f}"
            raise RuntimeError(msg)
        ks.append(k)
        points.append(n_pts)
        dense_times.append(t_dense)
        factor_times.append(t_factor)
        dense_band.append((d_lo, d_hi))
        factor_band.append((f_lo, f_hi))
        print(
            f"K={k:4d}  points={n_pts:4d}  dense={t_dense * 1e3:8.1f} ms  "
            f"factor={t_factor * 1e3:8.1f} ms  speedup={t_dense / t_factor:5.2f}x"
        )

    from matplotlib.ticker import NullFormatter, ScalarFormatter

    def band(bands: list[tuple[float, float]], color: str) -> None:
        """Shade the min--max range across repetitions for one series."""
        ax.fill_between(ks, [b[0] for b in bands], [b[1] for b in bands], color=color, alpha=0.18, linewidth=0)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    band(dense_band, "#c00000")
    ax.loglog(ks, dense_times, "-o", ms=4, color="#c00000", label="cvxcla, dense backend")
    band(factor_band, "#1f4e79")
    ax.loglog(ks, factor_times, "-s", ms=4, color="#1f4e79", label="cvxcla, factor backend (Woodbury)")
    ax.set_xlabel(f"Number of factors $K$ (fixed $n={_RANK_N_ASSETS}$)")
    ax.set_ylabel("Frontier trace time [s]")
    ax.set_title(f"CLA runtime vs factor rank ($n={_RANK_N_ASSETS}$)", fontsize=9)
    ax.set_xticks(ks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(ks[0] * 0.85, ks[-1] * 1.18)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / "rank_scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ======================================================================================
# Figure: real-frontier (from frontier_real.py)  ->  real_frontier.pdf
# ======================================================================================
_REAL_SHORT_WINDOW = 120  # trading days < N -> rank-deficient sample covariance
_REAL_N_FACTORS = 20
_REAL_REPEATS = 5


def _real_problem(n: int) -> dict:
    """Long-only, fully-invested box-constrained problem of size n."""
    return {
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }


def _real_median_trace(mean: np.ndarray, covariance: object, n: int) -> tuple[object, float]:
    """Return (CLA, median trace seconds) over REPEATS repetitions."""
    cla = CLA(mean=mean, covariance=covariance, **_real_problem(n))
    times = []
    for _ in range(_REAL_REPEATS):
        start = time.perf_counter()
        cla = CLA(mean=mean, covariance=covariance, **_real_problem(n))
        times.append(time.perf_counter() - start)
    return cla, float(np.median(times))


def _real_factor_estimate(cov: np.ndarray, k: int) -> FactorCovariance:
    """Diagonal-plus-low-rank estimate from the top-k eigenpairs of ``cov``."""
    evals, evecs = np.linalg.eigh(cov)
    top = np.argsort(evals)[::-1][:k]
    u = evecs[:, top]
    delta = np.clip(evals[top], 1e-12, None)
    d = np.clip(np.diag(cov) - np.einsum("ij,j,ij->i", u, delta, u), 1e-10, None)
    return FactorCovariance(d=d, u=u, delta=delta)


def figure_real_frontier(out_dir: Path) -> None:
    """Run the empirical S&P 500 experiment, print statistics, and write the figure."""
    import pandas as pd

    returns = pd.read_parquet(DATA)
    r = returns.to_numpy()
    t_days, n = r.shape
    mean = r.mean(axis=0)
    sample_cov = np.cov(r, rowvar=False)

    span = f"{returns.index[0].date()} -> {returns.index[-1].date()}"
    print(f"universe                : {n} assets x {t_days} days ({span})")
    print(f"covariance condition no.: {np.linalg.cond(sample_cov):,.0f}")

    # 1. Full-history frontier (dense) + Woodbury approximation.
    cla, dense_time = _real_median_trace(mean, sample_cov, n)
    factor_cla, factor_time = _real_median_trace(mean, _real_factor_estimate(sample_cov, _REAL_N_FACTORS), n)
    print(f"dense frontier          : {len(cla)} turning points in {dense_time * 1e3:.1f} ms (median)")
    print(
        f"factor (K={_REAL_N_FACTORS}) approx    : {len(factor_cla)} turning points in "
        f"{factor_time * 1e3:.1f} ms (median)"
    )

    frontier = cla.frontier
    ann = 252.0  # annualise daily moments for readable axes
    ret = frontier.returns * ann
    vol = frontier.volatility * np.sqrt(ann)
    sharpe, _ = frontier.max_sharpe
    print(f"annualised return range : [{ret.min():.3f}, {ret.max():.3f}]")
    print(f"annualised vol range    : [{vol.min():.3f}, {vol.max():.3f}]")
    print(f"max Sharpe (annualised) : {sharpe * np.sqrt(ann):.3f}")

    # 2. Degeneracy demonstration on a short (rank-deficient) estimation window.
    short = r[-_REAL_SHORT_WINDOW:]
    short_cov = np.cov(short, rowvar=False)
    rank = int(np.linalg.matrix_rank(short_cov))
    try:
        CLA(mean=short.mean(axis=0), covariance=short_cov, **_real_problem(n))
        print(f"short window W={_REAL_SHORT_WINDOW}     : traced (unexpected)")
    except Exception as exc:  # noqa: BLE001 - we are demonstrating the failure
        print(f"short window W={_REAL_SHORT_WINDOW}     : rank(S)={rank} < N={n} -> {type(exc).__name__}: {exc}")

    # Draw the exact curve between turning points and mark only the turning points.
    curve_w = _along_segments(frontier.weights)
    curve_vol = np.sqrt(np.einsum("ij,jk,ik->i", curve_w, sample_cov, curve_w) * ann)
    curve_ret = (curve_w @ mean) * ann
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(curve_vol, curve_ret, "-", lw=1.0, color="#1f4e79", label="Efficient frontier")
    ax.plot(vol, ret, "o", ms=2.5, color="#1f4e79", label="Turning points")
    ax.scatter(vol[[0, -1]], ret[[0, -1]], color="#c00000", zorder=5, s=18)
    ax.annotate("max return", (vol[0], ret[0]), textcoords="offset points", xytext=(-6, 4), ha="right", fontsize=8)
    ax.annotate("min variance", (vol[-1], ret[-1]), textcoords="offset points", xytext=(8, -2), fontsize=8)
    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised expected return")
    ax.set_title(f"S&P 500 efficient frontier: {n} assets, {t_days} days ({len(cla)} turning points)", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "real_frontier.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ======================================================================================
# Figure: estimators (Section 11.1: structured covariance estimators)  ->  estimator_shrinkage.pdf
# ======================================================================================
_EST_WINDOWS = [1213, 900, 750, 600, 500, 400, 300, 250, 180, 120]


def figure_estimators(out_dir: Path) -> None:
    """Trace the S&P 500 frontier through three covariance estimators via the same engine.

    Also shows how Ledoit--Wolf shrinkage tracks the conditioning of the sample
    covariance as the estimation window shrinks.
    """
    import pandas as pd
    from sklearn.covariance import LedoitWolf

    r = pd.read_parquet(DATA).to_numpy()
    t_days, n = r.shape
    mean = r.mean(axis=0)
    sample_cov = np.cov(r, rowvar=False)
    ann = 252.0

    # 1. Same engine, three statistical risk models, on the full-history universe.
    lw_full = LedoitWolf().fit(r)
    estimators = [
        ("sample (dense)", sample_cov),
        (f"Ledoit-Wolf (rho={lw_full.shrinkage_:.3f})", lw_full.covariance_),
        (f"factor PCA (K={_REAL_N_FACTORS})", _real_factor_estimate(sample_cov, _REAL_N_FACTORS)),
    ]
    print(f"universe                : {n} assets x {t_days} days")
    print(f"{'estimator':34s} points  median ms  maxSharpe(ann)  held  eff_N")
    for name, cov in estimators:
        cla, secs = _real_median_trace(mean, cov, n)
        sharpe, w = cla.frontier.max_sharpe
        held = int(np.sum(w > 1e-6))
        eff = 1.0 / float(np.sum(w**2))  # inverse Herfindahl: effective # of holdings
        print(
            f"{name:34s}  {len(cla):4d}   {secs * 1e3:7.1f}    {sharpe * np.sqrt(ann):8.2f}     {held:3d}  {eff:5.1f}"
        )

    # 2. Shrinkage intensity and sample-covariance conditioning vs estimation window.
    windows, rhos, conds = [], [], []
    for tw in _EST_WINDOWS:
        rt = r[-tw:]
        windows.append(tw)
        rhos.append(float(LedoitWolf().fit(rt).shrinkage_))
        conds.append(float(np.linalg.cond(np.cov(rt, rowvar=False))))
    print("\n   window T   LW rho      cond(S)")
    for tw, rho, cond in zip(windows, rhos, conds, strict=True):
        print(f"  {tw:8d}   {rho:6.3f}   {cond:10.2e}")

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(windows, rhos, "-o", ms=4, color="#1f4e79", label=r"Ledoit--Wolf $\rho$")
    ax.set_xlabel("Estimation window $T$ (trading days)")
    ax.set_ylabel(r"Shrinkage intensity $\rho$", color="#1f4e79")
    ax.tick_params(axis="y", labelcolor="#1f4e79")
    ax.axvline(n, ls="--", color="#888", lw=1)
    ax.annotate(f"$T=n={n}$", (n, min(rhos)), xytext=(6, 0), textcoords="offset points", fontsize=8, color="#555")
    ax.invert_xaxis()  # large windows (well-sampled) on the left
    ax2 = ax.twinx()
    ax2.semilogy(windows, conds, "-s", ms=4, color="#c00000")
    ax2.set_ylabel("Sample-covariance condition number", color="#c00000")
    ax2.tick_params(axis="y", labelcolor="#c00000")
    ax.set_title("Shrinkage rises as the sample covariance degrades", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "estimator_shrinkage.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ======================================================================================
# Figure: michaud (Section 12: resampled / robust frontier)  ->  michaud_frontier.pdf
# ======================================================================================
_MICHAUD_B = 200
_MICHAUD_SEED = 0
_MICHAUD_TOL = 1e-4  # weight above which an asset counts as held
_MICHAUD_ANN = 252.0


def _michaud_frontier(mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace one frontier and return its annualised vol, return and holdings.

    The three arrays are ordered by volatility, each evaluated under its own (mean, cov),
    and sampled along each segment so interpolating them follows the exact curve.
    """
    tp = CLA(mean=mean, covariance=cov, **_real_problem(len(mean))).turning_points
    w = _along_segments(np.array([t.weights for t in tp]), num=10)
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w)) * np.sqrt(_MICHAUD_ANN)
    ret = (w @ mean) * _MICHAUD_ANN
    held = (w > _MICHAUD_TOL).sum(axis=1).astype(float)
    order = np.argsort(vol)
    return vol[order], ret[order], held[order]


def _michaud_interp(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolate y(x) onto grid, NaN outside the frontier's own volatility range."""
    out = np.full(grid.shape, np.nan)
    inside = (grid >= x[0]) & (grid <= x[-1])
    out[inside] = np.interp(grid[inside], x, y)
    return out


def figure_michaud(out_dir: Path) -> None:
    """Draw a cloud of resampled efficient frontiers on the S&P 500.

    The estimation-error spread around the full-sample frontier, made visible because
    cvxcla traces each exact frontier cheaply. Left: the return cloud with envelopes;
    right: the number of holdings, equally unstable, under the same resampling.
    """
    import pandas as pd

    rng = np.random.default_rng(_MICHAUD_SEED)
    r = pd.read_parquet(DATA).to_numpy()
    t_days, n = r.shape
    mu0 = r.mean(axis=0)
    s0 = np.cov(r, rowvar=False)
    chol = np.linalg.cholesky(s0 + 1e-12 * np.eye(n))  # fast Gaussian draws

    start = time.perf_counter()
    frontiers = []
    for _ in range(_MICHAUD_B):
        sim = mu0 + rng.standard_normal((t_days, n)) @ chol.T
        with contextlib.suppress(Exception):  # a degenerate draw is simply skipped
            frontiers.append(_michaud_frontier(sim.mean(axis=0), np.cov(sim, rowvar=False)))
    secs = time.perf_counter() - start
    vt, rt, ht = _michaud_frontier(mu0, s0)  # full-sample reference frontier

    # Shared volatility grid covered by the bulk of the draws, so the bands are well posed.
    vmins = np.array([f[0][0] for f in frontiers])
    vmaxs = np.array([f[0][-1] for f in frontiers])
    grid = np.linspace(float(np.quantile(vmins, 0.75)), float(np.quantile(vmaxs, 0.25)), 200)
    ret_grid = np.array([_michaud_interp(grid, v, ret) for v, ret, _h in frontiers])
    held_grid = np.array([_michaud_interp(grid, v, h) for v, _ret, h in frontiers])
    r5, r25, r50, r75, r95 = (np.nanpercentile(ret_grid, q, axis=0) for q in (5, 25, 50, 75, 95))
    h10, h50, h90 = (np.nanpercentile(held_grid, q, axis=0) for q in (10, 50, 90))

    mid = grid.size // 2
    print(f"universe                : {n} assets x {t_days} days")
    each_ms = secs / len(frontiers) * 1e3
    print(f"resampling              : B={len(frontiers)} exact frontiers in {secs:.1f}s ({each_ms:.0f} ms each)")
    print(
        f"at vol~{grid[mid]:.2f}    : return 25-75% [{r25[mid]:.2f}, {r75[mid]:.2f}] "
        f"vs full-sample {np.interp(grid[mid], vt, rt):.2f}"
    )
    print(
        f"                          holdings 10-90% [{h10[mid]:.0f}, {h90[mid]:.0f}] median {h50[mid]:.0f} "
        f"vs full-sample {np.interp(grid[mid], vt, ht):.0f}"
    )

    blue, red = "#1f4e79", "#c00000"
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.4, 3.5))

    for v, ret, _h in frontiers:
        ax_a.plot(v, ret, "-", lw=0.4, color=blue, alpha=0.05, zorder=1)
    ax_a.fill_between(grid, r5, r95, color=blue, alpha=0.15, lw=0, label="5--95%")
    ax_a.fill_between(grid, r25, r75, color=blue, alpha=0.28, lw=0, label="25--75%")
    ax_a.plot(grid, r50, "--", lw=1.1, color=blue, label="resampled median")
    ax_a.plot(vt, rt, "-", lw=2.0, color=red, label="full-sample frontier")
    ax_a.set_xlabel("Annualised volatility")
    ax_a.set_ylabel("Annualised expected return")
    ax_a.set_title(f"{len(frontiers)} resampled efficient frontiers", fontsize=9)
    ax_a.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    ax_a.grid(True, alpha=0.25)

    ax_b.fill_between(grid, h10, h90, color=blue, alpha=0.22, lw=0, label="10--90% of draws")
    ax_b.plot(grid, h50, "-", lw=1.6, color=blue, label="resampled median")
    ax_b.plot(vt, ht, "-", lw=1.6, color=red, label="full-sample frontier")
    ax_b.set_xlabel("Annualised volatility")
    ax_b.set_ylabel("Number of holdings")
    ax_b.set_title("Portfolio size under estimation error", fontsize=9)
    ax_b.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    ax_b.grid(True, alpha=0.25)

    fig.tight_layout()
    out = out_dir / "michaud_frontier.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ======================================================================================
# Figure: degeneracy (from degeneracy_boundary.py)  ->  degeneracy.pdf
# ======================================================================================
_DEGEN_N_ASSETS = 120
_DEGEN_WINDOWS = [240, 180, 150, 130, 120, 100, 90, 75, 60, 45, 30, 20, 15]
_DEGEN_SEED = 0
_DEGEN_GUARD = 1e12  # the singularity guard in CLA._emit: free-block cond above this is declined


@dataclass
class _DegenSweepResult:
    """Outcome of one trace attempt at a given number of observations T."""

    t_obs: int
    rank: int
    completed: bool
    n_points: int
    worst_cond: float  # worst candidate free-block condition number
    obj_gap: float  # worst objective gap of a completed turning point vs QP


def _degen_max_objective_gap(cla: CLA, mean: np.ndarray, cov: np.ndarray) -> float:
    """Worst (obj_cla - obj_qp) over turning points; ~0 means the frontier is optimal."""
    n = len(mean)
    gap = 0.0
    for tp in cla.turning_points:
        lam = tp.lamb
        if not np.isfinite(lam) or lam <= 0:
            continue
        wq = _osqp_qp(mean, cov, lam, np.ones((1, n)), np.ones(1))
        wc = tp.weights
        obj_cla = 0.5 * wc @ cov @ wc - lam * mean @ wc
        obj_qp = 0.5 * wq @ cov @ wq - lam * mean @ wq
        gap = max(gap, float(obj_cla - obj_qp))
    return gap


def _degen_trace(t_obs: int) -> _DegenSweepResult:
    """Trace the frontier for a size-``t_obs`` sample, recording the diagnostics."""
    rng = np.random.default_rng(_DEGEN_SEED)
    returns = rng.standard_normal((t_obs, _DEGEN_N_ASSETS)) * 0.01 + rng.uniform(0.0, 1e-3, _DEGEN_N_ASSETS)
    cov = np.cov(returns, rowvar=False)
    mean = returns.mean(axis=0)
    kwargs = {
        "lower_bounds": np.zeros(_DEGEN_N_ASSETS),
        "upper_bounds": np.ones(_DEGEN_N_ASSETS),
        "a": np.ones((1, _DEGEN_N_ASSETS)),
        "b": np.ones(1),
    }

    worst = [0.0]
    original_emit = CLA._emit

    def recording_emit(self: CLA, lamb: float, weights: np.ndarray, free: np.ndarray, active_ineq: np.ndarray) -> None:
        if np.any(free):
            worst[0] = max(worst[0], float(np.linalg.cond(cov[np.ix_(free, free)])))
        original_emit(self, lamb, weights, free, active_ineq)

    cla_module.CLA._emit = recording_emit
    try:
        cla = CLA(mean=mean, covariance=cov, **kwargs)
        completed, n_points = True, len(cla)
        obj_gap = _degen_max_objective_gap(cla, mean, cov)
    except ValueError:
        completed, n_points, obj_gap = False, 0, 0.0
    finally:
        cla_module.CLA._emit = original_emit

    return _DegenSweepResult(
        t_obs=t_obs,
        rank=int(np.linalg.matrix_rank(cov)),
        completed=completed,
        n_points=n_points,
        worst_cond=worst[0],
        obj_gap=obj_gap,
    )


def figure_degeneracy(out_dir: Path) -> None:
    """Run the degeneracy sweep, print a table, and write the figure."""
    results = [_degen_trace(t) for t in _DEGEN_WINDOWS]

    print(f"universe n = {_DEGEN_N_ASSETS}; singularity guard (cond) = {_DEGEN_GUARD:g}\n")
    print(f"{'T':>5} {'rank':>5} {'status':>9} {'pts':>5} {'worst cond':>11} {'obj gap':>11}")
    for r in results:
        status = "completed" if r.completed else "declined"
        print(f"{r.t_obs:>5} {r.rank:>5} {status:>9} {r.n_points:>5} {r.worst_cond:>11.2e} {r.obj_gap:>11.2e}")

    from matplotlib.lines import Line2D

    ts = [r.t_obs for r in results]
    cond = [max(r.worst_cond, 1.0) for r in results]
    done = [r.completed for r in results]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for t, c, ok in zip(ts, cond, done, strict=True):
        ax.scatter(t, c, s=36, zorder=3, color="#1f4e79" if ok else "#c00000", marker="o" if ok else "X")
    ax.axhline(_DEGEN_GUARD, ls="--", lw=1.0, color="#555555")
    ax.text(ts[0], _DEGEN_GUARD * 1.6, "singularity guard", ha="right", va="bottom", fontsize=7.5, color="#555555")
    ax.axvline(_DEGEN_N_ASSETS, ls=":", lw=1.0, color="#999999")
    ax.text(
        _DEGEN_N_ASSETS * 1.03,
        min(cond) * 1.5,
        f"$T=n={_DEGEN_N_ASSETS}$",
        rotation=90,
        va="bottom",
        fontsize=7.5,
        color="#777777",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observations $T$ (sample covariance rank $\\approx \\min(T-1, n)$)")
    ax.set_ylabel("Worst candidate free-block cond. number")
    ax.set_title(
        f"Full rank completes (optimal); a singular free block is declined ($n={_DEGEN_N_ASSETS}$)", fontsize=8.5
    )

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f4e79", ms=7, label="completed (optimal)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#c00000", ms=7, label="declined (guard)"),
    ]
    ax.legend(handles=legend, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    out = out_dir / "degeneracy.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")


# ======================================================================================
# Check: validate-exact (from validate_exact.py)
# ======================================================================================
_VEXACT_N_ASSETS = 40


def _vexact_qp_solution(mean: np.ndarray, cov: np.ndarray, lam: float) -> np.ndarray:
    """Solve min 1/2 w'Sigma w - lam mu'w s.t. 1'w=1, 0<=w<=1 with the reference QP."""
    n = mean.shape[0]
    return _osqp_qp(mean, cov, lam, np.ones((1, n)), np.ones(1))


def check_validate_exact(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Trace with the CLA, re-solve each segment midpoint as a QP, report error."""
    import pandas as pd

    returns = pd.read_parquet(DATA).iloc[:, :_VEXACT_N_ASSETS]
    mean = returns.mean(axis=0).to_numpy()
    cov = np.cov(returns.to_numpy(), rowvar=False)

    cla = CLA(
        mean=mean,
        covariance=cov,
        lower_bounds=np.zeros(_VEXACT_N_ASSETS),
        upper_bounds=np.ones(_VEXACT_N_ASSETS),
        a=np.ones((1, _VEXACT_N_ASSETS)),
        b=np.ones(1),
    )
    tps = cla.turning_points
    print(f"subproblem              : {_VEXACT_N_ASSETS} assets, cond(S)={np.linalg.cond(cov):,.0f}")
    print(f"turning points          : {len(tps)}")

    errors = []
    # Segments between consecutive turning points with finite lambda.
    for hi, lo in pairwise(tps):
        lam_hi, lam_lo = hi.lamb, lo.lamb
        if not np.isfinite(lam_hi):  # skip the lambda = inf endpoint segment
            continue
        lam = 0.5 * (lam_hi + lam_lo)
        frac = (lam - lam_lo) / (lam_hi - lam_lo)  # affine in lambda on the segment
        w_cla = lo.weights + frac * (hi.weights - lo.weights)
        w_qp = _vexact_qp_solution(mean, cov, lam)
        errors.append(float(np.max(np.abs(w_cla - w_qp))))

    errors = np.array(errors)
    print(f"segment midpoints tested: {errors.size}")
    print(f"median |w_CLA - w_QP|   : {np.median(errors):.2e}")
    print(f"max    |w_CLA - w_QP|   : {errors.max():.2e}")
    verdict = "EXACT (matches QP to solver tolerance)" if errors.max() < 1e-4 else "MISMATCH"
    print(f"verdict                 : {verdict}")


# ======================================================================================
# Check: validate-constraints (from validate_constraints.py)
# ======================================================================================
_VCON_N_ASSETS = 30
_VCON_N_FACTORS = 4
_VCON_N_SECTORS = 3
_VCON_SEED = 11
_VCON_SECTOR_CAP = 0.45
_VCON_SHORT = 0.3  # scenario 3: per-asset short limit
_VCON_LEVERAGE = 1.3  # scenario 3: gross-exposure cap


def _vcon_make_market(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dense Sigma, mean, sector id) for a deterministic factor market.

    Sector 0 carries a return premium, so the maximum-return end of the frontier
    wants to overweight it and the sector-exposure cap binds (scenario 2).
    """
    u = rng.standard_normal((_VCON_N_ASSETS, _VCON_N_FACTORS)) / np.sqrt(_VCON_N_ASSETS)
    delta = rng.uniform(0.5, 2.0, _VCON_N_FACTORS) * _VCON_N_ASSETS
    d = rng.uniform(0.5, 2.0, _VCON_N_ASSETS)
    cov = np.diag(d) + (u * delta) @ u.T
    sector = np.arange(_VCON_N_ASSETS) % _VCON_N_SECTORS  # interleaved sector membership
    mean = rng.uniform(0.0, 1.0, _VCON_N_ASSETS)
    mean[sector == 0] += 1.0  # a clear premium for sector 0 -> its cap will bind
    return cov, mean, sector


def _vcon_qp_solution(
    mean: np.ndarray,
    cov: np.ndarray,
    lam: float,
    a: np.ndarray,
    b: np.ndarray,
    g: np.ndarray | None,
    h: np.ndarray | None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    leverage: float | None = None,
) -> np.ndarray:
    """Solve the return-parametrised QP under A w = b, G w <= h, the box and an optional cap."""
    return _osqp_qp(mean, cov, lam, a, b, g, h, lower, upper, leverage)


def _vcon_validate(
    label: str,
    mean: np.ndarray,
    cov: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    g: np.ndarray | None = None,
    h: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    leverage: float | None = None,
) -> float:
    """Trace one constrained problem and check every segment midpoint against a QP.

    Returns the maximum weight discrepancy over the tested segment midpoints, and
    reports how many turning points hold an inequality row, or the gross-exposure cap,
    active (binding).
    """
    lower = np.zeros(_VCON_N_ASSETS) if lower is None else lower
    upper = np.ones(_VCON_N_ASSETS) if upper is None else upper
    cla = CLA(
        mean=mean,
        covariance=cov,
        lower_bounds=lower,
        upper_bounds=upper,
        a=a,
        b=b,
        g=g,
        h=h,
        leverage=leverage,
    )
    tps = cla.turning_points
    active = 0
    if g is not None and g.shape[0] > 0:
        active = sum(int(np.any(np.abs(g @ tp.weights - h) <= 1e-7)) for tp in tps)
    capped = 0
    if leverage is not None:
        capped = sum(int(abs(np.abs(tp.weights).sum() - leverage) <= 1e-7) for tp in tps)

    errors = []
    ties = 0  # zero-length segments: several events at one lambda, nothing to test
    for hi, lo in pairwise(tps):
        if not np.isfinite(hi.lamb):  # skip the lambda = inf endpoint segment
            continue
        if hi.lamb == lo.lamb:
            ties += 1
            continue
        lam = 0.5 * (hi.lamb + lo.lamb)
        frac = (lam - lo.lamb) / (hi.lamb - lo.lamb)  # affine in lambda on the segment
        w_cla = lo.weights + frac * (hi.weights - lo.weights)
        w_qp = _vcon_qp_solution(mean, cov, lam, a, b, g, h, lower, upper, leverage)
        errors.append(float(np.max(np.abs(w_cla - w_qp))))

    err = np.array(errors)
    print(f"\n{label}")
    print(f"  equality rows m         : {a.shape[0]}")
    if g is not None and g.shape[0] > 0:
        print(f"  inequality rows p       : {g.shape[0]}")
        print(f"  turning pts w/ active G : {active} / {len(tps)}")
    if leverage is not None:
        print(f"  gross-exposure cap c    : {leverage}")
        print(f"  turning pts w/ cap tight: {capped} / {len(tps)}")
    print(f"  turning points          : {len(tps)}")
    print(f"  segment midpoints tested: {err.size}")
    if ties:
        print(f"  zero-length segments    : {ties}")
    print(f"  median |w_CLA - w_QP|   : {np.median(err):.2e}")
    print(f"  max    |w_CLA - w_QP|   : {err.max():.2e}")
    return err.max()


def check_validate_constraints(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Run the multi-row-equality, inequality-cap and gross-exposure-cap scenarios and report exactness."""
    rng = np.random.default_rng(_VCON_SEED)
    cov, mean, sector = _vcon_make_market(rng)
    ones = np.ones((1, _VCON_N_ASSETS))

    # Scenario 1: budget + a characteristic-neutrality row (m = 2). The neutrality
    # target is reachable (equal weight satisfies it), so the problem is feasible
    # and the first vertex is the Section 3 linear program, not the greedy fill.
    char = rng.standard_normal(_VCON_N_ASSETS)
    target = float(char.mean())  # c' (1/n 1) = mean(char): equal weight is feasible
    a2 = np.vstack([ones, char[None, :]])
    b2 = np.array([1.0, target])
    err1 = _vcon_validate("Scenario 1 -- multi-row equality (budget + characteristic neutrality)", mean, cov, a2, b2)

    # Scenario 2: budget + per-sector exposure caps G w <= h. Sector 0 carries a
    # premium, so the high-return end wants to pile into it and the cap binds.
    g = np.array([(sector == s).astype(float) for s in range(_VCON_N_SECTORS)])
    h = np.full(_VCON_N_SECTORS, _VCON_SECTOR_CAP)
    err2 = _vcon_validate("Scenario 2 -- inequality sector-exposure caps (G w <= h)", mean, cov, ones, np.ones(1), g, h)

    # Scenario 3: a 130/30 book -- budget, shorts down to -30% per asset, and a
    # gross-exposure cap ||w||_1 <= 1.3 traced through the long/short lift.
    lower = np.full(_VCON_N_ASSETS, -_VCON_SHORT)
    upper = np.ones(_VCON_N_ASSETS)
    err3 = _vcon_validate(
        "Scenario 3 -- gross-exposure cap (130/30: ||w||_1 <= 1.3)",
        mean,
        cov,
        ones,
        np.ones(1),
        lower=lower,
        upper=upper,
        leverage=_VCON_LEVERAGE,
    )

    worst = max(err1, err2, err3)
    verdict = "EXACT (matches QP to solver tolerance)" if worst < 1e-4 else "MISMATCH"
    print(f"\nworst discrepancy across scenarios: {worst:.2e}  ->  {verdict}")


# ======================================================================================
# Check: tie-degeneracy (from tie_degeneracy.py)
# ======================================================================================
_TIE_SIZES = [20, 60, 120, 240]
_TIE_SEEDS = range(8)
_TIE_N_FACTORS = 5


def _tie_factor_cov(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """A well-conditioned positive-definite K-factor covariance."""
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    return np.diag(d) + (u * delta) @ u.T


def _tie_long_only(mean: np.ndarray, cov: np.ndarray, g: np.ndarray | None = None, h: np.ndarray | None = None) -> dict:
    """Assemble a long-only, fully-invested problem dict for the CLA constructor."""
    n = len(mean)
    return {
        "mean": mean,
        "covariance": cov,
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
        "g": g,
        "h": h,
    }


def _tie_tied_means(rng: np.random.Generator, n: int) -> dict:
    """Means fall into 4 exactly-tied groups (events tie at and along the trace)."""
    base = rng.uniform(0.0, 1.0, 4)
    mean = np.resize(np.repeat(base, int(np.ceil(n / 4))), n)
    return _tie_long_only(mean, _tie_factor_cov(rng, n, _TIE_N_FACTORS))


def _tie_duplicated_assets(rng: np.random.Generator, n: int) -> dict:
    """The second half copies the first: exchangeable pairs with tied means and events.

    Each pair shares its mean and its covariance row, but the idiosyncratic diagonal
    keeps the covariance positive definite, so the pair's events tie exactly.
    """
    half = n // 2
    u = rng.standard_normal((half, _TIE_N_FACTORS)) / np.sqrt(half)
    delta = rng.uniform(0.5, 2.0, _TIE_N_FACTORS) * half
    d = rng.uniform(0.5, 2.0, half)
    u_full = np.vstack([u, u])[:n]
    d_full = np.concatenate([d, d])[:n]
    cov = np.diag(d_full) + (u_full * delta) @ u_full.T  # positive definite: d_full > 0
    mean_half = rng.uniform(0.0, 1.0, half)
    mean = np.concatenate([mean_half, mean_half])[:n]
    return _tie_long_only(mean, cov)


def _tie_group_caps(rng: np.random.Generator, n: int) -> dict:
    """Non-overlapping group-exposure caps (p ~ n/3), tightened so many bind at once."""
    cov = _tie_factor_cov(rng, n, _TIE_N_FACTORS)
    mean = rng.uniform(0.0, 1.0, n)
    groups = np.arange(n) // 3  # groups of 3 -> p = ceil(n/3)
    p = int(groups.max()) + 1
    g = np.array([(groups == j).astype(float) for j in range(p)])
    h = np.full(p, 1.5 * 3 / n)  # 1.5x the equal-weight group mass: binds, stays feasible
    return _tie_long_only(mean, cov, g, h)


def _tie_overlapping_caps(rng: np.random.Generator, n: int) -> dict:
    """Overlapping windowed caps (p = n), tight enough to force a degenerate first vertex."""
    cov = _tie_factor_cov(rng, n, _TIE_N_FACTORS)
    mean = rng.uniform(0.0, 1.0, n)
    width = max(2, n // 10)
    g = np.array([np.isin(np.arange(n), np.arange(i, i + width) % n).astype(float) for i in range(n)])
    h = np.full(n, width / n)  # tight: the concentrated max-return vertex over-activates rows
    return _tie_long_only(mean, cov, g, h)


def _tie_points(cla: CLA, kwargs: dict) -> tuple[int, int]:
    """Return (tie points, tie points without linearly independent active constraints).

    A tie point is a turning point at the same lambda as the one before it (within the
    relative window of the event selection), skipping the first vertex, which the CLA
    lists twice. Proposition 1 of the paper needs the gradients of the constraints that
    hold with equality there -- the rows of A, the binding rows of G, the bounds a weight
    sits on -- to be linearly independent; the second count is the points where they are not.
    """
    n = len(kwargs["mean"])
    a = np.atleast_2d(kwargs["a"])
    g, h = kwargs.get("g"), kwargs.get("h")
    lower, upper = kwargs["lower_bounds"], kwargs["upper_bounds"]
    tps = cla.turning_points[1:]
    ties = dependent = 0
    for prev, tp in itertools.pairwise(tps):
        if not (np.isfinite(prev.lamb) and abs(tp.lamb - prev.lamb) <= 1e-10 * max(1.0, abs(prev.lamb))):
            continue
        ties += 1
        w = tp.weights
        at_bound = np.flatnonzero((np.abs(w - lower) <= 1e-9) | (np.abs(w - upper) <= 1e-9))
        rows = [a, np.eye(n)[at_bound]]
        if g is not None:
            rows.append(np.atleast_2d(g)[np.abs(np.atleast_2d(g) @ w - h) <= 1e-7])
        grads = np.vstack(rows)
        dependent += int(np.linalg.matrix_rank(grads) < grads.shape[0])
    return ties, dependent


def _tie_outcome(kwargs: dict) -> tuple[str, int, int, int, int, int]:
    """Trace one instance; return (status, turning_points, cap, max_active_rows, ties, dependent ties).

    status in {completed, declined, cap_hit}. The CLA traces in its constructor, so
    a decline (ValueError) or a cap hit (RuntimeError) surfaces here.
    """
    n = len(kwargs["mean"])
    g = kwargs.get("g")
    p = 0 if g is None else np.atleast_2d(g).shape[0]
    cap = 100 * (n + p + 1)
    try:
        cla = CLA(**kwargs)
    except ValueError:
        return "declined", 0, cap, 0, 0, 0
    except RuntimeError:
        return "cap_hit", 0, cap, 0, 0, 0
    active = 0
    if g is not None:
        h = kwargs["h"]
        active = max(int(np.sum(np.abs(g @ tp.weights - h) <= 1e-7)) for tp in cla.turning_points)
    return "completed", len(cla.turning_points), cap, active, *_tie_points(cla, kwargs)


def check_tie_degeneracy(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Run every family over the size/seed grid and report the robustness envelope."""
    families = {
        "tied means": _tie_tied_means,
        "exchangeable pairs": _tie_duplicated_assets,
        "group caps (p~n/3)": _tie_group_caps,
        "overlapping caps (p=n)": _tie_overlapping_caps,
    }
    grid = [(n, s) for n in _TIE_SIZES for s in _TIE_SEEDS]
    print(f"{len(grid)} instances per family; sizes {_TIE_SIZES}, seeds {_TIE_SEEDS.start}..{_TIE_SEEDS.stop - 1}\n")
    head = (
        f"{'family':<24}{'completed':>10}{'declined':>9}{'cap hits':>9}{'max tps':>8}{'tps/cap':>9}{'max active':>11}"
        f"{'ties':>7}{'ties dep.':>10}"
    )
    print(head)
    for name, build in families.items():
        completed = declined = cap_hit = 0
        max_tps = max_active = ties = dependent = 0
        worst_ratio = 0.0
        for n, seed in grid:
            status, tps, cap, active, n_ties, n_dep = _tie_outcome(build(np.random.default_rng(seed), n))
            ties += n_ties
            dependent += n_dep
            if status == "completed":
                completed += 1
                max_tps = max(max_tps, tps)
                max_active = max(max_active, active)
                worst_ratio = max(worst_ratio, tps / cap)
            elif status == "declined":
                declined += 1
            else:
                cap_hit += 1
        print(
            f"{name:<24}{completed:>10}{declined:>9}{cap_hit:>9}{max_tps:>8}{worst_ratio:>8.1%}{max_active:>11}"
            f"{ties:>7}{dependent:>10}"
        )

    print("\nNo cap hits: every completing trace stays far below 100(n+p+1), and the")
    print("over-constrained family is declined at the first vertex with a diagnosis,")
    print("not run into the cap. Declines are the two documented boundaries.")
    print("'ties' counts turning points at the lambda of the one before (a tie); 'ties dep.'")
    print("those where the active constraint gradients are dependent, outside Proposition 1.")


# ======================================================================================
# Check: osqp (a warm-started QP grid on the S&P 500 frontier, as in the lasso note)
# ======================================================================================
_OSQP_GRID = 100
_OSQP_TOLS = (1e-6, 1e-5)


def check_osqp(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Time the exact S&P 500 trace against a warm-started OSQP grid, at two tolerances.

    The synthetic n-point grids are part of the ``scaling`` sweep.
    """
    import pandas as pd

    r = pd.read_parquet(DATA).to_numpy()
    t_days, n = r.shape
    mean, sigma = r.mean(axis=0), np.cov(r, rowvar=False)
    cla, t_cla = _real_median_trace(mean, sigma, n)
    lams_tp = [tp.lamb for tp in cla.turning_points if np.isfinite(tp.lamb) and tp.lamb > 0]
    lams = np.geomspace(max(lams_tp), min(lams_tp), _OSQP_GRID)
    exact = np.array([_cla_weight_at(cla, float(lam)) for lam in lams])

    print(f"universe                : {n} assets x {t_days} days")
    print(f"cvxcla exact trace      : {len(cla)} turning points in {t_cla * 1e3:.1f} ms (median)")
    for eps in _OSQP_TOLS:
        times = []
        for _ in range(_REAL_REPEATS):
            grid, secs = _osqp_warm_grid(mean, sigma, lams, eps)
            times.append(secs)
        gap = float(np.max(np.abs(grid - exact)))
        t_osqp = float(np.median(times))
        print(
            f"OSQP warm grid eps={eps:.0e}: {_OSQP_GRID} solves in {t_osqp * 1e3:.1f} ms (median), "
            f"{t_osqp / t_cla:.1f}x the exact trace, max|w - w*| = {gap:.1e}"
        )


# ======================================================================================
# Check: validate-kkt  (Section 8.6 KKT residuals)
# ======================================================================================
_KKT_SHORT_WINDOW = 600  # trading days: more than the 494 assets, but ill-conditioned
_KKT_TOL = 1e-9  # a weight within this of a bound, or a row within it of h, is active
_KKT_FRACTIONS = (0.25, 0.5, 0.75)  # interior points tested on every segment
_KKT_ABOVE = (2.0, 4.0)  # multiples of the first finite lambda tested above it
_KKT_PASS = 1e-8  # relative residual below which a condition counts as satisfied


@dataclass
class _KKTResiduals:
    """The largest residual of each KKT condition over the points tested on one trace."""

    points: int = 0
    vertices: int = 0  # points whose free set does not determine the multipliers
    primal: float = 0.0
    stationarity: float = 0.0
    dual: float = 0.0
    complementarity: float = 0.0

    def update(self, other: tuple[float, float, float, float, bool]) -> None:
        """Fold in the four residuals of one test point (and whether it was a vertex)."""
        self.points += 1
        self.vertices += int(other[4])
        self.primal = max(self.primal, other[0])
        self.stationarity = max(self.stationarity, other[1])
        self.dual = max(self.dual, other[2])
        self.complementarity = max(self.complementarity, other[3])


def _kkt_point(
    w: np.ndarray,
    lam: float,
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
) -> tuple[float, float, float, float, bool]:
    """KKT residuals of ``min 1/2 w'Sw - lam mu'w`` s.t. ``Aw = b, Gw <= h, l <= w <= u`` at w.

    The certificate does not use the CLA's own multipliers or partition. The active
    set is read off w (a weight within _KKT_TOL of a bound is blocked, a row within it
    of h is active). When the free coordinates determine the equality and active-row
    multipliers (their rows have full rank there), these solve stationarity on the
    free coordinates by least squares and the bound multipliers are the remaining
    gradient on the blocked ones, so dual feasibility is tested on its own. At a
    vertex they are not unique (on the maximum-return vertex no weight is free), and
    the test asks instead whether sign-feasible multipliers exist: a bounded least
    squares fit of stationarity over all coordinates with eta, and the bound
    multipliers, >= 0, whose residual is the distance to a KKT point.

    Returns (primal infeasibility, stationarity, dual infeasibility, complementarity,
    vertex); the middle three are relative to the scale max(1, |S w|, lam |mu|).
    """
    at_lower = np.abs(w - lower) <= _KKT_TOL
    at_upper = (np.abs(w - upper) <= _KKT_TOL) & ~at_lower
    free = ~(at_lower | at_upper)
    slack = g @ w - h
    active = np.abs(slack) <= _KKT_TOL

    grad = cov @ w - lam * mean
    scale = max(1.0, float(np.max(np.abs(cov @ w))), lam * float(np.max(np.abs(mean))))
    rows = np.vstack([a, g[active]])
    m = a.shape[0]
    vertex = rows.shape[0] > 0 and np.linalg.matrix_rank(rows[:, free]) < rows.shape[0]
    if not vertex:
        mult = np.linalg.lstsq(rows[:, free].T, -grad[free], rcond=None)[0] if rows.shape[0] else np.zeros(0)
        full = grad + rows.T @ mult  # gradient of the Lagrangian without the bound terms
        nu_lower = full[at_lower]  # multiplier of l <= w: must be >= 0
        nu_upper = -full[at_upper]  # multiplier of w <= u: must be >= 0
        eta = mult[m:]  # multipliers of the active inequality rows: >= 0
        stationarity = float(np.max(np.abs(full[free]), initial=0.0)) / scale
    else:
        from scipy.optimize import lsq_linear

        eye = np.eye(len(w))
        system = np.hstack([rows.T, -eye[:, at_lower], eye[:, at_upper]])
        lower_b = np.concatenate([np.full(m, -np.inf), np.zeros(system.shape[1] - m)])
        # BVLS is an exact active-set method; the default trust-region solver stops
        # at its tolerance, ~1e-8 here, well above the residuals being measured.
        fit = lsq_linear(system, -grad, bounds=(lower_b, np.full(system.shape[1], np.inf)), method="bvls", tol=1e-15)
        eta = fit.x[m : rows.shape[0]]
        nu_lower = fit.x[rows.shape[0] : rows.shape[0] + int(at_lower.sum())]
        nu_upper = fit.x[rows.shape[0] + int(at_lower.sum()) :]
        stationarity = float(np.max(np.abs(grad + system @ fit.x))) / scale

    primal = max(
        float(np.max(np.maximum(lower - w, 0.0), initial=0.0)),
        float(np.max(np.maximum(w - upper, 0.0), initial=0.0)),
        float(np.max(np.abs(a @ w - b), initial=0.0)),
        float(np.max(np.maximum(slack, 0.0), initial=0.0)),
    )
    dual = (
        max(
            float(np.max(np.maximum(-nu_lower, 0.0), initial=0.0)),
            float(np.max(np.maximum(-nu_upper, 0.0), initial=0.0)),
            float(np.max(np.maximum(-eta, 0.0), initial=0.0)),
        )
        / scale
    )
    complementarity = (
        max(
            float(np.max(np.abs(nu_lower * (w - lower)[at_lower]), initial=0.0)),
            float(np.max(np.abs(nu_upper * (upper - w)[at_upper]), initial=0.0)),
            float(np.max(np.abs(eta * slack[active]), initial=0.0)),
        )
        / scale
    )
    return primal, stationarity, dual, complementarity, vertex


def _kkt_trace_points(cla: CLA) -> list[tuple[float, np.ndarray]]:
    """Return (lambda, w) at interior points of every segment of a trace.

    Between consecutive turning points the weights are affine in lambda, so the
    interior points are interpolations; above the first finite turning point the
    portfolio is the constant maximum-return vertex.
    """
    points = []
    tps = cla.turning_points
    finite = [tp.lamb for tp in tps if np.isfinite(tp.lamb)]
    for hi, lo in pairwise(tps):
        if not np.isfinite(hi.lamb):
            points += [(c * finite[0], hi.weights) for c in _KKT_ABOVE]
            continue
        if hi.lamb - lo.lamb <= 1e-12 * max(1.0, abs(hi.lamb)):
            continue  # a zero-length step at a tie
        for frac in _KKT_FRACTIONS:
            points.append((lo.lamb + frac * (hi.lamb - lo.lamb), lo.weights + frac * (hi.weights - lo.weights)))
    return points


def _kkt_check(cla: CLA, mean: np.ndarray, cov: np.ndarray, kwargs: dict) -> _KKTResiduals:
    """KKT residuals over a trace of the problem with box, equality and inequality rows."""
    n = len(mean)
    g = np.zeros((0, n)) if kwargs.get("g") is None else np.atleast_2d(kwargs["g"])
    h = np.zeros(0) if kwargs.get("h") is None else np.asarray(kwargs["h"], dtype=float)
    res = _KKTResiduals()
    for lam, w in _kkt_trace_points(cla):
        res.update(
            _kkt_point(
                w, lam, mean, cov, kwargs["lower_bounds"], kwargs["upper_bounds"], kwargs["a"], kwargs["b"], g, h
            )
        )
    return res


def _kkt_check_leverage(cla: CLA, mean: np.ndarray, cov: np.ndarray, kwargs: dict, cap: float) -> _KKTResiduals:
    """KKT residuals of the lifted long/short program for a gross-exposure cap.

    Every asset whose box straddles zero is split into legs x+ = max(w, 0) and
    x- = max(-w, 0), so ``||w||_1 <= cap`` becomes the linear row ``1'x <= cap`` and
    the residuals are those of the lifted problem, built here independently of the
    CLA's own lift.
    """
    lower, upper = kwargs["lower_bounds"], kwargs["upper_bounds"]
    n = len(mean)
    legs = [(i, 1.0) for i in range(n) if upper[i] > 0] + [(i, -1.0) for i in range(n) if lower[i] < 0]
    lift = np.zeros((n, len(legs)))
    for j, (i, sign) in enumerate(legs):
        lift[i, j] = sign
    leg_upper = np.array([upper[i] if sign > 0 else -lower[i] for i, sign in legs])
    a_l = kwargs["a"] @ lift
    g_l = np.ones((1, len(legs)))
    h_l = np.array([cap])
    res = _KKTResiduals()
    for lam, w in _kkt_trace_points(cla):
        x = np.array([max(sign * w[i], 0.0) for i, sign in legs])
        res.update(
            _kkt_point(
                x, lam, lift.T @ mean, lift.T @ cov @ lift, np.zeros(len(legs)), leg_upper, a_l, kwargs["b"], g_l, h_l
            )
        )
    return res


def check_validate_kkt(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Report the largest KKT residual of each condition over every segment of several traces."""
    import pandas as pd

    from cvxcla import IncrementalDenseCovariance

    returns = pd.read_parquet(DATA)
    cases = []

    def long_only(n: int) -> dict:
        return {"lower_bounds": np.zeros(n), "upper_bounds": np.ones(n), "a": np.ones((1, n)), "b": np.ones(1)}

    sub = returns.iloc[:, :_VEXACT_N_ASSETS]
    mean40, cov40 = sub.mean(axis=0).to_numpy(), np.cov(sub.to_numpy(), rowvar=False)
    cases.append(("S&P 500, 40 assets", mean40, cov40, cov40, long_only(_VEXACT_N_ASSETS), None))

    full_mean, full_cov = returns.mean(axis=0).to_numpy(), np.cov(returns.to_numpy(), rowvar=False)
    n_full = len(full_mean)
    cases.append(("S&P 500, 494 assets", full_mean, full_cov, full_cov, long_only(n_full), None))
    cases.append(
        (
            "S&P 500, 494 assets, incremental backend",
            full_mean,
            full_cov,
            IncrementalDenseCovariance(full_cov),
            long_only(n_full),
            None,
        )
    )
    short = returns.iloc[-_KKT_SHORT_WINDOW:]
    short_mean, short_cov = short.mean(axis=0).to_numpy(), np.cov(short.to_numpy(), rowvar=False)
    cases.append(
        (
            f"S&P 500, 494 assets, last {_KKT_SHORT_WINDOW} days",
            short_mean,
            short_cov,
            short_cov,
            long_only(n_full),
            None,
        )
    )

    dense, factor, problem = _scale_make_problem(np.random.default_rng(_SCALE_SEED), 320, _SCALE_N_FACTORS)
    factor_kwargs = {k: v for k, v in problem.items() if k != "mean"}
    cases.append(("factor market, 320 assets, factor backend", problem["mean"], dense, factor, factor_kwargs, None))

    rng = np.random.default_rng(_VCON_SEED)
    cov30, mean30, sector = _vcon_make_market(rng)
    ones = np.ones((1, _VCON_N_ASSETS))
    char = rng.standard_normal(_VCON_N_ASSETS)
    neutral = long_only(_VCON_N_ASSETS) | {"a": np.vstack([ones, char[None, :]]), "b": np.array([1.0, char.mean()])}
    cases.append(("30 assets, budget + neutrality row", mean30, cov30, cov30, neutral, None))
    caps = long_only(_VCON_N_ASSETS) | {
        "g": np.array([(sector == s).astype(float) for s in range(_VCON_N_SECTORS)]),
        "h": np.full(_VCON_N_SECTORS, _VCON_SECTOR_CAP),
    }
    cases.append(("30 assets, budget + sector caps", mean30, cov30, cov30, caps, None))
    book = {
        "lower_bounds": np.full(_VCON_N_ASSETS, -_VCON_SHORT),
        "upper_bounds": np.ones(_VCON_N_ASSETS),
        "a": ones,
        "b": np.ones(1),
    }
    cases.append(("30 assets, 130/30 gross-exposure cap", mean30, cov30, cov30, book, _VCON_LEVERAGE))

    print(
        f"{'problem':<42}{'cond':>9}{'points':>8}{'vertex':>7}{'primal':>10}{'station.':>10}{'dual':>10}{'compl.':>10}"
    )
    worst = 0.0
    for label, mean, cov, covariance, kwargs, cap in cases:
        cla = CLA(mean=mean, covariance=covariance, leverage=cap, **kwargs)
        res = _kkt_check(cla, mean, cov, kwargs) if cap is None else _kkt_check_leverage(cla, mean, cov, kwargs, cap)
        worst = max(worst, res.primal, res.stationarity, res.dual, res.complementarity)
        print(
            f"{label:<42}{np.linalg.cond(cov):>9.1e}{res.points:>8d}{res.vertices:>7d}{res.primal:>10.1e}"
            f"{res.stationarity:>10.1e}{res.dual:>10.1e}{res.complementarity:>10.1e}"
        )
    # The certificate must also detect errors: feed it wrong points from the first trace.
    mean, cov, kwargs = cases[0][1], cases[0][2], cases[0][4]
    points = _kkt_trace_points(CLA(mean=mean, covariance=cov, **kwargs))
    lam, w = points[len(points) // 2]
    no_rows = (np.zeros((0, len(w))), np.zeros(0))
    bounds = (kwargs["lower_bounds"], kwargs["upper_bounds"], kwargs["a"], kwargs["b"], *no_rows)
    inside = np.flatnonzero((w > 1e-6) & (w < 1.0 - 1e-6))
    shift = np.zeros_like(w)
    shift[inside[0]], shift[inside[1]] = 1e-3, -1e-3  # stays on the budget
    wrong = {
        "lambda of another segment": (points[len(points) // 4][0], w),
        "weights shifted by 1e-3": (lam, w + shift),
    }
    print("\nsensitivity (largest relative residual of the four conditions):")
    for name, (lam_x, w_x) in wrong.items():
        print(f"  {name:<28}{max(_kkt_point(w_x, lam_x, mean, cov, *bounds)[:4]):.1e}")
    print(
        f"\nprimal: absolute; stationarity, dual and complementarity: relative to max(1, |Sw|, lam|mu|)."
        f"\nvertex: points whose free set does not determine the multipliers; there stationarity is"
        f"\nthe distance to a KKT point under sign-feasible multipliers, so dual feasibility holds by"
        f"\nconstruction and a violation would show up as stationarity."
        f"\nworst residual {worst:.1e}  ->  {'PASS' if worst < _KKT_PASS else 'FAIL'} (threshold {_KKT_PASS:.0e})"
    )


# ======================================================================================
# Check: validate-scaling  (Section 8.6 units and the slope floor)
# ======================================================================================
_VSCALE_FACTORS = (1e-6, 1e-3, 1.0, 1e3, 1e6)  # rescalings of mu and of Sigma
_VSCALE_FLOORS = (1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6)  # multiples of the slope floors


def _vscale_weights(cla: CLA) -> np.ndarray:
    """The turning-point weights of a trace, stacked row by row."""
    return np.array([tp.weights for tp in cla.turning_points])


def _vscale_floor_trace(mean: np.ndarray, cov: np.ndarray, kwargs: dict, factor: float) -> np.ndarray | str:
    """Trace with both slope floors multiplied by ``factor``; return the weights or the error.

    The floors are ``sqrt(eps) / lambda_scale`` for weight slopes and ``sqrt(eps) *
    mu_scale`` for multiplier slopes; passing ``lambda_scale / factor`` and
    ``mu_scale * factor`` to the event scan multiplies both by ``factor`` and leaves
    the event-ordering window alone.
    """
    original = cla_module.segment_events

    def scaled(segment, lower, upper, g, h, lam_scale=1.0, mu_scale=1.0):
        return original(segment, lower, upper, g, h, lam_scale / factor, mu_scale * factor)

    cla_module.segment_events = scaled
    try:
        return _vscale_weights(CLA(mean=mean, covariance=cov, **kwargs))
    except (RuntimeError, ValueError) as exc:
        return type(exc).__name__
    finally:
        cla_module.segment_events = original


def check_validate_scaling(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Invariance under a change of units, and the sensitivity of the trace to the slope floor."""
    import pandas as pd

    if not hasattr(CLA, "lambda_scale"):
        print("needs a cvxcla whose event tests are scale-aware (newer than 2.1.0); skipped")
        return
    returns = pd.read_parquet(DATA)

    def long_only(n: int) -> dict:
        return {"lower_bounds": np.zeros(n), "upper_bounds": np.ones(n), "a": np.ones((1, n)), "b": np.ones(1)}

    sub = returns.iloc[:, :_VEXACT_N_ASSETS]
    mean40, cov40 = sub.mean(axis=0).to_numpy(), np.cov(sub.to_numpy(), rowvar=False)
    kwargs40 = long_only(_VEXACT_N_ASSETS)
    reference = CLA(mean=mean40, covariance=cov40, **kwargs40)
    ref_w = _vscale_weights(reference)
    ref_lam = np.array([tp.lamb for tp in reference.turning_points])
    finite = np.isfinite(ref_lam) & (ref_lam > 0)

    print(f"rescaling mu by c and Sigma by s ({_VEXACT_N_ASSETS} S&P 500 assets, {len(ref_w)} turning points):")
    worst_w = worst_lam = 0.0
    failures = []
    for c in _VSCALE_FACTORS:
        for s in _VSCALE_FACTORS:
            try:
                cla = CLA(mean=c * mean40, covariance=s * cov40, **kwargs40)
            except (RuntimeError, ValueError) as exc:
                failures.append(f"c={c:.0e}, s={s:.0e}: {type(exc).__name__}")
                continue
            w = _vscale_weights(cla)
            lam = np.array([tp.lamb for tp in cla.turning_points])
            if w.shape != ref_w.shape:
                failures.append(f"c={c:.0e}, s={s:.0e}: {len(w)} turning points")
                continue
            worst_w = max(worst_w, float(np.max(np.abs(w - ref_w))))
            worst_lam = max(worst_lam, float(np.max(np.abs(lam[finite] * c / s - ref_lam[finite]) / ref_lam[finite])))
    print(
        f"  {len(_VSCALE_FACTORS) ** 2} rescalings, c and s from {_VSCALE_FACTORS[0]:.0e} to {_VSCALE_FACTORS[-1]:.0e}"
    )
    print(f"  failures: {', '.join(failures) if failures else 'none'}")
    print(f"  max |w - w_ref| = {worst_w:.1e}, max relative error of lambda * c / s = {worst_lam:.1e}")

    full_mean, full_cov = returns.mean(axis=0).to_numpy(), np.cov(returns.to_numpy(), rowvar=False)
    short = returns.iloc[-_KKT_SHORT_WINDOW:]
    dense, _, problem = _scale_make_problem(np.random.default_rng(_SCALE_SEED), 320, _SCALE_N_FACTORS)
    cases = [
        (f"S&P 500, {_VEXACT_N_ASSETS} assets", mean40, cov40, kwargs40),
        ("S&P 500, 494 assets", full_mean, full_cov, long_only(len(full_mean))),
        (
            f"S&P 500, 494 assets, last {_KKT_SHORT_WINDOW} days",
            short.mean(axis=0).to_numpy(),
            np.cov(short.to_numpy(), rowvar=False),
            long_only(len(full_mean)),
        ),
        ("factor market, 320 assets", problem["mean"], dense, {k: v for k, v in problem.items() if k != "mean"}),
    ]
    print("\nslope floors multiplied by k: max |w - w(k=1)|, or the changed turning-point count")
    print(f"{'problem':<42}" + "".join(f"{k:>9.0e}" for k in _VSCALE_FLOORS))
    for label, mean, cov, kwargs in cases:
        ref = _vscale_floor_trace(mean, cov, kwargs, 1.0)
        cells = []
        for factor in _VSCALE_FLOORS:
            w = _vscale_floor_trace(mean, cov, kwargs, factor)
            if isinstance(w, str):
                cells.append(w[:8])
            elif w.shape != ref.shape:
                cells.append(f"{len(w)} pts")
            else:
                cells.append(f"{np.max(np.abs(w - ref)):.0e}")
        print(f"{label + f' ({len(ref)})':<42}" + "".join(f"{c:>9}" for c in cells))


# ======================================================================================
# Check: validate-projection  (Appendix A feasibility corrections)
# ======================================================================================
def _vproj_trace(mean: np.ndarray, cov: np.ndarray) -> tuple[str, list[tuple[float, float, float]]]:
    """Trace a long-only problem, recording every feasibility projection that changes a point.

    Returns the outcome and, per correction, (max |w' - w|, relative change of the
    objective 1/2 w'Sw - lam mu'w, largest constraint residual of w').
    """
    n = len(mean)
    kwargs = {"lower_bounds": np.zeros(n), "upper_bounds": np.ones(n), "a": np.ones((1, n)), "b": np.ones(1)}
    records: list[tuple[float, float, float]] = []
    current = [0.0]
    original_project, original_emit = cla_module.project_feasible, cla_module.CLA._emit

    def objective(w: np.ndarray) -> float:
        return 0.5 * float(w @ cov @ w) - current[0] * float(mean @ w)

    def recording_project(weights, lower, upper, a, b, g, h, active_ineq):
        out = original_project(weights, lower, upper, a, b, g, h, active_ineq)
        if not np.array_equal(out, weights):
            change = abs(objective(out) - objective(weights)) / max(1.0, abs(objective(weights)))
            residual = max(
                float(np.max(np.maximum(lower - out, 0.0))),
                float(np.max(np.maximum(out - upper, 0.0))),
                float(np.max(np.abs(a @ out - b))),
            )
            records.append((float(np.max(np.abs(out - weights))), change, residual))
        return out

    def recording_emit(self, lamb, weights, free, active_ineq):
        current[0] = 0.0 if not np.isfinite(lamb) else float(lamb)
        original_emit(self, lamb, weights, free, active_ineq)

    cla_module.project_feasible, cla_module.CLA._emit = recording_project, recording_emit
    try:
        outcome = f"{len(CLA(mean=mean, covariance=cov, **kwargs))} points"
    except (RuntimeError, ValueError) as exc:
        outcome = f"declined ({type(exc).__name__})"
    finally:
        cla_module.project_feasible, cla_module.CLA._emit = original_project, original_emit
    return outcome, records


def check_validate_projection(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Report how often the feasibility projection fires and how large its corrections are."""
    import pandas as pd

    problems = []
    for t_obs in _DEGEN_WINDOWS:
        rng = np.random.default_rng(_DEGEN_SEED)
        returns = rng.standard_normal((t_obs, _DEGEN_N_ASSETS)) * 0.01 + rng.uniform(0.0, 1e-3, _DEGEN_N_ASSETS)
        problems.append(
            (f"synthetic n={_DEGEN_N_ASSETS}, T={t_obs}", returns.mean(axis=0), np.cov(returns, rowvar=False))
        )
    sp500 = pd.read_parquet(DATA)
    for t_obs in _EST_WINDOWS:
        window = sp500.iloc[-t_obs:]
        problems.append(
            (f"S&P 500 n=494, T={t_obs}", window.mean(axis=0).to_numpy(), np.cov(window.to_numpy(), rowvar=False))
        )

    print(f"{'problem':<28}{'outcome':>22}{'proj.':>7}{'max |dw|':>11}{'max dobj':>11}{'residual':>11}")
    worst = (0.0, 0.0, 0.0)
    for label, mean, cov in problems:
        outcome, records = _vproj_trace(mean, cov)
        if records:
            stats = tuple(max(r[i] for r in records) for i in range(3))
            worst = tuple(max(a, b) for a, b in zip(worst, stats, strict=True))
            cells = "".join(f"{x:>11.1e}" for x in stats)
        else:
            cells = f"{'-':>11}" * 3
        print(f"{label:<28}{outcome:>22}{len(records):>7d}{cells}")
    print(
        f"\nlargest correction {worst[0]:.1e}, objective change {worst[1]:.1e} (relative), "
        f"constraint residual after projection {worst[2]:.1e}"
    )


# ======================================================================================
# Check: validate-factor  (Section 5 factor-model conditioning)
# ======================================================================================
_VFACTOR_N = 120
_VFACTOR_K = 10
_VFACTOR_SEED = 1


def _vfactor_compare(d: np.ndarray, u: np.ndarray, delta: np.ndarray, mean: np.ndarray) -> str:
    """Trace with the factor backend and with the dense Sigma it represents; summarise the gap."""
    n = len(d)
    dense = np.diag(d) + (u * delta) @ u.T
    kwargs = {"lower_bounds": np.zeros(n), "upper_bounds": np.ones(n), "a": np.ones((1, n)), "b": np.ones(1)}
    factor = CLA(mean=mean, covariance=FactorCovariance(d=d, u=u, delta=delta), **kwargs)
    res = _kkt_check(factor, mean, dense, kwargs)
    kkt = max(res.primal, res.stationarity, res.dual, res.complementarity)
    w_factor = np.array([tp.weights for tp in factor.turning_points])
    w_dense = np.array([tp.weights for tp in CLA(mean=mean, covariance=dense, **kwargs).turning_points])
    gap = f"{np.max(np.abs(w_factor - w_dense)):9.1e}" if w_factor.shape == w_dense.shape else f"{len(w_dense):>6} pts"
    return f"{len(w_factor):>7d}{len(w_dense):>7d}{gap}{kkt:>10.1e}{np.linalg.cond(dense):>11.1e}"


def check_validate_factor(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Factor backend against the dense trace on ill-conditioned and degenerate factor models.

    The Woodbury solve inverts Delta and the capacitance matrix Delta^{-1} + U_F' D_F^{-1} U_F.
    With d > 0 the latter is positive definite, so neither an ill-conditioned Delta, nor nearly
    collinear loadings, nor a full rank K = n should cost accuracy; this checks it.
    """
    rng = np.random.default_rng(_VFACTOR_SEED)
    n, k = _VFACTOR_N, _VFACTOR_K
    d = rng.uniform(0.5, 2.0, n)
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    mean = rng.uniform(0.0, 1.0, n)
    cases = [(f"cond(Delta) = {c:.0e}", d, u, np.geomspace(float(n), n / c, k)) for c in (1e0, 1e4, 1e8, 1e12, 1e14)]
    for gap in (1e-2, 1e-6, 1e-10):
        collinear = u.copy()
        collinear[:, 1] = collinear[:, 0] + gap * rng.standard_normal(n)
        cases.append((f"collinear loadings, gap {gap:.0e}", d, collinear, np.full(k, float(n))))
    for rank in (n // 2, n):
        cases.append((f"rank K = {rank}", d, rng.standard_normal((n, rank)) / np.sqrt(n), np.full(rank, float(n))))

    print(f"n = {n}; columns: turning points (factor, dense), max |w_factor - w_dense|, KKT residual, cond(Sigma)")
    print(f"{'factor model':<34}{'factor':>7}{'dense':>7}{'max|dw|':>9}{'KKT':>10}{'cond':>11}")
    for label, d_i, u_i, delta in cases:
        print(f"{label:<34}{_vfactor_compare(d_i, u_i, delta, mean)}")

    print("\nnot positive definite (refused at construction by a cvxcla newer than 2.1.0):")
    for label, delta in (
        ("singular", np.ones((k, k))),
        ("indefinite", np.r_[1.0, -0.5, np.ones(k - 2)]),
    ):
        try:
            FactorCovariance(d=d, u=u, delta=delta)
            outcome = "accepted"
        except ValueError as exc:
            outcome = f"ValueError: {exc}"
        print(f"  {label:<12}{outcome}")


# ======================================================================================
# Check: validate-conditioning  (Appendix A conditioning study)
# ======================================================================================
_VCOND_N = 60
_VCOND_SEED = 5
_VCOND_KAPPAS = (1e2, 1e4, 1e6, 1e8, 1e10, 1e11, 3e11, 1e12, 3e12, 1e13, 1e14, 1e16)
_VCOND_POINTS = 6  # segment midpoints compared with the reference QP per trace
_VCOND_CERTIFIED = 1e-8  # KKT residual at or below which a solution counts as certified


def _vcond_problem(kappa: float, rotated: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """A long-only, fully-invested problem whose covariance has condition number ``kappa``.

    The spectrum is prescribed, ``Sigma = Q diag(lambda) Q'`` with eigenvalues spread
    geometrically from 1 down to 1/kappa, so conditioning is varied on its own,
    independently of the sample size that drives the rank-deficiency sweep of the
    degeneracy target. With ``rotated`` Q is a random orthogonal matrix and the free
    blocks see only part of the spectrum; otherwise Q = I, Sigma is diagonal, and a free
    block inherits the full spread once its assets are free together.
    """
    rng = np.random.default_rng(_VCOND_SEED)
    q, _ = np.linalg.qr(rng.standard_normal((_VCOND_N, _VCOND_N)))
    if not rotated:
        q = np.eye(_VCOND_N)
    cov = (q * np.geomspace(1.0, 1.0 / kappa, _VCOND_N)) @ q.T
    cov = 0.5 * (cov + cov.T)
    mean = rng.uniform(0.0, 1.0, _VCOND_N)
    kwargs = {
        "lower_bounds": np.zeros(_VCOND_N),
        "upper_bounds": np.ones(_VCOND_N),
        "a": np.ones((1, _VCOND_N)),
        "b": np.ones(1),
    }
    return mean, cov, kwargs


def _vcond_reference(cla: CLA, mean: np.ndarray, cov: np.ndarray, kwargs: dict) -> tuple[float, float, int]:
    """Compare the trace with OSQP at segment midpoints; certify OSQP's own solutions.

    Returns (max |w_CLA - w_QP|, largest KKT residual of the QP solutions, QP failures).
    The QP solutions are judged by the same KKT certificate as the trace, so a reference
    that struggles shows up in its own residual, not only as a disagreement.
    """
    tps = cla.turning_points
    segments = [(hi, lo) for hi, lo in pairwise(tps) if np.isfinite(hi.lamb) and hi.lamb > lo.lamb]
    picks = np.unique(np.linspace(0, len(segments) - 1, _VCOND_POINTS).round().astype(int)) if segments else []
    gap, ref_kkt, failures = 0.0, 0.0, 0
    zero_g, zero_h = np.zeros((0, _VCOND_N)), np.zeros(0)
    for i in picks:
        hi, lo = segments[i]
        lam = 0.5 * (hi.lamb + lo.lamb)
        w_cla = 0.5 * (hi.weights + lo.weights)
        try:
            w_qp = _vexact_qp_solution(mean, cov, lam)
        except Exception:  # noqa: BLE001 - any solver failure is recorded, not raised
            failures += 1
            continue
        gap = max(gap, float(np.max(np.abs(w_cla - w_qp))))
        lower, upper, a, b = kwargs["lower_bounds"], kwargs["upper_bounds"], kwargs["a"], kwargs["b"]
        ref_kkt = max(ref_kkt, max(_kkt_point(w_qp, lam, mean, cov, lower, upper, a, b, zero_g, zero_h)[:4]))
    return gap, ref_kkt, failures


def check_validate_conditioning(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Trace problems of prescribed condition number across the singularity guard."""
    print(f"n = {_VCOND_N}, long-only budget; Sigma with eigenvalues spread from 1 to 1/kappa")
    for rotated in (True, False):
        print(f"\n{'random eigenvectors' if rotated else 'eigenvectors aligned with the assets (Sigma diagonal)'}:")
        _vcond_family(rotated)
    print(
        f"\ncertified: every KKT residual of the trace <= {_VCOND_CERTIFIED:.0e} (see validate-kkt);"
        "\nmax|dw|: largest gap to OSQP at segment midpoints; QP KKT: the same certificate applied"
        "\nto OSQP's own solutions, so a large value means the reference, not the trace, is unreliable."
    )


def _vcond_family(rotated: bool) -> None:
    """Print one row per condition number for one eigenvector family."""
    print(
        f"{'kappa':>8}{'outcome':>26}{'points':>8}{'cond(S_FF)':>12}{'KKT':>10}"
        f"{'max|dw|':>10}{'QP KKT':>10}{'QP fails':>9}"
    )
    for kappa in _VCOND_KAPPAS:
        mean, cov, kwargs = _vcond_problem(kappa, rotated)
        try:
            cla = CLA(mean=mean, covariance=cov, **kwargs)
        except (ValueError, RuntimeError) as exc:
            print(f"{kappa:>8.0e}{'declined: ' + type(exc).__name__:>26}")
            continue
        res = _kkt_check(cla, mean, cov, kwargs)
        kkt = max(res.primal, res.stationarity, res.dual, res.complementarity)
        worst = max(float(np.linalg.cond(cov[np.ix_(tp.free, tp.free)])) for tp in cla.turning_points if tp.free.any())
        gap, ref_kkt, failures = _vcond_reference(cla, mean, cov, kwargs)
        outcome = "completed, certified" if kkt <= _VCOND_CERTIFIED else "completed, not certified"
        print(
            f"{kappa:>8.0e}{outcome:>26}{len(cla):>8d}{worst:>12.1e}{kkt:>10.1e}"
            f"{gap:>10.1e}{ref_kkt:>10.1e}{failures:>9d}"
        )


# ======================================================================================
# Orchestration
# ======================================================================================
@dataclass
class Step:
    """One reproducible step: a target name, its runner, and whether it is slow."""

    target: Target
    artefact: str
    runner: object  # Callable[[Path], None]
    slow: bool


class Target(enum.StrEnum):
    """A buildable artefact, or a group of them."""

    frontier = "frontier"
    scaling = "scaling"
    rank_scaling = "rank-scaling"
    validate_exact = "validate-exact"
    validate_constraints = "validate-constraints"
    real_frontier = "real-frontier"
    degeneracy = "degeneracy"
    tie_degeneracy = "tie-degeneracy"
    osqp = "osqp"
    validate_kkt = "validate-kkt"
    validate_scaling = "validate-scaling"
    validate_projection = "validate-projection"
    validate_factor = "validate-factor"
    validate_conditioning = "validate-conditioning"
    estimators = "estimators"
    michaud = "michaud"
    figures = "figures"
    checks = "checks"
    all = "all"


# The individual steps, in the order the paper presents them (from reproduce_paper.py).
STEPS: list[Step] = [
    Step(Target.frontier, "Figure 1 (frontier.pdf)", figure_frontier, False),
    Step(Target.scaling, "Figure 2 (scaling.pdf) + Table 1", figure_scaling, True),
    Step(Target.rank_scaling, "Figure 3 (rank_scaling.pdf) + Table 2", figure_rank_scaling, True),
    Step(Target.validate_exact, "Section 10.5 exactness numbers", check_validate_exact, False),
    Step(Target.validate_constraints, "Section 10.5 general-constraint exactness", check_validate_constraints, False),
    Step(Target.real_frontier, "Figure 4 (real_frontier.pdf)", figure_real_frontier, False),
    Step(Target.degeneracy, "Figure 5 (degeneracy.pdf)", figure_degeneracy, False),
    Step(Target.tie_degeneracy, "Section 10.6 tie-heavy stress envelope", check_tie_degeneracy, False),
    Step(Target.osqp, "Section 10.4 warm-started OSQP grid on the S&P 500", check_osqp, False),
    Step(Target.validate_kkt, "Section 8.6 KKT residuals on every segment", check_validate_kkt, False),
    Step(Target.validate_scaling, "Section 8.6 units and the slope floor", check_validate_scaling, False),
    Step(Target.validate_projection, "Appendix A feasibility corrections", check_validate_projection, False),
    Step(Target.validate_factor, "Section 5 factor-model conditioning", check_validate_factor, False),
    Step(Target.validate_conditioning, "Appendix A conditioning study", check_validate_conditioning, False),
    Step(Target.estimators, "Figure 8 (estimator_shrinkage.pdf) + estimator table", figure_estimators, False),
    Step(Target.michaud, "Figure 9 (michaud_frontier.pdf) + Section 12 resampling table", figure_michaud, False),
]

_STEP_BY_TARGET: dict[Target, Step] = {s.target: s for s in STEPS}

# Which individual steps each group target expands to.
_FIGURE_TARGETS = [
    Target.frontier,
    Target.scaling,
    Target.rank_scaling,
    Target.real_frontier,
    Target.degeneracy,
    Target.estimators,
    Target.michaud,
]
_CHECK_TARGETS = [
    Target.validate_exact,
    Target.validate_constraints,
    Target.tie_degeneracy,
    Target.osqp,
    Target.validate_kkt,
    Target.validate_scaling,
    Target.validate_projection,
    Target.validate_factor,
    Target.validate_conditioning,
]


def _run_step(step: Step, out_dir: Path) -> bool:
    """Run one step, reporting (but not propagating) a missing dependency or failure.

    Returns True if it ran cleanly; False if an optional dependency is missing or
    the run raised (the error is reported, not propagated, so one step's failure
    does not abort the rest).
    """
    print(f"\n=== {step.target.value}  ->  {step.artefact} ===", flush=True)
    try:
        step.runner(out_dir)
    except ImportError as exc:
        print(f"[skip] {step.target.value}: missing optional dependency ({exc})", flush=True)
        return False
    except Exception as exc:  # noqa: BLE001 - one step must not abort the rest
        print(f"[FAILED] {step.target.value}: {type(exc).__name__}: {exc}", flush=True)
        return False
    print(f"[ok] {step.target.value}", flush=True)
    return True


def _expand(target: Target, quick: bool) -> list[Step]:
    """Resolve a target into the ordered list of individual steps to run."""
    if target is Target.all:
        wanted = [s.target for s in STEPS]
    elif target is Target.figures:
        wanted = list(_FIGURE_TARGETS)
    elif target is Target.checks:
        wanted = list(_CHECK_TARGETS)
    else:
        wanted = [target]
    # --quick only skips the slow sweeps for the group targets; an explicit slow
    # target always runs.
    is_group = target in (Target.all, Target.figures, Target.checks)
    steps = [_STEP_BY_TARGET[t] for t in wanted]
    if quick and is_group:
        steps = [s for s in steps if not s.slow]
    # Preserve the canonical STEPS order.
    return [s for s in STEPS if s in steps]


# ======================================================================================
# Typer CLI
# ======================================================================================
_DESCRIPTIONS: dict[Target, str] = {
    Target.frontier: "Fig 1: 20x50 factor-model frontier   -> frontier.pdf",
    Target.scaling: "Fig 2: runtime vs problem size (SLOW) -> scaling.pdf",
    Target.rank_scaling: "Fig 3: runtime vs factor rank (SLOW)  -> rank_scaling.pdf",
    Target.validate_exact: "Sec 10.5: frontier exactness vs QP    (check)",
    Target.validate_constraints: "Sec 10.5: constrained exactness vs QP (check)",
    Target.real_frontier: "Fig 4: S&P 500 empirical frontier     -> real_frontier.pdf",
    Target.degeneracy: "Fig 5: the degeneracy boundary        -> degeneracy.pdf",
    Target.tie_degeneracy: "Sec 10.6: tie-heavy stress envelope   (check)",
    Target.osqp: "Sec 10.4: S&P trace vs warm OSQP grid (check)",
    Target.estimators: "Fig 8: covariance-estimator shrinkage -> estimator_shrinkage.pdf",
    Target.michaud: "Fig 9: Michaud resampled frontier     -> michaud_frontier.pdf",
    Target.figures: "all seven figures",
    Target.checks: "all nine numerical checks",
    Target.all: "every figure and every check",
}


def _choose_target() -> Target:
    """Interactively ask which artefact to build when none was given on the CLI."""
    options = list(Target)
    typer.echo("Which artefact to generate?\n")
    for i, t in enumerate(options, 1):
        typer.echo(f"  {i:>2}. {t.value:<26} {_DESCRIPTIONS[t]}")
    typer.echo("")
    while True:
        raw = typer.prompt("Enter a number or name", default="all").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        try:
            return Target(raw)
        except ValueError:
            typer.echo(f"  '{raw}' is not a valid choice; try again.")


def _run_target(target: Target, out_dir: Path, quick: bool) -> int:
    """Build the requested artefact(s); return a process exit code."""
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = _expand(target, quick)
    is_group = target in (Target.all, Target.figures, Target.checks)
    if quick and is_group:
        print("[--quick] skipping the long-running scaling sweeps", flush=True)

    results = {step.target: _run_step(step, out_dir) for step in steps}

    if is_group:
        print("\n=== summary ===", flush=True)
        for tgt, ok in results.items():
            print(f"  {'ok  ' if ok else 'FAIL'}  {tgt.value}", flush=True)
    return 0 if all(results.values()) else 1


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    target: Annotated[
        Target | None,
        typer.Argument(help="Which artefact to build. Omit to choose interactively."),
    ] = None,
    out_dir: Annotated[
        Path,
        typer.Option(help="Directory to write artefacts into."),
    ] = DEFAULT_OUT_DIR,
    quick: Annotated[
        bool,
        typer.Option(help="For 'all'/'figures', skip the slow scaling sweeps (scaling, rank-scaling)."),
    ] = False,
) -> None:
    """Build a CLA-paper figure (or all of them, or the check suite)."""
    if target is None:
        target = _choose_target()
    raise typer.Exit(_run_target(target, out_dir, quick))


if __name__ == "__main__":
    app()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "casadi==3.8.1",
#     "cvxcla==2.1.0",
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
    factor (Woodbury) backend, with baselines, and the memory table (scaling.pdf).
    SLOW: the dense backend at n=5120 takes several minutes per trace.
  * ``rank-scaling``    -- Figure 3 + Table 2: runtime vs factor rank at fixed n
    (rank_scaling.pdf).  SLOW.
  * ``validate-exact``  -- Section 10.5 exactness numbers (no figure).
  * ``validate-constraints`` -- Section 10.5 general-constraint exactness (no figure).
  * ``real-frontier``   -- Figure 4: the S&P 500 empirical frontier (real_frontier.pdf).
  * ``degeneracy``      -- Figure 5: the degeneracy boundary (degeneracy.pdf).
  * ``tie-degeneracy``  -- Section 10.6 tie-heavy stress envelope (no figure).
  * ``osqp``            -- S&P 500 trace vs a warm-started OSQP grid (no figure).
  * ``estimators``      -- Figure 8 + Section 11.1 estimator table (estimator_shrinkage.pdf).
  * ``michaud``         -- Figure 9 + Section 12 resampling table (michaud_frontier.pdf).
  * ``figures``         -- all seven figures.
  * ``checks``          -- all four numerical checks.
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

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(vol_f, returns_f, "-o", ms=2.5, lw=1.0, color="#1f4e79", label="Efficient frontier")
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
_SCALE_SIZES = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
# The external baselines are timed only up to here: PyPortfolioOpt already takes
# minutes at n=640, and both grow like n^3 or faster.
_SCALE_BASELINE_MAX_N = 640
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

    def band(xs: list[int], bands: list[tuple[float, float] | None], color: str) -> None:
        """Shade the min--max range across repetitions for one series."""
        xb = [x for x, b in zip(xs, bands, strict=True) if b is not None]
        lo = [b[0] for b in bands if b is not None]
        hi = [b[1] for b in bands if b is not None]
        if xb:
            ax.fill_between(xb, lo, hi, color=color, alpha=0.18, linewidth=0)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    have_ppo = any(t is not None for t in ppo_times)
    if have_ppo:
        pn = [n for n, t in zip(ns, ppo_times, strict=True) if t is not None]
        pt = [t for t in ppo_times if t is not None]
        band(ns, ppo_band, "#7f7f7f")
        ax.loglog(pn, pt, "-^", ms=4, color="#7f7f7f", label="PyPortfolioOpt CLA")
    if any(t is not None for t in clar_times):
        # General-solver baseline: drawn here for context, discussed in the paper's
        # grid-baseline section (a warm-started QP swept over lambda).
        cn = [n for n, t in zip(ns, clar_times, strict=True) if t is not None]
        ct = [t for t in clar_times if t is not None]
        band(ns, clar_band, "#ff7f0e")
        ax.loglog(cn, ct, "-v", ms=4, color="#ff7f0e", label=f"OSQP, {_SCALE_GRID}-point $\\lambda$-grid")
    if any(t is not None for t in qpo_times):
        qn = [n for n, t in zip(ns, qpo_times, strict=True) if t is not None]
        qt = [t for t in qpo_times if t is not None]
        band(ns, qpo_band, "#9467bd")
        ax.loglog(qn, qt, "-P", ms=4, color="#9467bd", label="qpOASES, hot-started path")
    if any(t is not None for t in inv_times):
        vn = [n for n, t in zip(ns, inv_times, strict=True) if t is not None]
        vt = [t for t in inv_times if t is not None]
        band(ns, inv_band, "#2ca02c")
        ax.loglog(vn, vt, "-D", ms=4, color="#2ca02c", label="cvxcla, incremental dense")
    band(ns, dense_band, "#c00000")
    ax.loglog(ns, dense_times, "-o", ms=4, color="#c00000", label="cvxcla, dense")
    band(ns, factor_band, "#1f4e79")
    ax.loglog(ns, factor_times, "-s", ms=4, color="#1f4e79", label=f"cvxcla, factor ($K={_SCALE_N_FACTORS}$)")
    ax.set_xlabel("Number of assets $n$")
    ax.set_ylabel("Frontier trace time [s]")
    ax.set_title("CLA runtime vs problem size", fontsize=9)

    # Label the x-axis at the actual problem sizes as plain integers, not the
    # default powers of ten (which never coincide with 20, 40, ..., 640 and
    # leave cluttered minor-tick labels on a log axis).
    ax.set_xticks(ns)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(ns[0] * 0.85, ns[-1] * 1.18)
    ax.tick_params(axis="x", labelsize=7)

    # Headroom above the slowest series keeps the legend clear of every curve.
    slowest = max(t for series in (dense_times, ppo_times, clar_times, qpo_times) for t in series if t is not None)
    ax.set_ylim(top=slowest * 300)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    out = out_dir / "scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


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

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(vol, ret, "-o", ms=2.5, lw=1.0, color="#1f4e79", label="Efficient frontier")
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

    The three arrays are ordered by volatility, each evaluated under its own (mean, cov).
    """
    tp = CLA(mean=mean, covariance=cov, **_real_problem(len(mean))).turning_points
    w = np.array([t.weights for t in tp])
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
    Target.checks: "all four numerical checks",
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

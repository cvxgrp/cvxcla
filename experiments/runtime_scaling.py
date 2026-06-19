"""Runtime-scaling experiment for the CLA paper.

Measures the wall-clock time to trace the entire long-only efficient frontier as
a function of the number of assets n, for n in {20, 40, 80, 160, 320, 640}. To
make the dense and factor (Woodbury) backends solve the *identical* matrix -- so
they trace the same frontier and the comparison is a pure per-solve cost
difference -- the covariance is a ground-truth K-factor model
``Sigma = diag(d) + U diag(delta) U^T`` with a fixed number of factors K. The
dense backend receives ``Sigma`` materialised; the factor backend receives the
structured ``FactorCovariance``.

For reference we also time PyPortfolioOpt's ``CLA`` (the Bailey & Lopez de Prado
critical-line implementation) on the same dense problem, when it is installed,
and an incremental-inverse baseline (``InverseCLA`` from ``inverse_cla.py``) that
shares cvxcla's vectorised event logic but maintains ``Sigma_FF^{-1}`` through
rank-1 updates instead of re-solving each step. The baseline-vs-cvxcla-dense
comparison cleanly isolates the linear-algebra strategy (maintained inverse vs
fresh block-eliminated solve), since only that differs between them. (PyPortfolioOpt
differs from both algorithmically --- it recomputes a full inverse per candidate
each step --- so it is reported for context, not as a clean vectorisation control.)
All implementations are timed with the same protocol --- the median of REPEATS
repetitions --- so the comparison is apples-to-apples.

As a further point of reference -- the route a general convex-optimisation user
would take rather than another critical-line implementation -- we optionally time
reconstructing the frontier with Clarabel, the interior-point conic solver CVXPY
selects by default, solving the return-parametrised QP at each of n lambda values
spanning the frontier (the same native-API sweep as ``clarabel_baseline.py``). The
Clarabel curve is drawn on the figure for context only; it is discussed in the
paper's Clarabel-baseline section (a QP swept over lambda), not the runtime-vs-size
text.

Prints a table and, when matplotlib is available, writes docs/paper/scaling.pdf
(log-log runtime vs n for each implementation).

Usage:
    uv run --with pyportfolioopt --with clarabel --with scipy --with matplotlib \
        python experiments/runtime_scaling.py
"""

from __future__ import annotations

import time

import numpy as np

from cvxcla import CLA, FactorCovariance

try:  # the explicit-inverse baseline lives alongside this script
    from inverse_cla import InverseCLA
except ImportError:  # pragma: no cover - allows import from other CWDs
    InverseCLA = None

SIZES = [20, 40, 80, 160, 320, 640]
N_FACTORS = 10
SEED = 7
REPEATS = 3


def make_problem(rng: np.random.Generator, n: int, k: int) -> tuple[np.ndarray, FactorCovariance, dict]:
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


def stats(times: list[float]) -> tuple[float, float, float]:
    """Return (median, min, max) seconds, the centre and min--max band over repetitions."""
    return float(np.median(times)), float(np.min(times)), float(np.max(times))


def median_trace(covariance: object, problem: dict) -> tuple[int, float, float, float]:
    """Return (turning points, median, min, max) trace seconds over REPEATS repetitions."""
    cla = CLA(covariance=covariance, **problem)
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        cla = CLA(covariance=covariance, **problem)
        times.append(time.perf_counter() - start)
    return (len(cla), *stats(times))


def median_trace_inverse(dense: np.ndarray, problem: dict) -> tuple[int, float, float, float] | None:
    """Time the vectorised explicit-inverse baseline over REPEATS repetitions.

    Returns (turning points, median, min, max) seconds or None when the baseline
    module is unavailable. Mirrors ``median_trace`` so the protocol is identical.
    """
    if InverseCLA is None:
        return None
    kw = {
        "mean": problem["mean"],
        "covariance": dense,
        "lower_bounds": problem["lower_bounds"],
        "upper_bounds": problem["upper_bounds"],
    }
    inv = InverseCLA(**kw)
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        inv = InverseCLA(**kw)
        times.append(time.perf_counter() - start)
    return (len(inv), *stats(times))


def pypfopt_trace(dense: np.ndarray, mean: np.ndarray) -> tuple[int, float, float, float] | None:
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
    for _ in range(REPEATS):
        start = time.perf_counter()
        ppo = PYPFOPT_CLA(mean, dense, weight_bounds=(0, 1))
        ppo._solve()  # compute the full critical line (all turning points)
        times.append(time.perf_counter() - start)
        n_pts = len(ppo.w)
    return (n_pts, *stats(times))


def clarabel_trace(dense: np.ndarray, problem: dict) -> tuple[int, float, float, float] | None:
    """Time reconstructing the frontier with Clarabel: n QP solves over a lambda-grid.

    Mirrors ``clarabel_baseline.py`` -- the interior-point conic solver CVXPY
    selects by default, driven through its native API (no modelling layer), solving
    the return-parametrised QP ``min 1/2 w'Sigma w - lambda mu'w`` s.t. ``1'w=1``,
    ``0<=w<=1`` at each of n lambda values spanning the frontier's lambda-range. The
    whole sweep is timed (median of REPEATS) against a single trace, since that is
    the honest cost of recovering the frontier with a general solver. Returns
    (grid points, median, min, max) seconds, or None when Clarabel/scipy are absent.
    """
    try:
        import clarabel
        from scipy import sparse
    except ImportError:
        return None
    n = dense.shape[0]
    mean = problem["mean"]
    # lambda-range from a single cvxcla trace: the finite turning points are the
    # frontier's lambda breakpoints, so [0, lam_max] spans the whole frontier.
    cla = CLA(covariance=dense, **problem)
    lam_max = max(tp.lamb for tp in cla.turning_points if np.isfinite(tp.lamb))
    lams = np.linspace(0.0, lam_max, n)
    p = sparse.triu(dense, format="csc")
    a = sparse.vstack([np.ones((1, n)), sparse.eye(n), -sparse.eye(n)], format="csc")
    b = np.concatenate([[1.0], np.ones(n), np.zeros(n)])
    cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(n), clarabel.NonnegativeConeT(n)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False

    def solve_grid() -> None:
        """One full lambda-sweep: a fresh solver per lambda (no warm start assumed)."""
        for lam in lams:
            clarabel.DefaultSolver(p, -lam * mean, a, b, cones, settings).solve()

    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        solve_grid()
        times.append(time.perf_counter() - start)
    return (len(lams), *stats(times))


def main() -> None:
    """Run the scaling sweep, print a table, and write the figure."""
    ns, dense_times, factor_times, ppo_times, inv_times, clar_times, points = [], [], [], [], [], [], []
    dense_band, factor_band, ppo_band, inv_band, clar_band = [], [], [], [], []
    for n in SIZES:
        rng = np.random.default_rng(SEED)
        dense, factor, problem = make_problem(rng, n, N_FACTORS)
        n_pts, t_dense, d_lo, d_hi = median_trace(dense, problem)
        n_pts_f, t_factor, f_lo, f_hi = median_trace(factor, problem)
        if n_pts != n_pts_f:
            msg = f"backends disagree at n={n}: {n_pts} vs {n_pts_f}"
            raise RuntimeError(msg)
        inv = median_trace_inverse(dense, problem)
        if inv and inv[0] != n_pts:
            msg = f"inverse baseline disagrees at n={n}: {inv[0]} vs {n_pts}"
            raise RuntimeError(msg)
        ppo = pypfopt_trace(dense, problem["mean"])
        clar = clarabel_trace(dense, problem)
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
        inv_str = f"  inverse={inv[1] * 1e3:9.1f} ms" if inv else "  inverse=n/a"
        ppo_str = f"  pypfopt={ppo[1] * 1e3:9.1f} ms (pts={ppo[0]})" if ppo else "  pypfopt=n/a"
        clar_str = f"  clarabel={clar[1] * 1e3:9.1f} ms ({clar[0]} solves)" if clar else "  clarabel=n/a"
        print(
            f"n={n:4d}  points={n_pts:4d}  dense={t_dense * 1e3:8.1f} ms  "
            f"factor={t_factor * 1e3:8.1f} ms  speedup={t_dense / t_factor:5.2f}x{inv_str}{ppo_str}{clar_str}"
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
        if ppo_times[-1] is not None:
            line += f"; dense/pypfopt (overall) = {ppo_times[-1] / dense_times[-1]:.0f}x"
        print(line)

    # Empirical log-log slope (runtime ~ n^p) over the largest few sizes.
    def slope(times: list[float]) -> float:
        x = np.log(np.array(ns[-4:], dtype=float))
        y = np.log(np.array(times[-4:]))
        return float(np.polyfit(x, y, 1)[0])

    print(f"\ndense   empirical exponent p (time ~ n^p): {slope(dense_times):.2f}")
    print(f"factor  empirical exponent p (time ~ n^p): {slope(factor_times):.2f}")
    if all(t is not None for t in inv_times):
        print(f"inverse empirical exponent p (time ~ n^p): {slope([t for t in inv_times if t]):.2f}")
    if all(t is not None for t in ppo_times):
        print(f"pypfopt empirical exponent p (time ~ n^p): {slope([t for t in ppo_times if t]):.2f}")
    if all(t is not None for t in clar_times):
        print(f"clarabel empirical exponent p (time ~ n^p): {slope([t for t in clar_times if t]):.2f}")

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter, ScalarFormatter
    except ImportError:
        print("matplotlib not available - skipping docs/paper/scaling.pdf")
        return

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
        ax.loglog(pn, pt, "-^", ms=4, color="#7f7f7f", label="PyPortfolioOpt CLA (Bailey–López de Prado)")  # noqa: RUF001
    if any(t is not None for t in clar_times):
        # General-solver baseline: drawn here for context, discussed in the paper's
        # Clarabel-baseline section (a QP swept over lambda), not the scaling text.
        cn = [n for n, t in zip(ns, clar_times, strict=True) if t is not None]
        ct = [t for t in clar_times if t is not None]
        band(ns, clar_band, "#ff7f0e")
        ax.loglog(cn, ct, "-v", ms=4, color="#ff7f0e", label="Clarabel, $n$ QP solves over a $\\lambda$-grid")
    if any(t is not None for t in inv_times):
        vn = [n for n, t in zip(ns, inv_times, strict=True) if t is not None]
        vt = [t for t in inv_times if t is not None]
        band(ns, inv_band, "#2ca02c")
        ax.loglog(vn, vt, "-D", ms=4, color="#2ca02c", label="vectorised explicit-inverse baseline")
    band(ns, dense_band, "#c00000")
    ax.loglog(ns, dense_times, "-o", ms=4, color="#c00000", label="cvxcla, dense backend")
    band(ns, factor_band, "#1f4e79")
    ax.loglog(ns, factor_times, "-s", ms=4, color="#1f4e79", label=f"cvxcla, factor backend (Woodbury, K={N_FACTORS})")
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
    ax.tick_params(axis="x", labelsize=8)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = "docs/paper/scaling.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

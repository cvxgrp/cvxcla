"""External baseline: reconstruct the frontier by sweeping a QP solver over lambda.

This is the head-to-head the paper's software comparison calls for (see the
"relation to portfolio-optimization software" discussion): a *general* convex QP
solver, swept over the risk-aversion parameter, against cvxcla's single exact
frontier trace. The solver is Clarabel -- the interior-point conic solver that
CVXPY selects by default -- driven through its **native** Python API, with no
CVXPY modelling layer in between, so the baseline is as fast as this solver gets
(no per-solve canonicalisation overhead) and the comparison is honest.

The contrast is the one the paper draws: cvxcla returns the *entire* efficient
frontier exactly, as a finite set of turning points, from a single call; a QP
solver returns *one* optimiser per solve, so recovering the frontier means
re-solving on a grid of risk-aversion values -- here ``n`` values for ``n`` assets,
a grid as fine as the universe is large -- and the result is still only a
piecewise sampling, not the exact piecewise-linear frontier.

For each universe size ``n`` we, at fixed seed:

1. trace the full long-only frontier once with cvxcla (the dense backend), and
2. solve the return-parametrised QP

       minimize  1/2 w' Sigma w - lambda mu' w
       s.t.      1' w = 1,  0 <= w <= 1

   with Clarabel at each of ``n`` lambda values spanning the frontier's
   lambda-range, timing the whole sweep.

As a correctness cross-check (and a second, independent exactness test alongside
``validate_exact.py``) every Clarabel solution is compared to the exact frontier
weights, obtained by linear interpolation in lambda between cvxcla's turning
points -- exact because the frontier is affine in lambda on each segment.

Prints a table and, when matplotlib is available, writes
docs/paper/clarabel_baseline.pdf (trace time vs n: one cvxcla trace against the
n-solve Clarabel sweep).

Usage:
    uv run --with clarabel --with matplotlib python experiments/clarabel_baseline.py
"""

from __future__ import annotations

import time
from itertools import pairwise

import numpy as np

from cvxcla import CLA

SIZES = [20, 40, 80, 160, 320]
N_FACTORS = 10
SEED = 7
REPEATS = 3


def make_problem(rng: np.random.Generator, n: int, k: int) -> tuple[np.ndarray, dict]:
    """Return (dense Sigma, problem dict) for a size-n, K-factor long-only problem.

    The same factor-model construction as ``runtime_scaling.py`` so the universes
    are comparable across the paper's experiments.
    """
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    dense = np.diag(d) + (u * delta) @ u.T
    problem = {
        "mean": rng.uniform(0.0, 1.0, n),
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }
    return dense, problem


def cla_weight_at(cla: CLA, lam: float) -> np.ndarray:
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


def median_time(fn: object, repeats: int = REPEATS) -> float:
    """Median wall-clock seconds of calling ``fn`` ``repeats`` times."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()  # type: ignore[operator]
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def main() -> None:
    """Run the lambda-sweep baseline against cvxcla and report the gap."""
    try:
        import clarabel
        from scipy import sparse
    except ImportError:
        print("clarabel/scipy not installed; skipping the Clarabel baseline.")
        print("  install with:  uv run --with clarabel --with matplotlib python experiments/clarabel_baseline.py")
        return

    def solve_grid(dense: np.ndarray, mean: np.ndarray, lams: np.ndarray) -> list[np.ndarray]:
        """Solve the QP once per lambda in ``lams`` with a fresh Clarabel solver each time.

        Only the linear term ``q = -lambda mu`` changes between solves; ``P`` and the
        constraint matrix are built once. A fresh solver per lambda is the faithful
        cost of reconstructing the frontier on a grid (no warm start is assumed).
        """
        n = dense.shape[0]
        p = sparse.triu(dense, format="csc")
        a = sparse.vstack([np.ones((1, n)), sparse.eye(n), -sparse.eye(n)], format="csc")
        b = np.concatenate([[1.0], np.ones(n), np.zeros(n)])
        cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(n), clarabel.NonnegativeConeT(n)]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        out = []
        for lam in lams:
            solver = clarabel.DefaultSolver(p, -lam * mean, a, b, cones, settings)
            out.append(np.array(solver.solve().x))
        return out

    ns, cla_ms, clar_ms, ratios, errs, points = [], [], [], [], [], []
    for n in SIZES:
        rng = np.random.default_rng(SEED)
        dense, problem = make_problem(rng, n, N_FACTORS)

        cla = CLA(covariance=dense, **problem)
        lam_max = max(tp.lamb for tp in cla.turning_points if np.isfinite(tp.lamb))
        lams = np.linspace(0.0, lam_max, n)  # n grid points for n assets

        t_cla = median_time(lambda d=dense, p=problem: CLA(covariance=d, **p))
        t_clar = median_time(lambda d=dense, m=problem["mean"], la=lams: solve_grid(d, m, la))

        # Exactness cross-check: Clarabel grid solution vs the exact frontier weights.
        grid = solve_grid(dense, problem["mean"], lams)
        err = max(float(np.max(np.abs(w - cla_weight_at(cla, lam)))) for w, lam in zip(grid, lams, strict=True))

        ns.append(n)
        points.append(len(cla))
        cla_ms.append(t_cla * 1e3)
        clar_ms.append(t_clar * 1e3)
        ratios.append(t_clar / t_cla)
        errs.append(err)
        print(
            f"n={n:4d}  points={len(cla):4d}  cvxcla(full frontier)={t_cla * 1e3:8.1f} ms   "
            f"clarabel({n} solves)={t_clar * 1e3:9.1f} ms   "
            f"({t_clar * 1e3 / n:5.2f} ms/solve)  ratio={t_clar / t_cla:6.1f}x  max|w-w*|={err:.1e}"
        )

    def slope(values: list[float]) -> float:
        x = np.log(np.array(ns[-4:], dtype=float))
        y = np.log(np.array(values[-4:]))
        return float(np.polyfit(x, y, 1)[0])

    print(f"\ncvxcla   empirical exponent p (time ~ n^p): {slope(cla_ms):.2f}")
    print(f"clarabel empirical exponent p (n solves)   : {slope(clar_ms):.2f}")
    print(f"max weight discrepancy (Clarabel vs exact frontier): {max(errs):.1e}")

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter, ScalarFormatter
    except ImportError:
        print("matplotlib not available - skipping docs/paper/clarabel_baseline.pdf")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.loglog(ns, clar_ms, "-^", ms=4, color="#7f7f7f", label="Clarabel, $n$ QP solves over a $\\lambda$-grid")
    ax.loglog(ns, cla_ms, "-o", ms=4, color="#c00000", label="cvxcla, one exact frontier trace")
    ax.set_xlabel("Number of assets $n$")
    ax.set_ylabel("Time to obtain the frontier [ms]")
    ax.set_title("Exact frontier trace vs a $\\lambda$-swept QP baseline", fontsize=9)
    ax.set_xticks(ns)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(ns[0] * 0.85, ns[-1] * 1.18)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = "docs/paper/clarabel_baseline.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

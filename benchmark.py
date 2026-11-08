# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cvxcla==2.7.1",
#     "numpy==2.4.6",
#     # scikit-base (a PyPortfolioOpt dependency) imports packaging without declaring it.
#     "packaging==26.3",
#     "pyportfolioopt==1.6.0",
# ]
# ///
r"""cvxcla vs PyPortfolioOpt: a community benchmark of the Critical Line Algorithm.

Run it with one command, no clone and no virtual environment needed::

    uv run https://raw.githubusercontent.com/cvxgrp/cvxcla/main/benchmark.py

and paste the block it prints at the end into
https://github.com/cvxgrp/cvxcla/discussions/1010.

Both libraries compute the *entire* efficient frontier of

    max  lambda mu'w - 1/2 w' Sigma w   s.t.  1'w = 1,  lb <= w <= ub

as a finite list of turning points. PyPortfolioOpt's ``CLA`` is the open-source code
of Bailey and Lopez de Prado (2013); cvxcla is a reformulation built on one KKT
solve per turning point. Every scenario below is in PyPortfolioOpt's own problem
class (dense covariance, box bounds, one budget row), so neither side is given a
feature the other lacks; the last scenario is a cvxcla-only capability demo and is
reported separately.

Speed is only worth comparing between correct answers, so each trace is also
checked, with numpy alone and independently of both libraries:

``kkt``     at every finite turning point, the weights are feasible and satisfy the
            optimality conditions of the problem at that point's own lambda (one
            multiplier for the budget row, signed ones for the bounds); the worst
            violation is reported, relative to the size of the gradient.
``agree``   the two frontiers are the same set of portfolios: every turning point of
            each lies within ``1e-6`` (max-norm) of the other's frontier, the
            polyline through its turning points. A lost turning point makes the
            other library's vertex sit off that polyline, so it shows here, and
            the report names the library that lost it.

Each run happens in a freshly spawned interpreter with a timeout (the PyPortfolioOpt
loop has no iteration cap, so a cycling trace would otherwise never return). The
time reported is the median of ``--repeats`` traces after one untimed warm-up
(a trace slower than five seconds is timed once, without a warm-up).

Usage::

    uv run benchmark.py                 # the default scenarios (a few minutes)
    uv run benchmark.py --quick         # smaller sizes, for a first look
    uv run benchmark.py --threads 1     # single-threaded BLAS
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import multiprocessing
import os
import platform
import statistics
import subprocess  # nosec B404
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

AGREE_TOL = 1e-6
KKT_TOL = 1e-6
FEAS_TOL = 1e-8
SLOW = 5.0  # seconds; a warm-up slower than this is not repeated
RESULTS_URL = "https://github.com/cvxgrp/cvxcla/discussions/1010"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# ======================================================================================
# Problems
# ======================================================================================
@dataclass(frozen=True)
class Scenario:
    """A named problem generator; ``head_to_head`` is False for cvxcla-only scenarios."""

    name: str
    description: str
    n: int
    build: Callable[[np.random.Generator, int], dict[str, np.ndarray]]
    head_to_head: bool = True


def _factor_model(rng: np.random.Generator, n: int, k: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(d, u, delta)`` of a K-factor covariance ``diag(d) + U diag(delta) U'``."""
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    return d, u, delta


def factor(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Long-only, fully invested; a 10-factor covariance and distinct means."""
    d, u, delta = _factor_model(rng, n)
    return {"mean": rng.uniform(0.0, 1.0, n), "cov": np.diag(d) + (u * delta) @ u.T, "ub": np.ones(n)}


def short_sample(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """A sample covariance from fewer observations than assets (T = n/2): Sigma is singular."""
    returns = rng.standard_normal((n // 2, n)) @ np.diag(rng.uniform(0.1, 0.3, n))
    return {"mean": rng.uniform(0.0, 0.1, n), "cov": np.cov(returns, rowvar=False), "ub": np.ones(n)}


def tied_means(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Expected returns in five exactly tied groups, as from a rating or a sector view."""
    problem = factor(rng, n)
    levels = rng.uniform(0.0, 1.0, 5)
    return {**problem, "mean": rng.permutation(np.resize(levels, n))}


def capped(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Position limits of 5/n per asset, so many bounds bind along the frontier."""
    return {**factor(rng, n), "ub": np.full(n, 5.0 / n)}


def scenarios(quick: bool) -> list[Scenario]:
    """The scenario list; ``quick`` keeps every kind but shrinks the sizes."""
    sizes = (50, 100, 200) if quick else (50, 100, 200, 400)
    out = [Scenario(f"factor-{n}", "10-factor covariance, long-only", n, factor) for n in sizes]
    m = 100 if quick else 200
    out += [
        Scenario(f"short-sample-{m}", "sample covariance, T = n/2 (singular)", m, short_sample),
        Scenario(f"tied-means-{m}", "expected returns in 5 tied groups", m, tied_means),
        Scenario(f"capped-{m}", "upper bound 5/n per asset", m, capped),
        Scenario("factor-op-5000", "cvxcla FactorCovariance, n x n never formed", 5000, factor, head_to_head=False),
    ]
    return out


# ======================================================================================
# The two libraries
# ======================================================================================
def trace_pypfopt(problem: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return (turning points as rows, lambdas) from PyPortfolioOpt's CLA."""
    from pypfopt.cla import CLA

    n = len(problem["mean"])
    cla = CLA(problem["mean"], problem["cov"], weight_bounds=(np.zeros(n), problem["ub"]))
    cla._solve()  # the frontier itself; max_sharpe() etc. call this first
    lambdas = np.array([np.inf if lam is None else float(lam) for lam in cla.ls])
    return np.array([w.ravel() for w in cla.w]), lambdas


def trace_cvxcla(problem: dict[str, np.ndarray], factor_operator: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (turning points as rows, lambdas) from cvxcla's CLA."""
    from cvxcla import CLA, FactorCovariance

    n = len(problem["mean"])
    covariance = FactorCovariance(**problem["factors"]) if factor_operator else problem["cov"]
    cla = CLA(
        mean=problem["mean"],
        covariance=covariance,
        lower_bounds=np.zeros(n),
        upper_bounds=problem["ub"],
        a=np.ones((1, n)),
        b=np.ones(1),
    )
    return np.array([tp.weights for tp in cla.turning_points]), np.array([tp.lamb for tp in cla.turning_points])


# ======================================================================================
# Independent checks (numpy only)
# ======================================================================================
def kkt_violation(points: np.ndarray, lambdas: np.ndarray, problem: dict[str, np.ndarray]) -> float:
    """Worst feasibility or optimality violation over the finite-lambda turning points.

    With ``g = lambda mu - Sigma w`` the conditions are: a single ``nu`` with
    ``g_i = nu`` on the free assets, ``g_i <= nu`` at the lower and ``g_i >= nu`` at
    the upper bound. Such a ``nu`` exists iff ``max(g_free, g_lower) <= min(g_free,
    g_upper)``; the shortfall, scaled by the size of the gradient, is the violation.
    """
    mean, cov, ub = problem["mean"], problem["cov"], problem["ub"]
    worst = 0.0
    for w, lam in zip(points, lambdas, strict=True):
        worst = max(worst, float(np.max(-w)), float(np.max(w - ub)), abs(float(w.sum()) - 1.0))
        if not np.isfinite(lam):
            continue
        g = lam * mean - cov @ w
        at_lb, at_ub = w <= FEAS_TOL, w >= ub - FEAS_TOL
        free = ~(at_lb | at_ub)
        lo = np.max(g[free | at_lb], initial=-np.inf)
        hi = np.min(g[free | at_ub], initial=np.inf)
        scale = 1.0 + np.max(np.abs(lam * mean)) + np.max(np.abs(cov @ w))
        worst = max(worst, float(max(0.0, lo - hi) / scale))
    return worst


def polyline_distance(points: np.ndarray, vertices: np.ndarray) -> float:
    """Largest distance (max-norm) from a point to the polyline through ``vertices``.

    Each point is projected onto every segment in the Euclidean norm and the max-norm
    of the residual taken: an upper bound on the true max-norm distance, so a small
    value certifies that the point lies on the polyline.
    """
    a, b = vertices[:-1], vertices[1:]
    ab = b - a
    length2 = np.maximum((ab * ab).sum(axis=1), 1e-300)
    worst = 0.0
    for p in points:
        t = np.clip(((p - a) * ab).sum(axis=1) / length2, 0.0, 1.0)
        residual = np.abs(a + t[:, None] * ab - p).max(axis=1)
        nearest = min(float(residual.min()), float(np.abs(vertices - p).max(axis=1).min()))
        worst = max(worst, nearest)
    return worst


def compare(pp: np.ndarray, cx: np.ndarray) -> str:
    """Say whether the two frontiers agree and, if not, which one is missing a vertex.

    If every point of A lies on B's polyline but not vice versa, B has a turning point
    that A's polyline cuts across: A lost it, and A's frontier is not efficient there.
    """
    pp_off, cx_off = polyline_distance(pp, cx), polyline_distance(cx, pp)
    if max(pp_off, cx_off) <= AGREE_TOL:
        return "yes"
    if pp_off <= AGREE_TOL:
        return f"pypfopt misses a vertex ({cx_off:.0e})"
    if cx_off <= AGREE_TOL:
        return f"cvxcla misses a vertex ({pp_off:.0e})"
    return f"differ ({max(pp_off, cx_off):.0e})"


# ======================================================================================
# Workers (each call runs in a freshly spawned interpreter)
# ======================================================================================
def worker(library: str, scenario: Scenario, seed: int, repeats: int) -> dict[str, object]:
    """Build the problem, time ``repeats`` traces after a warm-up, and check the last one."""
    rng = np.random.default_rng(seed)
    factor_operator = not scenario.head_to_head
    if factor_operator:  # only the factors: the n x n matrix is never formed
        d, u, delta = _factor_model(rng, scenario.n)
        problem = {
            "mean": rng.uniform(0.0, 1.0, scenario.n),
            "ub": np.ones(scenario.n),
            "factors": {"d": d, "u": u, "delta": delta},
        }
    else:
        problem = scenario.build(rng, scenario.n)

    def run() -> tuple[np.ndarray, np.ndarray]:
        if library == "pypfopt":
            return trace_pypfopt(problem)
        return trace_cvxcla(problem, factor_operator=factor_operator)

    try:
        start = time.perf_counter()
        points, lambdas = run()
        times = [time.perf_counter() - start]
        if times[0] > SLOW:  # a slow trace: its warm-up is the measurement
            repeats = 0
        else:
            times = []
        for _ in range(repeats):
            start = time.perf_counter()
            points, lambdas = run()
            times.append(time.perf_counter() - start)
    except Exception as exc:  # report a failure as a result rather than crash  # noqa: BLE001
        return {"status": type(exc).__name__}
    kkt = np.nan if factor_operator else kkt_violation(points, lambdas, problem)
    return {"status": "ok", "time": statistics.median(times), "points": points, "kkt": kkt}


def in_fresh_process(timeout: float, *args: object) -> dict[str, object]:
    """Run ``worker(*args)`` in a spawned interpreter; kill it after ``timeout`` seconds."""
    pool = multiprocessing.get_context("spawn").Pool(1)
    try:
        return pool.apply_async(worker, args).get(timeout)
    except multiprocessing.TimeoutError:
        return {"status": "timeout"}
    finally:
        pool.terminate()
        pool.join()


# ======================================================================================
# Reporting
# ======================================================================================
def environment() -> list[str]:
    """Describe the machine and numerical stack the timings were taken on."""
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        out = subprocess.run(  # nosec B603
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
        )
        cpu = out.stdout.strip() or cpu
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), cpu)
        except OSError:
            pass
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    threads = ", ".join(f"{k}={os.environ[k]}" for k in THREAD_VARS if k in os.environ) or "default"
    versions = ", ".join(f"{p} {md.version(p)}" for p in ("cvxcla", "pyportfolioopt", "numpy"))
    return [
        f"cpu:      {cpu} ({os.cpu_count()} logical cores)",
        f"system:   {platform.platform()}, Python {platform.python_version()}",
        f"packages: {versions}",
        f"blas:     {blas['name']} {blas['version']}, threads: {threads}",
    ]


def _seconds(result: dict[str, object]) -> str:
    return f"{result['time']:9.4f}" if result["status"] == "ok" else f"{result['status']:>9}"


def _kkt(result: dict[str, object]) -> str:
    if result["status"] != "ok" or np.isnan(result["kkt"]):
        return "      -"
    return "     ok" if result["kkt"] <= KKT_TOL else f"{result['kkt']:7.0e}"


def main() -> None:
    """Run every scenario with both libraries and print a pasteable report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--quick", action="store_true", help="smaller problem sizes")
    parser.add_argument("--repeats", type=int, default=3, help="timed traces per run (default 3)")
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per run (default 300)")
    parser.add_argument("--seed", type=int, default=2026, help="problem seed (default 2026)")
    parser.add_argument("--threads", type=int, help="limit BLAS threads (e.g. 1)")
    args = parser.parse_args()
    if args.threads is not None:  # inherited by the spawned workers
        for var in THREAD_VARS:
            os.environ[var] = str(args.threads)

    columns = f"{'pypfopt s':>9} {'cvxcla s':>9} {'speedup':>8} {'kkt pp':>7} {'kkt cx':>7}"
    header = f"{'scenario':<17} {'n':>5} {columns}  agree"
    lines = [header, "-" * len(header)]
    print("cvxcla vs PyPortfolioOpt: the Critical Line Algorithm, whole frontier\n", flush=True)
    print(header, flush=True)
    print(lines[1], flush=True)
    for scenario in scenarios(args.quick):
        cx = in_fresh_process(args.timeout, "cvxcla", scenario, args.seed, args.repeats)
        if scenario.head_to_head:
            pp = in_fresh_process(args.timeout, "pypfopt", scenario, args.seed, args.repeats)
        else:
            pp = {"status": "n/a"}
        both = pp["status"] == "ok" and cx["status"] == "ok"
        speedup = f"{pp['time'] / cx['time']:7.0f}x" if both else f"{'-':>8}"
        agree = compare(pp["points"], cx["points"]) if both else "-"
        columns = f"{_seconds(pp)} {_seconds(cx)} {speedup} {_kkt(pp)} {_kkt(cx)}"
        line = f"{scenario.name:<17} {scenario.n:>5} {columns}  {agree}"
        lines.append(line)
        print(line, flush=True)

    env = environment()
    print(
        "\nkkt pp / kkt cx: optimality check of each turning point, independent of both libraries;"
        "\nagree: both libraries return the same frontier (same set of portfolios) to 1e-6;"
        "\n       'misses a vertex' means that library's frontier cuts a corner off the other's."
        "\nAn exception name in a time column means that library raised it."
        "\nfactor-op-5000 is cvxcla only: PyPortfolioOpt needs the dense n x n matrix."
        "\nHow much of the speedup survives fixing PyPortfolioOpt's bugs: experiments/bldp_fair.py.\n"
    )
    print(f"Please paste this into {RESULTS_URL} :\n")
    print("```text")
    print("\n".join([*env, f"options:  seed={args.seed} repeats={args.repeats} quick={args.quick}", "", *lines]))
    print("```")


if __name__ == "__main__":
    main()

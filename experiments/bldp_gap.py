# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cvxcla==2.6.0",
#     "matplotlib==3.11.0",
#     "numpy==2.4.6",
#     "pyportfolioopt==1.6.0",
# ]
# ///
r"""Reproduce and attribute the runtime gap between BLdP's CLA and cvxcla (issue #869, track 1).

``comparison.md`` and the paper attribute a large runtime gap (about 340x at
n = 640) between the Bailey--Lopez de Prado (BLdP) CLA, as shipped by
PyPortfolioOpt, and cvxcla's dense backend to one structural fact: the branch
that decides which blocked asset re-enters the free set calls
``np.linalg.inv`` once per *candidate*, i.e. O(n) dense inverses per turning
point. This script checks that reading and then measures how much of the gap it
actually explains. It has three parts.

1. **Code reading.** Parse the source of ``pypfopt.cla.CLA._solve`` and report,
   for each ``np.linalg.inv`` call, whether it sits inside a ``for`` loop and
   which branch it belongs to. (PyPortfolioOpt names the branches by what happens
   to the *weight*: "case b) Free one bounded weight" is the addition branch,
   "case a) Bound one free weight" the removal branch.)

2. **Attribution ladder.** Time a sequence of implementations that each change
   one thing relative to the previous rung, all on the same problems:

   ``bldp``         PyPortfolioOpt 1.6.0 ``CLA``, unmodified (BLdP as shipped).
   ``bordered``     The same ``_solve`` loop, but in the addition branch the
                    inverse of each candidate block ``Sigma[f+i, f+i]`` is a rank-one
                    bordering of one ``inv(Sigma[f, f])`` per turning point, O(n_F^2)
                    per candidate instead of a fresh O(n_F^3) inverse. The Python
                    loop over candidates, the slicing and the per-candidate lambda
                    formula are kept.
   ``vec-fresh``    Vectorised event search (one matrix expression over all
                    assets, no Python loop over candidates) with a fresh
                    ``np.linalg.inv`` of the free block per turning point
                    (``InverseCLA(fresh=True)`` from ``inverse_cla.py``).
   ``vec-incr``     The same, with the free-block inverse maintained by rank-one
                    border/deletion updates (``InverseCLA``).
   ``cvxcla``       ``cvxcla.CLA`` with the dense backend (a fresh block-eliminated
                    KKT solve per turning point, plus its feasibility and
                    degeneracy guards).

   The consecutive ratios telescope: bldp/cvxcla = (bldp/bordered) x
   (bordered/vec-fresh) x (vec-fresh/cvxcla). The first factor is the
   inverse-in-the-loop, the second the Python-loop event search (with its
   per-candidate slicing), the third whatever remains between a minimal
   vectorised CLA and cvxcla.

3. **Phase profile.** One instrumented copy of the shipped ``_solve`` (same
   operations, wrapped in timers) splits the BLdP trace time into the addition
   branch (inverses / slicing / lambda formula), the removal branch, the step
   update and the final purges, and counts the inverses it performs.

Every variant is checked against ``cvxcla``: each of its turning points must lie
on cvxcla's frontier (max-norm distance to the nearest cvxcla turning point).

**BLdP can cycle.** Its re-entry test ``lam < self.ls[-1]`` has no tolerance and
its loop has no iteration cap. When an asset leaves the free set and its re-entry
lambda comes out a few ulps *below* the exit lambda, it re-enters, leaves again,
and the trace never terminates. The shipped code happens to avoid this on most of
these problems, but the bordered rung's different round-off (about 1e-15 in
lambda) triggers it on some seeds. A run that cycles is reported, not timed: the
instrumented copy stops after ``10 (n + 1)`` steps, and every measurement runs
under a hard ``--timeout``. Ratios use only the seeds on which both rungs finished.

Problems are the K = 10 factor-model, long-only, fully-invested problems of
``make_figures.py scaling`` (seed 7 there; seeds 7, 8, ... here), so the n = 640
point is directly comparable to the paper's figure. Each (variant, n, seed) is
measured in a freshly spawned interpreter, traced once untimed (warm-up and
verification), then timed ``--repeats`` times; the per-seed median is the
measurement. Error bars are the min--max of those medians over seeds, i.e. they
show problem-instance variation, which dominates timer noise here.

Usage:
    uv run experiments/bldp_gap.py                       # n = 20..320, 3 seeds
    uv run experiments/bldp_gap.py --sizes 20 40 80 160 320 640 --seeds 3
    uv run experiments/bldp_gap.py --threads 1           # single-threaded BLAS (all workers)
    uv run experiments/bldp_gap.py --json out.json --figure out.pdf

The hardware, Python, numpy and BLAS configuration are printed with the results;
a timing is only meaningful together with them.
"""

# The instrumented ``_solve`` below mirrors PyPortfolioOpt line for line, so it keeps
# upstream's names (covarF, wB, ...) and its manual index counter to stay diffable.
# ruff: noqa: N806, N811

from __future__ import annotations

import argparse
import ast
import inspect
import json
import multiprocessing
import os
import platform
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

VARIANTS = ["bldp", "bordered", "vec-fresh", "vec-incr", "cvxcla"]


class CycleError(RuntimeError):
    """Raised when the instrumented BLdP loop exceeds its step cap (it is cycling)."""


N_FACTORS = 10
BASE_SEED = 7


# ======================================================================================
# Problems
# ======================================================================================
def make_problem(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (dense covariance, mean) of the ``make_figures.py scaling`` factor model."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, N_FACTORS)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, N_FACTORS) * n
    d = rng.uniform(0.5, 2.0, n)
    dense = np.diag(d) + (u * delta) @ u.T
    mean = rng.uniform(0.0, 1.0, n)
    return dense, mean


# ======================================================================================
# Part 1: code reading
# ======================================================================================
def _is_inv_call(node: ast.AST) -> bool:
    """Return True for a call spelled ``np.linalg.inv(...)``."""
    return isinstance(node, ast.Call) and ast.unparse(node.func) in {"np.linalg.inv", "numpy.linalg.inv"}


def read_inverse_sites() -> list[dict[str, object]]:
    """Locate every ``np.linalg.inv`` call in ``pypfopt.cla.CLA._solve``.

    For each call, report its source line, whether it is lexically inside a
    ``for`` loop (and the loop's header), and the branch comment that precedes it.
    """
    from pypfopt.cla import CLA as PyPfOptCLA

    source = textwrap.dedent(inspect.getsource(PyPfOptCLA._solve))
    lines = source.splitlines()
    tree = ast.parse(source)
    sites: list[dict[str, object]] = []

    def visit(node: ast.AST, loops: list[ast.For]) -> None:
        for child in ast.iter_child_nodes(node):
            if _is_inv_call(child):
                # The nearest preceding "# n) case ..." style comment names the branch.
                branch = next(
                    (
                        lines[k].strip().lstrip("# ")
                        for k in range(child.lineno - 1, -1, -1)
                        if lines[k].strip().startswith("# ") and ")" in lines[k]
                    ),
                    "?",
                )
                sites.append(
                    {
                        "line": child.lineno,
                        "code": lines[child.lineno - 1].strip(),
                        "branch": branch,
                        "in_loop": bool(loops),
                        "loop": f"for {ast.unparse(loops[-1].target)} in {ast.unparse(loops[-1].iter)}"
                        if loops
                        else "",
                    }
                )
            visit(child, [*loops, child] if isinstance(child, ast.For) else loops)

    visit(tree, [])
    return sites


# ======================================================================================
# Part 2/3: BLdP variants (subclasses of the shipped PyPortfolioOpt CLA)
# ======================================================================================
def _bldp_classes() -> tuple[type, type]:
    """Build the ``bordered`` and instrumented subclasses of PyPortfolioOpt's CLA.

    Defined lazily so that a worker only imports PyPortfolioOpt when it needs it.
    Both reproduce ``pypfopt.cla.CLA._solve`` (v1.6.0) statement for statement; the
    only differences are marked ``# CHANGED``.
    """
    from pypfopt.cla import CLA as PyPfOptCLA

    class InstrumentedBLdP(PyPfOptCLA):
        """``_solve`` of PyPortfolioOpt 1.6.0 with phase timers and an inverse counter.

        With ``bordered=True`` the addition branch borders one inverse per turning
        point instead of inverting every candidate block from scratch.
        """

        def __init__(self, *args, bordered: bool = False, timed: bool = False, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.bordered = bordered
            self.timed = timed
            self.phases: dict[str, float] = defaultdict(float)
            self.inverses = 0  # full np.linalg.inv calls
            self.borders = 0  # rank-one bordered inverses (bordered=True only)

        def _clock(self, phase: str, start: float) -> float:
            now = time.perf_counter()
            if self.timed:
                self.phases[phase] += now - start
            return now

        def _inv(self, matrix: np.ndarray) -> np.ndarray:
            self.inverses += 1
            return np.linalg.inv(matrix)

        def _solve(self) -> None:
            t = time.perf_counter()
            f, w = self._init_algo()
            self.w.append(np.copy(w))
            self.ls.append(None)
            self.g.append(None)
            self.f.append(f[:])
            t = self._clock("init", t)
            cap = 10 * (self.mean.shape[0] + 1)  # CHANGED: BLdP itself has no cap
            while True:
                if len(self.ls) > cap:
                    msg = f"no termination after {cap} steps"
                    raise CycleError(msg)
                # 1) case a): Bound one free weight  (removal branch)
                l_in = None
                covarF_inv = None
                if len(f) > 1:
                    covarF, covarFB, meanF, wB = self._get_matrices(f)
                    covarF_inv = self._inv(covarF)
                    j = 0
                    for i in f:
                        lam, bi = self._compute_lambda(covarF_inv, covarFB, meanF, wB, j, [self.lB[i], self.uB[i]])
                        if PyPfOptCLA._infnone(lam) > PyPfOptCLA._infnone(l_in):
                            l_in, i_in, bi_in = lam, i, bi
                        j += 1  # noqa: SIM113
                t = self._clock("remove", t)
                # 2) case b): Free one bounded weight  (addition branch)
                l_out = None
                if len(f) < self.mean.shape[0]:
                    b = self._get_b(f)
                    if self.bordered:  # CHANGED: one inverse of the current free block
                        if covarF_inv is None:
                            covarF_inv = self._inv(self._reduce_matrix(self.cov_matrix, f, f))
                        base_inv = covarF_inv
                        t = self._clock("add.inverse", t)
                    for i in b:
                        covarF, covarFB, meanF, wB = self._get_matrices([*f, i])
                        t = self._clock("add.slice", t)
                        if self.bordered:  # CHANGED: rank-one border of base_inv
                            self.borders += 1
                            c = covarF[:-1, -1]
                            v = base_inv @ c
                            schur = covarF[-1, -1] - c @ v
                            col = (-v / schur).reshape(-1, 1)
                            covarF_inv = np.block(
                                [[base_inv + np.outer(v, v) / schur, col], [col.T, np.array([[1.0 / schur]])]]
                            )
                        else:
                            covarF_inv = self._inv(covarF)
                        t = self._clock("add.inverse", t)
                        lam, bi = self._compute_lambda(
                            covarF_inv, covarFB, meanF, wB, meanF.shape[0] - 1, self.w[-1][i]
                        )
                        if (self.ls[-1] is None or lam < self.ls[-1]) and lam > PyPfOptCLA._infnone(l_out):
                            l_out, i_out = lam, i
                        t = self._clock("add.lambda", t)
                if (l_in is None or l_in < 0) and (l_out is None or l_out < 0):
                    # 3) compute minimum variance solution
                    self.ls.append(0)
                    covarF, covarFB, meanF, wB = self._get_matrices(f)
                    covarF_inv = self._inv(covarF)
                    meanF = np.zeros(meanF.shape)
                else:
                    # 4) decide lambda
                    if PyPfOptCLA._infnone(l_in) > PyPfOptCLA._infnone(l_out):
                        self.ls.append(l_in)
                        f.remove(i_in)
                        w[i_in] = bi_in
                    else:
                        self.ls.append(l_out)
                        f.append(i_out)
                    covarF, covarFB, meanF, wB = self._get_matrices(f)
                    covarF_inv = self._inv(covarF)
                # 5) compute solution vector
                wF, g = self._compute_w(covarF_inv, covarFB, meanF, wB)
                for i in range(len(f)):
                    w[f[i]] = wF[i]
                self.w.append(np.copy(w))
                self.g.append(g)
                self.f.append(f[:])
                t = self._clock("step", t)
                if self.ls[-1] == 0:
                    break
            # 6) Purge turning points
            self._purge_num_err(10e-10)
            t = self._clock("purge.num_err", t)
            self._purge_excess()
            self._clock("purge.excess", t)

    return PyPfOptCLA, InstrumentedBLdP


def run_variant(variant: str, dense: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Trace the full frontier with one variant; return its turning points (rows)."""
    n = len(mean)
    if variant in {"bldp", "bordered"}:
        shipped, instrumented = _bldp_classes()
        cla = (
            shipped(mean, dense, weight_bounds=(0, 1))
            if variant == "bldp"
            else instrumented(mean, dense, weight_bounds=(0, 1), bordered=True)
        )
        cla._solve()
        return np.array([w.ravel() for w in cla.w])
    if variant in {"vec-fresh", "vec-incr"}:
        from inverse_cla import InverseCLA

        inv = InverseCLA(
            mean=mean,
            covariance=dense,
            lower_bounds=np.zeros(n),
            upper_bounds=np.ones(n),
            fresh=variant == "vec-fresh",
        )
        return np.array(inv.weights)
    if variant == "cvxcla":
        from cvxcla import CLA

        cla = CLA(
            mean=mean,
            covariance=dense,
            lower_bounds=np.zeros(n),
            upper_bounds=np.ones(n),
            a=np.ones((1, n)),
            b=np.ones(1),
        )
        return np.array([tp.weights for tp in cla.turning_points])
    msg = f"unknown variant {variant!r}"
    raise ValueError(msg)


def frontier_distance(points: np.ndarray, reference: np.ndarray) -> float:
    """Return the largest max-norm distance from a turning point to the nearest reference point."""
    gaps = np.abs(points[:, None, :] - reference[None, :, :]).max(axis=2)
    return float(gaps.min(axis=1).max())


# ======================================================================================
# Workers (each call runs in a freshly spawned interpreter)
# ======================================================================================
def measure(variant: str, n: int, seed: int, repeats: int) -> dict[str, object]:
    """Warm up, verify against cvxcla, then time ``repeats`` traces of one variant."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # for inverse_cla
    dense, mean = make_problem(n, seed)
    try:
        points = run_variant(variant, dense, mean)
    except CycleError:
        return {"variant": variant, "n": n, "seed": seed, "status": "cycled"}
    reference = points if variant == "cvxcla" else run_variant("cvxcla", dense, mean)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_variant(variant, dense, mean)
        times.append(time.perf_counter() - start)
    return {
        "variant": variant,
        "n": n,
        "seed": seed,
        "status": "ok",
        "points": len(points),
        "reference_points": len(reference),
        "distance": frontier_distance(points, reference),
        "times": times,
        "median": float(np.median(times)),
    }


def profile(n: int, seed: int) -> dict[str, object] | None:
    """Run the instrumented shipped ``_solve`` once and return its phase split."""
    dense, mean = make_problem(n, seed)
    _, instrumented = _bldp_classes()
    cla = instrumented(mean, dense, weight_bounds=(0, 1), timed=True)
    start = time.perf_counter()
    try:
        cla._solve()
    except CycleError:
        return None
    total = time.perf_counter() - start
    steps = len(cla.ls) - 1  # loop iterations (before purging)
    return {
        "n": n,
        "seed": seed,
        "total": total,
        "phases": dict(cla.phases),
        "inverses": cla.inverses,
        "steps": steps,
        "points": len(cla.w),
    }


# Environment variables the common BLAS builds read their thread count from at load
# time; set before the workers are spawned, so each worker's numpy picks them up.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _fresh(timeout: float, fn: object, *args: object) -> object | None:
    """Run ``fn(*args)`` in its own spawned interpreter (no state carried between runs).

    Returns None if it has not finished after ``timeout`` seconds; the worker is
    then killed, which is the only way to stop an unmodified BLdP trace that cycles.
    """
    pool = multiprocessing.get_context("spawn").Pool(1)
    try:
        return pool.apply_async(fn, args).get(timeout)
    except multiprocessing.TimeoutError:
        return None
    finally:
        pool.terminate()
        pool.join()


# ======================================================================================
# Reporting
# ======================================================================================
def environment(threads: int | None) -> dict[str, object]:
    """Describe the machine and numerical stack the timings were taken on."""
    import importlib.metadata as md

    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        import subprocess  # nosec B404

        out = subprocess.run(  # nosec B603
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
        )
        cpu = out.stdout.strip() or cpu
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "python": platform.python_version(),
        "packages": {p: md.version(p) for p in ("numpy", "pyportfolioopt", "cvxcla")},
        "blas": f"{blas['name']} {blas['version']}",
        "threads_limit": threads,
        "thread_env": {k: os.environ[k] for k in _THREAD_VARS if k in os.environ},
    }


def summarise(results: list[dict[str, object]]) -> dict[tuple[str, int], dict[str, float]]:
    """Collapse per-seed medians to (median, min, max) over seeds per (variant, n)."""
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in results:
        if r["status"] == "ok":
            groups[(r["variant"], r["n"])].append(r["median"])
    return {k: {"median": float(np.median(v)), "lo": float(min(v)), "hi": float(max(v))} for k, v in groups.items()}


def seed_ratios(results: list[dict[str, object]], num: str, den: str, n: int) -> list[float]:
    """Return the per-seed time ratio num/den at size n (same problem instance for both)."""
    by = {(r["variant"], r["seed"]): r["median"] for r in results if r["n"] == n and r["status"] == "ok"}
    seeds = sorted({s for (_, s) in by})
    return [by[(num, s)] / by[(den, s)] for s in seeds if (num, s) in by and (den, s) in by]


def _fmt_ratio(values: list[float]) -> str:
    if not values:
        return "-"
    med = float(np.median(values))
    return f"{med:.1f}x [{min(values):.1f}, {max(values):.1f}]" if len(values) > 1 else f"{med:.1f}x"


def report(
    env: dict[str, object],
    sites: list[dict[str, object]],
    results: list[dict[str, object]],
    profiles: list[dict[str, object]],
    sizes: list[int],
) -> None:
    """Print every table as markdown."""
    print("## Environment\n")
    print(f"- platform: {env['platform']}")
    print(f"- cpu: {env['cpu']}")
    print(f"- python {env['python']}, " + ", ".join(f"{k} {v}" for k, v in env["packages"].items()))
    print(f"- BLAS: {env['blas']}")
    print(f"- thread limit: {env['threads_limit'] or 'none (BLAS default)'}; environment: {env['thread_env']}")

    print("\n## 1. Code reading: `np.linalg.inv` in `pypfopt.cla.CLA._solve`\n")
    print("| line | branch | inside a loop | code |")
    print("|---:|---|---|---|")
    for s in sites:
        loop = f"yes, `{s['loop']}`" if s["in_loop"] else "no"
        print(f"| {s['line']} | {s['branch']} | {loop} | `{s['code']}` |")

    print("\n## 2. Frontier agreement with cvxcla (worst case over seeds)\n")
    print("Turning-point counts / largest distance to cvxcla's frontier; failed runs listed by seed.\n")
    print("| n | " + " | ".join(f"{v} pts / dist" for v in VARIANTS) + " |")
    print("|---:|" + "---:|" * len(VARIANTS))
    for n in sizes:
        cells = []
        for v in VARIANTS:
            every = [r for r in results if r["variant"] == v and r["n"] == n]
            rs = [r for r in every if r["status"] == "ok"]
            failed = [f"{r['status']} (seed {r['seed']})" for r in every if r["status"] != "ok"]
            cell = ""
            if rs:
                pts = "/".join(sorted({str(r["points"]) for r in rs}))
                cell = f"{pts} / {max(r['distance'] for r in rs):.0e}"
            cells.append(", ".join([c for c in [cell, *failed] if c]) or "-")
        print(f"| {n} | " + " | ".join(cells) + " |")

    summary = summarise(results)
    print("\n## 3. Trace time [ms]: median over seeds [min, max]\n")
    print("| n | " + " | ".join(VARIANTS) + " |")
    print("|---:|" + "---:|" * len(VARIANTS))
    for n in sizes:
        cells = []
        for v in VARIANTS:
            s = summary.get((v, n))
            cells.append(f"{s['median'] * 1e3:.1f} [{s['lo'] * 1e3:.1f}, {s['hi'] * 1e3:.1f}]" if s else "-")
        print(f"| {n} | " + " | ".join(cells) + " |")

    print("\n## 4. Attribution: per-seed ratios, median [min, max]\n")
    print("The first three columns multiply to the fourth.\n")
    print(
        "| n | inverse-in-loop (bldp/bordered) | Python loop (bordered/vec-fresh) | rest (vec-fresh/cvxcla) "
        "| total (bldp/cvxcla) | aside: fresh vs incremental (vec-fresh/vec-incr) |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    for n in sizes:
        cols = [
            seed_ratios(results, "bldp", "bordered", n),
            seed_ratios(results, "bordered", "vec-fresh", n),
            seed_ratios(results, "vec-fresh", "cvxcla", n),
            seed_ratios(results, "bldp", "cvxcla", n),
            seed_ratios(results, "vec-fresh", "vec-incr", n),
        ]
        print(f"| {n} | " + " | ".join(_fmt_ratio(c) for c in cols) + " |")

    print("\n## 5. Empirical exponents (time ~ n^p, least squares over the timed sizes)\n")
    for v in VARIANTS:
        pairs = [(n, summary[(v, n)]["median"]) for n in sizes if (v, n) in summary]
        if len(pairs) >= 3:
            x, y = np.log([p[0] for p in pairs]), np.log([p[1] for p in pairs])
            p_all = float(np.polyfit(x, y, 1)[0])
            p_last = float((y[-1] - y[-2]) / (x[-1] - x[-2]))
            print(f"- {v:9s} p = {p_all:.2f} overall, {p_last:.2f} over the last doubling")

    if profiles:
        print("\n## 6. Where the shipped BLdP trace spends its time (first terminating seed, % of trace)\n")
        phases = ["add.inverse", "add.slice", "add.lambda", "remove", "step", "purge.num_err", "purge.excess", "init"]
        print("| n | seed | steps | inverses | inverses / step | " + " | ".join(phases) + " | other |")
        print("|---:|---:|---:|---:|---:|" + "---:|" * (len(phases) + 1))
        for p in profiles:
            shares = [100 * p["phases"].get(k, 0.0) / p["total"] for k in phases]
            other = 100 - sum(shares)
            print(
                f"| {p['n']} | {p['seed']} | {p['steps']} | {p['inverses']} | {p['inverses'] / p['steps']:.1f} | "
                + " | ".join(f"{s:.1f}" for s in shares)
                + f" | {other:.1f} |"
            )


def plot(results: list[dict[str, object]], sizes: list[int], path: Path) -> None:
    """Write the scaling curves (left) and the attribution ratios (right)."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    summary = summarise(results)
    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(10, 3.8))
    styles = {
        "bldp": ("-^", "#7f7f7f", "BLdP (PyPortfolioOpt 1.6.0)"),
        "bordered": ("-v", "#ff7f0e", "BLdP, bordered inverse in loop"),
        "vec-fresh": ("-D", "#9467bd", "vectorised, fresh inverse"),
        "vec-incr": ("-s", "#2ca02c", "vectorised, incremental inverse"),
        "cvxcla": ("-o", "#c00000", "cvxcla, dense"),
    }
    for v, (marker, colour, label) in styles.items():
        ns = [n for n in sizes if (v, n) in summary]
        med = np.array([summary[(v, n)]["median"] for n in ns])
        lo = np.array([summary[(v, n)]["lo"] for n in ns])
        hi = np.array([summary[(v, n)]["hi"] for n in ns])
        ax_t.errorbar(ns, med, yerr=[med - lo, hi - med], fmt=marker, ms=4, color=colour, label=label, capsize=2)
    ax_t.set(xscale="log", yscale="log", xlabel="number of assets $n$", ylabel="frontier trace time [s]")
    ax_t.legend(fontsize=7)

    for (num, den), colour, label in [
        (("bldp", "bordered"), "#ff7f0e", "inverse in loop"),
        (("bordered", "vec-fresh"), "#9467bd", "Python-loop event search"),
        (("vec-fresh", "cvxcla"), "#c00000", "rest (vs cvxcla)"),
        (("bldp", "cvxcla"), "#000000", "total"),
    ]:
        ns, med, lo, hi = [], [], [], []
        for n in sizes:
            r = seed_ratios(results, num, den, n)
            if r:
                ns.append(n)
                med.append(np.median(r))
                lo.append(min(r))
                hi.append(max(r))
        med, lo, hi = np.array(med), np.array(lo), np.array(hi)
        ax_r.errorbar(ns, med, yerr=[med - lo, hi - med], fmt="-o", ms=4, color=colour, label=label, capsize=2)
    ax_r.axhline(1.0, color="0.6", lw=0.8)
    ax_r.set(xscale="log", yscale="log", xlabel="number of assets $n$", ylabel="time ratio")
    ax_r.legend(fontsize=7)
    for ax in (ax_t, ax_r):
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    fig.savefig(path)
    print(f"\nwrote {path}")


# ======================================================================================
# Main
# ======================================================================================
def main() -> None:
    """Run the three parts and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 80, 160, 320])
    parser.add_argument("--seeds", type=int, default=3, help="problem instances per size")
    parser.add_argument("--repeats", type=int, default=3, help="timed traces per instance")
    parser.add_argument("--threads", type=int, default=None, help="cap BLAS threads in every worker")
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--bldp-max-n", type=int, default=10_000, help="skip bldp/bordered above this n")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per measurement before giving up")
    parser.add_argument("--json", type=Path, default=None, help="write raw measurements here")
    parser.add_argument("--figure", type=Path, default=None, help="write the figure here (.pdf/.png)")
    args = parser.parse_args()

    if args.threads is not None:
        for var in _THREAD_VARS:
            os.environ[var] = str(args.threads)
    env = environment(args.threads)
    sites = read_inverse_sites()
    results: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    for n in args.sizes:
        for k in range(args.seeds):
            seed = BASE_SEED + k
            for v in args.variants:
                if v in {"bldp", "bordered"} and n > args.bldp_max_n:
                    continue
                r = _fresh(args.timeout, measure, v, n, seed, args.repeats)
                r = r or {"variant": v, "n": n, "seed": seed, "status": "timeout"}
                results.append(r)
                line = (
                    f"{r['median'] * 1e3:10.1f} ms  pts={r['points']:4d} (cvxcla {r['reference_points']:4d})  "
                    f"dist={r['distance']:.1e}"
                    if r["status"] == "ok"
                    else r["status"]
                )
                print(f"n={n:5d} seed={seed} {v:9s} {line}", file=sys.stderr, flush=True)
        if "bldp" in args.variants and n <= args.bldp_max_n:
            # Profile the first seed on which the shipped code finished.
            for k in range(args.seeds):
                p = _fresh(args.timeout, profile, n, BASE_SEED + k)
                if p is not None:
                    profiles.append(p)
                    break

    print()
    report(env, sites, results, profiles, args.sizes)
    if args.json:
        args.json.write_text(
            json.dumps({"environment": env, "sites": sites, "results": results, "profiles": profiles}, indent=2)
        )
        print(f"\nwrote {args.json}")
    if args.figure:
        plot(results, args.sizes, args.figure)


if __name__ == "__main__":
    main()

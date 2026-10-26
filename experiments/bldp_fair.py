# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "clarabel==0.11.1",
#     "cvxcla==2.7.1",
#     "cvxpy==1.9.3",
#     "numpy==2.4.6",
#     # scikit-base (a PyPortfolioOpt dependency) imports packaging without declaring it.
#     "packaging==26.3",
#     "pyportfolioopt==1.6.0",
# ]
# ///
r"""Is the BLdP-vs-cvxcla comparison apples to apples? (issue #869, track 3).

``bldp_gap.py`` (track 1) measured a runtime gap of about 400x between PyPortfolioOpt's
BLdP CLA and cvxcla, and ``bldp_agree.py`` (track 2) found that the shipped BLdP code
returns a wrong frontier on many of the same kind of problems. This script asks
which parts of the gap a fair comparison may credit to cvxcla. All problems are
BLdP's own problem class (dense covariance, box bounds, one budget row), the
K = 10 factor-model, long-only problems of the paper's scaling figure, so cvxcla
is never given a feature BLdP lacks. Four parts:

A. **Correctness-adjusted gap.** Time the shipped code (``bldp``) and the code with
   both track-2 bugs fixed (``bldp-fix-wp``, from ``bldp_agree.py``) against cvxcla,
   and check each path ``w(lambda)`` against cvxcla's (exact sup-norm, as in
   ``bldp_agree.py``). A speed ratio against a wrong answer is not a fair one.

B. **Algorithm or interpreter?** Track 1's ladder credited 12-141x to "vectorised
   event search". That rung changed two things at once, which this part separates
   with a telescoping chain of rungs that all return the correct frontier:

   ``bldp-fix-wp`` BLdP's ``_solve`` (bugs fixed): for every candidate asset, invert
                  the bordered free block and re-derive lambda from scratch, slicing
                  Sigma and recomputing the shared products each time. Per turning
                  point: O(n) candidates x O(n_F^3 + n n_F) work.
   ``loop``       The reduced-cost formulation: solve the current free system once
                  per turning point, then read every candidate's event lambda off
                  its intercept/slope ``-gamma_i / delta_i`` (one dot product per
                  candidate), visiting the assets in a Python loop
                  (``InverseCLA(fresh=True, loop=True)``). Per turning point:
                  O(n_F^3) once plus O(n) candidates x O(n) work.
   ``vec-fresh``  The same arithmetic as whole-vector numpy expressions
                  (``InverseCLA(fresh=True)``).

   ``bldp-fix-wp / loop`` is the change of *formula*, an algorithmic change that
   removes a factor of about n_F from each step (it includes the per-candidate
   inverse, which track 1 put at <= 1.6x); ``loop / vec-fresh`` is the
   *interpreter*: the same flops, dispatched once per asset instead of once per
   step. The ratio that grows with n is the asymptotic one; the slopes are reported.
   (Track 1's ``bordered`` rung is not used: it cycles on 9 of 30 of these
   problems, with or without the two track-2 fixes, so the cycle is its
   zero-tolerance re-entry test and not the stale weights.)

C. **What cvxcla's extra machinery costs.** ``vec-fresh / cvxcla`` is everything
   cvxcla does beyond a minimal vectorised CLA on this problem class. A cProfile of
   cvxcla splits its trace into *guards* (feasibility checks and projection,
   held-bound checks, the round-off level of the multipliers, the degeneracy and
   conditioning tests, the mean shift, tied-start handling, input validation),
   *generality* (the inequality-row event search, idle here since p = 0) and the
   *core* (KKT solve, event search, stepping, orchestration). Fractions are under
   the profiler, which inflates Python-call-heavy code; they are converted to
   milliseconds with the unprofiled time.

D. **Environment.** The gap between Python-loop code and BLAS-backed code depends on
   the stack. ``--threads 1`` re-times the ladder with single-threaded BLAS.
   ``--stacks`` re-times the shipped BLdP code under other (Python, numpy) pairs by
   re-running this file under ``uv run --python ... --with numpy==...`` (cvxcla
   needs numpy >= 2, PyPortfolioOpt 1.6.0 needs numpy >= 1.26, so 1.26.4 on Python
   3.12 is the oldest stack that runs both the code and this check). Set
   ``UV_OFFLINE=1`` to resolve them from the uv cache only.

Each (variant, n, seed) is measured in a freshly spawned interpreter: one untimed
trace, then ``--repeats`` timed ones; the per-seed median is the measurement and
ratios are medians over seeds. A run that cycles or times out is reported, not
timed, and ratios use only the seeds on which both rungs finished.

Usage:
    uv run experiments/bldp_fair.py                          # A-C, n = 20..320
    uv run experiments/bldp_fair.py --threads 1              # same, single-threaded BLAS
    uv run experiments/bldp_fair.py --stacks 3.12:1.26.4 3.12:2.4.6 3.14:2.4.6
    uv run experiments/bldp_fair.py --json out.json
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import subprocess  # nosec B404
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for bldp_gap, bldp_agree, inverse_cla
from bldp_gap import _THREAD_VARS, CycleError, _bldp_classes, _fresh, environment, frontier_distance, make_problem

VARIANTS = ["bldp", "bldp-fix-wp", "loop", "vec-fresh", "cvxcla"]
BASE_SEED = 7

# cvxcla functions whose cumulative time is "guards" (none of them calls another;
# checked with the profiler's caller lists) and "generality".
GUARDS = {
    "check_feasible",
    "project_feasible",
    "_check_held_bounds",
    "guard_active_rows",
    "_gamma_noise",
    "roundoff_events_at_zero",
    "_free_blocks_well_conditioned",
    "_tied_start",
    "_mean_shift",
    "_restore_mean_units",
    "_scales",
    "_independent_equalities",
    "_check_finite_inputs",
    "_check_shapes",
    "_check_options",
    "_own_inputs",
}
GENERALITY = {"ineq_event_ratios"}


# ======================================================================================
# Variants
# ======================================================================================
def run(variant: str, dense: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Trace one variant; return (turning points as rows, lambdas or None)."""
    n = len(mean)
    if variant in {"bldp", "bldp-fix-wp"}:
        shipped, capped = _bldp_classes()
        if variant == "bldp":
            cla = shipped(mean, dense, weight_bounds=(0, 1))
        else:
            from bldp_agree import _fixed

            cla = _fixed(capped, purge_tol=1e-12)(mean, dense, weight_bounds=(0, 1))
        cla._solve()
        lambdas = np.array([np.inf if lam is None else float(lam) for lam in cla.ls])
        return np.array([w.ravel() for w in cla.w]), lambdas
    if variant in {"loop", "vec-fresh"}:
        from inverse_cla import InverseCLA

        inv = InverseCLA(mean, dense, np.zeros(n), np.ones(n), fresh=True, loop=variant == "loop")
        return np.array(inv.weights), None
    from cvxcla import CLA

    cla = CLA(
        mean=mean, covariance=dense, lower_bounds=np.zeros(n), upper_bounds=np.ones(n), a=np.ones((1, n)), b=np.ones(1)
    )
    return np.array([tp.weights for tp in cla.turning_points]), np.array([tp.lamb for tp in cla.turning_points])


def correctness(trace: tuple[np.ndarray, np.ndarray | None], ref: tuple[np.ndarray, np.ndarray]) -> float:
    """Distance of a trace to cvxcla's: the exact path sup-norm where lambdas exist."""
    weights, lambdas = trace
    if lambdas is None:  # InverseCLA keeps no lambdas: turning points vs cvxcla's
        return max(frontier_distance(weights, ref[0]), frontier_distance(ref[0], weights))
    from bldp_agree import agreement

    return float(agreement((weights, lambdas), ref)["distance"])


# ======================================================================================
# Workers (each call runs in a freshly spawned interpreter)
# ======================================================================================
def measure(variant: str, n: int, seed: int, repeats: int) -> dict[str, object]:
    """Warm up, check against cvxcla, then time ``repeats`` traces of one variant."""
    dense, mean = make_problem(n, seed)
    try:
        trace = run(variant, dense, mean)
    except CycleError:
        return {"variant": variant, "n": n, "seed": seed, "status": "cycled"}
    ref = trace if variant == "cvxcla" else run("cvxcla", dense, mean)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run(variant, dense, mean)
        times.append(time.perf_counter() - start)
    return {
        "variant": variant,
        "n": n,
        "seed": seed,
        "status": "ok",
        "points": len(trace[0]),
        "distance": correctness(trace, ref),
        "median": float(np.median(times)),
    }


def profile(n: int, seed: int) -> dict[str, float]:
    """Split one cvxcla trace into guards / generality / core by cumulative time."""
    dense, mean = make_problem(n, seed)
    run("cvxcla", dense, mean)
    start = time.perf_counter()
    run("cvxcla", dense, mean)
    plain = time.perf_counter() - start
    prof = cProfile.Profile()
    prof.enable()
    run("cvxcla", dense, mean)
    prof.disable()
    split: dict[str, float] = defaultdict(float)
    total = 0.0
    for (path, _, name), (_, _, _, cumulative, _) in pstats.Stats(prof).stats.items():  # type: ignore[attr-defined]
        if "cvxcla" not in path:
            continue
        if name == "__post_init__" and path.endswith("cla.py"):
            total = cumulative
        elif name in GUARDS:
            split["guards"] += cumulative
            split[name] += cumulative
        elif name in GENERALITY:
            split["generality"] += cumulative
    out = {k: v / total for k, v in split.items()}
    out["core"] = 1.0 - out.get("guards", 0.0) - out.get("generality", 0.0)
    out["plain_ms"] = plain * 1e3
    return out


def bldp_only(n: int, seed: int, repeats: int) -> dict[str, object]:
    """Time the shipped BLdP code alone (the ``--stacks`` worker; needs no cvxcla)."""
    import importlib.metadata as md
    import platform

    dense, mean = make_problem(n, seed)
    run("bldp", dense, mean)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run("bldp", dense, mean)
        times.append(time.perf_counter() - start)
    return {"python": platform.python_version(), "numpy": md.version("numpy"), "median": float(np.median(times))}


def run_stack(stack: str, n: int, seeds: int, repeats: int, timeout: float) -> list[dict[str, object]]:
    """Re-run this file as a ``--bldp-only`` worker under ``uv run`` with another (Python, numpy)."""
    python, numpy_version = stack.split(":")
    cmd = [
        "uv", "run", "--no-project", "--python", python,
        "--with", f"numpy=={numpy_version}", "--with", "pyportfolioopt==1.6.0", "--with", "packaging==26.3",
        "python", str(Path(__file__).resolve()), "--bldp-only", str(n), str(seeds), str(repeats),
    ]  # fmt: skip
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)  # noqa: S603  # nosec B603 B607
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        return [{"stack": stack, "status": "failed", "error": detail.strip().splitlines()[-1][:160]}]
    return [
        {"stack": stack, "status": "ok", **json.loads(line)} for line in out.stdout.splitlines() if line.startswith("{")
    ]


# ======================================================================================
# Reporting
# ======================================================================================
def medians(results: list[dict[str, object]]) -> dict[tuple[str, int, int], float]:
    """Index the finished measurements by (variant, n, seed)."""
    return {(r["variant"], r["n"], r["seed"]): r["median"] for r in results if r["status"] == "ok"}


def ratio(times: dict, num: str, den: str, n: int, seeds: list[int]) -> float | None:
    """Median over seeds of num/den, on the seeds where both finished."""
    vals = [times[(num, n, s)] / times[(den, n, s)] for s in seeds if (num, n, s) in times and (den, n, s) in times]
    return float(np.median(vals)) if vals else None


def slope(times: dict, num: str, den: str, sizes: list[int], seeds: list[int]) -> float | None:
    """Exponent of a ratio's growth over the last doubling of n (0 = constant factor)."""
    if len(sizes) < 2:
        return None
    a, b = ratio(times, num, den, sizes[-2], seeds), ratio(times, num, den, sizes[-1], seeds)
    return float(np.log(b / a) / np.log(sizes[-1] / sizes[-2])) if a and b else None


def _fmt(x: float | None, unit: str = "x") -> str:
    return "     -" if x is None else f"{x:6.1f}{unit}" if x >= 10 else f"{x:6.2f}{unit}"


def report(
    env: dict, results: list[dict], profiles: dict[int, dict], stacks: list[dict], sizes: list[int], seeds: list[int]
) -> None:
    """Print parts A-D."""
    print("environment")
    for k, v in env.items():
        print(f"  {k:14s}: {v}")
    times = medians(results)

    print("\nper-variant median time (ms) over seeds, and the worst distance to cvxcla's frontier")
    print(f"  {'n':>5s}" + "".join(f"{v:>22s}" for v in VARIANTS))
    for n in sizes:
        cells = []
        for v in VARIANTS:
            rows = [r for r in results if r["variant"] == v and r["n"] == n]
            ok = [r for r in rows if r["status"] == "ok"]
            if not ok:
                cells.append("/".join(sorted({r["status"] for r in rows})) or "-")
                continue
            t = np.median([r["median"] for r in ok]) * 1e3
            bad = len(rows) - len(ok)
            cells.append(f"{t:9.1f} ({max(r['distance'] for r in ok):.0e})" + (f"+{bad}c" if bad else ""))
        print(f"  {n:5d}" + "".join(f"{c:>22s}" for c in cells))

    print("\nA. correctness-adjusted gap (median ratio over seeds)")
    print(f"  {'n':>5s} {'bldp/cvxcla':>12s} {'fixed/cvxcla':>13s} {'bldp/fixed':>11s}")
    for n in sizes:
        print(
            f"  {n:5d} {_fmt(ratio(times, 'bldp', 'cvxcla', n, seeds)):>12s} "
            f"{_fmt(ratio(times, 'bldp-fix-wp', 'cvxcla', n, seeds)):>13s} "
            f"{_fmt(ratio(times, 'bldp', 'bldp-fix-wp', n, seeds)):>11s}"
        )

    print("\nB. formula vs interpreter (telescoping: bldp/cvxcla = product of the four)")
    chain = [("bldp", "bldp-fix-wp"), ("bldp-fix-wp", "loop"), ("loop", "vec-fresh"), ("vec-fresh", "cvxcla")]
    labels = ["bug fixes", "formula", "interpreter", "cvxcla extra"]
    print(f"  {'n':>5s}" + "".join(f"{lab:>14s}" for lab in labels) + f"{'total':>10s}")
    for n in sizes:
        cells = [_fmt(ratio(times, a, b, n, seeds)) for a, b in chain]
        print(
            f"  {n:5d}" + "".join(f"{c:>14s}" for c in cells) + f"{_fmt(ratio(times, 'bldp', 'cvxcla', n, seeds)):>10s}"
        )
    exps = [slope(times, a, b, sizes, seeds) for a, b in chain]
    print(
        "  growth ~ n^k over the last doubling, k = "
        + "  ".join(f"{lab} {_fmt(k, '')}" for lab, k in zip(labels, exps, strict=True))
    )

    if profiles:
        print("\nC. cvxcla trace split (fractions under cProfile; ms = fraction x unprofiled time)")
        print(f"  {'n':>5s} {'total ms':>9s} {'core':>14s} {'guards':>14s} {'generality':>14s}   largest guards")
        for n, p in profiles.items():
            ms = p["plain_ms"]
            cells = [f"{p.get(k, 0.0):5.0%} {p.get(k, 0.0) * ms:6.1f}ms" for k in ("core", "guards", "generality")]
            top = sorted(((v, k) for k, v in p.items() if k in GUARDS), reverse=True)[:3]
            print(
                f"  {n:5d} {ms:9.1f} "
                + " ".join(f"{c:>14s}" for c in cells)
                + "   "
                + ", ".join(f"{k} {v:.0%}" for v, k in top)
            )

    if stacks:
        print("\nD. shipped BLdP under other stacks (median over seeds, ms)")
        for s in stacks:
            if s["status"] != "ok":
                print(f"  {s['stack']:14s} failed: {s['error']}")
        ok = [s for s in stacks if s["status"] == "ok"]
        by: dict[str, list[float]] = defaultdict(list)
        for s in ok:
            by[f"python {s['python']}, numpy {s['numpy']}"].append(s["median"])
        for k, v in by.items():
            print(f"  {k:36s} {np.median(v) * 1e3:9.1f}")


def main() -> None:
    """Run parts A-D and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 80, 160, 320])
    parser.add_argument("--seeds", type=int, default=3, help="problem instances per size")
    parser.add_argument("--repeats", type=int, default=3, help="timed traces per instance")
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--threads", type=int, default=None, help="cap BLAS threads in every worker")
    parser.add_argument("--stacks", nargs="*", default=[], help="python:numpy pairs to time the shipped code under")
    parser.add_argument("--stack-n", type=int, default=160, help="problem size for --stacks")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per measurement before giving up")
    parser.add_argument("--json", type=Path, default=None, help="write raw measurements here")
    parser.add_argument("--bldp-only", nargs=3, type=int, metavar=("N", "SEEDS", "REPEATS"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.bldp_only:  # worker mode for --stacks: one JSON line per seed
        n, seeds, repeats = args.bldp_only
        for k in range(seeds):
            print(json.dumps(bldp_only(n, BASE_SEED + k, repeats)), flush=True)
        return

    if args.threads is not None:
        for var in _THREAD_VARS:
            os.environ[var] = str(args.threads)
    env = environment(args.threads)
    seeds = [BASE_SEED + k for k in range(args.seeds)]
    results: list[dict[str, object]] = []
    profiles: dict[int, dict] = {}
    for n in args.sizes:
        for seed in seeds:
            for v in args.variants:
                r = _fresh(args.timeout, measure, v, n, seed, args.repeats)
                r = r or {"variant": v, "n": n, "seed": seed, "status": "timeout"}
                results.append(r)
                line = f"{r['median'] * 1e3:10.1f} ms  dist={r['distance']:.1e}" if r["status"] == "ok" else r["status"]
                print(f"n={n:5d} seed={seed} {v:12s} {line}", file=sys.stderr, flush=True)
        if "cvxcla" in args.variants:
            profiles[n] = _fresh(args.timeout, profile, n, seeds[0])
    stacks = [
        r for s in args.stacks for r in run_stack(s, args.stack_n, args.seeds, args.repeats, args.timeout * args.seeds)
    ]

    print()
    report(env, results, profiles, stacks, args.sizes, seeds)
    if args.json:
        payload = {"environment": env, "results": results, "profiles": profiles, "stacks": stacks}
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

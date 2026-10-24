# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "clarabel==0.11.1",
#     "cvxcla==2.7.0",
#     "cvxpy==1.9.3",
#     "numpy==2.4.6",
#     # scikit-base (a PyPortfolioOpt dependency) imports packaging without declaring it.
#     "packaging==26.3",
#     "pyportfolioopt==1.6.0",
# ]
# ///
r"""Check that BLdP's CLA and cvxcla agree, and find where they don't (issue #869, track 2).

Both implementations claim to compute the same mathematical object: the exact,
finite list of Markowitz turning points of

    max  lambda mu'w - 1/2 w' Sigma w   s.t.  1'w = 1,  lB <= w <= uB

for lambda from infinity down to 0. Both use the same lambda convention, so turning
points, weights and lambdas can be compared directly. This script does three things.

1. **Differential test.** The same ``mean``, ``covariance``, ``lB``, ``uB`` go into

   ``bldp``       PyPortfolioOpt 1.6.0 ``CLA`` (the BLdP code as shipped, including
                  its ``_purge_num_err`` and ``_purge_excess`` clean-up passes);
   ``bldp-2013``  the same code with the paper's all-equal-means hack
                  ``mean[-1] += 1e-5`` switched back on (PyPortfolioOpt ships it
                  commented out; it is a no-op unless every mean is equal);
   ``bldp-fix-w`` the shipped code with the stale-weight bug fixed (see below);
   ``bldp-fix-wp`` and also with a tolerance in ``_purge_excess`` (see below), so the
                  two bugs are attributed separately from each other and from
                  everything else;
   ``cvxcla``     ``cvxcla.CLA`` with the dense backend.

   Each finished trace defines a path ``w(lambda)``: linear in lambda between turning
   points, constant above the first finite one. Two traces *agree* when their paths
   are within ``1e-8`` in max-norm for every lambda. Both paths are piecewise
   linear, so this sup-norm is attained at a turning point of one of them and is
   computed exactly. Comparing paths rather than lists of turning points makes
   repeated points (BLdP emits some) and points that lie on a segment harmless,
   while a missing or misplaced vertex or a wrong lambda still shows.

2. **Independent validation.** Each finished trace is checked against CVXPY/Clarabel
   at tolerance ``1e-12``, not against the other implementation: the weights are read
   off the trace at every finite turning point and at every segment midpoint
   (interpolated linearly in lambda), and the QP is solved at that lambda from
   scratch. The primary measure is the *objective gap* ``f(w_CLA) - f(w_QP)`` (relative
   to ``1 + |f(w_QP)|``), which stays meaningful when Sigma is singular and the
   optimal weights are not unique; the weight distance is reported alongside. The
   ``lambda = inf`` point is checked against the LP ``max mu'w``. A trace is
   classified as

   ``exact``      feasible to ``1e-9``, spans lambda from inf to 0 with lambda
                  non-increasing (to ``1e-10`` relative), objective gap below
                  ``1e-8`` everywhere;
   ``wrong``      finished, but fails one of those;
   ``cycled``     did not terminate within ``10 (n + 1)`` steps (BLdP only: its loop
                  has no iteration cap of its own);
   ``<Error>``    raised (the exception type is shown).

3. **Degenerate inputs.** Beyond generic problems, families aimed at the weak spots
   ``comparison.md`` names: all-equal and tied expected returns, duplicated assets,
   rank-deficient sample covariances, a conditioning sweep (``cond(Sigma)`` from
   ``1e2`` to ``1e16``), and bounds that bind exactly, including a budget that
   ``uB = 1/k`` fills in exact (k = 4) and in rounded (k = 5) arithmetic, a feasible
   set that is a single point, and an infeasible one.

**The stale-weight bug.** When a free asset hits a bound, BLdP sets the working
vector ``w[i] = bound`` and then rebuilds the free/blocked split with
``_get_matrices``, which reads the blocked weights from ``self.w[-1]``: the
*previous* turning point, where asset ``i`` still has its old free value. The new
turning point is solved for the wrong blocked weights, so its budget is off by exactly
``bound - w_prev[i]``. ``_purge_num_err`` then deletes it as infeasible, and with it a
true turning point; the next step reads the corrected weight and is right again.
Sometimes the trace recovers by removing the asset, re-adding it and removing it
again at the same lambda (three copies of one point); sometimes the vertex is simply
lost and the frontier interpolates across it. ``bldp-fix-w`` reads the blocked
weights from the working vector instead and changes nothing else.

**The zero-tolerance purge.** ``_purge_excess`` deletes turning point ``i`` whenever
a *later* point has a strictly higher expected return, ``mu'w_i < mu'w_j``. Along an
exact trace ``mu'w`` never increases, but where it is flat (a point held still over
a lambda interval, a repeated point) rounding makes it rise by about ``1e-14``, and
the purge deletes a true turning point. ``bldp-fix-wp`` compares with a relative
tolerance of ``1e-12`` instead.

The BLdP traces run through the statement-for-statement copy of PyPortfolioOpt's
``_solve`` in ``bldp_gap.py``, which adds only the step cap; without it a cycling
trace would never return.

Usage:
    uv run experiments/bldp_agree.py
    uv run experiments/bldp_agree.py --seeds 5 --json out.json
    uv run experiments/bldp_agree.py --families generic conditioning --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import cvxpy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for bldp_gap
from bldp_gap import CycleError, _bldp_classes, environment

IMPLEMENTATIONS = ["bldp", "bldp-2013", "bldp-fix-w", "bldp-fix-wp", "cvxcla"]
BLDP = [i for i in IMPLEMENTATIONS if i.startswith("bldp")]

FEAS_TOL = 1e-9
GAP_TOL = 1e-8
AGREE_TOL = 1e-8
N_FACTORS = 5


# ======================================================================================
# Problems
# ======================================================================================
class Problem(dict):
    """``mean``, ``cov``, ``lb``, ``ub`` and a human-readable ``label``."""


def _problem(label: str, mean: np.ndarray, cov: np.ndarray, lb: float | np.ndarray, ub: float | np.ndarray) -> Problem:
    n = len(mean)
    return Problem(
        label=label,
        mean=np.asarray(mean, dtype=float),
        cov=np.asarray(cov, dtype=float),
        lb=np.broadcast_to(np.asarray(lb, dtype=float), (n,)).copy(),
        ub=np.broadcast_to(np.asarray(ub, dtype=float), (n,)).copy(),
    )


def _factor_cov(rng: np.random.Generator, n: int, k: int = N_FACTORS) -> np.ndarray:
    """A well-conditioned positive-definite K-factor covariance (as in tie_degeneracy.py)."""
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    return np.diag(d) + (u * delta) @ u.T


def generic(rng: np.random.Generator) -> list[Problem]:
    """Long-only factor-model problems with distinct means: both should be exact here."""
    return [_problem(f"n={n}", rng.uniform(0, 1, n), _factor_cov(rng, n), 0.0, 1.0) for n in (10, 25, 50, 100)]


def shorting(rng: np.random.Generator) -> list[Problem]:
    """Generic problems with short positions allowed, ``-1 <= w <= 1``."""
    return [_problem(f"n={n}", rng.uniform(0, 1, n), _factor_cov(rng, n), -1.0, 1.0) for n in (10, 25, 50)]


def all_equal(rng: np.random.Generator) -> list[Problem]:
    """Every expected return identical: the frontier is the single minimum-variance point."""
    return [_problem(f"n={n}", np.full(n, 0.1), _factor_cov(rng, n), 0.0, 1.0) for n in (5, 20)]


def tied(rng: np.random.Generator) -> list[Problem]:
    """Means in four exactly tied groups; the top group ties at the first vertex."""
    out = []
    for n in (12, 40):
        mean = np.resize(np.repeat(rng.uniform(0, 1, 4), int(np.ceil(n / 4))), n)
        out.append(_problem(f"n={n}", rng.permutation(mean), _factor_cov(rng, n), 0.0, 1.0))
    # Only the top two tie: a tied start, then a generic trace.
    n = 12
    mean = rng.uniform(0, 0.8, n)
    mean[:2] = 1.0
    out.append(_problem(f"n={n} top-2 tie", mean, _factor_cov(rng, n), 0.0, 1.0))
    return out


def duplicated(rng: np.random.Generator) -> list[Problem]:
    """The second half copies the first (same row/column of Sigma, same mean): Sigma is singular."""
    out = []
    for n in (10, 30):
        half = n // 2
        cov = _factor_cov(rng, half)
        mean = rng.uniform(0, 1, half)
        idx = np.r_[np.arange(half), np.arange(half)]
        out.append(_problem(f"n={n} all copied", mean[idx], cov[np.ix_(idx, idx)], 0.0, 1.0))
    # One copy of the highest-return asset only.
    n = 12
    cov = _factor_cov(rng, n)
    mean = rng.uniform(0, 1, n)
    top = int(np.argmax(mean))
    idx = np.r_[np.arange(n), top]
    out.append(_problem(f"n={n + 1} top copied", mean[idx], cov[np.ix_(idx, idx)], 0.0, 1.0))
    return out


def rank_deficient(rng: np.random.Generator) -> list[Problem]:
    """Sample covariance from T < n observations: rank T, distinct means."""
    out = []
    for n, t in ((20, 8), (40, 30)):
        x = rng.standard_normal((t, n)) @ np.diag(rng.uniform(0.5, 2.0, n))
        out.append(_problem(f"n={n} T={t}", rng.uniform(0, 1, n), np.cov(x, rowvar=False), 0.0, 1.0))
    return out


CONDITIONING = [1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16]


def conditioning(rng: np.random.Generator) -> list[Problem]:
    """Sigma = Q diag(geomspace(1, 1/kappa)) Q' with a random rotation, n = 20."""
    n = 20
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    mean = rng.uniform(0, 1, n)
    out = []
    for kappa in CONDITIONING:
        cov = (q * np.geomspace(1.0, 1.0 / kappa, n)) @ q.T
        out.append(_problem(f"cond=1e{np.log10(kappa):.0f}", mean, 0.5 * (cov + cov.T), 0.0, 1.0))
    return out


def binding(rng: np.random.Generator) -> list[Problem]:
    """Bounds that bind exactly at the first vertex, or leave one or no feasible point."""
    n = 20
    mean = rng.uniform(0, 1, n)
    cov = _factor_cov(rng, n)
    return [
        # 4 x 0.25 == 1 exactly: the "partially filled" asset lands on its upper bound.
        _problem("uB=1/4", mean, cov, 0.0, 1 / 4),
        # 5 x 0.2 != 1 in floating point: the same degeneracy, blurred by rounding.
        _problem("uB=1/5", mean, cov, 0.0, 1 / 5),
        # Nonzero lower bounds that sum to 0.5: the start fills the other half.
        _problem("lB=1/(2n)", mean, cov, 1 / (2 * n), 1.0),
        # The only feasible portfolio is w = 1/n (sum(uB) == 1 exactly for n = 20).
        _problem("uB=1/n", mean, cov, 0.0, 1 / n),
        # sum(uB) < 1: no feasible portfolio.
        _problem("uB=0.9/n", mean, cov, 0.0, 0.9 / n),
    ]


FAMILIES: dict[str, Callable[[np.random.Generator], list[Problem]]] = {
    "generic": generic,
    "shorting": shorting,
    "all-equal": all_equal,
    "tied": tied,
    "duplicated": duplicated,
    "rank-deficient": rank_deficient,
    "conditioning": conditioning,
    "binding": binding,
}


# ======================================================================================
# Running the implementations
# ======================================================================================
def _fixed(capped: type, purge_tol: float | None = None) -> type:
    """The capped BLdP copy with blocked weights read from the working vector.

    ``_solve`` mutates one working vector ``w`` (the one ``_init_algo`` returns)
    throughout, and appends copies of it to ``self.w``. Reading the blocked weights
    from that vector instead of ``self.w[-1]`` differs only right after a removal,
    where it sees the weight just set to its bound. With ``purge_tol``,
    ``_purge_excess`` also ignores rises in expected return below that relative size.
    """

    class FixedBLdP(capped):
        def _init_algo(self):
            f, w = super()._init_algo()
            self._live = w
            return f, w

        def _get_matrices(self, f):
            covarF, covarFB, meanF, _ = super()._get_matrices(f)  # noqa: N806 -- upstream names
            return covarF, covarFB, meanF, self._reduce_matrix(self._live, self._get_b(f), [0])

        def _purge_excess(self) -> None:
            if purge_tol is None:
                super()._purge_excess()
                return
            # PyPortfolioOpt 1.6.0 ``_purge_excess``; only the comparison is CHANGED.
            i, repeat = 0, False
            while True:
                if repeat is False:
                    i += 1
                if i == len(self.w) - 1:
                    break
                mu = np.dot(self.w[i].T, self.mean)[0, 0]
                j, repeat = i + 1, False
                while True:
                    if j == len(self.w):
                        break
                    mu_ = np.dot(self.w[j].T, self.mean)[0, 0]
                    if mu < mu_ - purge_tol * max(1.0, abs(mu)):  # CHANGED: was mu < mu_
                        del self.w[i], self.ls[i], self.g[i], self.f[i]
                        repeat = True
                        break
                    j += 1

    return FixedBLdP


def run(impl: str, p: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Trace the frontier; return (turning-point weights as rows, lambdas with inf first)."""
    mean, cov, lb, ub = p["mean"], p["cov"], p["lb"], p["ub"]
    n = len(mean)
    if impl in BLDP:
        mean = mean.copy()
        if impl == "bldp-2013" and np.all(mean == mean.mean()):
            mean[-1] += 1e-5  # Bailey & Lopez de Prado (2013), Snippet 1
        _, capped = _bldp_classes()
        cls = {"bldp-fix-w": _fixed(capped), "bldp-fix-wp": _fixed(capped, purge_tol=1e-12)}.get(impl, capped)
        cla = cls(mean, cov, weight_bounds=(lb, ub))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 0/0 inside a singular block
            cla._solve()
        lambdas = np.array([np.inf if lam is None else float(lam) for lam in cla.ls])
        return np.array([w.ravel() for w in cla.w]), lambdas
    from cvxcla import CLA

    cla = CLA(mean=mean, covariance=cov, lower_bounds=lb, upper_bounds=ub, a=np.ones((1, n)), b=np.ones(1))
    return np.array([tp.weights for tp in cla.turning_points]), np.array([tp.lamb for tp in cla.turning_points])


# ======================================================================================
# Independent validation against a convex QP solver
# ======================================================================================
def qp(p: Problem, lam: float) -> np.ndarray:
    """Solve the parametric QP at ``lam`` (the LP ``max mu'w`` at ``lam = inf``) with Clarabel."""
    n = len(p["mean"])
    w = cp.Variable(n)
    if np.isinf(lam):
        objective = cp.Maximize(p["mean"] @ w)
    else:
        objective = cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(p["cov"])) - lam * p["mean"] @ w)
    constraints = [cp.sum(w) == 1, w >= p["lb"], w <= p["ub"]]
    cp.Problem(objective, constraints).solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return np.asarray(w.value, dtype=float)


def objective(p: Problem, w: np.ndarray, lam: float) -> float:
    """The minimised objective 1/2 w'Sigma w - lam mu'w (or -mu'w at lam = inf)."""
    if np.isinf(lam):
        return -float(p["mean"] @ w)
    return 0.5 * float(w @ p["cov"] @ w) - lam * float(p["mean"] @ w)


def validate(p: Problem, weights: np.ndarray, lambdas: np.ndarray) -> dict[str, object]:
    """Check one finished trace against the QP at every turning point and segment midpoint."""
    infeas = float(
        max(
            np.abs(weights.sum(axis=1) - 1).max(),
            (p["lb"] - weights).max(),
            (weights - p["ub"]).max(),
        )
    )
    spans = bool(np.isinf(lambdas[0]) and lambdas[-1] == 0)
    finite = lambdas[np.isfinite(lambdas)]
    monotone = bool(np.all(np.diff(finite) <= 1e-10 * np.maximum(1, np.abs(finite[:-1]))))  # rounding may rise ~1e-13
    probes: list[tuple[float, np.ndarray]] = list(zip(lambdas, weights, strict=True))
    for (lam_hi, w_hi), (lam_lo, w_lo) in pairwise(zip(lambdas, weights, strict=True)):
        if np.isfinite(lam_hi) and lam_hi > lam_lo:
            lam = 0.5 * (lam_hi + lam_lo)
            probes.append((lam, w_lo + 0.5 * (w_hi - w_lo)))
    gap, dist = 0.0, 0.0
    for lam, w in probes:
        w_qp = qp(p, lam)
        f_qp = objective(p, w_qp, lam)
        gap = max(gap, (objective(p, w, lam) - f_qp) / (1 + abs(f_qp)))
        dist = max(dist, float(np.abs(w - w_qp).max()))
    repeats = len(weights) - len(distinct(weights, lambdas)[0])
    exact = infeas <= FEAS_TOL and spans and monotone and gap <= GAP_TOL
    return {
        "status": "exact" if exact else "wrong",
        "points": len(weights),
        "repeats": repeats,
        "infeasibility": infeas,
        "spans": spans,
        "monotone": monotone,
        "gap": gap,
        "distance": dist,
        "probes": len(probes),
    }


def distinct(weights: np.ndarray, lambdas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop turning points that repeat the previous portfolio (to max-norm 1e-12)."""
    keep = np.r_[True, np.abs(np.diff(weights, axis=0)).max(axis=1) > 1e-12]
    return weights[keep], lambdas[keep]


def path(weights: np.ndarray, lambdas: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Evaluate a trace's ``w(lambda)`` at ``at`` (rows), clamped outside its finite lambdas."""
    finite = np.isfinite(lambdas)
    order = np.argsort(lambdas[finite], kind="stable")
    lam, w = lambdas[finite][order], weights[finite][order]
    return np.stack([np.interp(at, lam, w[:, j]) for j in range(w.shape[1])], axis=1)


def agreement(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> dict[str, object]:
    """Return the exact sup-norm distance between two traces' paths ``w(lambda)``."""
    (wa, la), (wb, lb) = a, b
    at = np.union1d(la[np.isfinite(la)], lb[np.isfinite(lb)])
    dist = max(float(np.abs(wa[0] - wb[0]).max()), float(np.abs(path(wa, la, at) - path(wb, lb, at)).max()))
    points = (len(distinct(wa, la)[0]), len(distinct(wb, lb)[0]))
    return {"agree": dist <= AGREE_TOL, "points": points, "distance": dist}


def check(p: Problem) -> dict[str, object]:
    """Run every implementation on one problem, validate each, and compare them pairwise."""
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    verdicts: dict[str, dict[str, object]] = {}
    for impl in IMPLEMENTATIONS:
        try:
            traces[impl] = run(impl, p)
        except CycleError:
            verdicts[impl] = {"status": "cycled"}
        except Exception as exc:  # noqa: BLE001 -- every failure mode is a finding
            verdicts[impl] = {"status": type(exc).__name__, "message": str(exc).splitlines()[0][:120]}
        else:
            verdicts[impl] = validate(p, *traces[impl])
    pairs = {
        f"{a}|{b}": agreement(traces[a], traces[b]) for a in BLDP for b in ["cvxcla"] if a in traces and b in traces
    }
    return {"label": p["label"], "verdicts": verdicts, "pairs": pairs}


# ======================================================================================
# Reporting
# ======================================================================================
def _cell(v: dict[str, object]) -> str:
    """One implementation's verdict on one problem, compactly."""
    if v["status"] in {"exact", "wrong"}:
        flags = []
        if v["infeasibility"] > FEAS_TOL:
            flags.append(f"infeas {v['infeasibility']:.0e}")
        if not v["spans"]:
            flags.append("no λ=0")
        if not v["monotone"]:
            flags.append("λ not monotone")
        if v["gap"] > GAP_TOL:
            flags.append(f"gap {v['gap']:.0e}")
        repeats = f", {v['repeats']} repeated" if v["repeats"] else ""
        return f"{v['status']} ({v['points']} pts{repeats}{', ' if flags else ''}{', '.join(flags)})"
    return str(v["status"])


def report(env: dict[str, object], results: dict[str, list[dict[str, object]]], verbose: bool) -> None:
    """Print the environment, a per-family summary, and every non-exact or disagreeing case."""
    print("environment")
    for k, v in env.items():
        print(f"  {k:14s}: {v}")
    print()
    print("summary: number of problems per verdict (exact / wrong / cycled / raised)")
    agree_cols = [f"{i}=cvxcla" for i in BLDP]
    print(
        f"  {'family':15s} {'problems':>8s}  "
        + "  ".join(f"{i:>21s}" for i in IMPLEMENTATIONS)
        + "".join(f"  {c:>18s}" for c in agree_cols)
    )
    for family, rows in results.items():
        cells = []
        for impl in IMPLEMENTATIONS:
            c = Counter(r["verdicts"][impl]["status"] for r in rows)
            raised = sum(v for k, v in c.items() if k not in {"exact", "wrong", "cycled"})
            cells.append(f"{c['exact']:>3d} /{c['wrong']:>3d} /{c['cycled']:>3d} /{raised:>3d}")
        agree = []
        for impl in BLDP:
            pair = [r["pairs"][f"{impl}|cvxcla"]["agree"] for r in rows if f"{impl}|cvxcla" in r["pairs"]]
            agree.append(f"{sum(pair)} of {len(pair)}")
        print(
            f"  {family:15s} {len(rows):>8d}  "
            + "  ".join(f"{c:>21s}" for c in cells)
            + "".join(f"  {a:>18s}" for a in agree)
        )

    print()
    print("cases (all of them with --verbose; otherwise only those not exact everywhere or not agreeing)")
    for family, rows in results.items():
        for r in rows:
            clean = all(v["status"] == "exact" for v in r["verdicts"].values()) and all(
                a["agree"] for a in r["pairs"].values()
            )
            if clean and not verbose:
                continue
            print(f"  {family} {r['label']}")
            for impl in IMPLEMENTATIONS:
                v = r["verdicts"][impl]
                extra = f"  -- {v['message']}" if "message" in v else ""
                print(f"      {impl:10s} {_cell(v)}{extra}")
            for name, a in r["pairs"].items():
                print(
                    f"      {name:20s} {'agree' if a['agree'] else 'DISAGREE'}: distinct points {a['points'][0]} "
                    f"vs {a['points'][1]}, path distance {a['distance']:.1e}"
                )

    if "conditioning" in results:
        print()
        print("conditioning sweep: worst objective gap over seeds (raised / cycled shown by name)")
        by_kappa: dict[str, dict[str, list[object]]] = defaultdict(lambda: defaultdict(list))
        for r in results["conditioning"]:
            for impl in IMPLEMENTATIONS:
                v = r["verdicts"][impl]
                by_kappa[r["problem"]][impl].append(v["gap"] if "gap" in v else v["status"])
        print(f"  {'':10s}" + "".join(f"{i:>18s}" for i in IMPLEMENTATIONS))
        for label, row in by_kappa.items():
            cells = []
            for impl in IMPLEMENTATIONS:
                vals = row[impl]
                failed = sorted({str(x) for x in vals if isinstance(x, str)})
                gaps = [x for x in vals if not isinstance(x, str)]
                text = f"{max(gaps):.1e}" if gaps else ""
                if failed:
                    text = (text + " " if text else "") + "+" + "/".join(failed)
                cells.append(text)
            print(f"  {label:10s}" + "".join(f"{c:>18s}" for c in cells))


def main() -> None:
    """Build every family, run the three implementations, validate and compare."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--seeds", type=int, default=3, help="instances per problem in each family")
    parser.add_argument("--verbose", action="store_true", help="list every case, not just the interesting ones")
    parser.add_argument("--json", type=Path, default=None, help="write the raw verdicts here")
    args = parser.parse_args()

    import importlib.metadata as md

    env = environment(None)
    env["packages"] = {**env["packages"], **{p: md.version(p) for p in ("cvxpy", "clarabel")}}
    del env["threads_limit"], env["thread_env"]
    results: dict[str, list[dict[str, object]]] = {}
    for family in args.families:
        results[family] = []
        for seed in range(args.seeds):
            for p in FAMILIES[family](np.random.default_rng(seed)):
                r = check(p)
                r["problem"], r["label"] = p["label"], f"{p['label']} seed={seed}"
                results[family].append(r)
                cells = "  ".join(f"{i}={r['verdicts'][i]['status']}" for i in IMPLEMENTATIONS)
                print(f"{family:15s} {r['label']:28s} {cells}", file=sys.stderr, flush=True)
    print()
    report(env, results, args.verbose)
    if args.json:
        args.json.write_text(json.dumps({"environment": env, "results": results}, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

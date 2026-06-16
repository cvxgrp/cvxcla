"""Map the degeneracy boundary of the CLA, and show how projection resolves it.

The companion paper documents a real limitation on near-degenerate problems. This
experiment characterises it and demonstrates the fix in ``cvxcla.cla._emit``.

We fix the universe at ``n`` assets and estimate the sample covariance from ``T``
observations, sweeping ``T`` from ``T > n`` (full rank, well posed) down to
``T << n`` (severely rank deficient). For each ``T`` we trace the long-only
frontier and record:

* whether the trace completes,
* the worst box-bound violation of a *candidate* turning point (before the
  projection in ``_emit`` repairs it), and
* the objective gap of every completed turning point against a reference QP
  solve (so we can confirm the completed frontier is genuinely optimal, not just
  feasible).

The story (see the paper): a near-rank-deficient covariance has near-flat
directions; round-off pushes a candidate weight a hair (<= ~1e-2) outside its
box along one of them, so the candidate is optimal but not exactly feasible.
``_emit`` projects it back onto the box and the trace completes with objective
gap ~ 1e-9. Only when the free set grows past the covariance rank does the solve
become unreliable, emitting a gross (O(0.1)) violation; the projection then
refuses (a guard at ``GUARD``) rather than return a suboptimal frontier.

Writes docs/paper/degeneracy.pdf.

Usage:
    uv run --with cvxpy --with matplotlib python experiments/degeneracy_boundary.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import cvxcla.cla as cla_module
from cvxcla import CLA

N_ASSETS = 120
WINDOWS = [240, 180, 150, 130, 120, 100, 90, 75, 60, 45, 30, 20, 15]
SEED = 0
GUARD = 1e-2  # the gross-violation guard in CLA._emit (round-off vs broken solve)


@dataclass
class SweepResult:
    """Outcome of one trace attempt at a given number of observations T."""

    t_obs: int
    rank: int
    completed: bool
    n_points: int
    raw_violation: float  # worst candidate box violation before projection
    obj_gap: float  # worst objective gap of a completed turning point vs QP


def _max_objective_gap(cla: CLA, mean: np.ndarray, cov: np.ndarray) -> float:
    """Worst (obj_cla - obj_qp) over turning points; ~0 means the frontier is optimal.

    Returns 0.0 when cvxpy is unavailable so the sweep still runs (the figure then
    omits the optimality series).
    """
    try:
        import cvxpy as cp
    except ImportError:
        return 0.0
    n = len(mean)
    gap = 0.0
    for tp in cla.turning_points:
        lam = tp.lamb
        if not np.isfinite(lam) or lam <= 0:
            continue
        w = cp.Variable(n)
        cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(w, cov, assume_PSD=True) - lam * (mean @ w)),
            [cp.sum(w) == 1, w >= 0, w <= 1],
        ).solve(solver=cp.CLARABEL)
        if w.value is None:
            continue
        wc = tp.weights
        obj_cla = 0.5 * wc @ cov @ wc - lam * mean @ wc
        obj_qp = 0.5 * w.value @ cov @ w.value - lam * mean @ w.value
        gap = max(gap, float(obj_cla - obj_qp))
    return gap


def _trace(t_obs: int) -> SweepResult:
    """Trace the frontier for a size-``t_obs`` sample, recording the diagnostics."""
    rng = np.random.default_rng(SEED)
    returns = rng.standard_normal((t_obs, N_ASSETS)) * 0.01 + rng.uniform(0.0, 1e-3, N_ASSETS)
    cov = np.cov(returns, rowvar=False)
    mean = returns.mean(axis=0)
    kwargs = {
        "lower_bounds": np.zeros(N_ASSETS),
        "upper_bounds": np.ones(N_ASSETS),
        "a": np.ones((1, N_ASSETS)),
        "b": np.ones(1),
    }

    worst = [0.0]
    original_emit = CLA._emit

    def recording_emit(self: CLA, lamb: float, weights: np.ndarray, free: np.ndarray) -> None:
        below = float(np.max(self.lower_bounds - weights))
        above = float(np.max(weights - self.upper_bounds))
        worst[0] = max(worst[0], below, above, 0.0)
        original_emit(self, lamb, weights, free)

    cla_module.CLA._emit = recording_emit
    try:
        cla = CLA(mean=mean, covariance=cov, **kwargs)
        completed, n_points = True, len(cla)
        obj_gap = _max_objective_gap(cla, mean, cov)
    except ValueError:
        completed, n_points, obj_gap = False, 0, 0.0
    finally:
        cla_module.CLA._emit = original_emit

    return SweepResult(
        t_obs=t_obs,
        rank=int(np.linalg.matrix_rank(cov)),
        completed=completed,
        n_points=n_points,
        raw_violation=worst[0],
        obj_gap=obj_gap,
    )


def main() -> None:
    """Run the sweep, print a table, and write the figure."""
    results = [_trace(t) for t in WINDOWS]

    print(f"universe n = {N_ASSETS}; projection guard = {GUARD:g}\n")
    print(f"{'T':>5} {'rank':>5} {'status':>9} {'pts':>5} {'raw viol':>11} {'obj gap':>11}")
    for r in results:
        status = "completed" if r.completed else "aborted"
        print(f"{r.t_obs:>5} {r.rank:>5} {status:>9} {r.n_points:>5} {r.raw_violation:>11.2e} {r.obj_gap:>11.2e}")

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not available - skipping docs/paper/degeneracy.pdf")
        return

    ts = [r.t_obs for r in results]
    viol = [max(r.raw_violation, 1e-12) for r in results]
    done = [r.completed for r in results]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for t, v, ok in zip(ts, viol, done, strict=True):
        ax.scatter(t, v, s=36, zorder=3, color="#1f4e79" if ok else "#c00000", marker="o" if ok else "X")
    ax.axhline(GUARD, ls="--", lw=1.0, color="#555555")
    ax.text(ts[0], GUARD * 1.3, "projection guard", ha="right", va="bottom", fontsize=7.5, color="#555555")
    ax.axvline(N_ASSETS, ls=":", lw=1.0, color="#999999")
    ax.text(
        N_ASSETS * 1.03, min(viol) * 1.5, f"$T=n={N_ASSETS}$", rotation=90, va="bottom", fontsize=7.5, color="#777777"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observations $T$ (sample covariance rank $\\approx \\min(T-1, n)$)")
    ax.set_ylabel("Worst candidate box violation")
    ax.set_title(f"Projection repairs round-off; the guard refuses rank deficiency ($n={N_ASSETS}$)", fontsize=8.5)

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f4e79", ms=7, label="completed (optimal)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#c00000", ms=7, label="aborted (guard)"),
    ]
    ax.legend(handles=legend, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    out = "docs/paper/degeneracy.pdf"
    fig.savefig(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

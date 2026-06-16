"""Map the degeneracy boundary of the CLA event logic (paper figure).

The companion paper documents a real limitation: on tie-heavy or
near-degenerate problems the trace can stop early. This experiment characterises
*where* that happens and *why* it is not a conditioning failure.

Fixing the universe at the full S&P 500 (N assets), we estimate the sample
covariance from the last ``W`` trading days and trace the long-only frontier,
sweeping the window length ``W``. A short window (``W`` small relative to ``N``)
makes the sample covariance near-rank-deficient and its frontier events nearly
coincident; a long window is well separated. For each ``W`` we record whether
the full trace completes and, at the point where it stops, three diagnostics:

* the worst box-bound violation of the candidate turning point,
* the feasibility tolerance ``tol`` it is compared against,
* the condition number of the free-asset block ``Sigma_FF``.

The headline finding (see the paper): when the trace aborts, the box violation
is only just above ``tol`` (order 1e-5) while ``Sigma_FF`` is *well*
conditioned (order 10). The failure is therefore accumulated round-off at a
degenerate vertex, not a singular free block.

Writes docs/paper/degeneracy.pdf.

Usage:
    uv run python experiments/fetch_sp500.py        # once, to download the data
    uv run --with pyarrow --with matplotlib python experiments/degeneracy_boundary.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import cvxcla.cla as cla_module
from cvxcla import CLA

DATA = Path(__file__).parent / "data" / "sp500_pct_returns.parquet"
WINDOWS = [80, 100, 120, 160, 220, 300, 400, 550, 750, 1000, 1213]


@dataclass
class TraceResult:
    """Outcome of one trace attempt at a given estimation-window length."""

    window: int
    rank: int
    completed: bool
    n_points: int
    free_at_stop: int
    violation: float  # worst box-bound violation of the stopping candidate
    cond_ff: float  # condition number of Sigma_FF at the stopping candidate


def _trace_with_diagnostics(mean: np.ndarray, cov: np.ndarray, tol: float) -> TraceResult | None:
    """Trace the long-only frontier, recording the stopping diagnostics.

    Wraps ``CLA._append`` for the duration of the trace to capture, for the last
    candidate turning point reached, its worst box violation and the condition
    number of the free block. Returns ``None`` if the data window is unusable.
    """
    n = cov.shape[0]
    kwargs = {
        "mean": mean,
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }
    last: dict[str, float] = {}
    original_append = CLA._append

    def recording_append(self: CLA, tp: object, tol: float | None = None) -> None:
        free = np.flatnonzero(tp.free)  # type: ignore[attr-defined]
        sub = cov[np.ix_(free, free)]
        below = float(np.max(self.lower_bounds - tp.weights))  # type: ignore[attr-defined]
        above = float(np.max(tp.weights - self.upper_bounds))  # type: ignore[attr-defined]
        last["violation"] = max(below, above, 0.0)
        last["cond_ff"] = float(np.linalg.cond(sub)) if free.size else 1.0
        last["free"] = float(free.size)
        return original_append(self, tp, tol)

    cla_module.CLA._append = recording_append
    try:
        cla = CLA(covariance=cov, tol=tol, **kwargs)
        completed, n_points = True, len(cla)
    except ValueError:
        completed, n_points = False, 0
    finally:
        cla_module.CLA._append = original_append

    return TraceResult(
        window=0,  # placeholder, set by caller
        rank=int(np.linalg.matrix_rank(cov)),
        completed=completed,
        n_points=n_points,
        free_at_stop=int(last.get("free", 0)),
        violation=float(last.get("violation", 0.0)),
        cond_ff=float(last.get("cond_ff", 1.0)),
    )


def main() -> None:
    """Sweep the estimation window, print a table, and write the figure."""
    returns = pd.read_parquet(DATA).to_numpy()
    _, n = returns.shape
    tol = CLA.tol  # the default feasibility tolerance (1e-5)

    results: list[TraceResult] = []
    print(f"universe N = {n}; feasibility tol = {tol:g}\n")
    print(f"{'W':>5} {'rank':>5} {'status':>9} {'pts/free':>9} {'box viol':>11} {'cond(S_FF)':>11}")
    for window in WINDOWS:
        sample = returns[-window:]
        cov = np.cov(sample, rowvar=False)
        res = _trace_with_diagnostics(sample.mean(axis=0), cov, tol)
        if res is None:
            continue
        res.window = window
        results.append(res)
        status = "completed" if res.completed else "aborted"
        count = res.n_points if res.completed else res.free_at_stop
        print(f"{window:>5} {res.rank:>5} {status:>9} {count:>9} {res.violation:>11.2e} {res.cond_ff:>11.1f}")

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping docs/paper/degeneracy.pdf")
        return

    windows = [r.window for r in results]
    viol = [max(r.violation, 1e-12) for r in results]
    cond = [r.cond_ff for r in results]
    done = [r.completed for r in results]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    # Worst box-bound violation at the stopping candidate, on a log axis,
    # coloured by whether the full trace completed.
    for w, v, ok in zip(windows, viol, done, strict=True):
        ax.scatter(w, v, s=34, zorder=3, color="#1f4e79" if ok else "#c00000", marker="o" if ok else "X")
    ax.axhline(tol, ls="--", lw=1.0, color="#555555")
    ax.text(windows[-1], tol * 1.25, "feasibility tol", ha="right", va="bottom", fontsize=7.5, color="#555555")
    ax.axvline(n, ls=":", lw=1.0, color="#999999")
    ax.text(n * 1.02, min(viol) * 1.5, f"$W=N={n}$", rotation=90, va="bottom", fontsize=7.5, color="#777777")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Estimation window $W$ (trading days)")
    ax.set_ylabel("Worst box violation at stop")
    ax.set_title("Where the CLA trace aborts (S&P 500)", fontsize=9)

    # Twin axis: condition number of the free block at the stopping candidate,
    # to show it stays small (order 10) even when the trace aborts.
    ax2 = ax.twinx()
    ax2.plot(windows, cond, "-s", ms=3, lw=0.9, color="#2ca02c", alpha=0.7)
    ax2.set_ylabel(r"cond$(\Sigma_{FF})$ at stop", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f4e79", ms=7, label="trace completed"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#c00000", ms=7, label="trace aborted"),
        Line2D([0], [0], color="#2ca02c", marker="s", ms=4, label=r"cond$(\Sigma_{FF})$"),
    ]
    ax.legend(handles=legend, fontsize=7.5, loc="center right")
    fig.tight_layout()
    out = "docs/paper/degeneracy.pdf"
    fig.savefig(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

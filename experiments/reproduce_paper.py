r"""Reproduce every figure and table in the paper from a clean checkout.

This is the single entry point for the paper's empirical results. It imports each
figure- and table-producing experiment in order and calls its ``main()``,
reporting which paper artefact each one regenerates:

==================================  ==========================================
Experiment                          Paper artefact
==================================  ==========================================
frontier_20x50                      Figure 1  (frontier.pdf)
runtime_scaling                     Figure 2  (scaling.pdf) + Table 1
rank_scaling                        Figure 3  (rank_scaling.pdf) + Table 2
validate_exact                      Section 10.5 exactness numbers
frontier_real                       Figure 4  (real_frontier.pdf)
degeneracy_boundary                 Figure 5  (degeneracy.pdf)
==================================  ==========================================

The S&P 500 input is the frozen snapshot committed at
``experiments/data/sp500_pct_returns.parquet`` (see ``fetch_sp500.py``), so no
network access is required. PDFs are written into ``docs/paper/``; run this from
the repository root so those relative paths resolve.

Beyond ``cvxcla`` itself the experiments need a few third-party packages:
``matplotlib`` (all figures), ``pandas``/``pyarrow`` (the S&P 500 data),
``cvxpy`` (the QP cross-checks in ``validate_exact`` and ``degeneracy_boundary``),
and ``PyPortfolioOpt`` (the external baseline in ``runtime_scaling``). A step whose
dependency is missing is reported and skipped rather than aborting the whole run.

Usage::

    uv run --with matplotlib --with pandas --with pyarrow --with cvxpy \\
        --with pyportfolioopt python experiments/reproduce_paper.py

Note that ``runtime_scaling`` traces universes up to n=640 and times
PyPortfolioOpt, which takes several minutes; pass ``--quick`` to skip the two
long-running scaling sweeps and regenerate only the fast artefacts.
"""

from __future__ import annotations

import importlib
import sys

# (module, paper artefact, slow?) in the order the paper presents them. The
# modules are siblings of this file, which sys.path[0] makes importable by name.
STEPS: list[tuple[str, str, bool]] = [
    ("frontier_20x50", "Figure 1 (frontier.pdf)", False),
    ("runtime_scaling", "Figure 2 (scaling.pdf) + Table 1", True),
    ("rank_scaling", "Figure 3 (rank_scaling.pdf) + Table 2", True),
    ("validate_exact", "Section 10.5 exactness numbers", False),
    ("frontier_real", "Figure 4 (real_frontier.pdf)", False),
    ("degeneracy_boundary", "Figure 5 (degeneracy.pdf)", False),
]


def run_step(module_name: str, artefact: str) -> bool:
    """Import one experiment module, call its ``main()``, and report the outcome.

    Args:
        module_name: Name of a sibling experiment module (without ``.py``).
        artefact: Human-readable paper artefact the experiment regenerates.

    Returns:
        ``True`` if ``main()`` ran cleanly; ``False`` if an optional dependency is
        missing or the run raised (the error is reported, not propagated).
    """
    print(f"\n=== {module_name}  ->  {artefact} ===", flush=True)
    try:
        module = importlib.import_module(module_name)
        module.main()
    except ImportError as exc:
        print(f"[skip] {module_name}: missing optional dependency ({exc})", flush=True)
        return False
    except Exception as exc:  # noqa: BLE001 - one experiment must not abort the rest
        print(f"[FAILED] {module_name}: {type(exc).__name__}: {exc}", flush=True)
        return False
    print(f"[ok] {module_name}", flush=True)
    return True


def main() -> int:
    """Run every paper experiment and summarise which artefacts were regenerated.

    Returns:
        Process exit code: ``0`` if every attempted step succeeded, ``1`` if any
        step failed or was skipped for a missing dependency.
    """
    quick = "--quick" in sys.argv[1:]
    steps = [s for s in STEPS if not (quick and s[2])]
    if quick:
        print("[--quick] skipping the long-running scaling sweeps", flush=True)

    results = {module: run_step(module, artefact) for module, artefact, _ in steps}

    print("\n=== summary ===", flush=True)
    for module, ok in results.items():
        print(f"  {'ok  ' if ok else 'FAIL'}  {module}", flush=True)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

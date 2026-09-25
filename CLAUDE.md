# CLAUDE.md

Guidance for Claude Code (and human contributors) working in this repository.

## What this project is

`cvxcla` implements the **Critical Line Algorithm (CLA)** of Markowitz for
computing the efficient frontier of a portfolio-optimization problem. The
library lives in `src/cvxcla/`; everything else is either tests, docs, or
development infrastructure synced from the Rhiza template.

## The Rhiza split — read this before editing config

This repo syncs its development infrastructure (CI workflows, Makefile, linters,
test harness, release tooling) from the **mother repo `jebel-quant/rhiza`**.
The pinned template version, profile and exclusions live in
`.rhiza/template.yml` — read it there rather than trusting a copy in prose,
which drifts on every sync.

### Rhiza-owned (do NOT edit locally)

The authoritative, machine-generated list of synced files is the `files:` block
of `.rhiza/template.lock`:

```bash
sed -n '/^files:/,/^[a-z_]*:/p' .rhiza/template.lock
```

**Do not hand-edit any file in that list** — your change will be silently
overwritten on the next template sync (`/rhiza:update`). Fix Rhiza-owned
problems upstream in `jebel-quant/rhiza`, or adjust `.rhiza/template.yml`
(e.g. its `exclude:` list) and re-sync. Two entries are easy to miss: the
`Makefile` is template-owned, and so is `tests/test_rhiza_packaging.py`
despite living under `tests/`.

### Locally owned (edit freely — this is the actual project)

- `src/cvxcla/` — the library. Modules: `cla.py` (the CLA solver), `lasso.py`,
  the `operators/` package (`_core.py` — the `QuadraticForm`/`CovarianceOperator`
  protocols plus the `cross`/`bordered_solve` helpers; `builders.py` — factory
  functions that wrap the `cvx.linalg` operators, `DenseOperator`/`GramOperator`/
  `FactorOperator`, as covariance/quadratic-form backends), `_builders.py`
  (the `ProblemBuilder`/`LassoBuilder` fluent builders; `builder.py` is its
  public re-export), `types.py`, `pathtracer.py`, `first.py` (first turning point), `__init__.py`.
  The per-turning-point numeric kernels are factored out of `cla.py` into pure
  private modules: `_kkt.py` (`active_set`/`solve_kkt`, composed with the
  `Segment` bundle by `critical_segment`), `_events.py`
  (`event_ratios`/`ineq_event_ratios`, stacked by `segment_events`),
  `_projection.py` (`project_feasible` and its capped-simplex/alternating
  workers), and `_checks.py` (the `well_conditioned`/`guard_degeneracy`
  conditioning tests and the `check_feasible` constraint validation); the
  first-vertex dispatch is `first.first_turning_point`. `cla.py` is the orchestrator that
  wires them into the `ParametricProblem` hooks. Leverage caps `||w||_1 <= c`
  (`CLA(leverage=...)`) live in `_leverage.py`: the long/short leg split
  (`LeverageLift`), the lifted covariance `SignedLift`, and the leg-event mask;
  `cla.py` traces the lifted problem with its private `_LeveragedCLA` subclass
  and maps the turning points back to asset weights. `lasso.py` is factored the same
  way, over `_lasso.py` (the `LassoSegment`/`LassoState` kernel with
  `scan_events`/`solve_segment`) and `_lasso_validate.py` (the design, operator
  and constraint input validators). Equality-constrained LASSO paths
  (`Lasso(a=...)`, `A beta = 0`) bypass the homotopy: `_lasso_cla.py` traces one
  leverage-capped `CLA` on `X^T X`, `X^T y` and rescales its turning points
  (`beta = w / lam`), reading the penalty off the KKT system.
- `tests/` — the project test suite (unit, property-based `test_properties.py`,
  benchmarks `tests/benchmarks/`). **Note:** `tests/test_rhiza_packaging.py`
  is Rhiza-owned; `make rhiza-test` runs the template's own structure,
  README and docstring checks, not this library's suite.
- `pyproject.toml` — package metadata, dependencies, tool config
  (`[tool.interrogate]`, etc.).
- `.rhiza/template.yml` — the one file under `.rhiza/` you may edit; it pins the
  Rhiza version/profile.
- Top-level prose not in the lock: `README.md`, `comparison.md`, `CHANGELOG.md`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.

When unsure whether a file is owned locally, grep `.rhiza/template.lock` — if
it's in the `files:` block, it's Rhiza-owned.

## Command policy

Always drive tooling through `make <target>`. **Never invoke `.venv/bin/...`
directly** — the targets bootstrap the environment, install pinned tools, and
pass the right flags. The `Makefile` is a thin shim that forwards every target
to the pinned `rhiza-task` runner; repo-specific settings (thresholds, paths)
live in the `[tool.rhiza-task]` table of `pyproject.toml`. `make help` lists
every task. Useful targets:

| Target | Purpose |
|--------|---------|
| `make fmt` | pre-commit hooks: ruff format/check, markdownlint, bandit, actionlint, interrogate, secrets |
| `make typecheck` | `ty` + `mypy --strict` over `src/` |
| `make docs-coverage` | interrogate docstring coverage |
| `make deps` | unused/missing/misplaced dependency analysis (deptry) |
| `make security` | bandit security scan over `src/` |
| `make rhiza-test` | the template's structure, README and docstring checks |
| `make test` | full suite **with** the coverage gate |

`make test` enforces `coverage_fail_under` from `[tool.rhiza-task]` in
`pyproject.toml` (currently **100%**). Coverage on
`src/` must stay at 100%; the only acceptable exclusions are the existing
`# pragma: no cover` on Protocol/abstract stubs and untyped-import
`# type: ignore[import-untyped]` on scipy.

To assess overall repo quality against Rhiza standards, run the `/rhiza:quality`
slash command (from the rhiza Claude Code plugin). To bump the Rhiza pin and
re-sync, use `/rhiza:update`.

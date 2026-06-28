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
The pin lives in `.rhiza/template.yml`:

```yaml
repository: "jebel-quant/rhiza"
ref: "v0.19.5"
profiles:
  - github-project
```

The authoritative, machine-generated list of synced files is the `files:` block
of `.rhiza/template.lock`. **Do not hand-edit any file in that list** — your
change will be silently overwritten on the next `make rhiza-update` /
`rhiza_sync` run. Fix Rhiza-owned problems upstream in `jebel-quant/rhiza`, or
adjust `.rhiza/template.yml` and re-sync.

### Rhiza-owned (do NOT edit locally)

Synced from the template — treat as read-only. Highlights from
`.rhiza/template.lock`:

- **All of `.rhiza/`** except `.rhiza/template.yml` (its tests, `make.d/*.mk`,
  requirements, utils, completions, semgrep/bandit config).
- `Makefile`, `pytest.ini`, `ruff.toml`, `.pre-commit-config.yaml`, `.bandit`,
  `.editorconfig`, `.python-version`, `cliff.toml`.
- **All `.github/workflows/rhiza_*.yml`**, plus issue/PR/discussion templates,
  `dependabot.yml`, rulesets, and `secret_scanning.yml`.
- `.claude/commands/rhiza_*.md` (the `rhiza_quality`, `rhiza_book`,
  `rhiza_update` slash commands).
- `docs/index.md`, `docs/mkdocs-base.yml`, `docs/development/TESTS.md`,
  `docs/development/MARIMO.md`, `docs/assets/`.

### Locally owned (edit freely — this is the actual project)

- `src/cvxcla/` — the library. Modules: `cla.py` (the CLA solver), `lasso.py`,
  the `operators/` package (`_core.py`, `dense.py`, `factor.py`, `gram.py` —
  the covariance/quadratic-form backends), `builder.py`, `types.py`,
  `pathtracer.py`, `first.py` (first turning point), `__init__.py`.
- `tests/` — the project test suite (unit, property-based `test_properties.py`,
  fuzz `tests/fuzz/`, benchmarks `tests/benchmarks/`). **Note:**
  `.rhiza/tests/` is Rhiza-owned and tests the template itself, not this library.
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
directly** — the Make targets bootstrap the environment, install pinned tools,
and pass the right flags. Useful targets:

| Target | Purpose |
|--------|---------|
| `make fmt` | pre-commit hooks: ruff format/check, markdownlint, bandit, actionlint, interrogate, secrets |
| `make typecheck` | `ty` + `mypy --strict` over `src/` |
| `make docs-coverage` | interrogate docstring coverage |
| `make deptry` | unused/missing/misplaced dependency analysis |
| `make security` | pip-audit + bandit |
| `make validate` | validate project structure against `.rhiza/template.yml` |
| `make test` | full suite **with** the coverage gate |

`make test` enforces `COVERAGE_FAIL_UNDER` (currently **100%**). Coverage on
`src/` must stay at 100%; the only acceptable exclusions are the existing
`# pragma: no cover` on Protocol/abstract stubs and untyped-import
`# type: ignore[import-untyped]` on scipy.

To assess overall repo quality against Rhiza standards, run the `/rhiza_quality`
slash command. To bump the Rhiza pin and re-sync, use `/rhiza_update`.

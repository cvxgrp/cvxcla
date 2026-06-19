## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

DEFAULT_AI_MODEL=claude-sonnet-4.6
LOGO_FILE=.rhiza/assets/rhiza-logo.svg
GH_AW_ENGINE ?= copilot  # Default AI engine for gh-aw workflows (copilot, claude, or codex)

# Override template default: include mkdocstrings plugin for API docs
MKDOCS_EXTRA_PACKAGES = --with 'mkdocstrings[python]'

# Override template default (90): this project sustains full coverage, so the
# gate defends 100%. Set before the include so the template's `?=` keeps it,
# while `?=` here still lets a command-line override (e.g. bootstrapping) win.
COVERAGE_FAIL_UNDER ?= 100

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

##@ Paper benchmarks

# Pin BLAS/OpenMP threads so the paper timings are deterministic and comparable
# across runs (Apple Accelerate / OpenMP / MKL). Run on a quiesced machine
# (AC power, other apps closed) for paper-grade numbers.
BENCH_THREADS := VECLIB_MAXIMUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

.PHONY: bench-rank bench-runtime bench

bench-rank: ## Table 2 + Figure 3: trace time vs factor rank (cvxcla only, fast)
	$(BENCH_THREADS) uv run --with matplotlib python experiments/rank_scaling.py

bench-runtime: ## Table 1 + Figure 2: trace time vs problem size (incl. PyPortfolioOpt, ~15-20 min)
	$(BENCH_THREADS) uv run --with matplotlib --with pyportfolioopt==1.6.0 python experiments/runtime_scaling.py

bench: bench-rank bench-runtime ## Regenerate both paper timing tables and figures

# Optional: developer-local extensions (not committed)
-include local.mk

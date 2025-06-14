# Set the default target to help
.DEFAULT_GOAL := help

# Declare all phony targets (targets that don't represent files)
.PHONY: venv install fmt clean help test jupyter marimo

# Development Setup
venv: ## Create a Python virtual environment using uv
	curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv package manager
	uv venv --python '3.12'  # Create a virtual environment with Python 3.12

install: venv ## Install dependencies and setup environment
	uv pip install --upgrade pip  # Ensure pip is up to date
	uv sync --dev --frozen --all-extras  # Install dependencies from pyproject.toml

# Code Quality
fmt: venv ## Format and lint code
	uvx pre-commit run --all-files  # Run pre-commit hooks on all files

.PHONY: ty
ty: install
	@uvx ty check tests  #/jquantstats

# Cleanup
clean: ## Clean build artifacts and stale branches
	git clean -X -d -f  # Remove files ignored by git
	git branch -v | grep "\[gone\]" | cut -f 3 -d ' ' | xargs git branch -D  # Remove stale branches

# Testing
test: install ## Run tests
	uv pip install pytest pytest-cov # Install pytest
	uv run pytest --cov=cvxcla tests  # Run tests in src/tests directory

# Help
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)  # Extract and format targets with comments


marimo: install ## Run the current repo version
	@uv pip install --no-cache-dir marimo
	@uv run marimo edit book/marimo/cla.py

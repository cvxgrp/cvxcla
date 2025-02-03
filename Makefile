.DEFAULT_GOAL := help

.PHONY: venv install fmt clean help test jupyter

venv:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv --python '3.12'

install: venv ## Install dependencies and setup environment
	uv pip install --upgrade pip
	uv sync --dev --frozen

fmt: venv ## Format and lint code
	uv pip install pre-commit
	uv run pre-commit install
	uv run pre-commit run --all-files

clean: ## Clean build artifacts and stale branches
	git clean -X -d -f
	git branch -v | grep "\[gone\]" | cut -f 3 -d ' ' | xargs git branch -D

test: install ## Run tests
	uv pip install pytest
	uv run pytest src/tests

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

marimo: install ## Install a run a Jupyter Lab server
	uv pip install marimo
	#uv run jupyter lab

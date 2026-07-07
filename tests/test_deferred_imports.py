"""Guard the deferred ``builder`` <-> solver import invariant.

``builder`` imports ``CLA`` and ``Lasso`` at module level, so the solver modules
must *not* import ``builder`` at module level in return -- that would form a real
import cycle. Both ``CLA.problem`` and ``Lasso.problem`` therefore import
``.builder`` function-locally. These tests pin both halves of the contract:

* the solver modules keep the back-edge deferred (a hoisted top-level import
  would be caught here rather than as a runtime ``ImportError``); and
* the factories still build the right object at call time.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

from cvxcla import CLA, DenseCovariance, Lasso, LassoBuilder, ProblemBuilder


def _module_level_imports(module_name: str) -> set[str]:
    """Names imported at module level (top of file, not inside a function)."""
    source = Path(importlib.import_module(module_name).__file__).read_text()
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:  # only module-level statements, so function-local imports are excluded
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return names


@pytest.mark.parametrize("module_name", ["cvxcla.cla", "cvxcla.lasso"])
def test_solver_does_not_import_builder_at_module_level(module_name: str) -> None:
    """Neither solver module may import ``builder`` at module level (would cycle)."""
    assert ".builder" not in _module_level_imports(module_name)


def test_cla_problem_builds_without_import_error() -> None:
    """``CLA.problem`` resolves its deferred import and returns a ``ProblemBuilder``."""
    mean = np.array([0.1, 0.2, 0.3])
    covariance = np.eye(3)
    assert isinstance(CLA.problem(mean, covariance), ProblemBuilder)


def test_lasso_problem_builds_without_import_error() -> None:
    """``Lasso.problem`` resolves its deferred import and returns a ``LassoBuilder``."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 4))
    y = rng.standard_normal(10)
    assert isinstance(Lasso.problem(x, y), LassoBuilder)


def test_lasso_from_operator_builds_without_import_error() -> None:
    """``Lasso.from_operator`` constructs a traced ``Lasso`` in operator mode."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 4))
    y = rng.standard_normal(10)
    lasso = Lasso.from_operator(DenseCovariance(x.T @ x), x.T @ y)
    assert isinstance(lasso, Lasso)
    assert lasso.path

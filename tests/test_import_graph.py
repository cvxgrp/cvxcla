"""Pin the acyclic ``builder`` -> solver import direction.

Each fluent builder is co-located with the object it constructs:
:class:`ProblemBuilder` lives with ``CLA`` in ``cvxcla.cla`` and ``LassoBuilder``
lives with ``Lasso`` in ``cvxcla.lasso``. The thin ``cvxcla.builder`` module only
re-exports them. That makes the internal import graph acyclic -- the solver
modules never import ``builder`` -- so no ``TYPE_CHECKING`` guard or function-local
back-edge is needed. These tests pin both halves of the contract:

* the solver modules must not import ``builder`` at module level (a hoisted import
  would reintroduce the cycle and is caught here); and
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
def test_solver_does_not_import_builder(module_name: str) -> None:
    """Neither solver module may import ``builder`` (would form an import cycle)."""
    assert ".builder" not in _module_level_imports(module_name)


def test_builder_reexports_from_solvers() -> None:
    """``builder`` re-exports the co-located builders from the solver modules."""
    imports = _module_level_imports("cvxcla.builder")
    assert ".cla" in imports
    assert ".lasso" in imports


def test_cla_problem_returns_builder() -> None:
    """``CLA.problem`` returns a ``ProblemBuilder``."""
    mean = np.array([0.1, 0.2, 0.3])
    covariance = np.eye(3)
    assert isinstance(CLA.problem(mean, covariance), ProblemBuilder)


def test_lasso_problem_returns_builder() -> None:
    """``Lasso.problem`` returns a ``LassoBuilder``."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 4))
    y = rng.standard_normal(10)
    assert isinstance(Lasso.problem(x, y), LassoBuilder)


def test_lasso_from_operator_builds() -> None:
    """``Lasso.from_operator`` constructs a traced ``Lasso`` in operator mode."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 4))
    y = rng.standard_normal(10)
    lasso = Lasso.from_operator(DenseCovariance(x.T @ x), x.T @ y)
    assert isinstance(lasso, Lasso)
    assert lasso.path

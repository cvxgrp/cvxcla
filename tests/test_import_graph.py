"""Pin the acyclic ``solver`` -> ``_builders`` import direction.

Both fluent builders live in the leaf module :mod:`cvxcla._builders`. That module
never imports a solver: :class:`ProblemBuilder` and :class:`LassoBuilder` are
generic in the class they construct, and ``CLA.problem`` / ``Lasso.problem`` pass
``cls`` in as ``solver``. So the dependency runs strictly one way --
``cla``/``lasso`` -> ``_builders`` -> ``operators`` -- and no ``TYPE_CHECKING``
guard or function-local back-edge is needed anywhere.

These tests pin every half of that contract:

* ``_builders`` must not import either solver, at module level *or* inside a
  function body (a deferred import is the failure mode that survives for years,
  because unlike a module-level cycle it does not crash on import);
* the solvers must import ``_builders``, not the other way round;
* ``builder`` remains a pure re-export of the leaf module; and
* the factories still build the right object at call time, so the injected
  ``solver`` really is the class it claims to be.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

from cvxcla import CLA, DenseCovariance, Lasso, LassoBuilder, ProblemBuilder


def _source(module_name: str) -> str:
    """Read the on-disk source of an importable module."""
    return Path(importlib.import_module(module_name).__file__).read_text()


def _module_level_imports(module_name: str) -> set[str]:
    """Names imported at module level (top of file, not inside a function)."""
    tree = ast.parse(_source(module_name))
    names: set[str] = set()
    for node in tree.body:  # only module-level statements, so function-local imports are excluded
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return names


def _all_imports(module_name: str) -> set[str]:
    """Every name imported anywhere in the module, including inside function bodies."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(_source(module_name))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return names


@pytest.mark.parametrize("solver_module", [".cla", ".lasso"])
def test_builders_module_is_a_leaf(solver_module: str) -> None:
    """``_builders`` imports no solver -- not at module level, not deferred."""
    assert solver_module not in _all_imports("cvxcla._builders")


def test_builders_module_imports_only_operators_internally() -> None:
    """The only intra-package import in ``_builders`` is the leaf ``operators``."""
    internal = {name for name in _all_imports("cvxcla._builders") if name.startswith(".")}
    assert internal == {".operators"}


@pytest.mark.parametrize("module_name", ["cvxcla.cla", "cvxcla.lasso"])
def test_solver_imports_builders(module_name: str) -> None:
    """Each solver depends on ``_builders``, pinning the direction of the edge."""
    assert "._builders" in _module_level_imports(module_name)


@pytest.mark.parametrize("module_name", ["cvxcla.cla", "cvxcla.lasso"])
def test_solver_does_not_import_builder(module_name: str) -> None:
    """No solver may import the ``builder`` re-export shim (that would be a cycle)."""
    assert ".builder" not in _all_imports(module_name)


def test_builder_reexports_from_the_leaf_module() -> None:
    """``builder`` is a pure re-export of ``_builders``, not of the solvers."""
    imports = _module_level_imports("cvxcla.builder")
    assert "._builders" in imports
    assert ".cla" not in imports
    assert ".lasso" not in imports


def test_cla_problem_returns_builder() -> None:
    """``CLA.problem`` returns a ``ProblemBuilder``."""
    mean = np.array([0.1, 0.2, 0.3])
    covariance = np.eye(3)
    assert isinstance(CLA.problem(mean, covariance), ProblemBuilder)


def test_cla_builder_traces_to_a_cla() -> None:
    """The injected solver really is ``CLA``, so ``.trace()`` yields one."""
    mean = np.array([0.1, 0.2, 0.3])
    cla = CLA.problem(mean, np.eye(3)).long_only().budget().trace()
    assert isinstance(cla, CLA)


def test_lasso_problem_returns_builder() -> None:
    """``Lasso.problem`` returns a ``LassoBuilder``."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 4))
    y = rng.standard_normal(10)
    assert isinstance(Lasso.problem(x, y), LassoBuilder)


def test_lasso_builder_traces_to_a_lasso() -> None:
    """The injected solver really is ``Lasso``, so ``.trace()`` yields one."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 4))
    y = rng.standard_normal(10)
    assert isinstance(Lasso.problem(x, y).trace(), Lasso)


def test_lasso_from_operator_builds() -> None:
    """``Lasso.from_operator`` constructs a traced ``Lasso`` in operator mode."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 4))
    y = rng.standard_normal(10)
    lasso = Lasso.from_operator(DenseCovariance(x.T @ x), x.T @ y)
    assert isinstance(lasso, Lasso)
    assert lasso.path

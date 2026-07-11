"""Backward-compatible re-export of the fluent problem builders.

The builders are co-located with the objects they construct -- to keep the
internal import graph acyclic, :class:`ProblemBuilder` lives with
:class:`cvxcla.cla.CLA` in :mod:`cvxcla.cla` and :class:`LassoBuilder` lives with
:class:`cvxcla.lasso.Lasso` in :mod:`cvxcla.lasso`. This module re-exports both so
the historical import path ``cvxcla.builder.ProblemBuilder`` keeps working.

Each builder is a thin, chainable convenience layer over its solver's explicit
constructor. Construct one via :meth:`cvxcla.cla.CLA.problem` /
:meth:`cvxcla.lasso.Lasso.problem`, chain the constraint methods, and finish with
``.trace()``.

Examples:
    >>> import numpy as np
    >>> from cvxcla import CLA
    >>> rng = np.random.default_rng(0)
    >>> mean = rng.uniform(0.0, 1.0, 4)
    >>> covariance = np.eye(4)
    >>> cla = CLA.problem(mean, covariance).long_only().budget().trace()
    >>> len(cla) > 0
    True
"""

from __future__ import annotations

from .cla import ProblemBuilder
from .lasso import LassoBuilder

__all__ = ["LassoBuilder", "ProblemBuilder"]

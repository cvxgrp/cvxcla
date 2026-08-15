"""Backward-compatible re-export of the fluent problem builders.

Both builders live in the leaf module :mod:`cvxcla._builders`, which imports
neither :mod:`cvxcla.cla` nor :mod:`cvxcla.lasso` -- each solver injects itself
into its builder via ``.problem(...)``, so the dependency runs one way and the
internal import graph stays acyclic. This module re-exports both so the
historical import path ``cvxcla.builder.ProblemBuilder`` keeps working.

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

from ._builders import LassoBuilder, ProblemBuilder

__all__ = ["LassoBuilder", "ProblemBuilder"]

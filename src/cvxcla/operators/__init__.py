"""Covariance backend abstraction for the Critical Line Algorithm.

The turning-point loop of the CLA touches the covariance matrix through a
small number of operations: full matrix-vector products, solves against the
free-asset block, and cross-products between free and blocked assets. This
package defines that contract as the ``CovarianceOperator`` protocol (see
:mod:`cvxcla.operators._core`), together with several implementations:

- ``DenseCovariance`` / ``IncrementalDenseCovariance`` (:mod:`cvxcla.operators.dense`):
  adapters that reproduce the behaviour of a plain ``numpy`` covariance matrix,
  the latter maintaining the free-block inverse across turning points.
- ``FactorCovariance`` (:mod:`cvxcla.operators.factor`): a diagonal-plus-low-rank
  covariance ``Sigma = diag(d) + U @ Delta @ U.T`` whose solves go through the
  Woodbury identity, so no ``n x n`` matrix is ever materialised. Memory and
  per-solve cost are ``O(n * k)`` instead of ``O(n^2)``.
- ``GramCovariance`` (:mod:`cvxcla.operators.gram`): a sample covariance backed by
  the ``(T, n)`` data matrix, never forming ``Sigma``.

See https://github.com/cvxgrp/cvxcla/issues/646 for the roadmap.
"""

from ._core import CovarianceOperator, QuadraticForm, bordered_solve
from .dense import DenseCovariance, IncrementalDenseCovariance
from .factor import FactorCovariance
from .gram import GramCovariance

__all__ = [
    "CovarianceOperator",
    "DenseCovariance",
    "FactorCovariance",
    "GramCovariance",
    "IncrementalDenseCovariance",
    "QuadraticForm",
    "bordered_solve",
]

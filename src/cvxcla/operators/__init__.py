"""Operator layer for the Critical Line Algorithm.

The turning-point loop reaches its Hessian (the covariance ``Sigma``, or the Gram
matrix ``X.T @ X`` for a LASSO / LARS path) through the cvx-linalg
:class:`~cvx.linalg.SymmetricOperator` protocol: matrix-vector and sub-block
products, a free-block solve, and a reciprocal-condition check. This package no
longer defines its own operator classes -- the backends (dense, matrix-free Gram,
diagonal-plus-low-rank Woodbury, and a maintained-inverse dense variant) live in
:mod:`cvx.linalg`. What remains here is thin:

- :data:`QuadraticForm` / :data:`CovarianceOperator`: aliases of
  :class:`cvx.linalg.SymmetricOperator`, the loop's backend contract.
- builders (:mod:`cvxcla.operators.builders`) that assemble the right cvx-linalg
  operator from CLA / LASSO inputs, kept under the familiar ``*Covariance`` names.
- :func:`bordered_solve` and :func:`cross`, the parametric-path helpers built on
  the operator protocol.
"""

from ._core import CovarianceOperator, QuadraticForm, bordered_solve, cross
from .builders import (
    DenseCovariance,
    FactorCovariance,
    GramCovariance,
    IncrementalDenseCovariance,
    dense_covariance,
    factor_covariance,
    gram_covariance,
    incremental_dense_covariance,
)

__all__ = [
    "CovarianceOperator",
    "DenseCovariance",
    "FactorCovariance",
    "GramCovariance",
    "IncrementalDenseCovariance",
    "QuadraticForm",
    "bordered_solve",
    "cross",
    "dense_covariance",
    "factor_covariance",
    "gram_covariance",
    "incremental_dense_covariance",
]

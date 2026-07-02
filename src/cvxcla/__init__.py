"""Critical Line Algorithm (CLA) for Portfolio Optimization.

This package implements the Critical Line Algorithm introduced by Harry Markowitz
for computing the efficient frontier in portfolio optimization problems.
The algorithm efficiently computes the turning points of the efficient frontier,
which are the points where the set of assets at their bounds changes.

The main class to use is CLA, which implements the algorithm and provides
methods to compute and analyze the efficient frontier.
"""

import importlib.metadata

from .builder import LassoBuilder, ProblemBuilder
from .cla import CLA
from .lasso import Lasso
from .operators import (
    CovarianceOperator,
    DenseCovariance,
    FactorCovariance,
    GramCovariance,
    IncrementalDenseCovariance,
    QuadraticForm,
)
from .pathtracer import ParametricProblem, trace

__all__ = [
    "CLA",
    "CovarianceOperator",
    "DenseCovariance",
    "FactorCovariance",
    "GramCovariance",
    "IncrementalDenseCovariance",
    "Lasso",
    "LassoBuilder",
    "ParametricProblem",
    "ProblemBuilder",
    "QuadraticForm",
    "trace",
]

try:
    __version__ = importlib.metadata.version("cvxcla")
except importlib.metadata.PackageNotFoundError:
    # Package metadata not available (development/editable install)
    __version__ = "0.0.0"

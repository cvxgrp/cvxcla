"""
types
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

MATRIX: TypeAlias = NDArray[np.float64]
BOOLEAN_VECTOR: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class TurningPoint:
    weights: MATRIX
    lamb: float
    free: BOOLEAN_VECTOR

    @property
    def free_indices(self):
        return np.where(self.free)[0]

    @property
    def blocked_indices(self):
        return np.where(~self.free)[0]

    def mean(self, mean: MATRIX):
        return float(mean.T @ self.weights)

    def variance(self, covariance: MATRIX):
        return float(self.weights.T @ covariance @ self.weights)

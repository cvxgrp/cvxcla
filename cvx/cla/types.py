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
class Next:
    free: BOOLEAN_VECTOR
    weights: MATRIX
    lamb: float = np.inf
    mean: float = -np.inf
    gamma: float = np.inf

    def __eq__(self, other):
        return np.allclose(self.weights, other.weights, atol=1e-5) and np.allclose(
            self.free, other.free
        )

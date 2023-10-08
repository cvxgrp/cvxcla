#    Copyright 2023 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
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
    """
    A turning point is a vector of weights, a lambda value, and a boolean vector
    indicating which assets are free. All assets that are not free are blocked.
    """

    weights: MATRIX
    free: BOOLEAN_VECTOR
    lamb: float = np.inf

    @property
    def free_indices(self):
        """
        Returns the indices of the free assets
        """
        return np.where(self.free)[0]

    @property
    def blocked_indices(self):
        """
        Returns the indices of the blocked assets
        """
        return np.where(~self.free)[0]

    def mean(self, mean: MATRIX):
        """
        Computes the expected return for a turning point
        """
        return float(mean.T @ self.weights)

    def variance(self, covariance: MATRIX):
        """
        Computes the expected variance for a turning point
        """
        return float(self.weights.T @ covariance @ self.weights)

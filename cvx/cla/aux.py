from dataclasses import dataclass, field
from typing import List

import numpy as np

from cvx.cla.types import MATRIX, TurningPoint


@dataclass(frozen=True)
class CLAUX:
    mean: MATRIX
    covariance: MATRIX
    lower_bounds: MATRIX
    upper_bounds: MATRIX
    turning_points: List[TurningPoint] = field(default_factory=list)
    tol: float = 1e-5

    @property
    def num_points(self):
        return len(self.turning_points)

    def append(self, tp: TurningPoint, tol=1e-5):
        assert np.all(
            tp.weights >= self.lower_bounds - tol), f"{self.lower_bounds} - {tp.weights}"
        assert np.all(
            tp.weights <= self.upper_bounds + tol), f"-{self.upper_bounds} + {tp.weights}"
        assert np.allclose(np.sum(tp.weights), 1.0), f"{np.sum(tp.weights)}"

        self.turning_points.append(tp)

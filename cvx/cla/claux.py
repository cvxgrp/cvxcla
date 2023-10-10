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
import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

from cvx.cla.first import init_algo
from cvx.cla.types import MATRIX, Frontier, FrontierPoint, TurningPoint


@dataclass(frozen=True)
class CLAUX:
    mean: MATRIX
    covariance: MATRIX
    lower_bounds: MATRIX
    upper_bounds: MATRIX
    A: MATRIX
    b: MATRIX
    turning_points: List[TurningPoint] = field(default_factory=list)
    tol: float = 1e-5
    logger: logging.Logger = logging.getLogger(__name__)

    def __len__(self):
        """
        Returns the number of turning points
        """
        return len(self.turning_points)

    def _first_turning_point(self):
        first = init_algo(
            mean=self.mean,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )
        return first

    def _append(self, tp: TurningPoint, tol=None):
        tol = tol or self.tol

        assert np.all(
            tp.weights >= (self.lower_bounds - tol)
        ), f"{(tp.weights + tol) - self.lower_bounds}"
        assert np.all(
            tp.weights <= (self.upper_bounds + tol)
        ), f"{(self.upper_bounds + tol) - tp.weights}"
        assert np.allclose(np.sum(tp.weights), 1.0), f"{np.sum(tp.weights)}"

        self.turning_points.append(tp)

    @property
    def frontier(self):
        """
        Returns the frontier
        """
        return Frontier(
            covariance=self.covariance,
            mean=self.mean,
            frontier=[FrontierPoint(point.weights) for point in self.turning_points],
        )

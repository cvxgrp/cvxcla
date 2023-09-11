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
from dataclasses import dataclass, field
from logging import Logger
from typing import List

import cvxpy as cp
import numpy as np
from loguru import logger as loguru

from cvx.cla.first import init_algo
from cvx.cla.types import MATRIX, TurningPoint


@dataclass(frozen=True)
class CLAUX:
    mean: MATRIX
    covariance: MATRIX
    lower_bounds: MATRIX
    upper_bounds: MATRIX
    turning_points: List[TurningPoint] = field(default_factory=list)
    tol: float = 1e-5
    logger: Logger = loguru

    # def __post_init__(self):
    #    self.logger.info("Initializing CLA (from CLAUX)")

    def first_turning_point(self):
        first = init_algo(
            mean=self.mean,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )
        return first

    @property
    def num_points(self):
        return len(self.turning_points)

    def append(self, tp: TurningPoint, tol=None):
        tol = tol or self.tol

        assert np.all(
            tp.weights >= (self.lower_bounds - tol)
        ), f"{(tp.weights + tol) - self.lower_bounds}"
        assert np.all(
            tp.weights <= (self.upper_bounds + tol)
        ), f"{(self.upper_bounds + tol) - tp.weights}"
        assert np.allclose(np.sum(tp.weights), 1.0), f"{np.sum(tp.weights)}"

        self.turning_points.append(tp)

    def minimum_variance(self):
        x = cp.Variable(shape=(self.mean.shape[0]), name="weights")

        constraints = [cp.sum(x) == 1, x >= self.lower_bounds, x <= self.upper_bounds]
        chol = np.linalg.cholesky(self.covariance)

        cp.Problem(cp.Minimize(cp.norm(chol.T @ x)), constraints).solve(cp.ECOS)

        return x.value

        # self.append(TurningPoint(lamb=0, weights=x.value, free=last.free))

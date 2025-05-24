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
Critical Line Algorithm auxiliary class implementation.

This module provides the CLAUX class which implements the core functionality
for the Critical Line Algorithm (CLA) used to compute the efficient frontier
in portfolio optimization problems.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .first import init_algo
from .types import MATRIX, Frontier, FrontierPoint, TurningPoint


@dataclass(frozen=True)
class CLAUX:
    """
    Critical Line Algorithm auxiliary class.

    This class implements the core functionality for the Critical Line Algorithm (CLA)
    used to compute the efficient frontier in portfolio optimization problems.

    Attributes:
        mean: Vector of expected returns for each asset.
        covariance: Covariance matrix of asset returns.
        lower_bounds: Vector of lower bounds for asset weights.
        upper_bounds: Vector of upper bounds for asset weights.
        A: Matrix for linear equality constraints (Ax = b).
        b: Vector for linear equality constraints (Ax = b).
        turning_points: List of turning points on the efficient frontier.
        tol: Tolerance for numerical calculations.
        logger: Logger instance for logging information and errors.
    """

    mean: MATRIX
    covariance: MATRIX
    lower_bounds: MATRIX
    upper_bounds: MATRIX
    A: MATRIX
    b: MATRIX
    turning_points: List[TurningPoint] = field(default_factory=list)
    tol: float = 1e-5
    logger: logging.Logger = logging.getLogger(__name__)

    def __len__(self) -> int:
        """
        Get the number of turning points in the efficient frontier.

        Returns:
            The number of turning points currently stored in the object.
        """
        return len(self.turning_points)

    def _first_turning_point(self) -> TurningPoint:
        """
        Calculate the first turning point on the efficient frontier.

        This method uses the init_algo function to find the first turning point
        based on the mean returns and the bounds on asset weights.

        Returns:
            A TurningPoint object representing the first point on the efficient frontier.
        """
        first = init_algo(
            mean=self.mean,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )
        return first

    def _append(self, tp: TurningPoint, tol: Optional[float] = None) -> None:
        """
        Append a turning point to the list of turning points.

        This method validates that the turning point satisfies the constraints
        before adding it to the list.

        Args:
            tp: The turning point to append.
            tol: Tolerance for constraint validation. If None, uses the class's tol attribute.

        Raises:
            AssertionError: If the turning point violates any constraints.
        """
        tol = tol or self.tol

        assert np.all(tp.weights >= (self.lower_bounds - tol)), f"{(tp.weights + tol) - self.lower_bounds}"
        assert np.all(tp.weights <= (self.upper_bounds + tol)), f"{(self.upper_bounds + tol) - tp.weights}"
        assert np.allclose(np.sum(tp.weights), 1.0), f"{np.sum(tp.weights)}"

        self.turning_points.append(tp)

    @property
    def frontier(self) -> Frontier:
        """
        Get the efficient frontier constructed from the turning points.

        This property creates a Frontier object from the list of turning points,
        which can be used to analyze the risk-return characteristics of the
        efficient portfolios.

        Returns:
            A Frontier object representing the efficient frontier.
        """
        return Frontier(
            covariance=self.covariance,
            mean=self.mean,
            frontier=[FrontierPoint(point.weights) for point in self.turning_points],
        )

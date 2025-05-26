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
Markowitz implementation of the Critical Line Algorithm.

This module provides the CLA class, which implements the Critical Line Algorithm
as described by Harry Markowitz and colleagues. The algorithm computes the entire
efficient frontier by finding all turning points, which are the points where the
set of assets at their bounds changes.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .first import init_algo
from .types import Frontier, FrontierPoint, TurningPoint


@dataclass(frozen=True)
class CLA:
    """
    Critical Line Algorithm implementation based on Markowitz's approach.

    This class implements the Critical Line Algorithm as described by Harry Markowitz
    and colleagues. It computes the entire efficient frontier by finding all turning
    points, which are the points where the set of assets at their bounds changes.

    The algorithm starts with the first turning point (the portfolio with the highest
    expected return) and then iteratively computes the next turning point with a lower
    expected return until it reaches the minimum variance portfolio.

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

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]
    A: NDArray[np.float64]
    b: NDArray[np.float64]
    turning_points: List[TurningPoint] = field(default_factory=list)
    tol: float = 1e-5
    logger: logging.Logger = logging.getLogger(__name__)

    def __post_init__(self):
        """
        Initialize the CLA object and compute the efficient frontier.

        This method is automatically called after initialization. It computes
        the entire efficient frontier by finding all turning points, starting
        from the first turning point (highest expected return) and iteratively
        computing the next turning point with a lower expected return until
        it reaches the minimum variance portfolio.

        The algorithm uses a block matrix approach to solve the system of equations
        that determine the turning points.

        Raises:
            AssertionError: If all variables are blocked, which would make the
                            system of equations singular.
        """
        ns = self.mean.shape[0]
        m = self.A.shape[0]

        # Initialize the portfolio.
        first = self._first_turning_point()
        self._append(first)

        # Set the P matrix.
        P = np.block([self.covariance, self.A.T])
        M = np.block([[self.covariance, self.A.T], [self.A, np.zeros((m, m))]])

        lam = np.inf

        while lam > 0:
            last = self.turning_points[-1]

            blocked = ~last.free
            assert not np.all(blocked), "Not all variables can be blocked"

            # Create the UP, DN, and IN
            UP = blocked & np.isclose(last.weights, self.upper_bounds)
            DN = blocked & np.isclose(last.weights, self.lower_bounds)

            # a variable is out if it is UP or DN
            OUT = np.logical_or(UP, DN)
            IN = ~OUT

            up = np.zeros(ns)
            up[UP] = self.upper_bounds[UP]

            dn = np.zeros(ns)
            dn[DN] = self.lower_bounds[DN]

            top = np.copy(self.mean)
            top[OUT] = 0

            _IN = np.concatenate([IN, np.ones(m, dtype=np.bool_)])

            bbb = np.array([np.block([up + dn, self.b]), np.block([top, np.zeros(m)])]).T

            alpha, beta = CLA._solve(M, bbb, _IN)

            gamma = P @ alpha
            delta = P @ beta - self.mean

            # Prepare the ratio matrix.
            L = -np.inf * np.ones([ns, 4])

            r_beta = beta[range(ns)]
            r_alpha = alpha[range(ns)]

            # IN security possibly going UP.
            i = IN & (r_beta < -self.tol)
            L[i, 0] = (self.upper_bounds[i] - r_alpha[i]) / r_beta[i]

            # IN security possibly going DN.
            i = IN & (r_beta > +self.tol)
            L[i, 1] = (self.lower_bounds[i] - r_alpha[i]) / r_beta[i]

            # UP security possibly going IN.
            i = UP & (delta < -self.tol)
            L[i, 2] = -gamma[i] / delta[i]

            # DN security possibly going IN.
            i = DN & (delta > +self.tol)
            L[i, 3] = -gamma[i] / delta[i]

            # If all elements of ratio are negative,
            # we have reached the end of the efficient frontier.
            if np.max(L) < 0:
                break

            secchg, dirchg = np.unravel_index(np.argmax(L, axis=None), L.shape)

            # Set the new value of lambda_E.
            lam = L[secchg, dirchg]

            free = np.copy(last.free)
            if dirchg == 0 or dirchg == 1:
                free[secchg] = False
            else:
                free[secchg] = True

            # Compute the portfolio at this corner.
            x = r_alpha + lam * r_beta

            # Save the data computed at this corner.
            self._append(TurningPoint(lamb=lam, weights=x, free=free))

        self._append(TurningPoint(lamb=0, weights=r_alpha, free=last.free))

    @staticmethod
    def _solve(A: NDArray[np.float64], b: np.ndarray, IN: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve a system of linear equations with some variables fixed.

        This method solves the system Ax = b, where some variables (indicated by IN)
        are free to be determined by the solver, and others (indicated by OUT) are
        fixed to specific values.

        Args:
            A: The coefficient matrix of the system.
            b: The right-hand side of the system. This should be a matrix with two columns,
               representing two different right-hand sides.
            IN: A boolean vector indicating which variables are free (True) and which
                are fixed (False).

        Returns:
            A tuple of two vectors (alpha, beta), where alpha is the solution to the first
            right-hand side and beta is the solution to the second right-hand side.
        """
        OUT = ~IN
        n = A.shape[1]
        x = np.zeros((n, 2))

        # Set the fixed variables to their specified values
        x[OUT, :] = b[OUT, :]

        # Adjust the right-hand side to account for the fixed variables
        bbb = b[IN, :] - A[IN, :][:, OUT] @ x[OUT, :]

        # Solve the system for the free variables
        x[IN, :] = np.linalg.inv(A[IN, :][:, IN]) @ bbb

        # Return the two solution vectors
        return x[:, 0], x[:, 1]

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

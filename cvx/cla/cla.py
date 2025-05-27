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
from functools import cached_property
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

    @cached_property
    def P(self):
        """
        Construct the projection matrix used in computing Lagrange multipliers.

        P is formed by horizontally stacking the covariance matrix and the transpose
        of the equality constraint matrix A. It is used to compute:
            - gamma = P @ alpha
            - delta = P @ beta - mean

        This step helps identify which constraints are becoming active or inactive.
        """
        return np.block([self.covariance, self.A.T])

    @cached_property
    def M(self):
        """
        Construct the Karush-Kuhn-Tucker (KKT) system matrix.

        The KKT matrix is built by augmenting the covariance matrix with the
        equality constraints. It forms the linear system:
            [Σ  Aᵗ]
            [A   0 ]
        which we solve to get the optimal portfolio weights (alpha) and the
        Lagrange multipliers (lambda) corresponding to the constraints.

        This matrix is symmetric but not necessarily positive definite.
        """
        m = self.A.shape[0]
        return np.block([[self.covariance, self.A.T], [self.A, np.zeros((m, m))]])

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
        m = self.A.shape[0]
        ns = len(self.mean)

        # Compute and store the first turning point
        self._append(self._first_turning_point())

        lam = np.inf

        while lam > 0:
            last = self.turning_points[-1]

            # --- Identify active set ---
            blocked = ~last.free
            assert not np.all(blocked), "All variables cannot be blocked"

            at_upper = blocked & np.isclose(last.weights, self.upper_bounds)
            at_lower = blocked & np.isclose(last.weights, self.lower_bounds)

            OUT = at_upper | at_lower
            IN = ~OUT

            # --- Construct RHS for KKT system ---
            fixed_weights = np.zeros(ns)
            fixed_weights[at_upper] = self.upper_bounds[at_upper]
            fixed_weights[at_lower] = self.lower_bounds[at_lower]

            adjusted_mean = self.mean.copy()
            adjusted_mean[OUT] = 0.0

            free_mask = np.concatenate([IN, np.ones(m, dtype=bool)])
            rhs_alpha = np.concatenate([fixed_weights, self.b])
            rhs_beta = np.concatenate([adjusted_mean, np.zeros(m)])
            rhs = np.column_stack([rhs_alpha, rhs_beta])

            # --- Solve KKT system ---
            alpha, beta = CLA._solve(self.M, rhs, free_mask)

            # --- Compute Lagrange multipliers and directional derivatives ---
            gamma = self.P @ alpha
            delta = self.P @ beta - self.mean

            # --- Compute event ratios ---
            L = np.full((ns, 4), -np.inf)
            r_alpha, r_beta = alpha[:ns], beta[:ns]
            tol = self.tol

            L[IN & (r_beta < -tol), 0] = (
                self.upper_bounds[IN & (r_beta < -tol)] - r_alpha[IN & (r_beta < -tol)]
            ) / r_beta[IN & (r_beta < -tol)]
            L[IN & (r_beta > +tol), 1] = (
                self.lower_bounds[IN & (r_beta > +tol)] - r_alpha[IN & (r_beta > +tol)]
            ) / r_beta[IN & (r_beta > +tol)]
            L[at_upper & (delta < -tol), 2] = -gamma[at_upper & (delta < -tol)] / delta[at_upper & (delta < -tol)]
            L[at_lower & (delta > +tol), 3] = -gamma[at_lower & (delta > +tol)] / delta[at_lower & (delta > +tol)]

            # --- Determine next event ---
            if np.max(L) < 0:
                break

            secchg, dirchg = np.unravel_index(np.argmax(L), L.shape)
            lam = L[secchg, dirchg]

            # --- Update free set ---
            free = last.free.copy()
            free[secchg] = dirchg >= 2  # boundary → IN if dirchg in {2, 3}

            # --- Compute new turning point ---
            new_weights = r_alpha + lam * r_beta
            self._append(TurningPoint(lamb=lam, weights=new_weights, free=free))

        # Final point at lambda = 0
        self._append(TurningPoint(lamb=0, weights=r_alpha, free=last.free))

    @staticmethod
    def _solve(A: NDArray[np.float64], b: np.ndarray, IN: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the system A x = b with some variables fixed.

        Args:
            A: Coefficient matrix of shape (n, n).
            b: Right-hand side matrix of shape (n, 2).
            IN: Boolean array of shape (n,) indicating which variables are free.

        Returns:
            A tuple (alpha, beta) of solutions for the two RHS vectors.
        """
        OUT = ~IN
        n = A.shape[1]
        x = np.zeros((n, 2))

        x[OUT] = b[OUT]  # Set fixed variables
        reduced_A = A[IN][:, IN]
        reduced_b = b[IN] - A[IN][:, OUT] @ x[OUT]

        x[IN] = np.linalg.solve(reduced_A, reduced_b)

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

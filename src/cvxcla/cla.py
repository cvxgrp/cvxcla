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
"""Markowitz implementation of the Critical Line Algorithm.

This module provides the CLA class, which implements the Critical Line Algorithm
as described by Harry Markowitz and colleagues. The algorithm computes the entire
efficient frontier by finding all turning points, which are the points where the
set of assets at their bounds changes.
"""

import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .first import init_algo
from .operators import CovarianceOperator, DenseCovariance
from .types import Frontier, FrontierPoint, TurningPoint


@dataclass(frozen=True)
class CLA:
    """Critical Line Algorithm implementation based on Markowitz's approach.

    This class implements the Critical Line Algorithm as described by Harry Markowitz
    and colleagues. It computes the entire efficient frontier by finding all turning
    points, which are the points where the set of assets at their bounds changes.

    The algorithm starts with the first turning point (the portfolio with the highest
    expected return) and then iteratively computes the next turning point with a lower
    expected return until it reaches the minimum variance portfolio.

    Attributes:
        mean: Vector of expected returns for each asset.
        covariance: Covariance matrix of asset returns, either as a plain
            ``numpy`` array or as a ``CovarianceOperator`` backend
            (see ``cvxcla.operators``).
        lower_bounds: Vector of lower bounds for asset weights.
        upper_bounds: Vector of upper bounds for asset weights.
        a: Matrix for linear equality constraints (Ax = b).
        b: Vector for linear equality constraints (Ax = b).
        turning_points: List of turning points on the efficient frontier.
        tol: Tolerance for numerical calculations.
        logger: Logger instance for logging information and errors.

    """

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64] | CovarianceOperator
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    turning_points: list[TurningPoint] = field(default_factory=list)
    tol: float = 1e-5
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    @cached_property
    def covariance_operator(self) -> CovarianceOperator:
        """Return the covariance as a ``CovarianceOperator`` backend.

        A plain ``numpy`` covariance matrix is wrapped in ``DenseCovariance``;
        an object already implementing the protocol is passed through. This is
        the single point where the input form is normalised.
        """
        if isinstance(self.covariance, np.ndarray):
            return DenseCovariance(cast("NDArray[np.float64]", self.covariance))
        return self.covariance

    def __post_init__(self) -> None:
        """Initialize the CLA object and compute the efficient frontier.

        This method is automatically called after initialization. It computes
        the entire efficient frontier by finding all turning points, starting
        from the first turning point (highest expected return) and iteratively
        computing the next turning point with a lower expected return until
        it reaches the minimum variance portfolio.

        The reduced KKT system at each turning point is solved by block
        elimination: two multi-RHS solves against the free covariance block
        (via the covariance backend) and a small m x m Schur complement
        ``A_F @ Sigma_FF^{-1} @ A_F.T``, where m is the number of equality
        constraints. The covariance only enters through the
        ``CovarianceOperator`` interface, so structured backends (e.g.
        ``FactorCovariance``) never materialise an n x n matrix.

        Raises:
            RuntimeError: If all variables are blocked, which would make the
                          system of equations singular.

        """
        m = self.a.shape[0]
        ns = len(self.mean)
        cov = self.covariance_operator
        tol = self.tol

        # Compute and store the first turning point
        self._append(self._first_turning_point())

        lam = np.inf

        while lam > 0:
            last = self.turning_points[-1]

            # --- Identify active set ---
            blocked = ~last.free
            if np.all(blocked):
                msg = "All variables cannot be blocked"
                raise RuntimeError(msg)

            at_upper = blocked & (np.abs(last.weights - self.upper_bounds) <= tol)
            at_lower = blocked & (np.abs(last.weights - self.lower_bounds) <= tol)

            _out = at_upper | at_lower
            _in = ~_out

            fixed_weights = np.zeros(ns)
            fixed_weights[at_upper] = self.upper_bounds[at_upper]
            fixed_weights[at_lower] = self.lower_bounds[at_lower]

            # --- Solve the reduced KKT system by block elimination ---
            # [Sigma_FF  A_F.T] [x ]   [r1]      with r1, r2 the RHS for the
            # [A_F       0    ] [nu] = [r2]      alpha (weights) and beta system
            a_free = self.a[:, _in]

            # Free-block solves: Sigma_FF^{-1} [A_F.T | r1_alpha | r1_beta]
            rhs_free = np.column_stack(
                [
                    a_free.T,
                    -cov.cross(_in, fixed_weights),
                    self.mean[_in],
                ]
            )
            solved = cov.solve_free(_in, rhs_free)
            y = solved[:, :m]  # Sigma_FF^{-1} A_F.T
            z_alpha = solved[:, m]
            z_beta = solved[:, m + 1]

            # Schur complement A_F Sigma_FF^{-1} A_F.T and multipliers
            schur = a_free @ y
            r2_alpha = self.b - self.a[:, _out] @ fixed_weights[_out]
            nu = np.linalg.solve(schur, np.column_stack([a_free @ z_alpha - r2_alpha, a_free @ z_beta]))
            nu_alpha, nu_beta = nu[:, 0], nu[:, 1]

            # Back-substitute the free weights
            r_alpha = fixed_weights.copy()
            r_alpha[_in] = z_alpha - y @ nu_alpha
            r_beta = np.zeros(ns)
            r_beta[_in] = z_beta - y @ nu_beta

            # --- Compute Lagrange multipliers and directional derivatives ---
            gamma = cov.matvec(r_alpha) + self.a.T @ nu_alpha
            delta = cov.matvec(r_beta) + self.a.T @ nu_beta - self.mean

            # --- Compute event ratios ---
            # A free weight moves along w(lam) = r_alpha + lam * r_beta, so even
            # a tiny slope crosses a bound given a long enough lam range.
            # Filtering slopes at self.tol misses such crossings and lets
            # weights drift out of bounds; only slopes at floating-point noise
            # level are excluded: below sqrt(machine epsilon) a slope is
            # indistinguishable from solve noise, and the huge ratios it would
            # produce only amplify rounding errors. Spurious ratios above the
            # current lam are removed by the lam window below.
            eps = np.sqrt(np.finfo(np.float64).eps)
            l_mat = np.full((ns, 4), -np.inf)

            l_mat[_in & (r_beta < -eps), 0] = (
                self.upper_bounds[_in & (r_beta < -eps)] - r_alpha[_in & (r_beta < -eps)]
            ) / r_beta[_in & (r_beta < -eps)]
            l_mat[_in & (r_beta > +eps), 1] = (
                self.lower_bounds[_in & (r_beta > +eps)] - r_alpha[_in & (r_beta > +eps)]
            ) / r_beta[_in & (r_beta > +eps)]
            l_mat[at_upper & (delta < -eps), 2] = -gamma[at_upper & (delta < -eps)] / delta[at_upper & (delta < -eps)]
            l_mat[at_lower & (delta > +eps), 3] = -gamma[at_lower & (delta > +eps)] / delta[at_lower & (delta > +eps)]

            # --- Determine next event ---
            # The current segment w(lam) = r_alpha + lam * r_beta is only valid
            # for lam at or below the current value: the frontier is traced with
            # non-increasing lam, so ratios above it are spurious crossings
            # outside the segment and must not be selected. Ties at the current
            # lam are kept; degenerate problems resolve them one per iteration.
            l_mat[l_mat > lam + tol] = -np.inf

            lam_max = np.max(l_mat)
            if lam_max < 0:
                break

            # Bland-style anti-cycling rule: on degenerate problems (tied means,
            # duplicated assets) several events coincide at the same ratio. Among
            # all events within tol of the best ratio we pick the lowest asset
            # index (and, within an asset, the lowest event type), so the choice
            # is deterministic and cannot cycle.
            tied = np.argwhere(l_mat >= lam_max - tol)
            secchg, dirchg = tied[0]
            lam = l_mat[secchg, dirchg]

            # --- Update free set ---
            free = last.free.copy()
            free[secchg] = dirchg >= 2  # boundary → IN if dirchg in {2, 3}

            # --- Compute new turning point ---
            new_weights = r_alpha + lam * r_beta
            self._append(TurningPoint(lamb=lam, weights=new_weights, free=free))

        # Final point at lambda = 0
        self._append(TurningPoint(lamb=0, weights=r_alpha, free=last.free))

    def __len__(self) -> int:
        """Get the number of turning points in the efficient frontier.

        Returns:
            The number of turning points currently stored in the object.

        """
        return len(self.turning_points)

    def _first_turning_point(self) -> TurningPoint:
        """Calculate the first turning point on the efficient frontier.

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

    def _append(self, tp: TurningPoint, tol: float | None = None) -> None:
        """Append a turning point to the list of turning points.

        This method validates that the turning point satisfies the constraints
        before adding it to the list.

        Args:
            tp: The turning point to append.
            tol: Tolerance for constraint validation. If None, uses the class's
                tol attribute. Pass 0 for exact validation.

        Raises:
            ValueError: If the turning point violates any constraints.

        """
        tol = self.tol if tol is None else tol

        if not np.all(tp.weights >= (self.lower_bounds - tol)):
            msg = "Weights below lower bounds"
            raise ValueError(msg)
        if not np.all(tp.weights <= (self.upper_bounds + tol)):
            msg = "Weights above upper bounds"
            raise ValueError(msg)
        if not np.allclose(np.sum(tp.weights), 1.0):
            msg = "Weights do not sum to 1"
            raise ValueError(msg)

        self.turning_points.append(tp)

    @property
    def frontier(self) -> Frontier:
        """Get the efficient frontier constructed from the turning points.

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

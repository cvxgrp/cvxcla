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
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from cvx.cla.first import init_algo
from cvx.cla.types import Frontier, FrontierPoint, TurningPoint

"""
Bailey and Lopez de Prado implementation of the Critical Line Algorithm.

This module provides an implementation of the Critical Line Algorithm based on
the approach described by David Bailey and Marcos Lopez de Prado in their paper
"An Open-Source Implementation of the Critical-Line Algorithm for Portfolio Optimization".

The implementation is included for testing and educational purposes only.
"""


@dataclass(frozen=True)
class CLA:
    """
    Bailey and Lopez de Prado implementation of the Critical Line Algorithm.

    This class implements the Critical Line Algorithm as described by Bailey and Lopez de Prado.
    It computes the entire efficient frontier by finding all turning points, which are the
    points where the set of assets at their bounds changes.

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
    turning_points: list[TurningPoint] = field(default_factory=list)
    tol: float = 1e-5
    logger: logging.Logger = logging.getLogger(__name__)

    def __post_init__(self) -> None:
        """
        Initialize the CLA object and compute the efficient frontier.

        This method is automatically called after initialization. It computes
        the entire efficient frontier by finding all turning points, starting
        from the first turning point (highest expected return) and iteratively
        computing the next turning point with a lower expected return until
        it reaches the minimum variance portfolio.

        The algorithm follows these steps:
        1. Start with the first turning point
        2. For each turning point, consider:
           a) Binding a free weight (moving it to a bound)
           b) Freeing a bound weight
        3. Choose the option that gives the highest lambda value
        4. Compute the new weights and create a new turning point
        5. Repeat until no more valid turning points can be found
        6. Add the minimum variance portfolio as the final point
        """
        # Compute the turning points,free sets and weights
        self._append(self._first_turning_point())

        while True:
            last = self.turning_points[-1]

            # 1) case a): Bound one free weight
            l_in = -np.inf

            # only try to bound a free asset if there are least two of them
            if np.sum(last.free) > 1:
                schur = _Schur(
                    covariance=self.covariance,
                    mean=self.mean,
                    free=last.free,
                    weights=last.weights,
                )

                for i in last.free_indices:
                    # count the number of entries that are True below the ith entry in fff
                    j = np.sum(last.free[:i])

                    lamb, bi = schur.compute_lambda(
                        index=j,
                        bi=np.array([self.lower_bounds[i], self.upper_bounds[i]]),
                    )

                    if lamb > l_in:
                        l_in, i_in, bi_in = lamb, i, bi

            # 2) case b): Free one bounded weight
            l_out = -np.inf

            for i in last.blocked_indices:
                fff = np.copy(last.free)
                fff[i] = True

                schur = _Schur(
                    covariance=self.covariance,
                    mean=self.mean,
                    free=fff,
                    weights=last.weights,
                )

                # count the number of entries that are True below the ith entry in fff
                j = np.sum(fff[:i])

                lamb, bi = schur.compute_lambda(
                    index=j,
                    bi=np.array([last.weights[i]]),
                )

                if last.lamb > lamb > l_out:
                    l_out, i_out = lamb, i

            l_current = np.max([l_in, l_out])

            if l_current > 0:
                # 4) decide lambda
                logger.info(f"l_in: {l_in}")
                logger.info(f"l_out: {l_out}")
                logger.info(f"l_current: {l_current}")
                f = np.copy(last.free)
                w = np.copy(last.weights)

                if l_in > l_out:
                    lll = l_in
                    f[i_in] = False
                    w[i_in] = bi_in  # set value at the correct boundary
                else:
                    lll = l_out
                    f[i_out] = True
            else:
                break

            schur = _Schur(
                covariance=self.covariance,
                mean=self.mean,
                free=f,
                weights=w,
            )
            # 5) compute solution vector
            weights = schur.update_weights(lamb=lll)
            tp = TurningPoint(weights=weights, lamb=lll, free=f)

            # check the turning point
            self._append(tp)

            logger.info(f"weights: {tp.weights}")
            logger.info(f"free: {tp.free_indices}")

        # 6) compute minimum variance solution
        last = self.turning_points[-1]

        schur = _Schur(
            covariance=self.covariance,
            mean=self.mean,
            free=last.free,
            weights=last.weights,
        )
        w = schur.update_weights(lamb=0)
        # assert np.allclose(x, w, atol=1e-4)

        self._append(TurningPoint(lamb=0, weights=w, free=last.free))

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

    def _append(self, tp: TurningPoint, tol: float | None = None) -> None:
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


class _Schur:
    """
    Helper class for the Critical Line Algorithm.

    This class implements the Schur complement method for efficiently computing
    the lambda values and weight updates in the Critical Line Algorithm.

    Attributes:
        covariance: Covariance matrix of asset returns.
        mean: Vector of expected returns for each asset.
        free: Boolean vector indicating which assets are free.
        weights: Vector of portfolio weights.
        __free_inv: Cached inverse of the free part of the covariance matrix.
    """

    def __init__(
        self,
        covariance: NDArray[np.float64],
        mean: NDArray[np.float64],
        free: NDArray[np.bool_],
        weights: NDArray[np.float64],
    ) -> None:
        assert covariance.shape[0] == covariance.shape[1] == mean.shape[0] == free.shape[0] == weights.shape[0]
        self.covariance = covariance
        self.mean = mean
        self.free = free
        self.weights = weights
        self.__free_inv = np.linalg.inv(self.covariance_free)

    @property
    def covariance_free(self) -> NDArray[np.float64]:
        """
        Get the covariance matrix for free assets only.

        Returns:
            The submatrix of the covariance matrix corresponding to free assets.
        """
        return self.covariance[self.free][:, self.free]

    @property
    def covariance_free_blocked(self) -> NDArray[np.float64]:
        """
        Get the cross-covariance matrix between free and blocked assets.

        Returns:
            The submatrix of the covariance matrix with rows for free assets
            and columns for blocked assets.
        """
        return self.covariance[self.free][:, ~self.free]

    @property
    def covariance_free_inv(self) -> NDArray[np.float64]:
        """
        Get the inverse of the covariance matrix for free assets.

        Returns:
            The inverse of the covariance matrix for free assets.
        """
        return self.__free_inv

    @property
    def mean_free(self) -> NDArray[np.float64]:
        """
        Get the expected returns for free assets only.

        Returns:
            The subvector of the mean vector corresponding to free assets.
        """
        return self.mean[self.free]

    @property
    def weights_blocked(self) -> NDArray[np.float64]:
        """
        Get the weights for blocked assets only.

        Returns:
            The subvector of the weights vector corresponding to blocked assets.
        """
        return self.weights[~self.free]

    def compute_lambda(self, index: int, bi: NDArray[np.float64]) -> tuple[float, float]:
        """
        Compute the lambda value for a given index and boundary values.

        This method computes the lambda value that would make the weight at the given
        index reach one of the boundary values. It is used to determine the next
        turning point in the Critical Line Algorithm.

        Args:
            index: The index of the free asset to consider.
            bi: Array of boundary values (lower and upper bounds) for the asset.

        Returns:
            A tuple containing the lambda value and the boundary value that would be reached.
        """

        def compute_bi(c: float, bi: NDArray[np.float64]) -> float:
            """
            Determine which boundary value to use based on the sign of c.

            Args:
                c: A coefficient that determines which boundary to use.
                bi: Array of boundary values.

            Returns:
                The appropriate boundary value.
            """
            if np.shape(bi)[0] == 1 or c <= 0:
                return bi[0]
            return bi[1]

        c4 = np.sum(self.covariance_free_inv, axis=0)
        c1 = np.sum(c4)
        c2 = self.covariance_free_inv @ self.mean_free
        # c3 = np.sum(c2)
        # c3 = np.sum(self.covariance_free_inv, axis=1) @ self.mean_free
        # c4 = np.sum(self.covariance_free_inv, axis=0)

        aux = -np.sum(c4) * c2[index] + np.sum(c2) * c4[index]

        bi = compute_bi(aux, bi)

        if self.weights_blocked.size == 0:
            return float((c4[index] - c1 * bi) / aux), bi

        l1 = np.sum(self.weights_blocked)
        l2 = self.covariance_free_inv @ self.covariance_free_blocked
        l3 = l2 @ self.weights_blocked
        l2 = np.sum(l3)
        return ((1 - l1 + l2) * c4[index] - c1 * (bi + l3[index])) / aux, bi

    def _compute_weight(self, lamb: float) -> tuple[NDArray[np.float64], float]:
        """
        Compute the weights for free assets given a lambda value.

        This is an internal helper method that computes the weights for the free assets
        and the gamma value (Lagrange multiplier for the budget constraint) for a given
        lambda value.

        Args:
            lamb: The lambda value to use for computing the weights.

        Returns:
            A tuple containing the weights for free assets and the gamma value.
        """
        g1 = np.sum(self.covariance_free_inv @ self.mean_free, axis=0)
        g2 = np.sum(np.sum(self.covariance_free_inv))

        if self.weights_blocked.size == 0:
            gamma = -lamb * g1 / g2 + 1 / g2
            w1 = 0
        else:
            g3 = np.sum(self.weights_blocked)
            g4 = self.covariance_free_inv @ self.covariance_free_blocked
            w1 = g4 @ self.weights_blocked
            g4 = np.sum(w1)
            gamma = -lamb * g1 / g2 + (1 - g3 + g4) / g2

        w2 = np.sum(self.covariance_free_inv, axis=1)
        w3 = self.covariance_free_inv @ self.mean_free
        return -w1 + gamma * w2 + lamb * w3, gamma

    def update_weights(self, lamb: float) -> NDArray[np.float64]:
        """
        Update the portfolio weights for a given lambda value.

        This method computes the new portfolio weights for a given lambda value.
        It uses the _compute_weight method to compute the weights for the free assets
        and then updates the full weights vector.

        Args:
            lamb: The lambda value to use for updating the weights.

        Returns:
            The updated portfolio weights vector.
        """
        weights, _ = self._compute_weight(lamb)
        new_weights = np.copy(self.weights)
        new_weights[self.free] = weights
        return new_weights

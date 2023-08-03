"""
The Critical Line algorithm computes a set of points on the efficient frontier.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ._cla import CLA
from .types import MATRIX


@dataclass(frozen=True)
class FrontierPoint:
    """
    A frontier point is a vector of weights.
    """

    weights: MATRIX

    def __post_init__(self):
        # check that the sum is close to 1
        assert np.isclose(np.sum(self.weights), 1.0)

        # make sure the weights are non-negative
        assert np.all(self.weights >= -1e-7)

    # The final result will be a list of points on the efficient frontier.
    # Each point is described by a vector of weights.
    @staticmethod
    def from_turning_point(turning_point):
        """
        Constructs a frontier point given a turning point
        Args:
            turning_point: The turning point

        Returns: A frontier point

        """
        return FrontierPoint(weights=turning_point.weights)

    def expected_return(self, mean):
        """
        Computes the expected return for a frontier point

        Args:
            mean: the vector of expected returns per asset

        Returns:
            Computes the sum of the weighted expected returns
        """
        return mean @ self.weights

    def expected_variance(self, covariance):
        """
        Computes the expected variance for a frontier point

        Args:
            covariance: the covariance matrix

        Returns:
            Computes $w^T Covariance w$
        """
        return self.weights.T @ (covariance @ self.weights)


@dataclass(frozen=True)
class Frontier:
    """
    A frontier is a list of frontier points. Some of them might be turning points.
    """

    frontier: list[FrontierPoint]
    mean: MATRIX
    covariance: MATRIX

    @staticmethod
    def construct(mean, covariance, lower_bounds, upper_bounds):
        """
        Constructs a frontier by computing a list of turning points.

        Args:
            mean: a vector of expected returns per asset
            covariance: a covariance matrix
            lower_bounds: lower bounds per asset
            upper_bounds: upper bounds per asset

        Returns:
            A frontier of frontier points each of them a turning point
        """
        cla = CLA(mean=mean, covariance=covariance, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

        frontier_points = [
            FrontierPoint.from_turning_point(t_point) for t_point in cla.turning_points
        ]
        return Frontier(frontier=frontier_points, mean=mean, covariance=covariance)

    def interpolate(self, num=100):
        """
        Interpolate the frontier with num-1 points between adjacent frontier points
        Args:
            num: The number of new points in between each pair

        Returns:
            A frontier with many new points
        """

        def _interpolate():
            for w_left, w_right in zip(self.weights[0:-1], self.weights[1:]):
                for lamb in np.linspace(0, 1, num):
                    if lamb > 0:
                        yield FrontierPoint(
                            weights=lamb * w_left + (1 - lamb) * w_right
                        )

        points = list(_interpolate())
        return Frontier(frontier=points, mean=self.mean, covariance=self.covariance)

    def __iter__(self):
        """
        Iterator for all frontier points
        """
        yield from self.frontier

    def __len__(self):
        return len(self.frontier)


    @property
    def weights(self):
        """
        Matrix of weights. One row per point
        """
        return np.array([point.weights for point in self])

    @property
    def returns(self):
        """
        Vector of expected returns.
        """
        return np.array([point.expected_return(self.mean) for point in self])

    @property
    def variance(self):
        """
        Vector of expected variances.
        """
        return np.array([point.expected_variance(self.covariance) for point in self])

    @property
    def sharpe_ratio(self):
        """
        Vector of expected Sharpe ratios.
        """
        return self.returns / self.volatility

    @property
    def volatility(self):
        """
        Vector of expected volatilities.
        """
        return np.sqrt(self.variance)

    @property
    def max_sharpe(self):
        """
        Maximal Sharpe ratio on the frontier

        Returns:
            Tuple of maximal Sharpe ratio and the weights to achieve it
        """

        def neg_sharpe(alpha, w_left, w_right):
            # convex combination of left and right weights
            weight = alpha * w_left + (1 - alpha) * w_right
            # compute the variance
            var = weight.T @ self.covariance @ weight
            returns = self.mean.T @ weight
            return -returns / np.sqrt(var)

        sharpe_ratios = self.sharpe_ratio

        # where is the maximal Sharpe ratio?
        sr_position_max = np.argmax(sharpe_ratios)

        right = np.min([sr_position_max + 1, len(self) - 1])
        left = np.max([0, sr_position_max - 1])
        # Look to the left and look to the right

        if right > sr_position_max:
            out = minimize(
                neg_sharpe,
                0.5,
                args=(self.weights[sr_position_max], self.weights[right]),
                bounds=((0, 1),),
            )
            var = out["x"][0]
            w_right = (
                var * self.weights[sr_position_max] + (1 - var) * self.weights[right]
            )
            sharpe_ratio_right = -out["fun"]
        else:
            w_right = self.weights[sr_position_max]
            sharpe_ratio_right = sharpe_ratios[sr_position_max]

        if left < sr_position_max:
            out = minimize(
                neg_sharpe,
                0.5,
                args=(self.weights[left], self.weights[sr_position_max]),
                bounds=((0, 1),),
            )
            var = out["x"][0]
            w_left = (
                var * self.weights[left] + (1 - var) * self.weights[sr_position_max]
            )
            sharpe_ratio_left = -out["fun"]
        else:
            w_left = self.weights[sr_position_max]
            sharpe_ratio_left = sharpe_ratios[sr_position_max]

        if sharpe_ratio_left > sharpe_ratio_right:
            return sharpe_ratio_left, w_left

        return sharpe_ratio_right, w_right

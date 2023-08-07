"""
The Critical Line algorithm computes a set of points on the efficient frontier.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

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
        #assert np.all(self.weights >= -1e-7)

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


def _turning_points(solver, mean, covariance, lower_bounds, upper_bounds, tol=None):
    x = solver.build(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds, covariance=covariance, tol=tol)

    #x = solver(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds, covariance=covariance, tol=tol)
    for point in x.turning_points:
        yield FrontierPoint(weights = point.weights)

@dataclass(frozen=True)
class Frontier:
    """
    A frontier is a list of frontier points. Some of them might be turning points.
    """

    frontier: list[FrontierPoint]
    mean: MATRIX
    covariance: MATRIX
    name: str = "FRONTIER"

    @staticmethod
    def build(solver, mean, covariance, lower_bounds, upper_bounds, name, tol=float(1e-5)):
        """
        Constructs a frontier by computing a list of turning points.

        Args:
            mean: a vector of expected returns per asset
            covariance: a covariance matrix
            lower_bounds: lower bounds per asset
            upper_bounds: upper bounds per asset
            name: the name of the frontier
            tol: tolerance for the solver

        Returns:
            A frontier of frontier points each of them a turning point
        """
        frontier_points = list(_turning_points(solver=solver,
                                               mean=mean,
                                               covariance=covariance,
                                               lower_bounds=lower_bounds,
                                               upper_bounds=upper_bounds,
                                               tol=tol))

        return Frontier(frontier=frontier_points, mean=mean, covariance=covariance, name=name)

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

        # in which point is the maximal Sharpe ratio?
        sr_position_max = np.argmax(self.sharpe_ratio)

        # np.min only there for security...
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

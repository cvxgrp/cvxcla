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
types
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import plotly.express as px
from numpy.typing import NDArray
from scipy.optimize import minimize
from typing_extensions import TypeAlias

MATRIX: TypeAlias = NDArray[np.float64]
BOOLEAN_VECTOR: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class FrontierPoint:
    weights: MATRIX

    def __post_init__(self):
        # check that the sum is close to 1
        assert np.isclose(np.sum(self.weights), 1.0)

    def mean(self, mean: MATRIX):
        """
        Computes the expected return for a turning point
        """
        return float(mean.T @ self.weights)

    def variance(self, covariance: MATRIX):
        """
        Computes the expected variance for a turning point
        """
        return float(self.weights.T @ covariance @ self.weights)


@dataclass(frozen=True)
class TurningPoint(FrontierPoint):
    """
    A turning point is a vector of weights, a lambda value, and a boolean vector
    indicating which assets are free. All assets that are not free are blocked.
    """

    free: BOOLEAN_VECTOR
    lamb: float = np.inf

    @property
    def free_indices(self):
        """
        Returns the indices of the free assets
        """
        return np.where(self.free)[0]

    @property
    def blocked_indices(self):
        """
        Returns the indices of the blocked assets
        """
        return np.where(~self.free)[0]


@dataclass(frozen=True)
class Frontier:
    """
    A frontier is a list of frontier points. Some of them might be turning points.
    """

    mean: MATRIX
    covariance: MATRIX
    frontier: list[FrontierPoint] = field(default_factory=list)

    def interpolate(self, num=100):
        """
        Interpolate the frontier with num-1 points between adjacent frontier points
        Args:
            num: The number of new points in between each pair

        Returns:
            A frontier with many new points
        """

        def _interpolate():
            for w_right, w_left in zip(self.weights[0:-1], self.weights[1:]):
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
        """
        Number of frontier points
        """
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
        return np.array([point.mean(self.mean) for point in self])

    @property
    def variance(self):
        """
        Vector of expected variances.
        """
        return np.array([point.variance(self.covariance) for point in self])

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

    def plot(self, volatility=False, markers=True):
        """
        Plot the frontier
        """
        if not volatility:
            fig = px.line(
                x=self.variance,
                y=self.returns,
                markers=markers,
                labels={"x": "Expected variance", "y": "Expected Return"},
            )
        else:
            fig = px.line(
                x=self.volatility,
                y=self.returns,
                markers=markers,
                labels={"x": "Expected volatility", "y": "Expected Return"},
            )
        return fig

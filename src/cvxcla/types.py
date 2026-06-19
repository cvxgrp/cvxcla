"""Type definitions and classes for the Critical Line Algorithm.

This module defines the core data structures used in the Critical Line Algorithm:
- FrontierPoint: Represents a point on the efficient frontier.
- TurningPoint: Represents a turning point on the efficient frontier.
- Frontier: Represents the entire efficient frontier.

It also defines type aliases for commonly used types.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

import numpy as np
from numpy.typing import NDArray

from .operators import CovarianceOperator


def _covariance_matvec(
    covariance: NDArray[np.float64] | CovarianceOperator, x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute ``Sigma @ x`` for a dense matrix or a ``CovarianceOperator`` backend."""
    if isinstance(covariance, CovarianceOperator):
        return covariance.matvec(x)
    return covariance @ x


@dataclass(frozen=True)
class FrontierPoint:
    """A point on the efficient frontier.

    This class represents a portfolio on the efficient frontier, defined by its weights.
    It provides methods to compute the expected return and variance of the portfolio.

    Attributes:
        weights: Vector of portfolio weights for each asset.

    """

    weights: NDArray[np.float64]

    def mean(self, mean: NDArray[np.float64]) -> float:
        """Compute the expected return of the portfolio.

        Args:
            mean: Vector of expected returns for each asset.

        Returns:
            The expected return of the portfolio.

        Examples:
            >>> import numpy as np
            >>> fp = FrontierPoint(weights=np.array([0.5, 0.5]))
            >>> fp.mean(np.array([0.1, 0.2]))
            0.15000000000000002

        """
        return float(mean.T @ self.weights)

    def variance(self, covariance: NDArray[np.float64] | CovarianceOperator) -> float:
        """Compute the expected variance of the portfolio.

        Args:
            covariance: Covariance matrix of asset returns, either as a dense
                matrix or as a ``CovarianceOperator`` backend.

        Returns:
            The expected variance of the portfolio.

        """
        return float(self.weights.T @ _covariance_matvec(covariance, self.weights))


@dataclass(frozen=True)
class TurningPoint(FrontierPoint):
    """Turning point.

    A turning point is a vector of weights, a lambda value, and a boolean vector
    indicating which assets are free. All assets that are not free are blocked.

    For problems with general inequality constraints ``G w <= h`` the turning
    point also records which inequality *rows* are active (held at equality
    ``g_i w = h_i``) via ``active_ineq``. These are the row analogue of the box
    active set: a free weight on a bound is a per-variable active constraint, an
    active inequality row is a per-row one. The default is an empty mask, so
    box-and-equality problems (and the LASSO path) are unaffected.
    """

    free: NDArray[np.bool_]
    lamb: float = np.inf
    active_ineq: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def free_indices(self) -> np.ndarray:
        """Returns the indices of the free assets."""
        return np.where(self.free)[0]

    @property
    def blocked_indices(self) -> np.ndarray:
        """Returns the indices of the blocked assets."""
        return np.where(~self.free)[0]


@dataclass(frozen=True)
class Frontier:
    """A frontier is a list of frontier points. Some of them might be turning points."""

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64] | CovarianceOperator
    frontier: list[FrontierPoint] = field(default_factory=list)

    def interpolate(self, num: int = 100) -> Frontier:
        """Interpolate the frontier with additional points between existing points.

        This method creates a new Frontier object with additional points interpolated
        between the existing points. This is useful for creating a smoother representation
        of the efficient frontier for visualization or analysis.

        Args:
            num: The number of points to use in the interpolation. The method will create
                 num-1 new points between each pair of adjacent existing points.

        Returns:
            A new Frontier object with the interpolated points.

        """

        def _interpolate() -> Iterator[FrontierPoint]:
            """Yield interpolated frontier points between each adjacent pair."""
            for w_right, w_left in zip(self.weights[0:-1], self.weights[1:], strict=False):  # pragma: no mutate
                for lamb in np.linspace(0, 1, num):
                    if lamb > 0:
                        yield FrontierPoint(weights=lamb * w_left + (1 - lamb) * w_right)

        points = list(_interpolate())
        return Frontier(frontier=points, mean=self.mean, covariance=self.covariance)

    def __iter__(self) -> Iterator[FrontierPoint]:
        """Iterate over all frontier points."""
        yield from self.frontier

    def __len__(self) -> int:
        """Give number of frontier points."""
        return len(self.frontier)

    @property
    def weights(self) -> np.ndarray:
        """Matrix of weights. One row per point."""
        return np.array([point.weights for point in self])

    @property
    def returns(self) -> np.ndarray:
        """Vector of expected returns."""
        return np.array([point.mean(self.mean) for point in self])

    @property
    def variance(self) -> np.ndarray:
        """Vector of expected variances."""
        return np.array([point.variance(self.covariance) for point in self])

    @property
    def sharpe_ratio(self) -> np.ndarray:
        """Vector of expected Sharpe ratios."""
        ratios: np.ndarray = self.returns / self.volatility
        return ratios

    @property
    def volatility(self) -> np.ndarray:
        """Vector of expected volatilities."""
        vol: np.ndarray = np.sqrt(self.variance)
        return vol

    @property
    def max_sharpe(self) -> tuple[float, np.ndarray]:
        """Maximal Sharpe ratio on the frontier.

        The maximiser lies on one of the two affine segments adjacent to the
        turning point of largest discrete Sharpe ratio. On each segment the Sharpe
        ratio has a closed-form maximiser (see :meth:`_segment_max_sharpe`), so the
        result is exact rather than the product of a numerical line search.

        Returns:
            Tuple of maximal Sharpe ratio and the weights to achieve it

        """
        weights = self.weights
        sharpe_ratios = self.sharpe_ratio

        # The discrete maximum brackets the continuous one: the optimum sits on a
        # segment touching the turning point of largest Sharpe ratio.
        sr_position_max = int(np.argmax(sharpe_ratios))
        right = min(sr_position_max + 1, len(self) - 1)
        left = max(0, sr_position_max - 1)

        # Look to the left and to the right of the discrete maximum.
        if right > sr_position_max:
            sharpe_ratio_right, w_right = self._segment_max_sharpe(weights[sr_position_max], weights[right])
        else:
            w_right = weights[sr_position_max]
            sharpe_ratio_right = sharpe_ratios[sr_position_max]

        if left < sr_position_max:
            sharpe_ratio_left, w_left = self._segment_max_sharpe(weights[left], weights[sr_position_max])
        else:
            w_left = weights[sr_position_max]
            sharpe_ratio_left = sharpe_ratios[sr_position_max]

        if sharpe_ratio_left > sharpe_ratio_right:
            return sharpe_ratio_left, w_left

        return sharpe_ratio_right, w_right

    def _segment_max_sharpe(self, w0: np.ndarray, w1: np.ndarray) -> tuple[float, np.ndarray]:
        """Closed-form maximum Sharpe ratio on the affine segment between two points.

        Parametrise the segment as ``w(t) = (1 - t) w0 + t w1`` for ``t`` in
        ``[0, 1]``. The expected return is affine and the variance quadratic in
        ``t``, so the Sharpe ratio is::

            S(t) = (a0 + a1 t) / sqrt(c0 + c1 t + c2 t**2)

        Its derivative has a *linear* numerator (the ``t**2`` terms cancel), so
        there is a single stationary point
        ``t* = (a0 c1 - 2 a1 c0) / (a1 c1 - 2 a0 c2)``. The maximiser over the
        segment is therefore whichever of ``{0, 1, clamp(t*)}`` yields the largest
        Sharpe ratio, evaluated in closed form rather than by a bounded line search.

        Args:
            w0: Weights at the ``t = 0`` end of the segment.
            w1: Weights at the ``t = 1`` end of the segment.

        Returns:
            Tuple of the maximal Sharpe ratio on the segment and its weights.

        """
        delta = w1 - w0
        sigma_w0 = _covariance_matvec(self.covariance, w0)
        sigma_delta = _covariance_matvec(self.covariance, delta)
        a0 = float(self.mean @ w0)
        a1 = float(self.mean @ delta)
        c0 = float(w0 @ sigma_w0)
        c1 = 2.0 * float(w0 @ sigma_delta)
        c2 = float(delta @ sigma_delta)

        def sharpe_at(t: float) -> tuple[float, np.ndarray]:
            """Sharpe ratio and weights at position ``t`` along the segment."""
            weight = w0 + t * delta
            sharpe = (a0 + a1 * t) / np.sqrt(c0 + c1 * t + c2 * t * t)
            return float(sharpe), weight

        # Candidate positions: the two endpoints and the interior stationary point
        # (only when it falls strictly inside the segment).
        candidates = [0.0, 1.0]
        denominator = a1 * c1 - 2.0 * a0 * c2
        if denominator != 0.0:
            t_star = (a0 * c1 - 2.0 * a1 * c0) / denominator
            if 0.0 < t_star < 1.0:
                candidates.append(t_star)

        return max((sharpe_at(t) for t in candidates), key=lambda item: item[0])

    def plot(self, volatility: bool = False, markers: bool = True) -> go.Figure:
        """Plot the efficient frontier.

        This function creates a line plot of the efficient frontier, with expected return
        on the y-axis and either variance or volatility on the x-axis.

        Args:
            volatility: If True, plot volatility (standard deviation) on the x-axis.
                       If False, plot variance on the x-axis.
            markers: If True, show markers at each point on the frontier.

        Returns:
            A plotly Figure object that can be displayed or saved.

        """
        try:
            import plotly.graph_objects as go
        except ImportError as e:
            msg = "Plotting requires plotly. Install it with: pip install cvxcla[plot]"
            raise ImportError(msg) from e

        fig = go.Figure()

        x = self.volatility if volatility else self.variance
        axis_title = "Expected volatility" if volatility else "Expected variance"

        fig.add_trace(
            go.Scatter(x=x, y=self.returns, mode="lines+markers" if markers else "lines", name="Efficient Frontier")
        )

        fig.update_layout(
            xaxis_title=axis_title,
            yaxis_title="Expected Return",
        )

        return fig

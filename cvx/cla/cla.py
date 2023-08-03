# On 20130210, v0.2
# Critical Line Algorithm
# Personal property of Marcos Lopez de Prado
# by MLdP <lopezdeprado@gmail.com>
from dataclasses import dataclass, field
from typing import List

# Modifications by Thomas Schmelzer to make it work again
import numpy as np

from cvx.cla.schur import Schur
from cvx.cla.first import init_algo
from cvx.cla.types import MATRIX


# ---------------------------------------------------------------
# ---------------------------------------------------------------

@dataclass(frozen=True)
class TurningPoint:
    weights: np.ndarray
    lamb: float
    free: np.ndarray

    @property
    def free_indices(self):
        return np.where(self.free)[0]

    @property
    def blocked_indices(self):
        return np.where(~self.free)[0]


@dataclass(frozen=True)
class CLA:
    mean: MATRIX
    covariance: MATRIX
    lower_bounds: MATRIX
    upper_bounds: MATRIX
    turning_points: List[TurningPoint] = field(default_factory=list)

    def __post_init__(self):
        # Compute the turning points,free sets and weights
        first = init_algo(mean=self.mean, lower_bounds=self.lower_bounds, upper_bounds=self.upper_bounds)
        tp = TurningPoint(weights=first.weights, lamb=+np.inf, free=first.free)
        self.turning_points.append(tp)

        while True:
            last = self.turning_points[-1]

            if np.all(last.free):
                break

            # 1) case a): Bound one free weight
            l_in = -np.inf

            # only try to bound a free asset if there are least two of them
            if np.sum(last.free) > 1:
                schur = Schur(
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

                schur = Schur(
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

                if self.turning_points[-1].lamb > lamb > l_out:
                    l_out, i_out = lamb, i

            if l_in > 0 or l_out > 0:
                # 4) decide lambda
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

            schur = Schur(
                covariance=self.covariance,
                mean=self.mean,
                free=f,
                weights=w,
            )

            # 5) compute solution vector
            weights = schur.update_weights(lamb=lll)

            tp = TurningPoint(weights=weights, lamb=lll, free=f)
            self.turning_points.append(tp)

        # 6) compute minimum variance solution
        f = np.full_like(self.mean, True, dtype=np.bool_)

        schur = Schur(
            covariance=self.covariance,
            mean=self.mean,
            free=f,
            weights=np.zeros_like(self.mean)
        )

        weights = schur.update_weights(lamb=0)
        tp = TurningPoint(weights=weights, lamb=0, free=f)
        self.turning_points.append(tp)

"""
Critical Line algorithm
"""
from __future__ import annotations

import collections as col
from dataclasses import dataclass

import numpy as np

from cvx.cla.first import init_algo
from cvx.cla.types import MATRIX


@dataclass(frozen=True)
class TurningPoint:
    """A TurningPoint is a point a new asset comes in / is leaving..."""

    free: MATRIX
    weights: MATRIX
    lamb: float | None = None
    gamma: float | None = None

    Schur = col.namedtuple(
        "Schur",
        [
            "covariance_free",
            "covariance_free_inv",
            "covariance_free_blocked",
            "mean_free",
            "weights_blocked",
        ],
    )

    @property
    def free_assets(self):
        return self.free

    @property
    def blocked_assets(self):
        f = np.array([False for _ in self.weights])
        f[self.free_assets] = True
        return list(np.where(~f)[0])

    @staticmethod
    def construct(mean, lower_bounds, upper_bounds, covariance):
        first = init_algo(
            mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds
        )
        turning_point = TurningPoint(
            weights=first.weights, free=list(np.where(first.free)[0])
        )

        num = mean.shape[0]

        # compile a list of turning points
        turning_points = [turning_point]

        while True:
            # 1) case a):
            # find the best candidate to remove from the list of free variables,
            # e.g. bound one free weight
            tp_in, l_in = turning_points[-1].remove(
                covariance=covariance,
                mean=mean,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )

            # 2) case b):
            # find the best candidate to add to a variable to to the free variables,
            # e.g. free one bounded weight
            tp_out, l_out = turning_points[-1].add(covariance=covariance, mean=mean)

            # 3) decide lambda
            if l_in and l_out < 0:
                break

            if l_in > l_out:
                turning_point = tp_in
            else:
                turning_point = tp_out

            # 4) compute solution vector (at this stage we know lambda!)
            # Note, it's expensive to update the weights and
            # hence we do not do it yet in step 1 or step 2
            turning_points.append(
                turning_point.update_weights(covariance=covariance, mean=mean)
            )

            # 5) compute minimum variance portfolio
            if len(turning_points[-1].free) == num:
                turning_point = turning_points[-1].update_weights(
                    covariance=covariance, mean=np.zeros(num)
                )

                turning_points.append(turning_point)

        return turning_points

    # @property
    # def blocked(self):
    #    return list(set(range(self.weights.shape[0])).difference(set(self.free)))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def _update(self, weights, gamma=None):
        new_weights = np.copy(self.weights)
        for index, value in enumerate(self.free):
            new_weights[value] = weights[index]

        return TurningPoint(
            weights=new_weights, free=self.free, lamb=self.lamb, gamma=gamma
        )

    def __get_matrix(self, covariance, mean, blocked=None):
        def bisect(matrix, ix1, ix2):
            mat = np.zeros((2, 2), object)
            mat[0, 0] = matrix[np.ix_(ix1, ix1)]
            mat[0, 1] = matrix[np.ix_(ix1, ix2)]
            mat[1, 0] = matrix[np.ix_(ix2, ix1)]
            mat[1, 1] = matrix[np.ix_(ix2, ix2)]
            return mat

        if blocked is not None:
            free = self.free + [blocked]
        else:
            free = self.free

        # get the list of blocked variables
        # f <=> b
        num = covariance.shape[0]
        blocked = list(set(range(num)).difference(set(free)))

        mat = bisect(covariance, free, blocked)

        return self.Schur(
            covariance_free=mat[0, 0],
            covariance_free_blocked=mat[0, 1],
            covariance_free_inv=np.linalg.inv(mat[0, 0]),
            mean_free=mean[free],
            weights_blocked=self.weights[blocked],
        )

    @staticmethod
    def __weight(
        lambdas, covariance_inv, covariance_free_blocked, mean_free, weights_blocked
    ):
        # 1) compute gamma
        ones_f = np.ones(mean_free.shape)
        _g1 = np.dot(np.dot(ones_f.T, covariance_inv), mean_free)
        _g2 = np.dot(np.dot(ones_f.T, covariance_inv), ones_f)
        if weights_blocked is None:
            gamma = -lambdas * _g1 / _g2 + 1 / _g2
            _w1 = 0
        else:
            ones_b = np.ones(weights_blocked.shape)
            _g3 = np.dot(ones_b.T, weights_blocked)
            _g4 = np.dot(covariance_inv, covariance_free_blocked)
            _w1 = np.dot(_g4, weights_blocked)
            _g4 = np.dot(ones_f.T, _w1)
            gamma = -lambdas * _g1 / _g2 + (1 - _g3 + _g4) / _g2
        # 2) compute weights
        _w2 = np.dot(covariance_inv, ones_f)
        _w3 = np.dot(covariance_inv, mean_free)
        return -_w1 + gamma * _w2 + lambdas * _w3, gamma

    def update_weights(self, covariance, mean):
        schur = self.__get_matrix(covariance, mean)

        weights, gam = TurningPoint.__weight(
            self.lamb,
            covariance_inv=schur.covariance_free_inv,
            covariance_free_blocked=schur.covariance_free_blocked,
            mean_free=schur.mean_free,
            weights_blocked=schur.weights_blocked,
        )

        return self._update(weights=weights, gamma=gam)

    def remove(self, covariance, mean, lower_bounds, upper_bounds):
        free = list(self.free)
        weights = np.copy(self.weights)

        schur = self.__get_matrix(covariance, mean)

        l_in = -np.inf
        bi_in = None
        i_in = None

        j = 0
        for i in self.free:
            lamb, _bi = self.__compute_lambda(
                schur.covariance_free_inv,
                schur.covariance_free_blocked,
                schur.mean_free,
                schur.weights_blocked,
                j,
                [lower_bounds[i], upper_bounds[i]],
            )
            if lamb > l_in:
                l_in = lamb
                i_in = i
                bi_in = _bi
            j += 1

        if l_in == -np.inf:
            return None, -np.inf

        # compute a TurningPoint
        free.remove(i_in)
        weights[i_in] = bi_in  # set value at the correct boundary
        # this is a candidate for a turning point
        return TurningPoint(free=free, weights=weights, lamb=l_in), l_in

    def add(self, covariance, mean):
        l_out = -np.inf
        free = list(self.free)
        weights = np.copy(self.weights)
        i_out = None

        if len(self.free) < covariance.shape[0]:
            for i in self.blocked_assets:
                print(f"i {i}")
                schur = self.__get_matrix(covariance=covariance, mean=mean, blocked=i)
                lamb, _bi = self.__compute_lambda(
                    schur.covariance_free_inv,
                    schur.covariance_free_blocked,
                    schur.mean_free,
                    schur.weights_blocked,
                    schur.mean_free.shape[0] - 1,
                    self.weights[i],
                )
                if (self.lamb is None or lamb < self.lamb) and lamb > l_out:
                    l_out = lamb
                    i_out = i

        if l_out == -np.inf:
            return None, -np.inf

        # compute a TurningPoint
        free.append(i_out)
        # this is a candidate for a turning point
        return TurningPoint(free=free, weights=weights, lamb=l_out), l_out

    @staticmethod
    def __compute_lambda(
        covariance_inv, covariance_free_blocked, mean_free, weights_blocked, index, _bi
    ):
        # 1) C
        ones_f = np.ones(mean_free.shape)
        _c1 = np.dot(np.dot(ones_f.T, covariance_inv), ones_f)
        _c2 = np.dot(covariance_inv, mean_free)
        _c3 = np.dot(np.dot(ones_f.T, covariance_inv), mean_free)
        _c4 = np.dot(covariance_inv, ones_f)
        _aux = -_c1 * _c2[index] + _c3 * _c4[index]
        if _aux == 0:
            return -np.inf, None
        # 2) bi
        if isinstance(_bi, list):
            _bi = _bi[1] if _aux > 0 else _bi[0]
        # 3) Lambda
        if weights_blocked is None:
            # All free assets
            return float((_c4[index] - _c1 * _bi) / _aux), _bi

        ones_b = np.ones(weights_blocked.shape)
        _l1 = np.dot(ones_b.T, weights_blocked)
        _l2 = np.dot(covariance_inv, covariance_free_blocked)
        _l3 = np.dot(_l2, weights_blocked)
        _l2 = np.dot(ones_f.T, _l3)
        return (
            float(((1 - _l1 + _l2) * _c4[index] - _c1 * (_bi + _l3[index])) / _aux),
            _bi,
        )

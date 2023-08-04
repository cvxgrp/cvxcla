from dataclasses import dataclass, field
from typing import List

import numpy as np
import logging

from cvx.cla._first import init_algo
from cvx.cla.types import MATRIX, BOOLEAN_VECTOR, TurningPoint



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

            # check the turning point
            self.append(tp)

        # 6) compute minimum variance solution
        last = self.turning_points[-1]
        mean = np.copy(self.mean)
        mean[last.free] = 0.0
        f = last.free
        w = last.weights

        #x = cp.Variable(shape=(self.mean.shape[0]), name="weights")
        #constraints = [
        #    cp.sum(x) == 1,
        #    x >= self.lower_bounds,
        #    x <= self.upper_bounds
        #]
        #cp.Problem(cp.Minimize(cp.quad_form(x, self.covariance)), constraints).solve()

        schur = Schur(
            covariance=self.covariance,
            mean=mean,
            free=f,
            weights=w
        )

        weights = schur.update_weights(lamb=0)
        tp = TurningPoint(weights=weights, lamb=0, free=last.free)

        self.append(tp)

    def append(self, tp: TurningPoint, tol=1e-10):
        tol = 1e-10
        assert np.all(
            tp.weights >= self.lower_bounds - tol), f"{self.lower_bounds} - {tp.weights}"
        assert np.all(
            tp.weights <= self.upper_bounds + tol), f"-{self.upper_bounds} + {tp.weights}"
        assert np.allclose(np.sum(tp.weights), 1.0), f"{np.sum(tp.weights)}"

        print(tp.weights)

        self.turning_points.append(tp)
class Schur:
    def __init__(self, covariance, mean, free: BOOLEAN_VECTOR, weights: MATRIX):
        assert (
            covariance.shape[0]
            == covariance.shape[1]
            == mean.shape[0]
            == free.shape[0]
            == weights.shape[0]
        )
        self.covariance = covariance
        self.mean = mean
        self.free = free
        self.weights = weights

    @property
    def covariance_free(self):
        return self.covariance[self.free][:, self.free]

    @property
    def covariance_free_blocked(self):
        return self.covariance[self.free][:, ~self.free]

    @property
    def covariance_free_inv(self):
        return np.linalg.inv(self.covariance_free)

    @property
    def mean_free(self):
        return self.mean[self.free]

    @property
    def weights_blocked(self):
        return self.weights[~self.free]

    def compute_lambda(self, index, bi):
        def compute_bi(c, bi):
            if np.shape(bi)[0] == 1 or c <= 0:
                return bi[0]
            return bi[1]

        c1 = np.sum(np.sum(self.covariance_free_inv))
        c2 = np.dot(self.covariance_free_inv, self.mean_free)
        c3 = np.dot(np.sum(self.covariance_free_inv, axis=1), self.mean_free)
        c4 = np.sum(self.covariance_free_inv, axis=0)

        aux = -c1 * c2[index] + c3 * c4[index]

        bi = compute_bi(aux, bi)

        if self.weights_blocked.size == 0:
            return float((c4[index] - c1 * bi) / aux), bi

        l1 = np.sum(self.weights_blocked)
        l2 = np.dot(self.covariance_free_inv, self.covariance_free_blocked)
        l3 = np.dot(l2, self.weights_blocked)
        l2 = np.sum(l3)
        return ((1 - l1 + l2) * c4[index] - c1 * (bi + l3[index])) / aux, bi

    def _compute_weight(self, lamb):
        ones_f = np.ones(self.mean_free.shape)
        g1 = np.dot(np.sum(self.covariance_free_inv, axis=0), self.mean_free)
        g2 = np.sum(np.sum(self.covariance_free_inv))

        if self.weights_blocked.size == 0:
            gamma = -lamb * g1 / g2 + 1 / g2
            w1 = 0
        else:
            g3 = np.sum(self.weights_blocked)
            g4 = np.dot(self.covariance_free_inv, self.covariance_free_blocked)
            w1 = np.dot(g4, self.weights_blocked)
            g4 = np.sum(w1)
            gamma = -lamb * g1 / g2 + (1 - g3 + g4) / g2

        w2 = np.dot(self.covariance_free_inv, ones_f)
        w3 = np.dot(self.covariance_free_inv, self.mean_free)
        return -w1 + gamma * w2 + lamb * w3, gamma

    def  update_weights(self, lamb, logger=None):
        # schur = self.__get_matrix(covariance, mean)
        logger = logger or logging.getLogger(__name__)

        weights, _ = self._compute_weight(lamb)
        logger.info(f"CURRENTLY: {self.weights}")
        logger.info(f"UPDATE: {weights}")
        logger.info(f"FREE: {self.free}")

        new_weights = np.copy(self.weights)
        new_weights[self.free] = weights
        logger.info(f"NEW: {new_weights}")

        return new_weights

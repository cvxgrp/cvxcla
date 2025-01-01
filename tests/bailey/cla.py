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
from dataclasses import dataclass

import numpy as np
from loguru import logger

from cvx.cla.claux import CLAUX
from cvx.cla.types import BOOLEAN_VECTOR, MATRIX, TurningPoint


@dataclass(frozen=True)
class CLA(CLAUX):
    def __post_init__(self):
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


class _Schur:
    def __init__(self, covariance, mean, free: BOOLEAN_VECTOR, weights: MATRIX):
        assert covariance.shape[0] == covariance.shape[1] == mean.shape[0] == free.shape[0] == weights.shape[0]
        self.covariance = covariance
        self.mean = mean
        self.free = free
        self.weights = weights
        self.__free_inv = np.linalg.inv(self.covariance_free)

    @property
    def covariance_free(self):
        return self.covariance[self.free][:, self.free]

    @property
    def covariance_free_blocked(self):
        return self.covariance[self.free][:, ~self.free]

    @property
    def covariance_free_inv(self):
        return self.__free_inv

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

    def _compute_weight(self, lamb):
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

    def update_weights(self, lamb):
        weights, _ = self._compute_weight(lamb)
        new_weights = np.copy(self.weights)
        new_weights[self.free] = weights
        return new_weights

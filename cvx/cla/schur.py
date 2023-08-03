import logging

import numpy as np

from cvx.cla.types import BOOLEAN_VECTOR, MATRIX


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
            if np.shape(bi)[0] == 1 or c < 0:
                return bi[0]
            return bi[1]

        c1 = np.sum(np.sum(self.covariance_free_inv))
        c2 = np.dot(self.covariance_free_inv, self.mean_free)
        c3 = np.dot(np.sum(self.covariance_free_inv, axis=1), self.mean_free)
        c4 = np.sum(self.covariance_free_inv, axis=0)

        aux = -c1 * c2[index] + c3 * c4[index]
        if aux == 0:
            return -np.inf, None

        bi = compute_bi(aux, bi)

        if self.weights_blocked is None:
            return float((c4[index] - c1 * bi) / aux), bi

        l1 = np.sum(self.weights_blocked)
        l2 = np.dot(self.covariance_free_inv, self.covariance_free_blocked)
        l3 = np.dot(l2, self.weights_blocked)
        l2 = np.sum(l3)
        return float(((1 - l1 + l2) * c4[index] - c1 * (bi + l3[index])) / aux), bi

    def _compute_weight(self, lamb):
        ones_f = np.ones(self.mean_free.shape)
        g1 = np.dot(np.sum(self.covariance_free_inv, axis=0), self.mean_free)
        g2 = np.sum(np.sum(self.covariance_free_inv))

        if self.weights_blocked is None:
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

    def update_weights(self, lamb, logger=None):
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

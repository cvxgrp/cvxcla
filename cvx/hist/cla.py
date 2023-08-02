# On 20130210, v0.2
# Critical Line Algorithm
# Personal property of Marcos Lopez de Prado
# by MLdP <lopezdeprado@gmail.com>

# Modifications by Thomas Schmelzer to make it work again
import numpy as np

# ---------------------------------------------------------------
# ---------------------------------------------------------------


class CLA:
    def __init__(self, mean, covar, lB, uB):
        # Initialize the class
        self.mean = mean
        self.covar = covar
        self.lB = lB
        self.uB = uB
        self.w = []  # solution
        self.l = []  # lambdas
        self.g = []  # gammas
        self.f = []  # free weights

    # ---------------------------------------------------------------
    def solve(self):
        # Compute the turning points,free sets and weights

        f, w = CLA.init_algo(mean=self.mean, lB=self.lB, uB=self.uB)

        self.w.append(np.copy(w))  # store solution
        self.l.append(+np.inf)
        self.g.append(None)
        self.f.append(f[:])

        while True:
            # 1) case a): Bound one free weight
            l_in = -np.inf
            # only try to bound a free asset if there are least two of them
            if len(f) > 1:
                covarF, covarFB, meanF, wB = CLA.get_matrices(
                    f, covar=self.covar, mean=self.mean, w=self.w[-1]
                )

                covarF_inv = np.linalg.inv(covarF)
                j = 0
                for i in f:
                    lamb, bi = CLA.compute_lambda(
                        covarF_inv,
                        covarFB,
                        meanF,
                        wB,
                        j,
                        np.array([self.lB[i], self.uB[i]]),
                    )

                    if lamb > l_in:
                        l_in, i_in, bi_in = lamb, i, bi
                    j += 1

            # 2) case b): Free one bounded weight
            l_out = -np.inf
            # if len(f)<self.mean.shape[0]:
            b = CLA.getB(f, num=self.mean.shape[0])
            for i in b:
                covarF, covarFB, meanF, wB = self.get_matrices(
                    f + [i], covar=self.covar, mean=self.mean, w=self.w[-1]
                )
                covarF_inv = np.linalg.inv(covarF)
                lamb, bi = CLA.compute_lambda(
                    covarF_inv,
                    covarFB,
                    meanF,
                    wB,
                    meanF.shape[0] - 1,
                    np.array([self.w[-1][i]]),
                )

                if self.l[-1] > lamb > l_out:
                    l_out, i_out = lamb, i

            if l_in < 0 and l_out < 0:
                # 3) compute minimum variance solution
                self.l.append(0)
                covarF, covarFB, meanF, wB = CLA.get_matrices(
                    f, covar=self.covar, mean=self.mean, w=self.w[-1]
                )
                covarF_inv = np.linalg.inv(covarF)
                meanF = np.zeros(meanF.shape)
            else:
                # 4) decide lambda
                w = np.copy(self.w[-1])
                if l_in > l_out:
                    self.l.append(l_in)
                    f.remove(i_in)
                    w[i_in] = bi_in  # set value at the correct boundary
                else:
                    self.l.append(l_out)
                    f.append(i_out)

                covarF, covarFB, meanF, wB = CLA.get_matrices(
                    f, covar=self.covar, mean=self.mean, w=w
                )
                covarF_inv = np.linalg.inv(covarF)
            # 5) compute solution vector
            wF, g = CLA.computeW(covarF_inv, covarFB, meanF, wB, lamb=self.l[-1])

            for i in range(len(f)):
                w[f[i]] = wF[i]

            self.w.append(np.copy(w))  # store solution
            self.g.append(g)
            self.f.append(f[:])
            if self.l[-1] == 0:
                break

    # ---------------------------------------------------------------
    @staticmethod
    def init_algo(mean, lB, uB):
        # Initialize weights to lower bounds
        weights = np.copy(lB)

        assert np.all(lB <= uB), "Lower bound exceeds upper bound"

        # Move weights from lower to upper bound
        # until sum of weights hits or exceeds 1
        for index in np.argsort(mean)[::-1]:
            weights[index] = uB[index]
            if np.sum(weights) >= 1:
                weights[index] -= np.sum(weights) - 1
                return [index], weights

        raise ValueError("No fully invested solution exists")

    # ---------------------------------------------------------------
    @staticmethod
    def compute_bi(c, bi):
        if np.shape(bi)[0] == 1 or c < 0:
            return bi[0]
        return bi[1]

    # ---------------------------------------------------------------
    @staticmethod
    def computeW(covarF_inv, covarFB, meanF, wB, lamb):
        # 1) compute gamma
        onesF = np.ones(meanF.shape)
        g1 = np.dot(np.dot(onesF.T, covarF_inv), meanF)
        g2 = np.dot(np.dot(onesF.T, covarF_inv), onesF)
        if not wB.size > 0:
            g, w1 = float(-lamb * g1 / g2 + 1 / g2), 0
        else:
            # onesB=np.ones(wB.shape)
            g3 = np.sum(wB)
            g4 = np.dot(covarF_inv, covarFB)
            w1 = np.dot(g4, wB)
            g4 = np.sum(w1)
            g = float(-lamb * g1 / g2 + (1 - g3 + g4) / g2)
        # 2) compute weights
        w2 = np.dot(covarF_inv, onesF)
        w3 = np.dot(covarF_inv, meanF)
        return -w1 + g * w2 + lamb * w3, g

    # ---------------------------------------------------------------
    @staticmethod
    def compute_lambda(covarF_inv, covarFB, meanF, wB, i, bi):
        # 1) C
        onesF = np.ones(meanF.shape)
        c1 = np.dot(np.dot(onesF.T, covarF_inv), onesF)
        c2 = np.dot(covarF_inv, meanF)
        c3 = np.dot(np.dot(onesF.T, covarF_inv), meanF)
        c4 = np.dot(covarF_inv, onesF)
        c = -c1 * c2[i] + c3 * c4[i]
        if c == 0:
            return None, None
        # 2) bi
        bi = CLA.compute_bi(c, bi)
        assert isinstance(bi, float)

        # 3) Lambda
        if not wB.size > 0:
            # All free assets
            return float((c4[i] - c1 * bi) / c), bi
        else:
            l1 = np.sum(wB)
            l2 = np.dot(covarF_inv, covarFB)
            l3 = np.dot(l2, wB)
            l2 = np.sum(l3)
            return float(((1 - l1 + l2) * c4[i] - c1 * (bi + l3[i])) / c), bi

    # ---------------------------------------------------------------
    @staticmethod
    def get_matrices(f, covar, mean, w):
        assert f is not None

        # Slice covarF,covarFB,covarB,meanF,meanB,wF,wB
        covarF = covar[f, :][:, f]
        meanF = mean[f]
        b = CLA.getB(f, num=mean.shape[0])

        if not b:
            covarFB = np.array([])
            wB = np.array([])
        else:
            covarFB = covar[f, :][:, b]
            wB = w[b]

        return covarF, covarFB, meanF, wB

    # ---------------------------------------------------------------
    @staticmethod
    def getB(f, num):
        def diffLists(list1, list2):
            return list(set(list1) - set(list2))

        return diffLists(range(num), f)


if __name__ == "__main__":
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([0.3, 0.5, 0.4])
    mean = np.array([0.3, 0.2, 0.5])
    covar = np.array([[0.005, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])

    free, weights = CLA.init_algo(mean=mean, lB=lb, uB=ub)
    print(free)
    print(weights)

    x = CLA(covar=covar, mean=mean, lB=lb, uB=ub)
    # try:
    x.solve()
    # except AssertionError:
    #    pass

    assert np.allclose(x.lB, lb)
    assert np.allclose(x.uB, ub)
    assert np.allclose(x.mean, mean)
    assert np.allclose(x.covar, covar)

    print(x.f)
    print(x.w)
    print(x.l)

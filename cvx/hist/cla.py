# On 20130210, v0.2
# Critical Line Algorithm
# Personal property of Marcos Lopez de Prado
# by MLdP <lopezdeprado@gmail.com>

# Modifications by Thomas Schmelzer to make it work again
import numpy as np

from cvx.cla.schur import Schur
from cvx.cla.first import init_algo


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
        self.f = []  # free weights

    # ---------------------------------------------------------------
    def solve(self):
        # Compute the turning points,free sets and weights

        first = init_algo(mean=self.mean, lower_bounds=self.lB, upper_bounds=self.uB)

        self.w.append(first.weights)  # store solution
        self.l.append(+np.inf)
        self.f.append(first.free)

        while True:
            f = np.copy(self.f[-1])
            w = np.copy(self.w[-1])
            if np.all(f):
                print("All points are free now")
                break

            # 1) case a): Bound one free weight
            l_in = -np.inf

            # only try to bound a free asset if there are least two of them
            if np.sum(f) > 1:
                f = np.copy(self.f[-1])
                w = np.copy(self.w[-1])

                schur = Schur(
                    covariance=self.covar,
                    mean=self.mean,
                    free=f,
                    weights=w,
                )

                #j = 0

                for i in np.where(f)[0]:
                    # count the number of entries that are True below the ith entry in fff
                    j = np.sum(f[:i])

                    lamb, bi = schur.compute_lambda(
                        index=j,
                        bi=np.array([self.lB[i], self.uB[i]]),
                    )

                    if lamb > l_in:
                        l_in, i_in, bi_in = lamb, i, bi
                    #j += 1

            # 2) case b): Free one bounded weight
            l_out = -np.inf

            f = np.copy(self.f[-1])
            w = np.copy(self.w[-1])
            for i in np.where(~f)[0]:
                fff = np.copy(f)
                fff[i] = True

                schur = Schur(
                    covariance=self.covar,
                    mean=self.mean,
                    free=fff,
                    weights=w,
                )

                # count the number of entries that are True below the ith entry in fff
                j = np.sum(fff[:i])

                lamb, bi = schur.compute_lambda(
                    # index i in fff corresponds to index j in mean_free
                    index=j,
                    bi=np.array([w[i]]),
                )

                if self.l[-1] > lamb > l_out:
                    l_out, i_out = lamb, i

            if l_in > 0 or l_out > 0:
                print(l_in, l_out)
                # 4) decide lambda
                w = np.copy(self.w[-1])
                if l_in > l_out:
                    self.l.append(l_in)
                    f[i_in] = False
                    w[i_in] = bi_in  # set value at the correct boundary
                else:
                    self.l.append(l_out)
                    f[i_out] = True
            else:
                print(l_in, l_out)
                print("Both lambdas are negative")
                break

            schur = Schur(
                covariance=self.covar,
                mean=self.mean,
                free=f,
                weights=w,
            )

            # 5) compute solution vector
            weights = schur.update_weights(lamb=self.l[-1])

            self.w.append(np.copy(weights))  # store solution
            self.f.append(f)



if __name__ == "__main__":
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([0.3, 0.5, 0.4])
    mean = np.array([0.3, 0.2, 0.5])
    covar = np.array([[0.005, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])

    first = init_algo(mean=mean, lower_bounds=lb, upper_bounds=ub)
    print(first.free)
    print(first.weights)

    x = CLA(covar=covar, mean=mean, lB=lb, uB=ub)
    x.solve()

    assert np.allclose(x.lB, lb)
    assert np.allclose(x.uB, ub)
    assert np.allclose(x.mean, mean)
    assert np.allclose(x.covar, covar)

    print(x.f)
    print(x.w)
    print(x.l)

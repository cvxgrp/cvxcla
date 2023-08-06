from dataclasses import dataclass, field
from typing import List

import numpy as np
import cvxpy as cp

from loguru import logger as loguru

from cvx.cla.aux import CLAUX
from cvx.cla.first import init_algo
from cvx.cla.types import TurningPoint


def Mbar_matrix(C, A, io: List[bool]):
    m = A.shape[0]

    Cbar = np.copy(C)

    #if io is not None:
    Cbar[io, :] = 0
    Cbar[io, io] = 1

    Abar = np.copy(A)

    #if io is not None:
    Abar[:, io] = 0

    toprow = np.concatenate([Cbar, Abar.T], axis=1)
    bottomrow = np.concatenate([Abar, np.zeros((m,m))], axis=1)

    return np.concatenate([toprow, bottomrow], axis=0)



@dataclass(frozen=True)
class CLA(CLAUX):
    def __post_init__(self):
        logger = loguru

        ns = self.mean.shape[0]

        A = np.ones((1, ns))
        b = np.array([1.0])
        m = A.shape[0]

        # --A07-- Compute basic statistics of the data.
        C = self.covariance

        # --A08-- Initialize the portfolio.
        x = init_algo(self.mean, self.lower_bounds, self.upper_bounds)

        # --A10-- Set the P matrix.
        P = np.concatenate((C, A.T), axis=1)

        # --A11 -- Initialize storage for quantities # to be computed in the main loop.
        lam = np.inf

        self.append(x)
        logger.info("First turning point added")

        # --A12 -- The main CLA loop , which steps
        # from corner portfolio to corner portfolio.
        while lam > 0:
            last = self.turning_points[-1]

            # --A13-- Create the UP, DN, and IN
            # sets from the current state vector.
            UP = ~last.free & np.isclose(last.weights, self.upper_bounds)
            DN = ~last.free & np.isclose(last.weights, self.lower_bounds)

            # a variable is out if it UP or DN
            OUT = np.logical_or(UP, DN)
            IN = ~OUT

            Mbar = Mbar_matrix(C, A, OUT)

            up = np.zeros(ns)
            up[UP] = self.upper_bounds[UP]

            dn = np.zeros(ns)
            dn[DN] = self.lower_bounds[DN]

            # --A15-- Create the right-hand sides for alpha and beta.
            k = up + dn
            bot = b - A @ k

            top = np.copy(self.mean)
            top[OUT] = 0

            rhsb = np.concatenate([top, np.zeros(m)])
            rhsa = np.concatenate([up + dn, bot], axis=0)

            # --A16-- Compute alpha, beta, gamma, and delta.
            alpha = np.linalg.lstsq(Mbar, rhsa)[0]
            beta = np.linalg.lstsq(Mbar, rhsb)[0]

            gamma = P @ alpha
            delta = P @ beta - self.mean

            # -A17-- Prepare the ratio matrix.
            L = -np.inf*np.ones([ns, 4])

            r_beta = beta[range(ns)]
            r_alpha = alpha[range(ns)]

            # --A18-- IN security possibly going UP.
            i = IN & (r_beta < -self.tol)
            L[i, 0] = (self.upper_bounds[i] - r_alpha[i]) / r_beta[i]

            # --A19-- IN security possibly going DN.
            i = IN & (r_beta > + self.tol)
            L[i, 1] = (self.lower_bounds[i] - r_alpha[i]) / r_beta[i]

            # --A20--UP security possibly going IN.
            i = UP & (delta < -self.tol)
            L[i, 2] = -gamma[i] / delta[i]

            # --A21-- DN security possibly going IN.
            i = DN & (delta > +self.tol)
            L[i, 3] = -gamma[i] / delta[i]

            # --A22--If all elements of ratio are negative,
            # we have reached the end of the efficient frontier.
            if np.max(L) < 0:
                break

            secchg, dirchg = np.unravel_index(np.argmax(L, axis=None), L.shape)

            # --A25-- Set the new value of lambda_E.
            lam = L[secchg, dirchg]

            free = np.copy(last.free)
            if dirchg == 0 or dirchg == 1:
                free[secchg] = False
            else:
                free[secchg] = True

            # --A27-- Compute the portfolio at this corner.
            x = r_alpha + lam * r_beta

            # --A28-- Save the data computed at this corner.
            self.append(TurningPoint(lamb=lam, weights=x, free=free))

        x = cp.Variable(shape=(self.mean.shape[0]), name="weights")
        constraints = [
           cp.sum(x) == 1,
           x >= self.lower_bounds,
           x <= self.upper_bounds
        ]
        cp.Problem(cp.Minimize(cp.quad_form(x, self.covariance)), constraints).solve()

        self.append(TurningPoint(lamb=0, weights=x.value, free=last.free))

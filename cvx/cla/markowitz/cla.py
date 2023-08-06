from dataclasses import dataclass, field
from typing import List

import numpy as np
import cvxpy as cp

from loguru import logger as loguru

from cvx.cla.aux import CLAUX
from cvx.cla.first import init_algo
from cvx.cla.types import MATRIX, TurningPoint


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
        logger.info(f"Number of assets {ns}")

        A = np.ones((1, ns))
        b = np.array([1.0])
        m = A.shape[0]

        # --A07-- Compute basic statistics of the data.
        C = self.covariance

        # --A08-- Initialize the portfolio.
        x = init_algo(self.mean, self.lower_bounds, self.upper_bounds).weights

        #x = initport(self.mean, self.lower_bounds, self.upper_bounds)
        logger.info(f"First vector of weights {x}")

        # --A09-- Set the state vector.
        up = 1 * (np.abs(x - self.upper_bounds) < self.tol)
        dn = 1 * (np.abs(x - self.lower_bounds) < self.tol)
        s = np.subtract(up, dn)
        logger.info(f"State vector {s}")

        # --A10-- Set the P matrix.
        P = np.concatenate((C, A.T), axis=1)

        # --A11 -- Initialize storage for quantities # to be computed in the main loop.
        lam = np.inf

        self.turning_points.append(TurningPoint(weights=x, lamb=np.inf, free=s))
        logger.info("First turning point added")

        # --A12 -- The main CLA loop , which steps
        # from corner portfolio to corner portfolio.
        while lam > 0:

            # --A13-- Create the UP, DN, and IN
            # sets from the current state vector.
            UP = s > +0.9
            DN = s < -0.9

            # a variable is out if it UP or DN
            OUT = np.logical_or(UP, DN)

            Mbar = Mbar_matrix(C, A, OUT)

            up = np.zeros(ns)
            up[UP] = self.upper_bounds[UP]

            dn = np.zeros(ns)
            dn[DN] = self.lower_bounds[DN]

            # --A15-- Create the right-hand sides for alpha and beta.
            k = up + dn
            logger.info(f"K vector {k}")
            bot = b - A @ k
            logger.info(f"Bot vector {bot}")

            top = np.copy(self.mean)
            top[OUT] = 0

            rhsb = np.concatenate([top, np.zeros(m)])
            rhsa = np.concatenate([up + dn, bot], axis=0)

            logger.info(f"RHSa vector {rhsa}")
            logger.info(f"RHSb vector {rhsb}")


            # --A16-- Compute alpha, beta, gamma, and delta.
            alpha = np.linalg.lstsq(Mbar, rhsa)[0]
            beta = np.linalg.lstsq(Mbar, rhsb)[0]

            logger.info(f"Alpha vector \n{alpha}")
            logger.info(f"Beta vector \n{beta}")

            gamma = P @ alpha
            delta = P @ beta - self.mean
            logger.info(f"Gamma vector \n{gamma}")
            logger.info(f"Delta vector \n{delta}")

            # -A17-- Prepare the ratio matrix.
            L = -np.inf*np.ones([ns, 4])

            # --A18-- IN security possibly going UP.
            i = np.where(~OUT & (beta[range(ns)] < -self.tol))[0]
            L[i, 0] = (self.upper_bounds[i] - alpha[i]) / beta[i]

            # --A19-- IN security possibly going DN.
            i = np.where(~OUT & (beta[range(ns)] > +self.tol))[0]
            L[i, 1] = (self.lower_bounds[i] - alpha[i]) / beta[i]

            # --A20--UP security possibly going IN.
            i = np.logical_and(UP, delta < -self.tol)
            L[i, 2] = -gamma[i] / delta[i]
            logger.info(f"UP going IN: {i}")

            # --A21-- DN security possibly going IN.
            i = np.logical_and(DN, delta > +self.tol)
            L[i, 3] = -gamma[i] / delta[i]
            logger.info(f"DN going IN: {i}")

            logger.info(f"L matrix: \n{L}")
            # --A22--If all elements of ratio are negative,
            # we have reached the end of the efficient frontier.
            if np.max(L) < 0:
                break

            #secchg,dirchg \
            #ind = np.argmax(L
            secchg, dirchg = np.unravel_index(np.argmax(L, axis=None), L.shape)

            # --A23-- Find which security is changing state.
            #secmax = np.max(L, axis=1)
            #secchg = np.argmax(secmax)

            logger.info(f"Asset changing state: {secchg}")
            #logger.info(f"Type of change: {secmax}")



            # --A24-- Find in which direction it is changing.
            #dirmax = np.max(L, axis=0)
            #dirchg = np.argmax(dirmax)

            # --A25-- Set the new value of lambda_E.
            lam = L[secchg, dirchg]
            logger.info(f"Lambda: {lam}")

            #assert False

            # --A26-- Set the state vector for the next segment.
            s[secchg] = (+1 if dirchg == 0 else -1 if dirchg == 1 else 0)

            # --A27-- Compute the portfolio at this corner.
            x = alpha[range(ns)] + lam * beta[range(ns)]

            # --A28-- Save the data computed at this corner.
            self.turning_points.append(TurningPoint(lamb=lam, weights=x, free=s))

        x = cp.Variable(shape=(self.mean.shape[0]), name="weights")
        constraints = [
           cp.sum(x) == 1,
           x >= self.lower_bounds,
           x <= self.upper_bounds
        ]
        cp.Problem(cp.Minimize(cp.quad_form(x, self.covariance)), constraints).solve()

        self.turning_points.append(TurningPoint(lamb=0, weights=x.value, free=s))

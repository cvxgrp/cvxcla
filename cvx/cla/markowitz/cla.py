from dataclasses import dataclass, field
from typing import List

import numpy as np
from loguru import logger as loguru

from cvx.cla.first import init_algo
from cvx.cla.types import MATRIX, TurningPoint


@dataclass(frozen=True)
class CLA:
    mean: MATRIX
    covariance: MATRIX
    lower_bounds: MATRIX
    upper_bounds: MATRIX
    tol: float = 1e-5

    turning_points: List[TurningPoint] = field(default_factory=list)

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

        # --A10-- Set the P matrix.
        P = np.concatenate((C, A.T), axis=1)

        # --A11 -- Initialize storage for quantities # to be computed in the main loop.
        lam = np.inf

        self.turning_points.append(TurningPoint(weights=x, lamb=np.inf, free=s))


        # --A12 -- The main CLA loop , which steps
        # from corner portfolio to corner portfolio.
        while lam > 0:

            # --A13-- Create the UP, DN, and IN
            # sets from the current state vector.
            UP = s > +0.9
            DN = s < -0.9
            IN = np.invert(np.logical_or(UP, DN))

            # --A14-- Create the Abar, Cbar, and Mbar matrices.
            io = np.where(np.logical_not(IN))[0]
            Abar = np.copy(A)
            Abar[:, io] = 0

            Cbar = np.copy(C)
            Cbar[io, :] = 0
            Cbar[io, io] = 1

            toprow = np.concatenate((Cbar, Abar.T), axis=1)
            botrow = np.concatenate((Abar, np.zeros([m, m])), axis=1)
            Mbar = np.concatenate([toprow, botrow], axis=0)

            # --A15-- Create the right-hand sides for alpha and beta.
            up = np.multiply(1 * UP, self.upper_bounds)
            dn = np.multiply(1 * DN, self.lower_bounds)
            k = np.add(up, dn)
            bot = b - A @ k

            top = np.copy(self.mean)
            top[io] = 0

            rhsb = np.concatenate([top, np.zeros(m)])
            rhsa = np.concatenate([k, bot], axis=0)

            # --A16-- Compute alpha, beta, gamma, and delta.
            alpha = np.linalg.lstsq(Mbar, rhsa)[0]
            beta = np.linalg.lstsq(Mbar, rhsb)[0]
            gamma = P @ alpha
            delta = P @ beta - self.mean

            # -A17-- Prepare the ratio matrix.
            L = -np.inf*np.ones([ns, 4])

            # --A18-- IN security possibly going UP.
            i = np.where(IN & (beta[range(ns)] < -self.tol))[0]
            L[i, 0] = (self.upper_bounds[i] - alpha[i]) / beta[i]

            # --A19-- IN security possibly going DN.
            i = np.where(IN & (beta[range(ns)] > +self.tol))[0]
            L[i, 1] = (self.lower_bounds[i] - alpha[i]) / beta[i]

            # --A20-- DN security possibly going IN.
            i = np.where(UP & (delta < -self.tol))[0]
            L[i, 2] = -gamma[i] / delta[i]

            # --A21-- UP security possibly going IN.
            i = np.where(DN & (delta > + self.tol))[0]
            L[i, 3] = -gamma[i] / delta[i]

            # --A22--If all elements of ratio are negative,
            # we have reached the end of the efficient frontier.
            if np.max(L) < 0:
                break

            # --A23-- Find which security is changing state.
            secmax = np.max(L, axis=1)
            secchg = np.argmax(secmax)

            # --A24-- Find in which direction it is changing.
            dirmax = np.max(L, axis=0)
            dirchg = np.argmax(dirmax)

            # --A25-- Set the new value of lambda_E.
            lam = np.max(secmax)

            # --A26-- Set the state vector for the next segment.
            s[secchg] = (+1 if dirchg == 0 else -1 if dirchg == 1 else 0)

            # --A27-- Compute the portfolio at this corner.
            x = alpha[range(ns)] + lam * beta[range(ns)]

            # --A28-- Save the data computed at this corner.
            self.turning_points.append(TurningPoint(lamb=lam, weights=x, free=s))

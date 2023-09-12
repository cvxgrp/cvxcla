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

from cvx.cla.claux import CLAUX
from cvx.cla.linalg.algebra import solve
from cvx.cla.types import TurningPoint


@dataclass(frozen=True)
class CLA(CLAUX):
    def __post_init__(self):
        self.logger.info("Initializing CLA")

        ns = self.mean.shape[0]

        A = np.ones((1, ns))
        b = np.array([1.0])
        m = A.shape[0]

        # --A07-- Compute basic statistics of the data.
        C = self.covariance

        # --A08-- Initialize the portfolio.
        first = self.first_turning_point()
        self.append(first)

        # --A10-- Set the P matrix.
        P = np.block([C, A.T])
        M = np.block([[C, A.T], [A, np.zeros((m, m))]])

        # --A11 -- Initialize storage for quantities # to be computed in the main loop.
        lam = np.inf

        self.logger.info("First turning point added")

        # --A12 -- The main CLA loop , which steps
        # from corner portfolio to corner portfolio.
        last = self.turning_points[-1]

        while lam > 0:
            last = self.turning_points[-1]

            blocked = ~last.free
            assert not np.all(blocked), "Not all variables can be blocked"

            # --A13-- Create the UP, DN, and IN
            # sets from the current state vector.
            UP = blocked & np.isclose(last.weights, self.upper_bounds)
            DN = blocked & np.isclose(last.weights, self.lower_bounds)

            # a variable is out if it is UP or DN
            OUT = np.logical_or(UP, DN)
            IN = ~OUT

            up = np.zeros(ns)
            up[UP] = self.upper_bounds[UP]

            dn = np.zeros(ns)
            dn[DN] = self.lower_bounds[DN]

            top = np.copy(self.mean)
            top[OUT] = 0

            _IN = np.concatenate([IN, np.ones(m, dtype=np.bool_)])

            bbb = np.zeros((ns + m, 2))
            bbb[:, 0] = np.concatenate([up + dn, b], axis=0)
            bbb[:, 1] = np.concatenate([top, np.zeros(m)])

            alpha, beta = solve(M, bbb, _IN)

            gamma = P @ alpha
            delta = P @ beta - self.mean

            # -A17-- Prepare the ratio matrix.
            L = -np.inf * np.ones([ns, 4])

            r_beta = beta[range(ns)]
            r_alpha = alpha[range(ns)]

            # --A18-- IN security possibly going UP.
            i = IN & (r_beta < -self.tol)
            L[i, 0] = (self.upper_bounds[i] - r_alpha[i]) / r_beta[i]

            # --A19-- IN security possibly going DN.
            i = IN & (r_beta > +self.tol)
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

        x = self.minimum_variance()
        self.append(TurningPoint(lamb=0, weights=x, free=last.free))

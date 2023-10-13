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
from cvx.cla.types import TurningPoint


@dataclass(frozen=True)
class CLA(CLAUX):
    def __post_init__(self):
        ns = self.mean.shape[0]
        m = self.A.shape[0]

        # Initialize the portfolio.
        first = self._first_turning_point()
        self._append(first)

        # Set the P matrix.
        P = np.block([self.covariance, self.A.T])
        M = np.block([[self.covariance, self.A.T], [self.A, np.zeros((m, m))]])

        lam = np.inf

        while lam > 0:
            last = self.turning_points[-1]

            blocked = ~last.free
            assert not np.all(blocked), "Not all variables can be blocked"

            # Create the UP, DN, and IN
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

            bbb = np.array(
                [np.block([up + dn, self.b]), np.block([top, np.zeros(m)])]
            ).T

            alpha, beta = CLA._solve(M, bbb, _IN)

            gamma = P @ alpha
            delta = P @ beta - self.mean

            # Prepare the ratio matrix.
            L = -np.inf * np.ones([ns, 4])

            r_beta = beta[range(ns)]
            r_alpha = alpha[range(ns)]

            # IN security possibly going UP.
            i = IN & (r_beta < -self.tol)
            L[i, 0] = (self.upper_bounds[i] - r_alpha[i]) / r_beta[i]

            # IN security possibly going DN.
            i = IN & (r_beta > +self.tol)
            L[i, 1] = (self.lower_bounds[i] - r_alpha[i]) / r_beta[i]

            # UP security possibly going IN.
            i = UP & (delta < -self.tol)
            L[i, 2] = -gamma[i] / delta[i]

            # DN security possibly going IN.
            i = DN & (delta > +self.tol)
            L[i, 3] = -gamma[i] / delta[i]

            # If all elements of ratio are negative,
            # we have reached the end of the efficient frontier.
            if np.max(L) < 0:
                break

            secchg, dirchg = np.unravel_index(np.argmax(L, axis=None), L.shape)

            # Set the new value of lambda_E.
            lam = L[secchg, dirchg]

            free = np.copy(last.free)
            if dirchg == 0 or dirchg == 1:
                free[secchg] = False
            else:
                free[secchg] = True

            # Compute the portfolio at this corner.
            x = r_alpha + lam * r_beta

            # Save the data computed at this corner.
            self._append(TurningPoint(lamb=lam, weights=x, free=free))

        self._append(TurningPoint(lamb=0, weights=r_alpha, free=last.free))

    @staticmethod
    def _solve(A, b, IN):
        OUT = ~IN
        n = A.shape[1]
        x = np.zeros((n, 2))

        x[OUT, :] = b[OUT, :]
        bbb = b[IN, :] - A[IN, :][:, OUT] @ x[OUT, :]

        x[IN, :] = np.linalg.inv(A[IN, :][:, IN]) @ bbb
        return x[:, 0], x[:, 1]

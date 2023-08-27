from enum import Enum

from cvx.cla.bailey.cla import CLA as Bailey
from cvx.cla.markowitz.cla import CLA as Markowitz


class Solver(Enum):
    BAILEY = Bailey
    MARKOWITZ = Markowitz

    def build(self, mean, lower_bounds, upper_bounds, covariance, tol=float(1e-5)):
        return self.value(
            mean,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            covariance=covariance,
            tol=tol,
        )

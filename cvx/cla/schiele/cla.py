from dataclasses import dataclass

from cvx.cla.aux import CLAUX


@dataclass(frozen=True)
class CLA(CLAUX):
    def __post_init__(self):
        self.logger.info("Initializing CLA")
        self.logger.info(self.mean)

        # compute a first turning point
        self.first_turning_point()

        # implement the algorithm here...

        # compute the value of the minimum variance portfolio
        self.minimum_variance()

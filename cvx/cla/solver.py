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
from enum import Enum

from cvx.cla.bailey.cla import CLA as Bailey
from cvx.cla.markowitz.cla import CLA as Markowitz
from cvx.cla.niedermayer.cla import CLA as Niedermayer


class Solver(Enum):
    BAILEY = Bailey
    MARKOWITZ = Markowitz
    NIEDERMAYER = Niedermayer

    def build(self, mean, lower_bounds, upper_bounds, covariance, tol=float(1e-5)):
        return self.value(
            mean,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            covariance=covariance,
            tol=tol,
        )

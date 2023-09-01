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

from cvx.cla.claux import CLAUX


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

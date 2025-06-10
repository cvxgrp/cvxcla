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
"""
Critical Line Algorithm (CLA) for Portfolio Optimization.

This package implements the Critical Line Algorithm introduced by Harry Markowitz
for computing the efficient frontier in portfolio optimization problems.
The algorithm efficiently computes the turning points of the efficient frontier,
which are the points where the set of assets at their bounds changes.

The main class to use is CLA, which implements the algorithm and provides
methods to compute and analyze the efficient frontier.
"""
import importlib.metadata

from .cla import CLA

__version__ = importlib.metadata.version("cvxcla")

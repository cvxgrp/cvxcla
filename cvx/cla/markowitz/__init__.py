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
Markowitz implementation of the Critical Line Algorithm.

This package provides an implementation of the Critical Line Algorithm based on
the approach described by Harry Markowitz and colleagues in their paper
"Avoiding the Downside: A Practical Review of the Critical Line Algorithm for
Mean-Semivariance Portfolio Optimization".

The implementation efficiently computes the turning points of the efficient frontier
by solving a sequence of linear systems.
"""

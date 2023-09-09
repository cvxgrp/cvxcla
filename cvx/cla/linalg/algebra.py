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
import numpy as np


def bilinear(mat, left=None, right=None):
    n, m = mat.shape
    assert n == m, "Matrix must be square"

    if left is None:
        left = np.ones(n)

    if right is None:
        right = np.ones(n)

    return left.T @ (mat @ right)


def sssolve(A, b, IN):
    OUT = ~IN
    n = A.shape[0]

    x = np.zeros(n)
    x[OUT] = b[OUT]

    bbb = b[IN] - A[IN, :][:, OUT] @ x[OUT]
    x[IN] = np.linalg.solve(A[IN, :][:, IN], bbb)

    np.isclose(A @ x - b, 0.0)

    return x

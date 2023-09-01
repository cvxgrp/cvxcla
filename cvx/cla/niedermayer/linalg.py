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

k = 3

if __name__ == "__main__":
    A = np.random.rand(k, k)
    A = A @ A.T

    a = np.atleast_2d(np.random.rand(k)).T
    alpha = np.atleast_2d(np.random.rand(1))

    print(A)
    print(a)
    print(alpha)

    X1 = np.concatenate([A, a], axis=1)
    X2 = np.concatenate([a.T, alpha], axis=1)

    print(X1)
    print(X2)

    X = np.concatenate([X1, X2], axis=0)
    print(X)

    invA = np.linalg.inv(A)

    c = invA @ a

    print(c)

    beta = 1 / (alpha - c.T @ a)

    print(beta)
    betascalar = beta[0][0]

    print(betascalar * c @ c.T)

    Y1 = np.concatenate([invA + betascalar * c @ c.T, -betascalar * c], axis=1)
    Y2 = np.concatenate([-betascalar * c.T, beta], axis=1)

    Y = np.concatenate([Y1, Y2], axis=0)
    print(Y)

    print(np.linalg.inv(X))
    print(X @ Y)

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


def replace_submatrix(mat, rows, columns, mat_replace):
    # if rows is a numpy array
    if isinstance(rows, np.ndarray):
        if rows.dtype == bool:
            rows = np.where(rows)[0]
        rows = rows.tolist()

    if isinstance(columns, np.ndarray):
        if columns.dtype == bool:
            columns = np.where(columns)[0]
        columns = columns.tolist()

    for i, row in enumerate(rows):
        mat[row, columns] = mat_replace[i, :]

    return mat


def get_submatrix(A, rows, columns):
    if isinstance(rows, np.ndarray):
        if rows.dtype == bool:
            rows = np.where(rows)[0]
        rows = rows.tolist()

    if isinstance(columns, np.ndarray):
        if columns.dtype == bool:
            columns = np.where(columns)[0]
        columns = columns.tolist()

    return A[rows, :][:, columns]


def ssssolve(A, b, IN):
    OUT = ~IN
    n = A.shape[1]
    x = np.zeros((n, 2))

    x[OUT, :] = b[OUT, :]
    bbb = b[IN, :] - A[IN, :][:, OUT] @ x[OUT, :]

    # lu, piv = scl.lu_factor(A[IN, :][:, IN])
    # lu, piv = lu_factor(A)
    for i in range(2):
        # x[IN, i] = scl.lu_solve((lu, piv), bbb[:, i])

        x[IN, i] = np.linalg.solve(A[IN, :][:, IN], bbb[:, i])

    return x[:, 0], x[:, 1]


class Solver:
    def __init__(self, C, A, IN):
        m = A.shape[0]
        self.M = np.block([[C, A.T], [A, np.zeros((m, m))]])

        self.__IN = np.concatenate([IN, np.ones(m, dtype=bool)])

        self.__inv = replace_submatrix(
            np.empty_like(self.M),
            rows=self.__IN,
            columns=self.__IN,
            mat_replace=np.linalg.inv(
                get_submatrix(self.M, rows=self.__IN, columns=self.__IN)
            ),
        )

    @property
    def active(self):
        return self.__IN

    @property
    def sub_M(self):
        return get_submatrix(self.M, rows=self.active, columns=self.active)

    @property
    def inv(self):
        return get_submatrix(self.__inv, rows=self.active, columns=self.active)
        # return self.__inv[, :][:, self.__IN]

    def solve(self, b):
        m = b.shape[1]
        x = np.zeros((self.M.shape[0], m))
        x[self.active, :] = self.inv @ b[self.active, :]

        return x[:, 0], x[:, 1]

    def free(self, new):
        """free the variable at index new"""
        if self.active[new]:
            "Variable is already free"
            return self.inv

        def enhance(A_inv, a, alpha):
            c = np.atleast_2d(A_inv @ a).T
            beta = 1 / (alpha - c.T @ a)

            return np.block([[A_inv + beta * c @ c.T, -beta * c], [-beta * c.T, beta]])

        a2 = enhance(
            A_inv=get_submatrix(self.__inv, rows=self.active, columns=self.active),
            a=self.M[self.active, new],
            alpha=self.M[new][new],
        )

        self.__inv[new, new] = a2[-1, -1]
        self.__inv[self.__IN, new] = a2[:-1, -1]
        self.__inv[new, self.__IN] = a2[-1, :-1]
        self.__inv = replace_submatrix(
            mat=self.__inv, rows=self.__IN, columns=self.__IN, mat_replace=a2[:-1, :-1]
        )
        self.__IN[new] = True
        return self.inv

    def block(self, new):
        """block the variable at index new"""
        if not self.__IN[new]:
            "Variable is already blocked"
            return self.inv

        def remove(B, b, beta):
            # see Lemma 2 in the Niedermayer paper
            b = np.atleast_2d(b).T
            return B - (b @ b.T) / beta

        self.__IN[new] = False
        B = self.inv
        # B = get_submatrix(self.__inv, rows=self.__IN, columns=self.__IN)
        b = self.__inv[self.__IN, new]
        beta = self.__inv[new, new]

        invA = remove(B=B, b=b, beta=beta)
        self.__inv = replace_submatrix(
            mat=self.__inv, rows=self.__IN, columns=self.__IN, mat_replace=invA
        )

        return self.inv

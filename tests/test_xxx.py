import numpy as np

from cvx.cla.linalg.algebra import sssolve


def test_xxx():
    M = np.array(
        [
            [0.01554826, 0.01988885, 0.01534229, 1.0],
            [0.01988885, 0.09052958, 0.01105402, 1.0],
            [0.01534229, 0.01105402, 0.0305299, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )

    Mbar = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.01534229, 0.01105402, 0.0305299, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    rhsa = np.array([0.1, 0.5, 0.0, 0.4])

    xx = np.linalg.solve(Mbar, rhsa)
    print(xx)

    rhsa = np.array([0.1, 0.5, 0.0, 1.0])
    xxx = sssolve(M, np.array(rhsa), np.array([False, False, True, True]))
    print(xxx)
    np.allclose(xx, xxx)

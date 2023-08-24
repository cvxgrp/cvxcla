import numpy as np

k = 3

if __name__ == '__main__':
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

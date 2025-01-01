import numpy as np
from loguru import logger


def f(lamb):
    denom = np.sum(np.sum(np.linalg.inv(Sigma), axis=1))
    numerator = 1 - lamb * np.sum(np.linalg.inv(Sigma) @ mu)

    gamma = numerator / denom

    w = lamb * np.linalg.inv(Sigma) @ mu + gamma * np.sum(np.linalg.inv(Sigma), axis=1)
    logger.info("********************************************************")
    logger.info(f"Lambda: {lamb}")
    logger.info(f"Gamma: {gamma}")
    logger.info(f"Weights: {w}")
    logger.info(f"Sum of weights: {np.sum(w)}")
    logger.info(f"Variance: {w.T @ Sigma @ w}")


if __name__ == "__main__":
    # https://www.cs.ubc.ca/~schmidtm/Courses/Notes/linearQuadraticGradients.pdf

    n = 3
    A = np.random.randn(n, n)
    Sigma = A @ A.T

    # ones = np.ones(n)
    mu = np.random.randn(n)

    f(lamb=+0.0)
    f(lamb=+0.1)
    f(lamb=+0.2)
    f(lamb=-0.1)
    f(lamb=-0.2)

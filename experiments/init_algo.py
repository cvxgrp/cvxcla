import numpy as np
from loguru import logger

from cvx.cla.first import init_algo

if __name__ == "__main__":
    n = 10000
    mean = 0.01 * np.random.randn(n)
    upper_bound = 0.03 * np.ones(n)

    logger.info("Hello")

    tp = init_algo(
        mean=mean, lower_bounds=np.zeros_like(upper_bound), upper_bounds=upper_bound
    )
    print(np.where(tp.free)[0])

import numpy as np
from loguru import logger

from cvx.cla.first import init_algo, init_algo_lp

if __name__ == "__main__":
    n = 10000
    mean = np.ones(n)
    upper_bound = np.ones(n)

    logger.info("Hello")

    tp = init_algo(mean=mean, lower_bounds=np.zeros_like(upper_bound), upper_bounds=upper_bound)
    print(np.where(tp.free)[0])

    tp = init_algo_lp(mean=mean, lower_bounds=np.zeros_like(upper_bound), upper_bounds=upper_bound)
    print(np.where(tp.free)[0])

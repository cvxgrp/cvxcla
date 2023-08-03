from loguru import logger
import numpy as np

from cvx.cla._first import init_algo, init_algo_lp

if __name__ == "__main__":
    # mean = np.array([1.0, 1.0, 1.0])
    # tp = init_algo(mean=mean)
    # tp_lp = init_algo_cvx(mean=mean)

    n = 10000
    mean = 0.01 * np.random.randn(n)
    upper_bound = 0.03 * np.ones(n)

    logger.info("Hello")

    tp = init_algo(mean=mean, lower_bounds=np.zeros_like(upper_bound), upper_bounds=upper_bound)
    print(np.where(tp.free)[0])
    print(tp.mean)

    logger.info("Hello")

    A_eq = np.atleast_2d(np.ones_like(mean))
    b_eq = np.array([1.0])

    tp_lp = init_algo_lp(mean=mean, upper_bounds=upper_bound, A_eq=A_eq, b_eq=b_eq)
    print(np.where(tp_lp.free)[0])
    print(tp_lp.mean)

    logger.info("Hello")
    # print(tp == tp_lp)

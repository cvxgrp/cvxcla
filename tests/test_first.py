import numpy as np
import pytest

from cvx.cla.first import init_algo, init_algo_lp
from cvx.cla.types import TurningPoint


@pytest.mark.parametrize("n", [5, 20, 100, 1000, 10000, 100000])
def test_init_algo(n):
    """Compute a first turning point"""
    mean = np.random.randn(n)
    lower_bounds = np.zeros(n)
    upper_bounds = np.random.rand(n)

    first = init_algo_lp(
        mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds, solver="ECOS"
    )

    assert np.sum(first.free) == 1
    assert np.sum(first.weights) == pytest.approx(1.0)
    assert isinstance(first, TurningPoint)

    first = init_algo(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

    assert np.sum(first.free) == 1
    assert np.sum(first.weights) == pytest.approx(1.0)
    assert isinstance(first, TurningPoint)


def test_small():
    """
    Test a first turning point
    """
    mean = np.array([1.0, 2.0, 3.0])
    lower_bound = np.array([0.0, 0.0, 0.0])
    upper_bound = np.array([0.4, 0.4, 0.4])
    tp = init_algo_lp(
        mean=mean, lower_bounds=lower_bound, upper_bounds=upper_bound, solver="ECOS"
    )

    assert np.allclose(tp.weights, [0.2, 0.4, 0.4])
    assert tp.lamb == np.inf
    assert np.allclose(tp.free, [True, False, False])

    tp = init_algo(mean=mean, lower_bounds=lower_bound, upper_bounds=upper_bound)

    assert np.allclose(tp.weights, [0.2, 0.4, 0.4])
    assert tp.lamb == np.inf
    assert np.allclose(tp.free, [True, False, False])


def test_no_fully_invested_portfolio():
    """
    Test that the algorithm fails if no fully invested portfolio
    can be constructed
    """
    mean = np.array([1.0, 2.0, 3.0])
    lower_bounds = np.array([0.0, 0.0, 0.0])
    upper_bounds = np.array([0.2, 0.2, 0.2])

    with pytest.raises(
        ValueError, match="Could not construct a fully invested portfolio"
    ):
        init_algo_lp(
            mean=mean,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            solver="ECOS",
        )

    with pytest.raises(
        ValueError, match="Could not construct a fully invested portfolio"
    ):
        init_algo(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)


def test_lb_ub_mixed():
    """
    Test that the algorithm fails if lower bounds are greater than upper bounds
    """
    upper_bounds = np.zeros(3)
    lower_bounds = np.ones(3)
    mean = np.ones(3)

    with pytest.raises(ValueError):
        init_algo_lp(
            mean=mean,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            solver="ECOS",
        )

    with pytest.raises(ValueError):
        init_algo(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

from __future__ import annotations

import numpy as np
import pytest

from cvx.cla.first import init_algo, init_algo_lp
from cvx.cla.types import TurningPoint


@pytest.mark.parametrize("n", [5, 20, 100, 1000, 10000])
def test_init_algo(n: int) -> None:
    """
    Test computing a first turning point with different problem sizes.

    This test verifies that both init_algo and init_algo_lp can compute a valid
    first turning point for problems of different sizes.

    Args:
        n: The number of assets in the portfolio
    """
    mean = np.random.randn(n)
    lower_bounds = np.zeros(n)
    upper_bounds = np.random.rand(n)

    first = init_algo_lp(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

    assert np.sum(first.free) == 1
    assert np.sum(first.weights) == pytest.approx(1.0)
    assert isinstance(first, TurningPoint)

    first = init_algo(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

    assert np.sum(first.free) == 1
    assert np.sum(first.weights) == pytest.approx(1.0)
    assert isinstance(first, TurningPoint)


def test_small() -> None:
    """
    Test computing a first turning point with a small problem.

    This test verifies that both init_algo and init_algo_lp compute the correct
    first turning point for a small problem with known solution.
    """
    mean = np.array([1.0, 2.0, 3.0])
    lower_bound = np.array([0.0, 0.0, 0.0])
    upper_bound = np.array([0.4, 0.4, 0.4])
    tp = init_algo_lp(mean=mean, lower_bounds=lower_bound, upper_bounds=upper_bound)

    assert np.allclose(tp.weights, [0.2, 0.4, 0.4])
    assert tp.lamb == np.inf
    assert np.allclose(tp.free, [True, False, False])

    tp = init_algo(mean=mean, lower_bounds=lower_bound, upper_bounds=upper_bound)

    assert np.allclose(tp.weights, [0.2, 0.4, 0.4])
    assert tp.lamb == np.inf
    assert np.allclose(tp.free, [True, False, False])


def test_no_fully_invested_portfolio() -> None:
    """
    Test that the algorithm fails when no fully invested portfolio can be constructed.

    This test verifies that both init_algo and init_algo_lp raise a ValueError
    when the upper bounds are too restrictive to allow a fully invested portfolio.
    """
    mean = np.array([1.0, 2.0, 3.0])
    lower_bounds = np.array([0.0, 0.0, 0.0])
    upper_bounds = np.array([0.2, 0.2, 0.2])

    with pytest.raises(ValueError, match="Could not construct a fully invested portfolio"):
        init_algo_lp(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

    with pytest.raises(ValueError, match="Could not construct a fully invested portfolio"):
        init_algo(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)


def test_lb_ub_mixed() -> None:
    """
    Test that the algorithm fails when lower bounds exceed upper bounds.

    This test verifies that both init_algo and init_algo_lp raise a ValueError
    when the lower bounds are greater than the upper bounds.
    """
    upper_bounds = np.zeros(3)
    lower_bounds = np.ones(3)
    mean = np.ones(3)

    with pytest.raises(ValueError):
        init_algo_lp(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

    with pytest.raises(ValueError):
        init_algo(mean=mean, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

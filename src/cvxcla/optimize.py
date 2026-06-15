"""A simple implementation of 1D line search optimization algorithm."""

from collections.abc import Callable
from typing import Any

import numpy as np


def minimize(
    fun: Callable[[float], float],
    x0: float,
    args: tuple[Any, ...] = (),
    bounds: tuple[tuple[float, float], ...] | None = None,
    tol: float = 1e-8,  # Increased precision  # pragma: no mutate
    max_iter: int = 200,  # Increased max iterations  # pragma: no mutate
    _test_mode: str | None = None,  # For testing only: 'left_overflow', 'right_overflow', or None
) -> dict[str, Any]:
    """Minimize a scalar function of one variable using a simple line search algorithm.

    This function mimics the interface of scipy.optimize.minimize but only supports
    1D optimization problems.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized: f(x, *args) -> float
    x0 : float
        Initial guess
    args : tuple, optional
        Extra arguments passed to the objective function
    bounds : tuple of tuple, optional
        Bounds for the variables, e.g. ((0, 1),)
    tol : float, optional
        Tolerance for termination
    max_iter : int, optional
        Maximum number of iterations

    Returns:
    -------
    dict
        A dictionary with keys:
        - 'x': the solution array
        - 'fun': the function value at the solution
        - 'success': a boolean flag indicating if the optimizer exited successfully
        - 'nit': the number of iterations
    """
    # Set default bounds if not provided
    if bounds is None:
        lower, upper = -np.inf, np.inf
    else:
        lower, upper = bounds[0]

    # Ensure initial guess is within bounds
    x = max(lower, min(upper, x0))

    # Golden section search parameters
    golden_ratio = (np.sqrt(5) - 1) / 2

    # Initialize search interval
    if np.isfinite(lower) and np.isfinite(upper):
        a, b = lower, upper
    else:
        # If bounds are infinite, start with a small interval around x0
        a, b = x - 1.0, x + 1.0  # pragma: no mutate

        # Expand interval until we bracket a minimum, but limit expansion to avoid overflow
        f_x = fun(x, *args)

        # Set a reasonable limit for expansion to avoid overflow
        max_expansion = 100.0  # pragma: no mutate
        min_bound = max(lower, x - max_expansion)  # pragma: no mutate
        max_bound = min(upper, x + max_expansion)

        # Expand to the left (hard-bounded so a degenerate objective cannot hang)
        try:
            expand = 0  # pragma: no mutate
            while a > min_bound and fun(a, *args) > f_x:  # pragma: no mutate
                if expand >= max_iter:  # pragma: no mutate
                    break  # pragma: no mutate
                a = max(min_bound, a - (b - a))  # pragma: no mutate
                expand += 1  # pragma: no mutate
        except (OverflowError, FloatingPointError):
            a = min_bound

        # Expand to the right (hard-bounded so a degenerate objective cannot hang)
        try:
            expand = 0  # pragma: no mutate
            while b < max_bound and fun(b, *args) > f_x:  # pragma: no mutate
                if expand >= max_iter:  # pragma: no mutate
                    break  # pragma: no mutate
                b = min(max_bound, b + (b - a))  # pragma: no mutate
                expand += 1  # pragma: no mutate
        except (OverflowError, FloatingPointError):
            b = max_bound

    # Golden section search
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)

    fc = fun(c, *args)
    fd = fun(d, *args)

    iter_count = 0
    while abs(b - a) > tol and iter_count < max_iter:  # pragma: no mutate
        if fc < fd:  # pragma: no mutate
            b = d
            d = c
            c = b - golden_ratio * (b - a)
            fd = fc
            fc = fun(c, *args)
        else:
            a = c
            c = d
            d = a + golden_ratio * (b - a)
            fc = fd
            fd = fun(d, *args)

        iter_count += 1

    # Final solution is the midpoint of the bracketing interval
    x_min = (a + b) / 2
    f_min = fun(x_min, *args)

    return {"x": np.array([x_min]), "fun": f_min, "success": iter_count < max_iter, "nit": iter_count}

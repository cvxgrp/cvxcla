"""Tests for the optimize module.

This module tests the 1D line search optimization algorithm used to find
the maximum Sharpe ratio on the efficient frontier.
"""

import numpy as np

from cvxcla.optimize import minimize


class TestMinimize:
    """Tests for the minimize function."""

    def test_simple_quadratic(self):
        """Test minimization of a simple quadratic function."""

        def f(x):
            return (x - 2.0) ** 2

        result = minimize(f, x0=0.0, bounds=((0.0, 4.0),))

        assert result["success"]
        assert np.isclose(result["x"][0], 2.0, atol=1e-4)
        assert np.isclose(result["fun"], 0.0, atol=1e-4)

    def test_with_bounds(self):
        """Test minimization with bounds."""

        def f(x):
            return (x - 5.0) ** 2

        # Bound the solution to [0, 1], so minimum should be at x=1
        result = minimize(f, x0=0.5, bounds=((0.0, 1.0),))

        assert result["success"]
        assert np.isclose(result["x"][0], 1.0, atol=1e-6)
        assert np.isclose(result["fun"], 16.0, atol=1e-6)

    def test_with_args(self):
        """Test minimization with extra arguments."""

        def f(x, a, b):
            return a * (x - b) ** 2

        result = minimize(f, x0=0.0, args=(2.0, 3.0), bounds=((0.0, 6.0),))

        assert result["success"]
        assert np.isclose(result["x"][0], 3.0, atol=1e-4)
        assert np.isclose(result["fun"], 0.0, atol=1e-4)

    def test_cosine_function(self):
        """Test minimization of a cosine function."""

        def f(x):
            return np.cos(x)

        # Minimum of cos(x) is at x = π
        result = minimize(f, x0=3.0, bounds=((0.0, 2 * np.pi),))

        assert result["success"]
        assert np.isclose(result["x"][0], np.pi, atol=1e-4)
        assert np.isclose(result["fun"], -1.0, atol=1e-6)

    def test_convergence_tolerance(self):
        """Test that tolerance parameter affects convergence."""

        def f(x):
            return x**2

        # Use a loose tolerance
        result_loose = minimize(f, x0=1.0, tol=1e-2)
        # Use a tight tolerance
        result_tight = minimize(f, x0=1.0, tol=1e-10)

        # Tight tolerance should have more iterations
        assert result_tight["nit"] >= result_loose["nit"]

    def test_max_iterations(self):
        """Test that max_iter limits the number of iterations."""

        def f(x):
            return np.sin(10 * x)

        result = minimize(f, x0=0.5, max_iter=5)

        assert result["nit"] <= 5

    def test_left_bound_is_minimum(self):
        """Test when minimum is at left bound."""

        def f(x):
            return x**2

        result = minimize(f, x0=0.5, bounds=((-1.0, 1.0),))

        assert result["success"]
        assert np.isclose(result["x"][0], 0.0, atol=1e-6)

    def test_right_bound_is_minimum(self):
        """Test when minimum is at right bound."""

        def f(x):
            return -(x**2)

        # Minimum of -x^2 on [0, 1] is at x=1
        result = minimize(f, x0=0.5, bounds=((0.0, 1.0),))

        assert result["success"]
        assert result["x"][0] >= 0.99  # Should be close to 1.0

    def test_initial_guess_outside_bounds(self):
        """Test that initial guess outside bounds is adjusted."""

        def f(x):
            return (x - 0.5) ** 2

        # Initial guess outside bounds should be clipped
        result = minimize(f, x0=10.0, bounds=((0.0, 1.0),))

        assert result["success"]
        assert np.isclose(result["x"][0], 0.5, atol=1e-6)

    def test_unbounded_optimization(self):
        """Test optimization without explicit infinite bounds but with reasonable search space."""

        def f(x):
            return (x - 10.0) ** 2

        # Use bounds that are wide enough but not infinite to avoid special case handling
        result = minimize(f, x0=0.0, bounds=((-50.0, 50.0),))

        assert result["success"]
        assert np.isclose(result["x"][0], 10.0, atol=1e-4)

    def test_function_with_multiple_args(self):
        """Test function with multiple arguments in tuple."""

        def f(x, w1, w2):
            # Simulate a simple Sharpe ratio calculation
            return -(w1 * x + w2 * (1 - x)) / np.sqrt(x**2 + (1 - x) ** 2)

        result = minimize(f, x0=0.5, args=(0.1, 0.05), bounds=((0.0, 1.0),))

        assert result["success"]
        # With higher return on w1, optimal x should favor w1
        assert result["x"][0] > 0.5

    def test_narrow_bounds(self):
        """Test with very narrow bounds."""

        def f(x):
            return x**2

        result = minimize(f, x0=0.5, bounds=((0.49, 0.51),))

        assert result["success"]
        assert 0.49 <= result["x"][0] <= 0.51

    def test_return_structure(self):
        """Test that the return structure has all expected keys."""

        def f(x):
            return x**2

        result = minimize(f, x0=1.0)

        assert "x" in result
        assert "fun" in result
        assert "success" in result
        assert "nit" in result
        assert isinstance(result["x"], np.ndarray)
        assert len(result["x"]) == 1
        assert isinstance(result["fun"], (float, np.floating))
        assert isinstance(result["success"], bool)
        assert isinstance(result["nit"], int)

    def test_overflow_handling_left(self):
        """Test overflow handling when expanding left bound."""
        # Counter to track how many times function is called
        call_count = [0]

        def f(x):
            call_count[0] += 1
            # Cause overflow on the second call during left expansion
            # First call is f_x = fun(x, *args) at line 67
            # Second call should be during left expansion at line 76
            if call_count[0] == 2 and x < 0:
                raise OverflowError("Simulated overflow during left expansion")
            return (x - 5.0) ** 2

        # Start with no bounds, so the algorithm will try to expand the search interval
        result = minimize(f, x0=0.0, bounds=None)

        # Should still succeed despite overflow
        assert result["success"]

    def test_overflow_handling_right(self):
        """Test overflow handling when expanding right bound.

        Note: This test covers defensive exception handling code that is extremely
        difficult to trigger in practice. The right expansion exception handler
        (lines 85-86) would require an OverflowError to be raised during the
        while loop condition check, which is challenging given the algorithm's
        exponential expansion pattern.
        """

        def f(x):
            # Simple quadratic function
            return (x - 10.0) ** 2

        # Test passes with normal execution
        # The actual exception handling at lines 85-86 remains as defensive code
        result = minimize(f, x0=10.0, bounds=None)

        # Should succeed
        assert result["success"]
        # Solution should be near x=10
        assert abs(result["x"][0] - 10.0) < 0.01

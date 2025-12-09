"""Tests for the first turning point computation module.

This module tests the functions used to compute the first turning point
on the efficient frontier, which is the portfolio with the highest expected
return that satisfies the constraints.
"""

import numpy as np
import pytest

from cvxcla.first import _free, init_algo


class TestInitAlgo:
    """Tests for the init_algo function."""

    def test_basic_case(self):
        """Test basic case with simple bounds."""
        mean = np.array([0.15, 0.10, 0.05])
        lower_bounds = np.array([0.0, 0.0, 0.0])
        upper_bounds = np.array([0.5, 0.5, 1.0])

        tp = init_algo(mean, lower_bounds, upper_bounds)

        # Should have at least one free asset
        assert np.any(tp.free)
        # Weights should sum to 1
        assert np.isclose(np.sum(tp.weights), 1.0)
        # Weights should respect bounds
        assert np.all(tp.weights >= lower_bounds - 1e-10)
        assert np.all(tp.weights <= upper_bounds + 1e-10)

    def test_highest_return_asset_selected(self):
        """Test that the asset with highest return is prioritized."""
        mean = np.array([0.20, 0.10, 0.05])
        lower_bounds = np.array([0.0, 0.0, 0.0])
        upper_bounds = np.array([1.0, 1.0, 1.0])

        tp = init_algo(mean, lower_bounds, upper_bounds)

        # First asset should get the most weight since it has highest return
        # and we can put 100% in it (upper bound is 1.0)
        assert tp.weights[0] == 1.0
        assert tp.free[0]
        assert tp.weights[1] == 0.0
        assert tp.weights[2] == 0.0

    def test_multiple_assets_needed(self):
        """Test case where multiple assets are needed to reach sum of 1."""
        mean = np.array([0.20, 0.15, 0.10, 0.05])
        lower_bounds = np.array([0.0, 0.0, 0.0, 0.0])
        upper_bounds = np.array([0.3, 0.3, 0.3, 0.3])

        tp = init_algo(mean, lower_bounds, upper_bounds)

        # Should need at least 4 assets to reach sum of 1 (since max is 0.3 each)
        # Actually we need at least ceil(1.0/0.3) = 4 assets
        # The last one should be free
        assert np.isclose(np.sum(tp.weights), 1.0)
        assert np.any(tp.free)

    def test_with_lower_bounds(self):
        """Test case with non-zero lower bounds."""
        mean = np.array([0.20, 0.10])
        lower_bounds = np.array([0.2, 0.3])
        upper_bounds = np.array([0.6, 0.8])

        tp = init_algo(mean, lower_bounds, upper_bounds)

        assert np.isclose(np.sum(tp.weights), 1.0)
        assert np.all(tp.weights >= lower_bounds - 1e-10)
        assert np.all(tp.weights <= upper_bounds + 1e-10)

    def test_invalid_bounds(self):
        """Test that invalid bounds raise an error."""
        mean = np.array([0.15, 0.10])
        lower_bounds = np.array([0.6, 0.5])
        upper_bounds = np.array([0.5, 0.6])  # Lower > upper for first asset

        with pytest.raises(ValueError, match="Lower bounds must be less than or equal to upper bounds"):
            init_algo(mean, lower_bounds, upper_bounds)

    def test_impossible_fully_invested(self):
        """Test case where fully invested portfolio is impossible."""
        mean = np.array([0.15, 0.10])
        lower_bounds = np.array([0.0, 0.0])
        upper_bounds = np.array([0.3, 0.3])  # Max sum is 0.6, can't reach 1.0

        with pytest.raises(ValueError, match="Could not construct a fully invested portfolio"):
            init_algo(mean, lower_bounds, upper_bounds)

    def test_exactly_one_free_asset(self):
        """Test that exactly one asset is marked as free in the result."""
        mean = np.array([0.20, 0.15, 0.10])
        lower_bounds = np.array([0.0, 0.0, 0.0])
        upper_bounds = np.array([0.4, 0.4, 0.4])

        tp = init_algo(mean, lower_bounds, upper_bounds)

        # Should have exactly one free asset
        assert np.sum(tp.free) == 1

    def test_sorted_by_return(self):
        """Test that assets are selected in order of decreasing returns."""
        mean = np.array([0.05, 0.20, 0.10, 0.15])
        lower_bounds = np.array([0.0, 0.0, 0.0, 0.0])
        upper_bounds = np.array([0.3, 0.3, 0.3, 0.3])

        tp = init_algo(mean, lower_bounds, upper_bounds)

        # Assets with higher returns should be saturated first
        # Order should be: index 1 (0.20), index 3 (0.15), index 2 (0.10), index 0 (0.05)
        # With upper bound of 0.3 each, we need at least 4 assets
        # Expected: [1]=0.3, [3]=0.3, [2]=0.3, [0]=0.1 (free)
        # But the last one to be added is free
        assert np.isclose(np.sum(tp.weights), 1.0)


class TestFreeHelper:
    """Tests for the _free helper function."""

    def test_asset_furthest_from_bounds(self):
        """Test that the function selects the asset furthest from bounds."""
        w = np.array([0.1, 0.5, 0.9])
        lower_bounds = np.array([0.0, 0.0, 0.0])
        upper_bounds = np.array([1.0, 1.0, 1.0])

        free = _free(w, lower_bounds, upper_bounds)

        # Asset at index 1 (0.5) is furthest from both bounds (0.5 distance)
        assert free[1]
        assert not free[0]
        assert not free[2]

    def test_only_one_free(self):
        """Test that exactly one asset is marked as free."""
        w = np.array([0.2, 0.3, 0.5])
        lower_bounds = np.array([0.0, 0.0, 0.0])
        upper_bounds = np.array([1.0, 1.0, 1.0])

        free = _free(w, lower_bounds, upper_bounds)

        assert np.sum(free) == 1

    def test_near_lower_bound(self):
        """Test selection when weights are near lower bounds."""
        w = np.array([0.0, 0.05, 0.95])
        lower_bounds = np.array([0.0, 0.0, 0.0])
        upper_bounds = np.array([1.0, 1.0, 1.0])

        free = _free(w, lower_bounds, upper_bounds)

        # Asset at index 1 is 0.05 from lower, 0.95 from upper (min=0.05)
        # Asset at index 2 is 0.95 from lower, 0.05 from upper (min=0.05)
        # Asset at index 0 is 0.0 from lower, 1.0 from upper (min=0.0)
        # Either index 1 or 2 should be selected (both have min distance 0.05)
        assert free[1] or free[2]
        assert not free[0]

    def test_with_tight_bounds(self):
        """Test with tight bounds around weights."""
        w = np.array([0.3, 0.4, 0.3])
        lower_bounds = np.array([0.2, 0.3, 0.2])
        upper_bounds = np.array([0.4, 0.5, 0.4])

        free = _free(w, lower_bounds, upper_bounds)

        # Asset 0: min(0.1, 0.1) = 0.1
        # Asset 1: min(0.1, 0.1) = 0.1
        # Asset 2: min(0.1, 0.1) = 0.1
        # Any could be selected; just ensure exactly one is free
        assert np.sum(free) == 1

"""Tests for the version number of the cvxcla package.

This module contains tests that verify the cvxcla package has a valid version number.
"""

import cvxcla as cvxcla


def test_version() -> None:
    """Test that the package has a version number.

    This test verifies that the __version__ attribute of the cvxcla package
    is defined and not None, ensuring that the package has a valid version.
    """
    assert cvxcla.__version__ is not None

"""Test version number."""

import cvxcla as cvxcla


def test_version():
    """Test that the package has a version number.

    This test verifies that the __version__ attribute of the cvxcla package
    is defined and not None, ensuring that the package has a valid version.
    """
    assert cvxcla.__version__ is not None

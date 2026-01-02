"""Tests for package metadata handling.

This module tests that the package can be imported even when metadata
is not available (e.g., in development mode or when not properly installed).
"""

import sys
from unittest.mock import patch


class TestPackageMetadata:
    """Tests for package metadata handling."""

    def test_import_cvxcla_succeeds(self):
        """Test that cvxcla can be imported."""
        import cvxcla

        assert cvxcla is not None
        assert hasattr(cvxcla, "CLA")

    def test_version_attribute_exists(self):
        """Test that __version__ attribute exists."""
        import cvxcla

        assert hasattr(cvxcla, "__version__")
        assert isinstance(cvxcla.__version__, str)

    def test_version_fallback_when_metadata_not_found(self):
        """Test that version falls back to 0.0.0 when metadata is not available."""
        # Remove cvxcla from sys.modules to force reimport
        if "cvxcla" in sys.modules:
            del sys.modules["cvxcla"]

        # Mock importlib.metadata.version to raise PackageNotFoundError
        import importlib.metadata

        original_version = importlib.metadata.version

        def mock_version(package_name):
            if package_name == "cvxcla":
                raise importlib.metadata.PackageNotFoundError(package_name)
            return original_version(package_name)

        with patch("importlib.metadata.version", side_effect=mock_version):
            # Import cvxcla with mocked metadata
            import cvxcla

            # Should fall back to "0.0.0" when metadata is not found
            assert cvxcla.__version__ == "0.0.0"

        # Clean up - reimport without mock for other tests
        if "cvxcla" in sys.modules:
            del sys.modules["cvxcla"]
        import cvxcla  # noqa: F401

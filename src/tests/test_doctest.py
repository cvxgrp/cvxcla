"""Tests for doctest examples in the project documentation.

This module contains tests that verify the code examples in the project's
documentation can be executed successfully using the doctest module.
"""

import doctest
from pathlib import Path


def test_doc(readme_path: Path) -> None:
    """Test that the README.md file contains valid doctests.

    This function uses the doctest module to verify that the code examples
    in the README.md file can be executed successfully.

    Args:
        readme_path: Path to the README.md file

    """
    doctest.testfile(str(readme_path), module_relative=False, verbose=True, optionflags=doctest.ELLIPSIS)

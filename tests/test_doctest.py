"""Test that all docstrings in the project can be run with doctest."""

import doctest


def test_doc(readme_path):
    """Test that the README.md file contains valid doctests.

    This function uses the doctest module to verify that the code examples
    in the README.md file can be executed successfully.

    Args:
        readme_path: Path to the README.md file

    """
    doctest.testfile(str(readme_path), module_relative=False, verbose=True, optionflags=doctest.ELLIPSIS)

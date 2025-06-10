"""Global fixtures for testing the cvxcla package.

This module contains fixtures that are used across multiple test files.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from cvx.bson import read_bson
from pandas import DataFrame


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture() -> Path:
    """Fixture that provides the path to the test resources directory.

    Returns:
        Path to the resources directory

    """
    return Path(__file__).parent / "resources"


@pytest.fixture()
def input_data(resource_dir: Path) -> SimpleNamespace:
    """Fixture that loads input data from a BSON file.

    Args:
        resource_dir: Path to the resources directory

    Returns:
        SimpleNamespace containing the input data (covariance, mean, bounds)

    """
    data = read_bson(file=resource_dir / "input_data.bson")
    return SimpleNamespace(**data)


@pytest.fixture()
def results(resource_dir: Path) -> SimpleNamespace:
    """Fixture that loads test results from a BSON file.

    Args:
        resource_dir: Path to the resources directory

    Returns:
        SimpleNamespace containing the expected test results

    """
    data = read_bson(file=resource_dir / "results.bson")
    return SimpleNamespace(**data)


@pytest.fixture()
def example(resource_dir: Path) -> DataFrame:
    """Fixture that loads example data from a CSV file.

    Args:
        resource_dir: Path to the resources directory

    Returns:
        DataFrame containing the example data

    """
    return pd.read_csv(resource_dir / "example.csv", header=0, index_col=0)


@pytest.fixture()
def example_solution(resource_dir: Path) -> DataFrame:
    """Fixture that loads example solution data from a CSV file.

    Args:
        resource_dir: Path to the resources directory

    Returns:
        DataFrame containing the example solution data

    """
    return pd.read_csv(resource_dir / "example_solution.csv", header=0, index_col=None)


@pytest.fixture()
def readme_path() -> Path:
    """Provide the path to the project's README.md file.

    This fixture searches for the README.md file by starting in the current
    directory and moving up through parent directories until it finds the file.

    Returns
    -------
    Path
        Path to the README.md file

    Raises
    ------
    FileNotFoundError
        If the README.md file cannot be found in any parent directory

    """
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        candidate = current_dir / "README.md"
        if candidate.is_file():
            return candidate
        current_dir = current_dir.parent
    raise FileNotFoundError("README.md not found in any parent directory")

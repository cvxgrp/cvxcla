"""global fixtures"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cvx.bson import read_bson


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture()
def input_data(resource_dir):
    data = read_bson(file=resource_dir / "input_data.bson")
    return SimpleNamespace(**data)


@pytest.fixture()
def results(resource_dir):
    data = read_bson(file=resource_dir / "results.bson")
    return SimpleNamespace(**data)


@pytest.fixture()
def example(resource_dir):
    return pd.read_csv(resource_dir / "example.csv", header=0, index_col=0)


@pytest.fixture()
def example_solution(resource_dir):
    return pd.read_csv(resource_dir / "example_solution.csv", header=0, index_col=None)

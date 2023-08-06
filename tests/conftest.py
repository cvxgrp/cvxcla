"""global fixtures"""
from __future__ import annotations

from pathlib import Path

import pytest
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cvx.cla.types import MATRIX, TurningPoint


@dataclass(frozen=True)
class InputData:
    mean: MATRIX
    covariance: MATRIX
    lower_bounds: MATRIX
    upper_bounds: MATRIX

@dataclass(frozen=True)
class OutputData:
    mean: MATRIX
    weights: MATRIX
    variance: MATRIX
    lamb: MATRIX

@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"

@pytest.fixture()
def input_data(resource_dir):
    path = resource_dir / "CLA_Data.csv"
    # 2) Load data, set seed
    data = np.genfromtxt(path, delimiter=",",
                         skip_header=1)  # load as numpy array
    mean = data[:1][0]
    lB = data[1:2][0]
    uB = data[2:3][0]
    covar = np.array(data[3:])

    return InputData(mean=mean, covariance=covar, lower_bounds=lB, upper_bounds=uB)

@pytest.fixture()
def results(resource_dir):
    results = pd.read_csv(resource_dir / "results.csv", header=0, index_col=None)
    return OutputData(
        mean=results["Return"].values,
        weights=results.values[:, 3:],
        variance=results["Risk"].values,
        lamb=results["Lambda"].values
    )

@pytest.fixture()
def example(resource_dir):
    return pd.read_csv(resource_dir / "example.csv", header=0, index_col=0)

@pytest.fixture()
def example_solution(resource_dir):
    return pd.read_csv(resource_dir / "example_solution.csv", header=0, index_col=None)
    #for row in frame.iterrows():
    #    yield TurningPoint(lamb=row["Lambda"], weights=row[1]["Weights"], mean=row[1]["Mean"], variance=row[1]["Variance"])

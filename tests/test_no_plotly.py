"""Test that plot() raises a helpful ImportError when plotly is not installed."""

import sys
from unittest.mock import patch

import numpy as np
import pytest

from cvxcla.types import Frontier, FrontierPoint


@pytest.fixture
def frontier():
    """Create a minimal frontier for testing."""
    mean = np.array([0.1, 0.2])
    covariance = np.array([[0.1, 0.0], [0.0, 0.1]])
    points = [FrontierPoint(weights=np.array([0.5, 0.5]))]
    return Frontier(mean=mean, covariance=covariance, frontier=points)


def test_plot_raises_import_error_without_plotly(frontier):
    """Test that plot() raises a helpful ImportError when plotly is not installed."""
    with (
        patch.dict(sys.modules, {"plotly.graph_objects": None}),
        pytest.raises(ImportError, match="pip install cvxcla\\[plot\\]"),
    ):
        frontier.plot()

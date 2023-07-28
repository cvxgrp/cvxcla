"""
types
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

MATRIX: TypeAlias = npt.NDArray[np.float64]
BLOCKMATRIX: TypeAlias = npt.NDArray[Any]

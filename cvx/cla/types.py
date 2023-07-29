"""
types
"""
from __future__ import annotations

from collections import namedtuple
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

MATRIX: TypeAlias = npt.NDArray[np.float64]
BLOCKMATRIX: TypeAlias = npt.NDArray[Any]
BOOLVECTOR: TypeAlias = npt.NDArray[np.bool_]

Next = namedtuple("Next", ["lamb", "free", "weights"])

"""Fuzz the cvxcla DenseCovariance quadratic-form operator.

``DenseCovariance`` wraps a symmetric covariance matrix and exposes ``matvec``
and ``solve_free`` (the free-block linear solve at the heart of the critical-line
algorithm). Neither should crash with an unexpected exception on adversarial
input — non-square/asymmetric matrices are rejected in ``__post_init__``, and
singular free blocks should raise a documented error (or numpy's
``LinAlgError``). This harness exercises that contract with coverage-guided
input.

Run locally:
    pip install atheris numpy scipy
    python tests/fuzz/fuzz_operators.py -atheris_runs=20000

Run in ClusterFuzzLite: this file is built by .clusterfuzzlite/build.sh.
"""

from __future__ import annotations

import contextlib
import sys

import atheris

# Pre-import the native dependencies OUTSIDE the instrumentation block so they
# load uninstrumented; only the first-party package under test is instrumented.
import numpy as np
import scipy.linalg  # noqa: F401  # pre-imported uninstrumented

with atheris.instrument_imports():
    from cvxcla.operators import DenseCovariance

_ALLOWED = (ValueError, TypeError, np.linalg.LinAlgError)


def test_one_input(data: bytes) -> None:
    """Build a DenseCovariance and exercise matvec/solve_free with fuzzed data."""
    fdp = atheris.FuzzedDataProvider(data)
    n = fdp.ConsumeIntInRange(1, 6)
    raw = np.array([fdp.ConsumeFloat() for _ in range(n * n)], dtype=np.float64).reshape(n, n)
    # Symmetrise so construction passes its symmetry check and the numeric paths
    # (matvec/solve_free) are actually exercised rather than only the guard.
    matrix = (raw + raw.T) / 2.0

    with contextlib.suppress(_ALLOWED):
        cov = DenseCovariance(matrix=matrix)
        cov.matvec(np.array([fdp.ConsumeFloat() for _ in range(n)], dtype=np.float64))
        free = np.array([fdp.ConsumeBool() for _ in range(n)], dtype=np.bool_)
        rhs = np.array([fdp.ConsumeFloat() for _ in range(int(free.sum()))], dtype=np.float64)
        cov.solve_free(free, rhs)


def main() -> None:
    """Run the Atheris fuzz loop."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

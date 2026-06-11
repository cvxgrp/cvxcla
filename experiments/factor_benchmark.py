"""Benchmark the dense vs the factor covariance backend of the CLA.

Traces the full efficient frontier of random diagonal-plus-low-rank problems
(n assets, k factors, long-only with upper bounds and a budget constraint)
once with the dense backend and once with the Woodbury-based
``FactorCovariance`` backend, and reports wall-clock times.

Usage:
    uv run python experiments/factor_benchmark.py [--sizes 500 2000 5000] [--skip-dense-above N]
"""

import argparse
import time

import numpy as np

from cvxcla import CLA, FactorCovariance


def make_problem(rng: np.random.Generator, n: int, k: int) -> tuple[FactorCovariance, dict]:
    """Create a random factor model and a long-only CLA problem of size n.

    Means and idiosyncratic variances are widely dispersed so that turning
    points are well separated; densely packed events trip a known limitation
    of the degeneracy handling that predates the factor backend.
    """
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 5.0, n)
    factor = FactorCovariance(d=d, u=u, delta=delta)
    problem = {
        "mean": rng.uniform(0.0, 1.0, n),
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }
    return factor, problem


def main() -> None:
    """Run the benchmark and print a markdown table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 2000, 5000])
    parser.add_argument("--skip-dense-above", type=int, default=10_000)
    args = parser.parse_args()

    rows = []
    for n in args.sizes:
        k = max(1, n // 20)
        rng = np.random.default_rng(0)
        factor, problem = make_problem(rng, n, k)

        start = time.perf_counter()
        cla_factor = CLA(covariance=factor, **problem)
        factor_time = time.perf_counter() - start

        if n <= args.skip_dense_above:
            dense = np.diag(factor.d) + (factor.u * factor.delta) @ factor.u.T
            start = time.perf_counter()
            cla_dense = CLA(covariance=dense, **problem)
            dense_time = time.perf_counter() - start
            if len(cla_dense) != len(cla_factor):
                msg = f"backends disagree: {len(cla_dense)} vs {len(cla_factor)} turning points"
                raise RuntimeError(msg)
            dense_cell = f"{dense_time:8.2f}"
            speedup = f"{dense_time / factor_time:6.1f}x"
        else:
            dense_cell = "skipped"
            speedup = "-"

        rows.append((n, k, len(cla_factor), dense_cell, factor_time, speedup))
        print(f"n={n:6d} k={k:4d} points={len(cla_factor):5d} dense={dense_cell} factor={factor_time:8.2f} {speedup}")

    print("\n| n | k | turning points | dense [s] | factor [s] | speedup |")
    print("|---:|---:|---:|---:|---:|---:|")
    for n, k, points, dense_cell, factor_time, speedup in rows:
        print(f"| {n} | {k} | {points} | {dense_cell.strip()} | {factor_time:.2f} | {speedup.strip()} |")


if __name__ == "__main__":
    main()

"""Score CLA frontier portfolios out-of-sample with jQuantStats.

The CLA reports what a portfolio is *expected* to do: `Frontier` exposes the
expected return, volatility and Sharpe ratio implied by the mean/covariance it
was handed. This experiment closes the loop and asks what those portfolios
actually *did*, on data the optimiser never saw.

  1. Split the committed S&P 500 return snapshot in half by time.
  2. Estimate the mean and the sample covariance on the first half only.
  3. Trace the long-only, fully-invested frontier with the CLA.
  4. Pick two portfolios off it -- the maximum-Sharpe one (exact, via
     ``Frontier.max_sharpe``) and the minimum-variance turning point -- and
     hold both fixed through the second half.
  5. Hand the three realised daily return series (the two frontier portfolios
     plus an equal-weight benchmark) to ``jquantstats`` and print its report.

The point is the contrast between the two blocks of output: the expected
figures come from ``cvxcla``, the realised ones from ``jquantstats``, and the
gap between them is estimation error, not a defect in either library.

The return matrix is the frozen snapshot at
``experiments/data/sp500_pct_returns.parquet``, so this reproduces offline;
``experiments/fetch_sp500.py`` is only for refreshing that snapshot.

Usage:
    uv run python experiments/frontier_stats.py [--assets 100]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from jquantstats import Data

from cvxcla import CLA

DATA = Path(__file__).parent / "data" / "sp500_pct_returns.parquet"
DATE_COL = "Date"
TRADING_DAYS = 260


def load(assets: int) -> pl.DataFrame:
    """Read the snapshot and keep the date column plus the first ``assets`` names."""
    frame = pl.read_parquet(DATA)
    names = [c for c in frame.columns if c != DATE_COL][:assets]
    return frame.select([DATE_COL, *names]).sort(DATE_COL)


def frontier_portfolios(train: pl.DataFrame) -> dict[str, np.ndarray]:
    """Trace the frontier on the training window and pick portfolios off it.

    Returns the maximum-Sharpe and minimum-variance weight vectors, plus an
    equal-weight benchmark, keyed by the column name they get in the report.
    """
    matrix = train.drop(DATE_COL).to_numpy()
    mean = matrix.mean(axis=0)
    covariance = np.cov(matrix, rowvar=False)
    n = mean.size

    frontier = CLA.problem(mean=mean, covariance=covariance).long_only().budget().trace().frontier
    print(f"assets                  : {n}")
    print(f"turning points          : {len(frontier)}")

    _, w_sharpe = frontier.max_sharpe
    # Frontier order runs from maximum return towards minimum variance, but read
    # the minimiser off the variance vector rather than relying on that order.
    w_minvar = frontier.weights[int(np.argmin(frontier.variance))]

    portfolios = {"max_sharpe": w_sharpe, "min_variance": w_minvar, "equal_weight": np.full(n, 1.0 / n)}
    for name, weights in portfolios.items():
        # Annualise the daily estimates so they are comparable with the
        # annualised figures jquantstats reports for the test window.
        expected = float(mean @ weights) * TRADING_DAYS
        volatility = float(np.sqrt(weights @ covariance @ weights)) * np.sqrt(TRADING_DAYS)
        print(
            f"{name:<24}: return {expected:7.2%}  volatility {volatility:6.2%}  "
            f"Sharpe {expected / volatility:5.2f}  names held {int(np.sum(weights > 1e-8)):3d}"
        )
    return portfolios


def realised(test: pl.DataFrame, portfolios: dict[str, np.ndarray]) -> Data:
    """Build the jQuantStats view of the buy-and-hold return series.

    Each portfolio is held fixed over the test window, so its realised return
    on a given day is the weighted average of that day's asset returns.
    """
    matrix = test.drop(DATE_COL).to_numpy()
    frame = pl.DataFrame({DATE_COL: test[DATE_COL], **{name: matrix @ w for name, w in portfolios.items()}})
    return Data.from_returns(frame, date_col=DATE_COL)


def main() -> None:
    """Fit on the first half of the snapshot, report the second half."""
    parser = argparse.ArgumentParser(description="Score CLA frontier portfolios with jQuantStats.")
    parser.add_argument("--assets", type=int, default=100, help="number of S&P 500 names to use (default: 100)")
    args = parser.parse_args()

    returns = load(args.assets)
    split = len(returns) // 2
    train, test = returns.head(split), returns.tail(len(returns) - split)
    for label, window in (("train", train), ("test", test)):
        first, last = window[DATE_COL][0].date(), window[DATE_COL][-1].date()
        print(f"{label + ' window':<24}: {first} -> {last} ({len(window)} days)")

    print("\n--- expected, in-sample (cvxcla) ---")
    portfolios = frontier_portfolios(train)

    print("\n--- realised, out-of-sample (jquantstats) ---")
    data = realised(test, portfolios)
    with pl.Config(tbl_rows=-1, tbl_width_chars=100, float_precision=4):
        print(data.stats.summary())


if __name__ == "__main__":
    main()

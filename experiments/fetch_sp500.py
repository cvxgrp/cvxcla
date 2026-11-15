# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lxml>=6.0.2",
#     "pandas>=2.3.3",
#     "polars>=1.20",
#     "polars-yahoo==0.2.0",
#     "pyarrow>=21.0.0",
#     "requests>=2.32.5",
# ]
# ///
"""Fetch S&P 500 constituent daily returns via Wikipedia + polars-yahoo.

Pulls the current S&P 500 ticker list from Wikipedia, downloads ~5 years of
adjusted-close prices from Yahoo Finance through polars-yahoo, cleans
thinly-traded / late-listed names, and saves the daily percentage-return matrix to
``experiments/data/sp500_pct_returns.parquet`` (one column per ticker, indexed by
``Date``, as pandas reads it).

This is a *provenance* script, not part of the reproducible path: a frozen
snapshot is already committed to the repository, and the paper's experiments
(``frontier_real.py``, ``validate_exact.py``) read that committed file directly.
Run this only to refresh the snapshot from live sources, which will produce a
*different* dataset (constituents and prices drift over time). The committed
snapshot was fetched with yfinance, before this script moved to polars-yahoo:
N=494 assets x T=1213 days (2021-07-30 to 2026-05-29),
SHA-256 b5faa5222555f28d77bad5404565297bb25bbe92b0b813f416cd2bebac79937e.

Usage:
    uv run experiments/fetch_sp500.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import polars as pl
import polars_yahoo as yh
import requests

START = "2021-06-01"
END = "2026-06-01"
MISSING_THRESHOLD = 0.05  # drop tickers missing more than 5% of trading days


def main() -> None:
    """Download, clean, and persist the S&P 500 daily-return matrix."""
    # 1. S&P 500 tickers from Wikipedia.
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (research script)"},
        timeout=30,
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"Found {len(tickers)} tickers in the S&P 500 index")

    # 2. Download adjusted-close prices (symbols Yahoo can't serve are skipped with a warning).
    bars = yh.download(tickers, start=START, end=END)
    raw = bars.pivot(on="ticker", index="date", values="adj_close").sort("date")
    names = sorted(c for c in raw.columns if c != "date")
    print(f"Raw download: {raw.height} trading days x {len(names)} tickers")

    # 3. Clean: keep well-populated tickers, forward-fill short gaps.
    keep = [c for c in names if raw[c].null_count() / raw.height <= MISSING_THRESHOLD]
    prices = raw.select("date", *keep).with_columns(pl.col(keep).forward_fill()).drop_nulls()
    print(f"After cleaning: {prices.height} days x {len(keep)} assets")

    # 4. Percentage returns.
    pct_returns = prices.with_columns(pl.col(keep).pct_change()).drop_nulls()

    # 5. Save with a pandas ``Date`` index: the experiments read it with pd.read_parquet.
    frame = pct_returns.select(*keep, pl.col("date").cast(pl.Datetime("ms")).alias("Date"))
    out = Path(__file__).parent / "data" / "sp500_pct_returns.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pandas().set_index("Date").to_parquet(out)
    print(f"Saved {out}")
    print(f"Date range: {frame['Date'][0].date()} -> {frame['Date'][-1].date()}")
    print(f"Shape: T={frame.height} days, N={len(keep)} assets")


if __name__ == "__main__":
    main()

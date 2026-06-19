"""Fetch S&P 500 constituent daily returns via Wikipedia + yfinance.

Pulls the current S&P 500 ticker list from Wikipedia, downloads ~5 years of
adjusted-close prices through yfinance, cleans thinly-traded / late-listed names,
and saves the daily percentage-return matrix to
``experiments/data/sp500_pct_returns.parquet``.

This is a *provenance* script, not part of the reproducible path: a frozen
snapshot is already committed to the repository, and the paper's experiments
(``frontier_real.py``, ``validate_exact.py``) read that committed file directly.
Run this only to refresh the snapshot from live sources, which will produce a
*different* dataset (constituents and prices drift over time). The committed
snapshot is N=494 assets x T=1213 days (2021-07-30 to 2026-05-29),
SHA-256 b5faa5222555f28d77bad5404565297bb25bbe92b0b813f416cd2bebac79937e.

Usage:
    uv run python experiments/fetch_sp500.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

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

    # 2. Download adjusted-close prices.
    raw = yf.download(
        tickers,
        auto_adjust=True,
        progress=True,
        threads=True,
        start=pd.Timestamp(START),
        end=pd.Timestamp(END),
    )["Close"]
    print(f"Raw download: {raw.shape[0]} trading days x {raw.shape[1]} tickers")

    # 3. Clean: keep well-populated tickers, forward-fill short gaps.
    missing_frac = raw.isna().mean()
    keep = missing_frac[missing_frac <= MISSING_THRESHOLD].index
    raw = raw[keep].ffill().dropna()
    print(f"After cleaning: {raw.shape[0]} days x {raw.shape[1]} assets")

    # 4. Percentage returns.
    pct_returns = raw.pct_change().dropna()

    # 5. Save.
    out = Path(__file__).parent / "data" / "sp500_pct_returns.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pct_returns.to_parquet(out)
    print(f"Saved {out}")
    print(f"Date range: {pct_returns.index[0].date()} -> {pct_returns.index[-1].date()}")
    print(f"Shape: T={pct_returns.shape[0]} days, N={pct_returns.shape[1]} assets")


if __name__ == "__main__":
    main()

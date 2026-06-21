"""Preprocessing utilities for OHLCV snapshots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_symbol_ohlcv(frame: pd.DataFrame, output_dir: Path, symbol: str) -> Path:
    """Write one symbol OHLCV frame to parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol}.parquet"
    frame.to_parquet(path, index=False)
    return path


def read_symbol_ohlcv(raw_dir: Path, symbol: str) -> pd.DataFrame:
    """Read one symbol OHLCV parquet file."""
    return pd.read_parquet(raw_dir / f"{symbol}.parquet")


def build_price_matrix(
    raw_dir: Path,
    symbols: list[str],
    min_coverage: float = 0.95,
    forward_fill_limit: int = 3,
) -> pd.DataFrame:
    """Build a close-price matrix from per-symbol OHLCV parquet files."""
    series: list[pd.Series] = []
    for symbol in symbols:
        frame = read_symbol_ohlcv(raw_dir, symbol)
        close = frame[["timestamp", "close"]].copy()
        close["timestamp"] = pd.to_datetime(close["timestamp"], utc=True)
        close = close.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        series.append(close["close"].rename(symbol))

    prices = pd.concat(series, axis=1).sort_index()
    min_non_missing = max(1, int(len(symbols) * min_coverage))
    prices = prices.dropna(thresh=min_non_missing)
    prices = prices.ffill(limit=forward_fill_limit)
    prices = prices.dropna(how="any")
    prices.index.name = "timestamp"
    return prices


def build_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Build simple returns from a price matrix."""
    returns = prices.pct_change(fill_method=None)
    returns = returns.replace([float("inf"), float("-inf")], pd.NA)
    returns = returns.dropna(how="any")
    returns.index.name = "timestamp"
    return returns


def write_processed_matrices(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    output_dir: Path,
    prefix: str = "",
) -> dict[str, Path]:
    """Write price and return matrices to parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{prefix}_" if prefix else ""
    price_path = output_dir / f"{prefix}prices_1m.parquet"
    return_path = output_dir / f"{prefix}returns_1m.parquet"
    prices.to_parquet(price_path)
    returns.to_parquet(return_path)
    return {"prices": price_path, "returns": return_path}

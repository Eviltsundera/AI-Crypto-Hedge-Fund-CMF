"""Common OHLCV schema helpers."""

from __future__ import annotations

import pandas as pd

BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

OHLCV_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
]


def normalize_binance_klines(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize raw Binance kline rows to the project OHLCV schema."""
    if frame.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    frame = frame.copy()
    if len(frame.columns) >= len(BINANCE_KLINE_COLUMNS):
        frame = frame.iloc[:, : len(BINANCE_KLINE_COLUMNS)]
        frame.columns = BINANCE_KLINE_COLUMNS

    if str(frame.iloc[0]["open_time"]).lower() == "open_time":
        frame = frame.iloc[1:].copy()

    numeric_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    timestamp_unit = "us" if frame["open_time"].dropna().median() > 10**14 else "ms"
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit=timestamp_unit, utc=True)
    frame["symbol"] = symbol
    frame = frame[OHLCV_COLUMNS]
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return frame

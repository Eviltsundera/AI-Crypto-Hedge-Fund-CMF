"""Binance public market-data ingestion."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

from ai_crypto_hedge_fund.data.schema import BINANCE_KLINE_COLUMNS, normalize_binance_klines

API_BASE_URL = "https://api.binance.com"
ARCHIVE_BASE_URL = "https://data.binance.vision/data/spot"


@dataclass(frozen=True)
class BinanceKlineRequest:
    """Parameters for a Binance kline archive request."""

    symbol: str
    interval: str
    start: pd.Timestamp
    end: pd.Timestamp


class BinancePublicDataClient:
    """Small wrapper around Binance public metadata and archive endpoints."""

    def __init__(self, timeout: int = 60, retries: int = 3) -> None:
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()

    def exchange_info(self) -> dict:
        """Return Binance exchange metadata."""
        response = self.session.get(f"{API_BASE_URL}/api/v3/exchangeInfo", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def ticker_24h(self) -> list[dict]:
        """Return 24h ticker statistics for all symbols."""
        response = self.session.get(f"{API_BASE_URL}/api/v3/ticker/24hr", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def download_archive(self, url: str, destination: Path) -> bool:
        """Download an archive if available. Return False for missing Binance archives."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0:
            return True

        temporary = destination.with_suffix(destination.suffix + ".tmp")
        for attempt in range(1, self.retries + 1):
            try:
                if temporary.exists():
                    temporary.unlink()
                with self.session.get(url, timeout=self.timeout, stream=True) as response:
                    if response.status_code == 404:
                        return False
                    response.raise_for_status()
                    with temporary.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                    temporary.replace(destination)
                return True
            except requests.RequestException:
                if attempt == self.retries:
                    raise
        return False


def utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Parse a value as a UTC timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def iter_months(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[pd.Timestamp]:
    """Yield month starts touched by the half-open interval [start, end)."""
    current = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    last = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    while current <= last:
        yield current
        current = current + pd.DateOffset(months=1)


def iter_days(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[pd.Timestamp]:
    """Yield days touched by the half-open interval [start, end)."""
    current = start.floor("D")
    while current < end:
        yield current
        current = current + pd.DateOffset(days=1)


def monthly_archive_url(symbol: str, interval: str, month: pd.Timestamp) -> str:
    """Return the Binance monthly kline archive URL."""
    return (
        f"{ARCHIVE_BASE_URL}/monthly/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{month:%Y-%m}.zip"
    )


def daily_archive_url(symbol: str, interval: str, day: pd.Timestamp) -> str:
    """Return the Binance daily kline archive URL."""
    return (
        f"{ARCHIVE_BASE_URL}/daily/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{day:%Y-%m-%d}.zip"
    )


def read_kline_zip(path: Path, symbol: str) -> pd.DataFrame:
    """Read a Binance kline archive and normalize it to OHLCV."""
    with ZipFile(path) as archive:
        csv_members = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_members:
            return normalize_binance_klines(pd.DataFrame(columns=BINANCE_KLINE_COLUMNS), symbol)
        with archive.open(csv_members[0]) as handle:
            frame = pd.read_csv(handle, header=None)
    return normalize_binance_klines(frame, symbol)


def download_symbol_klines(
    client: BinancePublicDataClient,
    request: BinanceKlineRequest,
    cache_dir: Path,
) -> pd.DataFrame:
    """Download monthly archives with daily fallback and return a filtered symbol frame."""
    start = utc_timestamp(request.start)
    end = utc_timestamp(request.end)
    frames: list[pd.DataFrame] = []

    for month in iter_months(start, end):
        monthly_url = monthly_archive_url(request.symbol, request.interval, month)
        monthly_path = cache_dir / "monthly" / request.symbol / f"{request.symbol}-{request.interval}-{month:%Y-%m}.zip"
        if client.download_archive(monthly_url, monthly_path):
            frames.append(read_kline_zip(monthly_path, request.symbol))
            continue

        month_end = min(month + pd.DateOffset(months=1), end)
        month_start = max(month, start)
        for day in iter_days(month_start, month_end):
            daily_url = daily_archive_url(request.symbol, request.interval, day)
            daily_path = (
                cache_dir
                / "daily"
                / request.symbol
                / f"{request.symbol}-{request.interval}-{day:%Y-%m-%d}.zip"
            )
            if client.download_archive(daily_url, daily_path):
                frames.append(read_kline_zip(daily_path, request.symbol))

    if not frames:
        return normalize_binance_klines(pd.DataFrame(columns=BINANCE_KLINE_COLUMNS), request.symbol)

    klines = pd.concat(frames, ignore_index=True)
    klines = klines[(klines["timestamp"] >= start) & (klines["timestamp"] < end)]
    return klines.drop_duplicates(["symbol", "timestamp"]).sort_values("timestamp").reset_index(drop=True)

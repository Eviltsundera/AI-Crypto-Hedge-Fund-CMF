#!/usr/bin/env python
"""Prepare Binance 1m data snapshots for the project.

The full 120-pair snapshot is intentionally written under data/external/ and ignored by git.
The sample snapshot is small enough to commit and is used for smoke tests and notebook demos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ai_crypto_hedge_fund.data.binance import (
    BinanceKlineRequest,
    BinancePublicDataClient,
    download_symbol_klines,
    utc_timestamp,
)
from ai_crypto_hedge_fund.data.loaders import FULL_SNAPSHOT_NAME
from ai_crypto_hedge_fund.data.preprocess import (
    build_price_matrix,
    build_return_matrix,
    write_processed_matrices,
    write_symbol_ohlcv,
)
from ai_crypto_hedge_fund.data.universe import (
    DEFAULT_SMALL_UNIVERSE,
    UniverseConfig,
    make_small_universe,
    select_spot_universe,
)
from ai_crypto_hedge_fund.paths import data_dir, external_data_dir, project_root, sample_data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start", help="UTC start date, inclusive. Defaults to end minus 365 days.")
    parser.add_argument("--end", help="UTC end date, exclusive. Defaults to yesterday 00:00 UTC.")
    parser.add_argument("--large-limit", type=int, default=120)
    parser.add_argument(
        "--symbols",
        help="Comma-separated symbols. Defaults to the small universe for sample and top liquidity for full.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def default_date_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp.now(tz="UTC").floor("D")
    start = end - pd.DateOffset(days=365)
    return start, end


def parse_date_range(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    default_start, default_end = default_date_range()
    start = utc_timestamp(args.start) if args.start else default_start
    end = utc_timestamp(args.end) if args.end else default_end
    if start >= end:
        raise ValueError("--start must be earlier than --end")
    return start, end


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(project_root())),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_universe_files(client: BinancePublicDataClient, large_limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    exchange_info = client.exchange_info()
    tickers = client.ticker_24h()
    large = select_spot_universe(
        exchange_info,
        tickers,
        config=UniverseConfig(large_limit=large_limit),
    )
    small = make_small_universe(large)

    processed_dir = data_dir() / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    large.to_csv(processed_dir / "universe_large.csv", index=False)
    small.to_csv(processed_dir / "universe_small.csv", index=False)
    return large, small


def requested_symbols(args: argparse.Namespace, large: pd.DataFrame, small: pd.DataFrame) -> list[str]:
    if args.symbols:
        return [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    if args.mode == "sample":
        symbols = small["symbol"].tolist()
        return symbols or list(DEFAULT_SMALL_UNIVERSE)
    return large["symbol"].tolist()


def snapshot_root(mode: str) -> Path:
    if mode == "sample":
        return sample_data_dir()
    return external_data_dir() / FULL_SNAPSHOT_NAME


def prepare_snapshot(args: argparse.Namespace) -> dict:
    start, end = parse_date_range(args)
    client = BinancePublicDataClient(timeout=args.timeout, retries=args.retries)
    large, small = write_universe_files(client, large_limit=args.large_limit)
    symbols = requested_symbols(args, large=large, small=small)

    root = snapshot_root(args.mode)
    raw_dir = root / "raw"
    processed_dir = root if args.mode == "sample" else root / "processed"
    cache_dir = root / "archives"

    raw_records: list[dict] = []
    row_counts: dict[str, int] = {}
    for index, symbol in enumerate(symbols, start=1):
        print(f"[{index}/{len(symbols)}] downloading {symbol} {args.interval} {start.date()}..{end.date()}")
        frame = download_symbol_klines(
            client,
            BinanceKlineRequest(symbol=symbol, interval=args.interval, start=start, end=end),
            cache_dir=cache_dir,
        )
        row_counts[symbol] = len(frame)
        raw_path = write_symbol_ohlcv(frame, output_dir=raw_dir, symbol=symbol)
        raw_records.append(file_record(raw_path))

    prices = build_price_matrix(raw_dir=raw_dir, symbols=symbols)
    returns = build_return_matrix(prices)
    processed_paths = write_processed_matrices(prices, returns, output_dir=processed_dir)

    files = {
        "universe_large": file_record(data_dir() / "processed" / "universe_large.csv"),
        "universe_small": file_record(data_dir() / "processed" / "universe_small.csv"),
        "prices": file_record(processed_paths["prices"]),
        "returns": file_record(processed_paths["returns"]),
    }
    if args.mode == "sample":
        files["raw_symbol_files"] = raw_records

    manifest = {
        "description": "Data manifest for the AI Crypto Hedge Fund MVP.",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": {
            "name": "Binance public spot market data",
            "metadata_api": "https://api.binance.com/api/v3/exchangeInfo",
            "klines_archive": "https://data.binance.vision/data/spot",
            "api_keys_required": False,
        },
        "policy": {
            "full_snapshot_storage": "Ignored local/server bundle; move with scp and publish externally if needed.",
            "git_storage": "Only universe files, manifest, code, and a compact sample snapshot are committed.",
            "external_snapshot_target": str(
                (external_data_dir() / FULL_SNAPSHOT_NAME).relative_to(project_root())
            ),
        },
        "snapshots": {
            args.mode: {
                "status": "prepared",
                "interval": args.interval,
                "start_utc": start.isoformat(),
                "end_utc_exclusive": end.isoformat(),
                "symbols": symbols,
                "symbol_count": len(symbols),
                "row_counts": row_counts,
                "price_rows": len(prices),
                "return_rows": len(returns),
                "missing_value_policy": "Require 95% timestamp coverage, forward-fill up to 3 bars, then drop incomplete rows.",
                "files": files,
            },
            "full": {
                "status": "external_bundle_expected" if args.mode == "sample" else "prepared",
                "target_interval": "1m",
                "target_symbol_count": args.large_limit,
                "target_horizon": "Last 12 months at preparation time.",
                "yandex_disk_url": None,
                "notes": [
                    "Full data is intentionally not committed to git.",
                    "After upload, add the public artifact URL and checksum here.",
                ],
            },
        },
        "known_limitations": [
            "Binance archive availability can lag for the current month; daily archives are used as fallback.",
            "Universe ranking uses latest 24h quote volume at preparation time, not historical average liquidity.",
            "Sample data is for smoke tests; final experiments should use the full external snapshot.",
        ],
    }

    manifest_path = data_dir() / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    prepare_snapshot(args)


if __name__ == "__main__":
    main()

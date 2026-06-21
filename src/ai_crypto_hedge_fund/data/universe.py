"""Universe selection for liquid Binance spot pairs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")
STABLE_OR_FIAT_BASE_ASSETS = {
    "AEUR",
    "BIDR",
    "BUSD",
    "BRL",
    "DAI",
    "EUR",
    "FDUSD",
    "GBP",
    "PAX",
    "TUSD",
    "TRY",
    "UAH",
    "USDC",
    "USDP",
    "USDS",
    "UST",
    "VAI",
}

DEFAULT_SMALL_UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]


@dataclass(frozen=True)
class UniverseConfig:
    """Configuration for selecting a liquid spot universe."""

    quote_asset: str = "USDT"
    large_limit: int = 120
    small_symbols: tuple[str, ...] = tuple(DEFAULT_SMALL_UNIVERSE)


def _is_plain_crypto_base(base_asset: str) -> bool:
    if base_asset in STABLE_OR_FIAT_BASE_ASSETS:
        return False
    return not base_asset.endswith(LEVERAGED_SUFFIXES)


def select_spot_universe(
    exchange_info: dict,
    tickers_24h: list[dict],
    config: UniverseConfig | None = None,
) -> pd.DataFrame:
    """Select liquid spot pairs from Binance exchange metadata and 24h tickers."""
    config = config or UniverseConfig()
    ticker_by_symbol = {row["symbol"]: row for row in tickers_24h}
    rows: list[dict] = []

    for symbol_info in exchange_info.get("symbols", []):
        symbol = symbol_info.get("symbol", "")
        base_asset = symbol_info.get("baseAsset", "")
        quote_asset = symbol_info.get("quoteAsset", "")
        permissions = set(symbol_info.get("permissions", []))

        if quote_asset != config.quote_asset:
            continue
        if symbol_info.get("status") != "TRADING":
            continue
        if not symbol_info.get("isSpotTradingAllowed", True) and "SPOT" not in permissions:
            continue
        if not _is_plain_crypto_base(base_asset):
            continue

        ticker = ticker_by_symbol.get(symbol, {})
        rows.append(
            {
                "symbol": symbol,
                "base_asset": base_asset,
                "quote_asset": quote_asset,
                "status": symbol_info.get("status"),
                "quote_volume_24h": float(ticker.get("quoteVolume", 0.0) or 0.0),
                "trade_count_24h": int(ticker.get("count", 0) or 0),
            }
        )

    universe = pd.DataFrame(rows)
    if universe.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "symbol",
                "base_asset",
                "quote_asset",
                "status",
                "quote_volume_24h",
                "trade_count_24h",
            ]
        )

    universe = universe.sort_values(["quote_volume_24h", "trade_count_24h"], ascending=False)
    universe = universe.head(config.large_limit).reset_index(drop=True)
    universe.insert(0, "rank", universe.index + 1)
    return universe


def make_small_universe(
    large_universe: pd.DataFrame,
    small_symbols: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Create the 5-7 coin portfolio universe from the larger candidate list."""
    small_symbols = list(small_symbols or DEFAULT_SMALL_UNIVERSE)
    small = large_universe[large_universe["symbol"].isin(small_symbols)].copy()
    order = {symbol: index for index, symbol in enumerate(small_symbols)}
    small["small_rank"] = small["symbol"].map(order) + 1
    small = small.sort_values("small_rank").reset_index(drop=True)
    columns = ["small_rank", *[column for column in small.columns if column != "small_rank"]]
    return small[columns]

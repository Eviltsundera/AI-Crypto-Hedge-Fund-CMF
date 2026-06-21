import pandas as pd

from ai_crypto_hedge_fund.data.preprocess import build_price_matrix, build_return_matrix, write_symbol_ohlcv
from ai_crypto_hedge_fund.data.schema import normalize_binance_klines
from ai_crypto_hedge_fund.data.universe import UniverseConfig, make_small_universe, select_spot_universe


def test_normalize_binance_klines() -> None:
    raw = pd.DataFrame(
        [
            [
                1_700_000_000_000,
                "100.0",
                "110.0",
                "95.0",
                "105.0",
                "42.0",
                1_700_000_059_999,
                "4410.0",
                12,
                "20.0",
                "2100.0",
                "0",
            ]
        ]
    )

    normalized = normalize_binance_klines(raw, symbol="BTCUSDT")

    assert normalized.loc[0, "symbol"] == "BTCUSDT"
    assert normalized.loc[0, "close"] == 105.0
    assert str(normalized.loc[0, "timestamp"].tzinfo) == "UTC"


def test_universe_selection_filters_and_ranks() -> None:
    exchange_info = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "isSpotTradingAllowed": True,
            },
            {
                "symbol": "ETHUSDT",
                "baseAsset": "ETH",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "isSpotTradingAllowed": True,
            },
            {
                "symbol": "USDCUSDT",
                "baseAsset": "USDC",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "isSpotTradingAllowed": True,
            },
        ]
    }
    tickers = [
        {"symbol": "BTCUSDT", "quoteVolume": "100", "count": 10},
        {"symbol": "ETHUSDT", "quoteVolume": "200", "count": 20},
        {"symbol": "USDCUSDT", "quoteVolume": "300", "count": 30},
    ]

    universe = select_spot_universe(exchange_info, tickers, UniverseConfig(large_limit=2))

    assert universe["symbol"].tolist() == ["ETHUSDT", "BTCUSDT"]
    assert make_small_universe(universe, ["BTCUSDT"])["symbol"].tolist() == ["BTCUSDT"]


def test_price_and_return_matrix_builders(tmp_path) -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
    for symbol, offset in {"BTCUSDT": 100.0, "ETHUSDT": 10.0}.items():
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": symbol,
                "open": [offset, offset + 1, offset + 2, offset + 3],
                "high": [offset, offset + 1, offset + 2, offset + 3],
                "low": [offset, offset + 1, offset + 2, offset + 3],
                "close": [offset, offset + 1, offset + 2, offset + 3],
                "volume": [1, 1, 1, 1],
                "quote_volume": [offset, offset, offset, offset],
                "trade_count": [1, 1, 1, 1],
            }
        )
        write_symbol_ohlcv(frame, tmp_path, symbol)

    prices = build_price_matrix(tmp_path, ["BTCUSDT", "ETHUSDT"])
    returns = build_return_matrix(prices)

    assert prices.shape == (4, 2)
    assert returns.shape == (3, 2)
    assert list(prices.columns) == ["BTCUSDT", "ETHUSDT"]

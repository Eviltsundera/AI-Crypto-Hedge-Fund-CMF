import pandas as pd

from ai_crypto_hedge_fund.strategies import (
    BaselineConfig,
    moving_average_crossover_signals,
    run_single_asset_baseline,
)


def test_moving_average_crossover_signal_is_long_only_after_slow_window() -> None:
    prices = pd.Series(
        [1.0, 1.0, 1.0, 2.0, 3.0],
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
        name="BTCUSDT",
    )

    signals = moving_average_crossover_signals(prices, fast_window=2, slow_window=3)

    assert signals.iloc[:2].tolist() == [0.0, 0.0]
    assert signals.iloc[-1] == 1.0


def test_single_asset_baseline_uses_only_test_period_returns() -> None:
    index = pd.date_range("2024-01-01", periods=12, freq="D")
    prices = pd.DataFrame({"BTCUSDT": [100, 101, 102, 103, 104, 105, 104, 106, 107, 109, 108, 110]}, index=index)
    config = BaselineConfig(
        symbol="BTCUSDT",
        fast_window=2,
        slow_window=3,
        test_size=0.25,
        transaction_cost_bps=0.0,
        periods_per_year=252,
    )

    result = run_single_asset_baseline(prices, config=config)

    assert result.split_timestamp == index[9]
    assert result.buy_and_hold.returns.index.min() == index[10]
    assert result.moving_average.returns.index.min() == index[10]
    assert result.equity_curves.columns.tolist() == ["buy_and_hold", "moving_average_crossover"]
    assert {"total_return", "sharpe_ratio", "max_drawdown", "hit_rate"}.issubset(
        result.metrics_payload("sample")["strategies"]["moving_average_crossover"]
    )

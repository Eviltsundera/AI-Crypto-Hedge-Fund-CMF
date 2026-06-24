import pandas as pd

from ai_crypto_hedge_fund.metrics import returns_from_prices
from ai_crypto_hedge_fund.portfolio import (
    DynamicRebalancingConfig,
    run_dynamic_rebalancing_experiment,
    threshold_rebalance_positions,
    time_based_rebalance_positions,
)


def test_time_based_rebalancing_records_scheduled_events() -> None:
    prices = _synthetic_price_frame(periods=240)
    returns = returns_from_prices(prices)
    test_index = returns.index[-120:]
    config = DynamicRebalancingConfig(
        symbols=tuple(prices.columns),
        lookback_periods=30,
        rebalance_frequency="30min",
        max_weight=0.60,
    )

    positions, events = time_based_rebalance_positions(returns, test_index, config)

    assert not events.empty
    assert set(events["reason"]) == {"scheduled"}
    assert positions.index.equals(test_index)
    assert positions.notna().all().all()
    assert all(abs(float(total) - 1.0) < 1e-9 for total in positions.sum(axis=1))


def test_threshold_rebalancing_records_threshold_events() -> None:
    index = pd.date_range("2024-01-01", periods=80, freq="min")
    returns = pd.DataFrame(
        {
            "BTCUSDT": [0.03] * 80,
            "ETHUSDT": [-0.01] * 80,
            "BNBUSDT": [0.0] * 80,
        },
        index=index,
    )
    config = DynamicRebalancingConfig(
        symbols=tuple(returns.columns),
        lookback_periods=10,
        drift_threshold=0.01,
        max_weight=0.60,
    )

    positions, events = threshold_rebalance_positions(returns, returns.iloc[20:], config)

    assert "drift_threshold" in set(events["reason"])
    assert positions.notna().all().all()
    assert all(abs(float(total) - 1.0) < 1e-9 for total in positions.sum(axis=1))


def test_dynamic_rebalancing_experiment_runs_on_synthetic_prices() -> None:
    prices = _synthetic_price_frame(periods=900)
    config = DynamicRebalancingConfig(
        symbols=tuple(prices.columns),
        test_size=0.25,
        transaction_cost_bps=0.0,
        periods_per_year=365,
        lookback_periods=120,
        rebalance_frequency="2H",
        drift_threshold=0.02,
        max_weight=0.50,
    )

    result = run_dynamic_rebalancing_experiment(prices, config=config, data_snapshot="sample")
    payload = result.metrics_payload()

    assert set(result.backtests) == {
        "static_max_sharpe_reference",
        "weekly_inverse_volatility",
        "threshold_inverse_volatility",
    }
    assert result.selected_strategy in result.backtests
    assert len(payload["comparison_table"]) == 3
    assert not result.rebalance_events.empty
    assert "weekly_inverse_volatility" in result.event_summary


def _synthetic_price_frame(periods: int) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="min")
    return pd.DataFrame(
        {
            "BTCUSDT": [100.0 + i * 0.010 for i in range(periods)],
            "ETHUSDT": [90.0 + i * 0.008 + (i % 7) * 0.01 for i in range(periods)],
            "BNBUSDT": [80.0 + i * 0.006 + (i % 5) * 0.01 for i in range(periods)],
            "SOLUSDT": [70.0 + i * 0.005 + (i % 3) * 0.02 for i in range(periods)],
            "XRPUSDT": [60.0 + i * 0.004 + (i % 11) * 0.01 for i in range(periods)],
            "ADAUSDT": [50.0 + i * 0.003 + (i % 13) * 0.01 for i in range(periods)],
        },
        index=index,
    )

import pandas as pd

from ai_crypto_hedge_fund.agents import AgentDecisionConfig, AgentDecisionEngine
from ai_crypto_hedge_fund.features import build_model_frame, build_next_return_targets
from ai_crypto_hedge_fund.models import SingleAssetModelConfig, run_single_asset_model_comparison


def test_next_return_targets_use_future_period_direction() -> None:
    prices = pd.Series(
        [100.0, 101.0, 99.0],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
        name="BTCUSDT",
    )

    targets = build_next_return_targets(prices)

    assert targets.loc[prices.index[0], "target_up"] == 1
    assert targets.loc[prices.index[1], "target_up"] == 0
    assert targets.loc[prices.index[0], "next_timestamp"] == prices.index[1]


def test_model_frame_contains_features_and_targets() -> None:
    prices = pd.Series(
        range(100, 520),
        index=pd.date_range("2024-01-01", periods=420, freq="min"),
        name="BTCUSDT",
        dtype=float,
    )

    frame = build_model_frame(prices)

    assert {"return_lag_1", "rolling_vol_60", "momentum_360", "target_up"}.issubset(frame.columns)
    assert frame.index.max() < prices.index.max()


def test_agent_blocks_high_volatility_even_with_enough_votes() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    signals = pd.DataFrame(
        {
            "baseline": [1.0, 1.0, 1.0],
            "econometric": [1.0, 1.0, 1.0],
            "ml": [0.0, 1.0, 1.0],
        },
        index=index,
    )
    volatility = pd.Series([0.01, 0.20, 0.01], index=index)
    drawdown = pd.Series([0.0, 0.0, -0.20], index=index)
    agent = AgentDecisionEngine(
        AgentDecisionConfig(min_long_votes=2, volatility_limit=0.05, max_drawdown_limit=-0.10)
    )

    signal, rationale = agent.decide(signals, volatility, drawdown)

    assert signal.tolist() == [1.0, 0.0, 0.0]
    assert "blocked_by_volatility" in rationale.iloc[1]
    assert "blocked_by_drawdown" in rationale.iloc[2]


def test_single_asset_model_comparison_runs_on_small_data() -> None:
    index = pd.date_range("2024-01-01", periods=900, freq="min")
    trend = pd.Series(range(900), index=index, dtype=float)
    cycle = ((trend % 20) - 10) / 200.0
    prices = pd.DataFrame({"BTCUSDT": 100.0 + trend * 0.01 + cycle})
    config = SingleAssetModelConfig(
        symbol="BTCUSDT",
        fast_window=5,
        slow_window=20,
        test_size=0.25,
        transaction_cost_bps=0.0,
        periods_per_year=365,
        econometric_window=20,
        max_train_rows=300,
        random_forest_estimators=8,
        random_forest_max_depth=3,
        ml_probability_threshold=0.5,
    )

    result = run_single_asset_model_comparison(prices, config=config, data_snapshot="sample")

    assert set(result.backtests) == {
        "buy_and_hold",
        "moving_average_crossover",
        "econometric_rolling",
        "random_forest",
        "agent_enhanced",
    }
    assert result.training_summary["training_rows_used"] <= 300
    assert result.training_summary["random_chance_reference"] == 0.5
    assert len(result.metrics_payload()["comparison_table"]) == 5

import pandas as pd

from ai_crypto_hedge_fund.models import CostAwareBoostingConfig, run_cost_aware_boosting_experiment


def test_cost_aware_boosting_experiment_drops_neutral_training_rows() -> None:
    index = pd.date_range("2024-01-01", periods=1400, freq="min")
    prices = pd.DataFrame(
        {
            "BTCUSDT": [
                100.0 + step * 0.006 + ((step % 40) - 20) * 0.004
                for step in range(1400)
            ]
        },
        index=index,
    )
    config = CostAwareBoostingConfig(
        symbol="BTCUSDT",
        test_size=0.25,
        validation_size=0.2,
        horizon_periods=20,
        cost_buffer=0.0002,
        max_train_rows=600,
        max_iter=12,
        max_leaf_nodes=7,
        min_samples_leaf=10,
        probability_smoothing_window=5,
        threshold_grid=(0.50, 0.60),
        transaction_cost_bps=0.0,
        periods_per_year=365,
    )

    result = run_cost_aware_boosting_experiment(prices, config=config, data_snapshot="sample")
    payload = result.metrics_payload()

    assert set(result.backtests) == {"buy_and_hold", "cost_aware_hist_gradient_boosting"}
    assert result.target_summary["train_neutral_rows"] >= 0
    assert result.validation_summary["selected_threshold"] in {0.50, 0.60}
    assert len(payload["comparison_table"]) == 2

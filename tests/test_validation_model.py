import pandas as pd

from ai_crypto_hedge_fund.models import ValidationModelConfig, run_validation_model_experiment


def test_validation_model_experiment_uses_validation_threshold() -> None:
    index = pd.date_range("2024-01-01", periods=1200, freq="min")
    prices = pd.DataFrame(
        {
            "BTCUSDT": [
                100.0 + step * 0.01 + ((step % 30) - 15) * 0.002
                for step in range(1200)
            ]
        },
        index=index,
    )
    config = ValidationModelConfig(
        symbol="BTCUSDT",
        test_size=0.25,
        validation_size=0.2,
        horizon_periods=20,
        max_train_rows=500,
        random_forest_estimators=8,
        random_forest_max_depth=3,
        min_samples_leaf=10,
        probability_smoothing_window=5,
        threshold_grid=(0.50, 0.55),
        transaction_cost_bps=0.0,
        periods_per_year=365,
    )

    result = run_validation_model_experiment(prices, config=config, data_snapshot="sample")
    payload = result.metrics_payload()

    assert set(result.backtests) == {"buy_and_hold", "validation_tuned_rf_60m"}
    assert result.validation_summary["selected_threshold"] in {0.50, 0.55}
    assert len(result.validation_summary["threshold_grid_results"]) == 2
    assert len(payload["comparison_table"]) == 2

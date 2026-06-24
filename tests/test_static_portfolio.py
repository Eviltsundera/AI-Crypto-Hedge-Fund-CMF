import pandas as pd

from ai_crypto_hedge_fund.portfolio import (
    StaticPortfolioConfig,
    equal_weight_weights,
    inverse_volatility_weights,
    max_sharpe_weights,
    run_static_portfolio_experiment,
)


def test_static_weight_methods_are_long_only_and_fully_invested() -> None:
    index = pd.date_range("2024-01-01", periods=120, freq="min")
    returns = pd.DataFrame(
        {
            "BTCUSDT": [0.001] * 120,
            "ETHUSDT": [0.001 if i % 2 == 0 else -0.001 for i in range(120)],
            "SOLUSDT": [0.002 if i % 3 == 0 else -0.001 for i in range(120)],
        },
        index=index,
    )

    for weights in [
        equal_weight_weights(returns.columns),
        inverse_volatility_weights(returns),
        max_sharpe_weights(returns, max_weight=0.60),
    ]:
        assert abs(float(weights.sum()) - 1.0) < 1e-9
        assert float(weights.min()) >= 0.0
        assert float(weights.max()) <= 0.60 + 1e-9


def test_inverse_volatility_allocates_less_to_higher_volatility_asset() -> None:
    index = pd.date_range("2024-01-01", periods=120, freq="min")
    returns = pd.DataFrame(
        {
            "LOWVOL": [0.0002 if i % 2 == 0 else -0.0002 for i in range(120)],
            "HIGHVOL": [0.002 if i % 2 == 0 else -0.002 for i in range(120)],
        },
        index=index,
    )

    weights = inverse_volatility_weights(returns)

    assert weights["LOWVOL"] > weights["HIGHVOL"]


def test_static_portfolio_experiment_runs_on_synthetic_prices() -> None:
    index = pd.date_range("2024-01-01", periods=900, freq="min")
    prices = pd.DataFrame(
        {
            "BTCUSDT": [100.0 + i * 0.010 for i in range(900)],
            "ETHUSDT": [90.0 + i * 0.008 + (i % 7) * 0.01 for i in range(900)],
            "BNBUSDT": [80.0 + i * 0.006 + (i % 5) * 0.01 for i in range(900)],
            "SOLUSDT": [70.0 + i * 0.005 + (i % 3) * 0.02 for i in range(900)],
            "XRPUSDT": [60.0 + i * 0.004 + (i % 11) * 0.01 for i in range(900)],
            "ADAUSDT": [50.0 + i * 0.003 + (i % 13) * 0.01 for i in range(900)],
        },
        index=index,
    )
    config = StaticPortfolioConfig(test_size=0.25, transaction_cost_bps=0.0, periods_per_year=365)

    result = run_static_portfolio_experiment(prices, config=config, data_snapshot="sample")
    payload = result.metrics_payload()

    assert set(result.backtests) == {"equal_weight", "inverse_volatility", "max_sharpe_constrained"}
    assert result.split_timestamp == prices.index[int(len(prices) * 0.75)]
    assert result.selected_method in result.backtests
    assert len(payload["comparison_table"]) == 3
    assert all(summary["effective_assets"] >= 1.0 for summary in result.diversification.values())

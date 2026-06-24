import pandas as pd

from ai_crypto_hedge_fund.metrics import returns_from_prices
from ai_crypto_hedge_fund.portfolio import (
    LargeUniverseConfig,
    rank_large_universe_assets,
    run_large_universe_experiment,
    sparse_rebalance_positions,
)


def test_large_universe_ranking_prefers_stronger_momentum() -> None:
    prices = _synthetic_large_prices(symbol_count=8, periods=300)
    returns = returns_from_prices(prices, dropna=False).fillna(0.0)
    config = LargeUniverseConfig(
        symbols=tuple(prices.columns),
        lookback_periods=120,
        momentum_periods=60,
        min_universe_size=8,
        active_count=4,
        max_weight=0.40,
    )

    ranking = rank_large_universe_assets(returns, returns.index[-1], config, risk_adjusted=False)

    assert ranking.index[0] == "ASSET07USDT"
    assert ranking.iloc[0]["score"] >= ranking.iloc[-1]["score"]


def test_sparse_rebalance_positions_are_capped_and_sparse() -> None:
    prices = _synthetic_large_prices(symbol_count=10, periods=500)
    returns = returns_from_prices(prices, dropna=False).fillna(0.0)
    config = LargeUniverseConfig(
        symbols=tuple(prices.columns),
        lookback_periods=120,
        momentum_periods=60,
        active_count=4,
        max_weight=0.30,
        min_universe_size=10,
    )

    positions, selected = sparse_rebalance_positions(
        returns,
        returns.index[-180:],
        config,
        strategy_name="top_momentum_weekly",
        risk_adjusted=True,
    )

    assert not selected.empty
    assert positions.max().max() <= 0.30 + 1e-9
    assert positions.astype(bool).sum(axis=1).max() <= 4
    assert all(float(total) <= 1.0 + 1e-9 for total in positions.sum(axis=1))


def test_large_universe_experiment_runs_on_synthetic_prices() -> None:
    prices = _synthetic_large_prices(symbol_count=12, periods=900)
    config = LargeUniverseConfig(
        symbols=tuple(prices.columns),
        test_size=0.25,
        transaction_cost_bps=0.0,
        periods_per_year=365,
        lookback_periods=180,
        momentum_periods=60,
        rebalance_frequency="2h",
        active_count=5,
        max_weight=0.30,
        min_universe_size=12,
    )

    result = run_large_universe_experiment(prices, config=config, data_snapshot="sample")
    payload = result.metrics_payload()

    assert result.universe_diagnostics["meets_100_pair_requirement"] is True
    assert set(result.backtests) == {
        "large_universe_equal_weight",
        "top_momentum_weekly",
        "risk_adjusted_momentum_weekly",
    }
    assert result.selected_strategy in result.backtests
    assert len(payload["comparison_table"]) == 3
    assert not result.selected_assets.empty


def _synthetic_large_prices(symbol_count: int, periods: int) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="min")
    data = {}
    for asset_id in range(symbol_count):
        symbol = f"ASSET{asset_id:02d}USDT"
        trend = 0.001 + asset_id * 0.0002
        cycle = (asset_id + 1) * 0.001
        data[symbol] = [
            100.0 + trend * step + cycle * (step % 17)
            for step in range(periods)
        ]
    return pd.DataFrame(data, index=index)

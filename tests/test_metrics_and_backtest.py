import math

import pandas as pd

from ai_crypto_hedge_fund.backtest import (
    backtest_prices,
    backtest_returns,
    signals_to_positions,
    time_train_test_split,
)
from ai_crypto_hedge_fund.metrics import (
    annualized_volatility,
    max_drawdown,
    performance_summary,
    returns_from_prices,
    sharpe_ratio,
    total_return,
)


def test_time_train_test_split_preserves_time_order() -> None:
    data = pd.Series(range(10), index=pd.date_range("2024-01-01", periods=10, freq="D"))

    train, test = time_train_test_split(data, test_size=3)

    assert train.index.max() < test.index.min()
    assert train.tolist() == list(range(7))
    assert test.tolist() == [7, 8, 9]


def test_returns_total_return_and_drawdown() -> None:
    prices = pd.Series(
        [100.0, 110.0, 99.0, 118.8],
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
        name="BTCUSDT",
    )

    returns = returns_from_prices(prices)
    equity = (1.0 + returns).cumprod()

    assert returns.round(6).tolist() == [0.1, -0.1, 0.2]
    assert total_return(returns) == 0.18800000000000017
    assert round(max_drawdown(equity), 6) == -0.1


def test_sharpe_and_volatility_behavior() -> None:
    flat_returns = pd.Series([0.01, 0.01, 0.01])
    mixed_returns = pd.Series([0.02, -0.01, 0.02, -0.01])

    assert math.isinf(sharpe_ratio(flat_returns, periods_per_year=252))
    assert annualized_volatility(flat_returns, periods_per_year=252) == 0.0
    assert sharpe_ratio(mixed_returns, periods_per_year=252) != 0.0


def test_signals_are_lagged_before_becoming_positions() -> None:
    signals = pd.Series(
        [1.0, 0.0, -1.0],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    long_only_positions = signals_to_positions(signals, allow_short=False, lag=1)
    long_short_positions = signals_to_positions(signals, allow_short=True, lag=1)

    assert long_only_positions.tolist() == [0.0, 1.0, 0.0]
    assert long_short_positions.tolist() == [0.0, 1.0, 0.0]


def test_backtest_position_alignment_and_costs() -> None:
    returns = pd.Series(
        [-0.5, 1.0],
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
        name="BTCUSDT",
    )
    signals = pd.Series([1.0, 0.0], index=returns.index)

    result = backtest_returns(returns, signals=signals, transaction_cost_bps=10)

    assert result.gross_returns.tolist() == [0.0, 1.0]
    assert result.turnover.tolist() == [0.0, 1.0]
    assert result.costs.tolist() == [0.0, 0.001]
    assert result.returns.tolist() == [0.0, 0.999]
    assert round(result.equity_curve.iloc[-1], 3) == 1.999


def test_backtest_prices_accepts_multicolumn_static_weights() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame(
        {
            "BTCUSDT": [100.0, 110.0, 121.0],
            "ETHUSDT": [100.0, 90.0, 99.0],
        },
        index=index,
    )
    positions = pd.DataFrame(
        {"BTCUSDT": [0.5, 0.5], "ETHUSDT": [0.5, 0.5]},
        index=index[1:],
    )

    result = backtest_prices(prices, positions=positions, periods_per_year=252)

    assert result.returns.round(6).tolist() == [0.0, 0.1]
    assert result.metrics["total_return"] == 0.10000000000000009


def test_performance_summary_includes_turnover_and_benchmark() -> None:
    returns = pd.Series([0.01, -0.02, 0.03])
    positions = pd.Series([0.0, 1.0, 1.0])
    benchmark = pd.Series([0.01, 0.0, 0.01])

    summary = performance_summary(
        returns,
        positions=positions,
        benchmark_returns=benchmark,
        periods_per_year=252,
    )

    assert {"total_return", "max_drawdown", "sharpe_ratio", "turnover"}.issubset(summary)
    assert summary["turnover"] == 1.0
    assert "benchmark_total_return" in summary

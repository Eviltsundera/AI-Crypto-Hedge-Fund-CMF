"""Backtesting primitives for trading strategies and portfolios."""

from ai_crypto_hedge_fund.backtest import (
    BacktestResult,
    backtest_prices,
    backtest_returns,
    signals_to_positions,
    time_train_test_split,
)

__all__ = [
    "BacktestResult",
    "backtest_prices",
    "backtest_returns",
    "signals_to_positions",
    "time_train_test_split",
]

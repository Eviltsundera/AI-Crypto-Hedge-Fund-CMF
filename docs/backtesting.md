# Backtesting and Metrics Core

The project uses shared utilities for all strategy and portfolio experiments. Keep metric calculations in these modules instead of duplicating formulas in notebooks or task-specific code.

## Modules

- `ai_crypto_hedge_fund.metrics`
  - `returns_from_prices`
  - `equity_curve`
  - `performance_summary`
  - individual metrics such as `total_return`, `sharpe_ratio`, `max_drawdown`, `hit_rate`, `calculate_turnover`
- `ai_crypto_hedge_fund.backtest`
  - `time_train_test_split`
  - `signals_to_positions`
  - `backtest_returns`
  - `backtest_prices`

## Position Alignment

`signals_to_positions` applies a one-period lag by default. A signal observed at timestamp `t` becomes a tradable position from the next return period, which avoids lookahead bias in deterministic strategy tests.

`backtest_returns` assumes passed positions are already executable for the return timestamp. If signals are passed instead, they are converted to lagged positions inside the backtest.

## Transaction Costs

Backtests support a simple basis-point cost model:

```python
from ai_crypto_hedge_fund.backtest import backtest_returns

result = backtest_returns(
    returns,
    signals=signals,
    transaction_cost_bps=10,
)
```

Costs are calculated from absolute position turnover per period and subtracted from gross strategy returns.

## Time Split

Use `time_train_test_split` for train/test separation. It sorts by index and never shuffles observations.

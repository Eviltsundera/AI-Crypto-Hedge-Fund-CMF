# Single-Asset Baseline Strategy

The baseline experiment evaluates `BTCUSDT` with two transparent policies:

- buy-and-hold benchmark;
- dual moving average crossover, long when the fast average is above the slow average and flat otherwise.

The implementation lives in `ai_crypto_hedge_fund.strategies.baseline` and uses the shared Task 04 backtesting and metrics utilities.

## Reproduce

Generate metrics and the equity/drawdown figure:

```bash
uv run python scripts/run_baseline_strategy.py --snapshot auto
```

`--snapshot auto` uses the ignored full data bundle when it exists under `data/external/`; otherwise it falls back to the committed sample snapshot.

Outputs:

- `reports/metrics/baseline_metrics.json`
- `reports/figures/baseline_equity_curve.png`

## Bias Controls

- Train/test split is time-ordered and never shuffled.
- Strategy signals are shifted by one period before execution.
- Metrics are calculated on the out-of-sample test period.
- Transaction costs are charged from absolute position turnover.

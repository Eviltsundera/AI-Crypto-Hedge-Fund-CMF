# Single-Asset Econometric, ML, and Agent Strategy

The single-asset model comparison extends the BTCUSDT baseline with three additional policies:

- econometric rolling-mean forecast;
- RandomForest direction classifier on lagged and rolling features, with smoothed probability signals;
- deterministic agent-enhanced strategy that combines baseline, econometric, and ML votes with volatility and drawdown risk rules.

All strategies use the shared `ai_crypto_hedge_fund.backtest` and `ai_crypto_hedge_fund.metrics` modules.

## Features and Target

Features are calculated from information available at the current timestamp:

- lagged 1-minute returns;
- rolling return means and volatilities;
- momentum over multiple windows;
- moving-average distance.

The ML target is the next 1-minute return direction: `1` when the next return is positive, otherwise `0`.

## Reproduce

Generate the model comparison report:

```bash
uv run python scripts/run_single_asset_models.py --snapshot auto
```

Outputs:

- `reports/metrics/single_asset_model_comparison.json`
- `reports/figures/single_asset_model_comparison.png`

## Full-Snapshot Result Summary

The committed report was generated on `beleriand` with the full ignored BTCUSDT data snapshot.

Out-of-sample test period: `2026-03-05 03:36:00+00:00` to `2026-06-20 23:59:00+00:00`.

| Strategy | Total Return | Sharpe | Max Drawdown | Turnover |
|---|---:|---:|---:|---:|
| Buy-and-hold | -0.1126 | -0.7224 | -0.2859 | 1 |
| Moving-average crossover | -0.4040 | -5.8061 | -0.4091 | 657 |
| Econometric rolling | -0.4125 | -7.2193 | -0.4207 | 773 |
| RandomForest | -0.0613 | -4.8854 | -0.0684 | 140 |
| Agent-enhanced | -0.1321 | -4.3925 | -0.1399 | 194 |

The RandomForest policy reduced drawdown and improved total return versus buy-and-hold in this down test period, while the deterministic agent reduced drawdown versus buy-and-hold but lagged it on total return.

## Bias Controls

- Training labels are restricted to rows whose next-return timestamp is before the test split.
- The ML model uses only training data.
- Strategy signals are shifted by one period before execution.
- Econometric and ML signals use short confirmation/smoothing windows to reduce 1-minute microstructure churn.
- Comparison metrics are calculated on the same out-of-sample test period as the Task 05 baseline.
- The report includes a 50% random-chance reference, majority-class reference, test direction accuracy, and ROC AUC where defined.

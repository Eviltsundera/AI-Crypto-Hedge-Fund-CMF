# Static Multi-Asset Portfolio

The static portfolio experiment compares long-only allocations across a small liquid crypto universe:

- BTCUSDT;
- ETHUSDT;
- BNBUSDT;
- SOLUSDT;
- XRPUSDT;
- ADAUSDT.

Weights are fitted on the training period only and evaluated on the out-of-sample test period. The experiment uses the shared `ai_crypto_hedge_fund.backtest` and `ai_crypto_hedge_fund.metrics` layers, including transaction-cost accounting.

## Allocation Methods

- Equal weight: each asset receives the same allocation.
- Inverse volatility: assets with lower training-period volatility receive larger weights.
- Constrained max-Sharpe: long-only SLSQP optimization with `sum(weights)=1`, no shorting, and a maximum per-asset weight cap. If optimization fails, the method falls back to capped inverse-volatility weights.

The selected portfolio is the method with the highest out-of-sample Sharpe ratio in the report.

## Reproduce

Generate the static portfolio report:

```bash
uv run python scripts/run_static_portfolio.py --snapshot auto
```

Outputs:

- `reports/metrics/static_portfolio_metrics.json`
- `reports/figures/static_portfolio_equity_curve.png`
- `reports/figures/static_portfolio_weights.png`

## Full-Snapshot Result Summary

The committed report was generated on `beleriand` with the full ignored 1-minute data snapshot.

Train period: `2025-06-26 12:00:00+00:00` to `2026-03-05 03:35:00+00:00`.

Out-of-sample test period: `2026-03-05 03:36:00+00:00` to `2026-06-20 23:59:00+00:00`.

Selection criterion: highest out-of-sample Sharpe ratio. The selected method is `max_sharpe_constrained`.

| Strategy | Total Return | Annualized Volatility | Sharpe | Max Drawdown | Effective Assets | Max Weight |
|---|---:|---:|---:|---:|---:|---:|
| Equal weight | -0.1941 | 0.4884 | -1.2514 | -0.3285 | 6.00 | 0.1667 |
| Inverse volatility | -0.1728 | 0.4697 | -1.1319 | -0.3108 | 5.56 | 0.2562 |
| Max-Sharpe constrained | -0.1497 | 0.4593 | -0.9649 | -0.2931 | 2.99 | 0.3500 |

Selected max-Sharpe constrained weights:

| Asset | Weight |
|---|---:|
| ETHUSDT | 0.3500 |
| BNBUSDT | 0.3500 |
| XRPUSDT | 0.3000 |
| BTCUSDT | 0.0000 |
| SOLUSDT | 0.0000 |
| ADAUSDT | 0.0000 |

## Trading Interpretation

The portfolio is a research allocation, not an execution system. In a real trading setup, the static weights would need exchange-specific order sizing, liquidity checks, slippage assumptions, fee tiers, custody limits, and periodic re-estimation. The MVP assumes immediate fills at minute close prices and applies a simple transaction-cost model on the initial allocation only.

## Bias Controls

- Portfolio weights are estimated only from the train split.
- Test metrics are calculated on unseen data after the split timestamp.
- All methods are evaluated on the same test returns.
- The optimizer is long-only and fully invested, with a per-asset cap to avoid excessive concentration.
- Sample mode remains available for smoke tests, but the committed full report is generated on `beleriand` from the ignored 12-month minute data bundle.

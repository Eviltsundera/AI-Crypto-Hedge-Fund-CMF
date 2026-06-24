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

## Trading Interpretation

The portfolio is a research allocation, not an execution system. In a real trading setup, the static weights would need exchange-specific order sizing, liquidity checks, slippage assumptions, fee tiers, custody limits, and periodic re-estimation. The MVP assumes immediate fills at minute close prices and applies a simple transaction-cost model on the initial allocation only.

## Bias Controls

- Portfolio weights are estimated only from the train split.
- Test metrics are calculated on unseen data after the split timestamp.
- All methods are evaluated on the same test returns.
- The optimizer is long-only and fully invested, with a per-asset cap to avoid excessive concentration.
- Sample mode remains available for smoke tests, but the committed full report is generated on `beleriand` from the ignored 12-month minute data bundle.

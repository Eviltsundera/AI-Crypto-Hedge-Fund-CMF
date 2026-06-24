# Dynamic Portfolio Rebalancing

The dynamic rebalancing experiment extends the six-asset static portfolio from Task 07 with two adaptive long-only policies:

- weekly inverse-volatility rebalancing;
- threshold inverse-volatility rebalancing when simulated passive holding drift exceeds 2 percentage points.

Both policies estimate new target weights from trailing data only. They are evaluated with the shared `ai_crypto_hedge_fund.backtest` and `ai_crypto_hedge_fund.metrics` layers, including transaction costs from position turnover.

## Reproduce

Generate the dynamic rebalancing report:

```bash
uv run python scripts/run_dynamic_rebalancing.py --snapshot auto
```

Outputs:

- `reports/metrics/rebalancing_metrics.json`
- `reports/metrics/rebalance_events.csv`
- `reports/figures/rebalancing_comparison.png`

## Full-Snapshot Result Summary

The committed report was generated on `beleriand` with the full ignored 1-minute data snapshot.

Train period: `2025-06-26 12:00:00+00:00` to `2026-03-05 03:35:00+00:00`.

Out-of-sample test period: `2026-03-05 03:36:00+00:00` to `2026-06-20 23:59:00+00:00`.

Selection criterion: highest out-of-sample Sharpe ratio. The selected strategy is `static_max_sharpe_reference`.

| Strategy | Total Return | Annualized Volatility | Sharpe | Max Drawdown | Turnover | Events |
|---|---:|---:|---:|---:|---:|---:|
| Static max-Sharpe reference | -0.1497 | 0.4593 | -0.9649 | -0.2931 | 1.0000 | 0 |
| Weekly inverse volatility | -0.1801 | 0.4725 | -1.1857 | -0.3120 | 1.3230 | 16 |
| Threshold inverse volatility | -0.1755 | 0.4736 | -1.1427 | -0.3102 | 1.2210 | 5 |

The threshold strategy traded less often than weekly rebalancing and performed better than weekly rebalancing, but the static max-Sharpe reference still had the best Sharpe ratio and drawdown on this test period.

## Bias Controls

- Dynamic weights use only trailing history available before the effective rebalance timestamp.
- Threshold drift is simulated from passive holding weights; trades are recorded only when the drift threshold is crossed.
- Strategy selection is based on out-of-sample Sharpe, not only return.
- Every strategy is long-only, fully invested, and cost-aware.
- Rebalance events are persisted in `reports/metrics/rebalance_events.csv` for auditability.

## Trading Interpretation

Dynamic rebalancing is useful operationally because it creates explicit trade dates, turnover, and event logs. In this experiment, simple inverse-volatility rebalancing did not improve the six-asset portfolio versus the static max-Sharpe allocation. A production system should combine rebalancing with signal ranking, liquidity checks, and portfolio risk limits before increasing turnover.

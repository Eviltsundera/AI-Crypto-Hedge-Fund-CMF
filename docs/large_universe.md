# Large-Universe Portfolio

The large-universe experiment scales the portfolio layer from the six-asset Task 08 setup to the full 120-pair Binance spot snapshot. It intentionally avoids dense covariance optimization across all assets and uses sparse, auditable allocation rules instead.

## Method

The experiment compares three strategies:

- large-universe equal weight across all available pairs;
- weekly top-momentum allocation to the top 20 assets;
- weekly risk-adjusted momentum allocation to the top 20 assets.

The sparse strategies use only trailing information at each rebalance timestamp:

- 30-day trailing lookback;
- 7-day momentum window;
- minimum data coverage filter;
- 8% maximum asset weight;
- gross exposure scaling when trailing portfolio volatility exceeds the target.

## Reproduce

Generate the large-universe report:

```bash
uv run python scripts/run_large_universe.py --snapshot auto
```

Outputs:

- `reports/metrics/large_universe_metrics.json`
- `reports/metrics/large_universe_selected_assets.csv`
- `reports/figures/large_universe_equity_curve.png`

## Full-Snapshot Result Summary

The committed report was generated on `beleriand` with the ignored full 1-minute data bundle.

Universe size: 120 pairs.

Train period: `2025-06-26 12:00:00+00:00` to `2026-03-05 03:35:00+00:00`.

Out-of-sample test period: `2026-03-05 03:36:00+00:00` to `2026-06-20 23:59:00+00:00`.

Selection criterion: highest out-of-sample Sharpe ratio. The selected strategy is `large_universe_equal_weight`.

| Strategy | Total Return | Annualized Volatility | Sharpe | Max Drawdown | Turnover | Events | Active Assets |
|---|---:|---:|---:|---:|---:|---:|---:|
| Large-universe equal weight | 0.1439 | 0.4868 | 1.1781 | -0.2932 | 1.0000 | 0 | 120 |
| Top momentum weekly | 0.0149 | 0.6572 | 0.4050 | -0.2980 | 24.2638 | 16 | 20 |
| Risk-adjusted momentum weekly | -0.0378 | 0.6417 | 0.1176 | -0.2783 | 24.6452 | 16 | 20 |

The sparse momentum policies create an auditable selection layer, but in this test period they did not beat the broad equal-weight universe after turnover and costs.

## Data Bundle Checksums

Full processed bundle on `beleriand`:

```text
/home/gajnanovda/AI-Crypto-Hedge-Fund-CMF/data/external/binance_spot_1m_120_12mo/processed
```

Processed file checksums:

| File | Size Bytes | SHA256 |
|---|---:|---|
| `prices_1m.parquet` | 95258376 | `027a8a0ba6841e04e39513f8262a5d451bcb27369bb07a308e6059d617ef9dff` |
| `returns_1m.parquet` | 193142203 | `2c4b8ed154f959fe7cb598524b214fcb423acab61ef3ae3169a4c717af6fe491` |

The public URL for this ignored bundle is intentionally deferred to the final external data publication task.

## Monitoring and Fail-Safes

Production monitoring should track:

- universe coverage and stale symbols;
- realized turnover, costs, and slippage versus assumptions;
- realized volatility versus target volatility;
- concentration, max weight, and active asset count;
- drift between desired and actual exchange balances;
- drawdown and stop-rule activation.

Operational fail-safes should include max order size, max portfolio turnover per rebalance, exchange outage handling, stale-data checks, order rejection handling, and a manual kill switch. Model quality should be reviewed over rolling windows with benchmark-relative return, drawdown, hit rate, active-set stability, and transaction-cost attribution.

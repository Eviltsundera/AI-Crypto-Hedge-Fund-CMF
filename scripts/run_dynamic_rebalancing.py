"""Generate dynamic portfolio rebalancing comparison reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ai_crypto_hedge_fund.data.loaders import load_price_matrix, snapshot_processed_dir
from ai_crypto_hedge_fund.paths import reports_dir
from ai_crypto_hedge_fund.portfolio import (
    DEFAULT_STATIC_UNIVERSE,
    DynamicRebalancingConfig,
    run_dynamic_rebalancing_experiment,
)


def main() -> None:
    args = parse_args()
    snapshot = resolve_snapshot(args.snapshot)
    symbols = tuple(args.symbols.split(",") if args.symbols else DEFAULT_STATIC_UNIVERSE)
    prices = load_price_matrix(snapshot=snapshot)
    config = DynamicRebalancingConfig(
        symbols=symbols,
        test_size=args.test_size,
        transaction_cost_bps=args.transaction_cost_bps,
        lookback_periods=args.lookback_periods,
        rebalance_frequency=args.rebalance_frequency,
        drift_threshold=args.drift_threshold,
        max_weight=args.max_weight,
        selection_metric=args.selection_metric,
    )
    result = run_dynamic_rebalancing_experiment(prices, config=config, data_snapshot=snapshot)

    metrics_path = Path(args.metrics_output)
    events_path = Path(args.events_output)
    figure_path = Path(args.figure_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    payload = result.metrics_payload()
    payload["notes"] = [
        "Dynamic targets use only trailing history available before the effective rebalance timestamp.",
        "Threshold drift is simulated as passive holding drift and trades are recorded only when the threshold is crossed.",
        "All strategies are long-only, fully invested, and evaluated by the shared Task 04 cost accounting.",
    ]
    metrics_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    result.rebalance_events.to_csv(events_path, index=False)
    write_rebalancing_figure(result.equity_curves, result.drawdowns, figure_path)

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote events to {events_path}")
    print(f"Wrote figure to {figure_path}")
    print(json.dumps(_json_safe(payload["comparison_table"]), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    default_metrics = reports_dir() / "metrics" / "rebalancing_metrics.json"
    default_events = reports_dir() / "metrics" / "rebalance_events.csv"
    default_figure = reports_dir() / "figures" / "rebalancing_comparison.png"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", choices=["auto", "sample", "full"], default="auto")
    parser.add_argument("--symbols", default=",".join(DEFAULT_STATIC_UNIVERSE))
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--lookback-periods", type=int, default=30 * 24 * 60)
    parser.add_argument("--rebalance-frequency", default="7d")
    parser.add_argument("--drift-threshold", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--selection-metric", default="sharpe_ratio")
    parser.add_argument("--metrics-output", default=str(default_metrics))
    parser.add_argument("--events-output", default=str(default_events))
    parser.add_argument("--figure-output", default=str(default_figure))
    return parser.parse_args()


def resolve_snapshot(snapshot: str) -> str:
    if snapshot != "auto":
        return snapshot
    full_prices = snapshot_processed_dir(snapshot="full") / "prices_1m.parquet"
    return "full" if full_prices.exists() else "sample"


def write_rebalancing_figure(equity_curves, drawdowns, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    equity_curves.plot(ax=axes[0], linewidth=1.2)
    axes[0].set_title("Dynamic portfolio rebalancing comparison")
    axes[0].set_ylabel("Equity")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    drawdowns.plot(ax=axes[1], linewidth=1.0)
    axes[1].set_title("Drawdowns")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Timestamp")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    main()

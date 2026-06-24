"""Generate single-asset econometric, ML, and agent comparison reports."""

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
from ai_crypto_hedge_fund.models import SingleAssetModelConfig, run_single_asset_model_comparison
from ai_crypto_hedge_fund.paths import reports_dir


def main() -> None:
    args = parse_args()
    snapshot = resolve_snapshot(args.snapshot)
    prices = load_price_matrix(snapshot=snapshot)
    config = SingleAssetModelConfig(
        symbol=args.symbol,
        fast_window=args.fast_window,
        slow_window=args.slow_window,
        test_size=args.test_size,
        transaction_cost_bps=args.transaction_cost_bps,
        max_train_rows=args.max_train_rows,
        random_forest_estimators=args.random_forest_estimators,
        random_forest_max_depth=args.random_forest_max_depth,
        ml_probability_threshold=args.ml_probability_threshold,
        ml_probability_smoothing_window=args.ml_probability_smoothing_window,
    )
    result = run_single_asset_model_comparison(prices, config=config, data_snapshot=snapshot)

    metrics_path = Path(args.metrics_output)
    figure_path = Path(args.figure_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    payload = result.metrics_payload()
    payload["notes"] = [
        "All strategies use the Task 04 backtest and metrics layer.",
        "ML labels are next-period return directions, and training rows are strictly before the test split.",
        "The deterministic agent combines baseline, econometric, and ML votes with volatility/drawdown risk rules.",
    ]
    metrics_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    write_model_comparison_figure(result.equity_curves, result.drawdowns, figure_path)

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote figure to {figure_path}")
    print(json.dumps(_json_safe(payload["comparison_table"]), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    default_metrics = reports_dir() / "metrics" / "single_asset_model_comparison.json"
    default_figure = reports_dir() / "figures" / "single_asset_model_comparison.png"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", choices=["auto", "sample", "full"], default="auto")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--fast-window", type=int, default=60)
    parser.add_argument("--slow-window", type=int, default=360)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--max-train-rows", type=int, default=120_000)
    parser.add_argument("--random-forest-estimators", type=int, default=80)
    parser.add_argument("--random-forest-max-depth", type=int, default=7)
    parser.add_argument("--ml-probability-threshold", type=float, default=0.54)
    parser.add_argument("--ml-probability-smoothing-window", type=int, default=30)
    parser.add_argument("--metrics-output", default=str(default_metrics))
    parser.add_argument("--figure-output", default=str(default_figure))
    return parser.parse_args()


def resolve_snapshot(snapshot: str) -> str:
    if snapshot != "auto":
        return snapshot
    full_prices = snapshot_processed_dir(snapshot="full") / "prices_1m.parquet"
    return "full" if full_prices.exists() else "sample"


def write_model_comparison_figure(equity_curves, drawdowns, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    equity_curves.plot(ax=axes[0], linewidth=1.2)
    axes[0].set_title("BTCUSDT single-asset model comparison")
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

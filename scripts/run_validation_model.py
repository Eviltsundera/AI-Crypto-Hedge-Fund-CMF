"""Generate validation-tuned single-asset ML report."""

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
from ai_crypto_hedge_fund.models import ValidationModelConfig, run_validation_model_experiment
from ai_crypto_hedge_fund.paths import reports_dir


def main() -> None:
    args = parse_args()
    snapshot = resolve_snapshot(args.snapshot)
    prices = load_price_matrix(snapshot=snapshot)
    config = ValidationModelConfig(
        symbol=args.symbol,
        horizon_periods=args.horizon_periods,
        max_train_rows=args.max_train_rows,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    result = run_validation_model_experiment(prices, config=config, data_snapshot=snapshot)

    metrics_path = Path(args.metrics_output)
    figure_path = Path(args.figure_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    payload = result.metrics_payload()
    payload["notes"] = [
        "This is an improvement experiment, not a replacement for the original Task 06 comparison.",
        "The probability threshold is selected on validation data inside the train period.",
        "The final test period is evaluated once after threshold selection.",
    ]
    metrics_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    write_validation_figure(result.equity_curves, result.drawdowns, figure_path)

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote figure to {figure_path}")
    print(json.dumps(_json_safe(payload["comparison_table"]), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    default_metrics = reports_dir() / "metrics" / "single_asset_validation_model.json"
    default_figure = reports_dir() / "figures" / "single_asset_validation_model.png"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", choices=["auto", "sample", "full"], default="auto")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--horizon-periods", type=int, default=60)
    parser.add_argument("--max-train-rows", type=int, default=160_000)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--metrics-output", default=str(default_metrics))
    parser.add_argument("--figure-output", default=str(default_figure))
    return parser.parse_args()


def resolve_snapshot(snapshot: str) -> str:
    if snapshot != "auto":
        return snapshot
    full_prices = snapshot_processed_dir(snapshot="full") / "prices_1m.parquet"
    return "full" if full_prices.exists() else "sample"


def write_validation_figure(equity_curves, drawdowns, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    equity_curves.plot(ax=axes[0], linewidth=1.2)
    axes[0].set_title("Validation-tuned BTCUSDT ML experiment")
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

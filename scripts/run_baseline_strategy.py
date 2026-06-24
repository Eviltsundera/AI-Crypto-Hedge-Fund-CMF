"""Generate single-asset baseline strategy reports."""

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
from ai_crypto_hedge_fund.strategies import BaselineConfig, run_single_asset_baseline


def main() -> None:
    args = parse_args()
    snapshot = resolve_snapshot(args.snapshot)
    prices = load_price_matrix(snapshot=snapshot)
    config = BaselineConfig(
        symbol=args.symbol,
        fast_window=args.fast_window,
        slow_window=args.slow_window,
        test_size=args.test_size,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    result = run_single_asset_baseline(prices, config=config)

    metrics_path = Path(args.metrics_output)
    figure_path = Path(args.figure_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    payload = result.metrics_payload(data_snapshot=snapshot)
    payload["notes"] = [
        "Signals are shifted by one period inside the backtest to avoid lookahead bias.",
        "Both strategies are evaluated only on the out-of-sample test period.",
        "Transaction costs are charged from absolute position turnover.",
    ]
    metrics_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    write_baseline_figure(result.equity_curves, result.drawdowns, figure_path)

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote figure to {figure_path}")
    print(json.dumps(_json_safe(payload["comparison"]), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    default_metrics = reports_dir() / "metrics" / "baseline_metrics.json"
    default_figure = reports_dir() / "figures" / "baseline_equity_curve.png"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", choices=["auto", "sample", "full"], default="auto")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--fast-window", type=int, default=60)
    parser.add_argument("--slow-window", type=int, default=360)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--metrics-output", default=str(default_metrics))
    parser.add_argument("--figure-output", default=str(default_figure))
    return parser.parse_args()


def resolve_snapshot(snapshot: str) -> str:
    if snapshot != "auto":
        return snapshot
    full_prices = snapshot_processed_dir(snapshot="full") / "prices_1m.parquet"
    return "full" if full_prices.exists() else "sample"


def write_baseline_figure(equity_curves, drawdowns, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    equity_curves.plot(ax=axes[0], linewidth=1.4)
    axes[0].set_title("BTCUSDT baseline equity curves")
    axes[0].set_ylabel("Equity")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    drawdowns.plot(ax=axes[1], linewidth=1.2)
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

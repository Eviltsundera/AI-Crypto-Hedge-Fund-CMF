"""Generate static multi-asset portfolio comparison reports."""

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
from ai_crypto_hedge_fund.portfolio import DEFAULT_STATIC_UNIVERSE, StaticPortfolioConfig
from ai_crypto_hedge_fund.portfolio import run_static_portfolio_experiment


def main() -> None:
    args = parse_args()
    snapshot = resolve_snapshot(args.snapshot)
    symbols = tuple(args.symbols.split(",") if args.symbols else DEFAULT_STATIC_UNIVERSE)
    prices = load_price_matrix(snapshot=snapshot)
    config = StaticPortfolioConfig(
        symbols=symbols,
        test_size=args.test_size,
        transaction_cost_bps=args.transaction_cost_bps,
        max_weight=args.max_weight,
        selection_metric=args.selection_metric,
    )
    result = run_static_portfolio_experiment(prices, config=config, data_snapshot=snapshot)

    metrics_path = Path(args.metrics_output)
    equity_path = Path(args.equity_figure_output)
    weights_path = Path(args.weights_figure_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    equity_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    payload = result.metrics_payload()
    payload["notes"] = [
        "Weights are fitted only on the train period and evaluated on the out-of-sample test period.",
        "All portfolio methods are long-only, fully invested, and use the shared Task 04 cost accounting.",
        "Static portfolios pay transaction cost on initial allocation only; no intratest rebalancing is applied.",
    ]
    metrics_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    write_equity_curve_figure(result.equity_curves, result.drawdowns, equity_path)
    write_weights_figure(result.weights, result.selected_method, weights_path)

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote equity figure to {equity_path}")
    print(f"Wrote weights figure to {weights_path}")
    print(json.dumps(_json_safe(payload["comparison_table"]), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    default_metrics = reports_dir() / "metrics" / "static_portfolio_metrics.json"
    default_equity = reports_dir() / "figures" / "static_portfolio_equity_curve.png"
    default_weights = reports_dir() / "figures" / "static_portfolio_weights.png"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", choices=["auto", "sample", "full"], default="auto")
    parser.add_argument("--symbols", default=",".join(DEFAULT_STATIC_UNIVERSE))
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--selection-metric", default="sharpe_ratio")
    parser.add_argument("--metrics-output", default=str(default_metrics))
    parser.add_argument("--equity-figure-output", default=str(default_equity))
    parser.add_argument("--weights-figure-output", default=str(default_weights))
    return parser.parse_args()


def resolve_snapshot(snapshot: str) -> str:
    if snapshot != "auto":
        return snapshot
    full_prices = snapshot_processed_dir(snapshot="full") / "prices_1m.parquet"
    return "full" if full_prices.exists() else "sample"


def write_equity_curve_figure(equity_curves, drawdowns, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    equity_curves.plot(ax=axes[0], linewidth=1.2)
    axes[0].set_title("Static portfolio comparison")
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


def write_weights_figure(weights, selected_method: str, output_path: Path) -> None:
    ax = weights.T.plot(kind="bar", figsize=(12, 5), width=0.78)
    ax.set_title(f"Static portfolio weights; selected: {selected_method}")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Asset")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=160)
    plt.close(ax.figure)


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

"""Generate large-universe sparse allocation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ai_crypto_hedge_fund.data.loaders import load_price_matrix, snapshot_processed_dir
from ai_crypto_hedge_fund.paths import reports_dir
from ai_crypto_hedge_fund.portfolio import LargeUniverseConfig, run_large_universe_experiment


def main() -> None:
    args = parse_args()
    snapshot = resolve_snapshot(args.snapshot)
    prices = load_price_matrix(snapshot=snapshot)
    symbols = tuple(args.symbols.split(",")) if args.symbols else tuple(prices.columns)
    min_universe_size = args.min_universe_size
    if snapshot == "sample" and len(symbols) < min_universe_size:
        min_universe_size = len(symbols)
    config = LargeUniverseConfig(
        symbols=symbols,
        test_size=args.test_size,
        transaction_cost_bps=args.transaction_cost_bps,
        lookback_periods=args.lookback_periods,
        momentum_periods=args.momentum_periods,
        rebalance_frequency=args.rebalance_frequency,
        active_count=args.active_count,
        max_weight=args.max_weight,
        min_coverage=args.min_coverage,
        min_universe_size=min_universe_size,
        target_annualized_volatility=args.target_annualized_volatility,
        min_gross_exposure=args.min_gross_exposure,
        selection_metric=args.selection_metric,
    )
    result = run_large_universe_experiment(prices, config=config, data_snapshot=snapshot)

    metrics_path = Path(args.metrics_output)
    selected_assets_path = Path(args.selected_assets_output)
    figure_path = Path(args.figure_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    selected_assets_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    payload = result.metrics_payload()
    payload["data_files"] = data_file_payload(snapshot)
    payload["notes"] = [
        "Large-universe allocation uses sparse top-N ranking instead of dense covariance optimization.",
        "Ranking uses trailing momentum and volatility only; no future test-period information is used.",
        "Risk control caps per-asset weight and scales gross exposure down when trailing portfolio volatility exceeds the target.",
    ]
    metrics_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    result.selected_assets.to_csv(selected_assets_path, index=False)
    write_large_universe_figure(result.equity_curves, result.drawdowns, figure_path)

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote selected assets to {selected_assets_path}")
    print(f"Wrote figure to {figure_path}")
    print(json.dumps(_json_safe(payload["comparison_table"]), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    default_metrics = reports_dir() / "metrics" / "large_universe_metrics.json"
    default_selected_assets = reports_dir() / "metrics" / "large_universe_selected_assets.csv"
    default_figure = reports_dir() / "figures" / "large_universe_equity_curve.png"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", choices=["auto", "sample", "full"], default="auto")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--lookback-periods", type=int, default=30 * 24 * 60)
    parser.add_argument("--momentum-periods", type=int, default=7 * 24 * 60)
    parser.add_argument("--rebalance-frequency", default="7D")
    parser.add_argument("--active-count", type=int, default=20)
    parser.add_argument("--max-weight", type=float, default=0.08)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--min-universe-size", type=int, default=100)
    parser.add_argument("--target-annualized-volatility", type=float, default=0.60)
    parser.add_argument("--min-gross-exposure", type=float, default=0.25)
    parser.add_argument("--selection-metric", default="sharpe_ratio")
    parser.add_argument("--metrics-output", default=str(default_metrics))
    parser.add_argument("--selected-assets-output", default=str(default_selected_assets))
    parser.add_argument("--figure-output", default=str(default_figure))
    return parser.parse_args()


def resolve_snapshot(snapshot: str) -> str:
    if snapshot != "auto":
        return snapshot
    full_prices = snapshot_processed_dir(snapshot="full") / "prices_1m.parquet"
    return "full" if full_prices.exists() else "sample"


def data_file_payload(snapshot: str) -> dict[str, Any]:
    processed_dir = snapshot_processed_dir(snapshot=snapshot)
    payload: dict[str, Any] = {"processed_dir": str(processed_dir)}
    for name in ["prices_1m.parquet", "returns_1m.parquet"]:
        path = processed_dir / name
        if path.exists():
            payload[name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return payload


def write_large_universe_figure(equity_curves, drawdowns, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    equity_curves.plot(ax=axes[0], linewidth=1.2)
    axes[0].set_title("Large-universe sparse allocation comparison")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

"""Dynamic long-only portfolio rebalancing policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ai_crypto_hedge_fund.backtest import BacktestResult, backtest_returns, time_train_test_split
from ai_crypto_hedge_fund.metrics import CRYPTO_MINUTE_PERIODS_PER_YEAR, drawdown_series, returns_from_prices
from ai_crypto_hedge_fund.portfolio.static import (
    DEFAULT_STATIC_UNIVERSE,
    diversification_summary,
    inverse_volatility_weights,
    max_sharpe_weights,
    select_best_method,
    select_static_universe,
    static_positions,
)


@dataclass(frozen=True)
class DynamicRebalancingConfig:
    """Configuration for dynamic portfolio rebalancing comparisons."""

    symbols: tuple[str, ...] = DEFAULT_STATIC_UNIVERSE
    test_size: float = 0.3
    transaction_cost_bps: float = 5.0
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR
    lookback_periods: int = 30 * 24 * 60
    rebalance_frequency: str = "7d"
    drift_threshold: float = 0.05
    max_weight: float = 0.35
    min_weight: float = 0.0
    covariance_ridge: float = 1e-10
    selection_metric: str = "sharpe_ratio"


@dataclass(frozen=True)
class DynamicRebalancingResult:
    """Result bundle for dynamic rebalancing experiments."""

    config: DynamicRebalancingConfig
    data_snapshot: str
    universe: list[str]
    split_timestamp: pd.Timestamp
    train_period: dict[str, Any]
    test_period: dict[str, Any]
    static_reference_weights: pd.Series
    backtests: dict[str, BacktestResult]
    equity_curves: pd.DataFrame
    drawdowns: pd.DataFrame
    rebalance_events: pd.DataFrame
    event_summary: dict[str, dict[str, float]]
    diversification: dict[str, dict[str, float]]
    selected_strategy: str
    selection_criterion: str

    def metrics_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable report payload."""
        strategies = {name: result.metrics for name, result in self.backtests.items()}
        return {
            "data_snapshot": self.data_snapshot,
            "universe": self.universe,
            "config": asdict(self.config),
            "split_timestamp_utc": self.split_timestamp.isoformat(),
            "train_period": self.train_period,
            "test_period": self.test_period,
            "static_reference_weights": self.static_reference_weights.to_dict(),
            "selected_strategy": self.selected_strategy,
            "selection_criterion": self.selection_criterion,
            "event_summary": self.event_summary,
            "diversification": self.diversification,
            "strategies": strategies,
            "comparison_table": [
                {
                    "strategy": name,
                    "total_return": metrics["total_return"],
                    "annualized_volatility": metrics["annualized_volatility"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "turnover": metrics["turnover"],
                    "rebalance_events": self.event_summary.get(name, {}).get("event_count", 0.0),
                    "average_event_turnover": self.event_summary.get(name, {}).get(
                        "average_event_turnover", 0.0
                    ),
                }
                for name, metrics in strategies.items()
            ],
        }


def run_dynamic_rebalancing_experiment(
    prices: pd.DataFrame,
    config: DynamicRebalancingConfig | None = None,
    data_snapshot: str = "sample",
) -> DynamicRebalancingResult:
    """Compare static and dynamic long-only rebalancing policies."""
    config = config or DynamicRebalancingConfig()
    price_frame = select_static_universe(prices, config.symbols)
    _validate_config(config, price_frame.shape[1])

    train_prices, test_prices = time_train_test_split(price_frame, test_size=config.test_size)
    split_timestamp = test_prices.index[0]
    all_returns = returns_from_prices(price_frame)
    train_returns = all_returns.loc[all_returns.index < split_timestamp]
    test_returns = returns_from_prices(test_prices)
    if train_returns.empty or test_returns.empty:
        raise ValueError("Train and test partitions must produce non-empty return matrices.")

    static_weights = max_sharpe_weights(
        train_returns,
        min_weight=config.min_weight,
        max_weight=config.max_weight,
        covariance_ridge=config.covariance_ridge,
    )
    static_backtest = backtest_returns(
        test_returns,
        positions=static_positions(test_returns.index, static_weights),
        transaction_cost_bps=config.transaction_cost_bps,
        periods_per_year=config.periods_per_year,
    )

    weekly_positions, weekly_events = time_based_rebalance_positions(
        all_returns=all_returns,
        test_index=test_returns.index,
        config=config,
        strategy_name="weekly_inverse_volatility",
    )
    threshold_positions, threshold_events = threshold_rebalance_positions(
        all_returns=all_returns,
        test_returns=test_returns,
        config=config,
        strategy_name="threshold_inverse_volatility",
    )

    backtests = {
        "static_max_sharpe_reference": static_backtest,
        "weekly_inverse_volatility": backtest_returns(
            test_returns,
            positions=weekly_positions,
            transaction_cost_bps=config.transaction_cost_bps,
            periods_per_year=config.periods_per_year,
            benchmark_returns=static_backtest.returns,
        ),
        "threshold_inverse_volatility": backtest_returns(
            test_returns,
            positions=threshold_positions,
            transaction_cost_bps=config.transaction_cost_bps,
            periods_per_year=config.periods_per_year,
            benchmark_returns=static_backtest.returns,
        ),
    }
    events = pd.concat([weekly_events, threshold_events], ignore_index=True)
    event_summary = summarize_rebalance_events(events, backtests)
    diversification = {
        "static_max_sharpe_reference": diversification_summary(static_weights, train_returns),
        "weekly_inverse_volatility": diversification_summary(
            weekly_positions.iloc[-1],
            train_returns,
        ),
        "threshold_inverse_volatility": diversification_summary(
            threshold_positions.iloc[-1],
            train_returns,
        ),
    }
    selected_strategy = select_best_method(backtests, config.selection_metric)
    equity_curves = pd.concat(
        [result.equity_curve.rename(method) for method, result in backtests.items()],
        axis=1,
    )
    drawdowns = pd.concat(
        [drawdown_series(result.equity_curve).rename(method) for method, result in backtests.items()],
        axis=1,
    )

    return DynamicRebalancingResult(
        config=config,
        data_snapshot=data_snapshot,
        universe=list(price_frame.columns),
        split_timestamp=split_timestamp,
        train_period=_period_payload(train_prices),
        test_period=_period_payload(test_prices),
        static_reference_weights=static_weights,
        backtests=backtests,
        equity_curves=equity_curves,
        drawdowns=drawdowns,
        rebalance_events=events,
        event_summary=event_summary,
        diversification=diversification,
        selected_strategy=selected_strategy,
        selection_criterion=f"highest out-of-sample {config.selection_metric}",
    )


def time_based_rebalance_positions(
    all_returns: pd.DataFrame,
    test_index: pd.Index,
    config: DynamicRebalancingConfig,
    strategy_name: str = "weekly_inverse_volatility",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build target positions that rebalance on a fixed clock schedule."""
    if len(test_index) == 0:
        raise ValueError("test_index must be non-empty.")
    frequency = pd.Timedelta(config.rebalance_frequency)
    positions = pd.DataFrame(index=test_index, columns=all_returns.columns, dtype=float)
    events: list[dict[str, Any]] = []
    current_weights: pd.Series | None = None
    last_rebalance_at: pd.Timestamp | None = None

    for timestamp in test_index:
        should_rebalance = (
            current_weights is None
            or last_rebalance_at is None
            or timestamp - last_rebalance_at >= frequency
        )
        if should_rebalance:
            new_weights = _estimate_inverse_volatility_target(all_returns, timestamp, config)
            turnover = float(new_weights.abs().sum()) if current_weights is None else float(
                (new_weights - current_weights).abs().sum()
            )
            current_weights = new_weights
            last_rebalance_at = timestamp
            events.append(
                _event_payload(
                    timestamp=timestamp,
                    effective_timestamp=timestamp,
                    strategy=strategy_name,
                    reason="scheduled",
                    turnover=turnover,
                    drift=0.0,
                    weights=current_weights,
                )
            )
        positions.loc[timestamp] = current_weights

    return positions.astype(float), pd.DataFrame(events)


def threshold_rebalance_positions(
    all_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    config: DynamicRebalancingConfig,
    strategy_name: str = "threshold_inverse_volatility",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build positions that rebalance when simulated holding-weight drift is too large."""
    if test_returns.empty:
        raise ValueError("test_returns must be non-empty.")
    positions = pd.DataFrame(index=test_returns.index, columns=test_returns.columns, dtype=float)
    events: list[dict[str, Any]] = []
    current_target = _estimate_inverse_volatility_target(all_returns, test_returns.index[0], config)
    simulated_weights = current_target.copy()
    events.append(
        _event_payload(
            timestamp=test_returns.index[0],
            effective_timestamp=test_returns.index[0],
            strategy=strategy_name,
            reason="initial",
            turnover=float(current_target.abs().sum()),
            drift=0.0,
            weights=current_target,
        )
    )

    timestamps = list(test_returns.index)
    for position, timestamp in enumerate(timestamps):
        positions.loc[timestamp] = current_target
        simulated_weights = _apply_return_drift(simulated_weights, test_returns.loc[timestamp])
        drift = float((simulated_weights - current_target).abs().max())
        next_timestamp = timestamps[position + 1] if position + 1 < len(timestamps) else None
        if drift >= config.drift_threshold and next_timestamp is not None:
            new_target = _estimate_inverse_volatility_target(all_returns, next_timestamp, config)
            turnover = float((new_target - simulated_weights).abs().sum())
            events.append(
                _event_payload(
                    timestamp=timestamp,
                    effective_timestamp=next_timestamp,
                    strategy=strategy_name,
                    reason="drift_threshold",
                    turnover=turnover,
                    drift=drift,
                    weights=new_target,
                )
            )
            current_target = new_target
            simulated_weights = new_target.copy()

    return positions.astype(float), pd.DataFrame(events)


def summarize_rebalance_events(
    events: pd.DataFrame,
    backtests: dict[str, BacktestResult],
) -> dict[str, dict[str, float]]:
    """Summarize rebalance event counts and event-level turnover by strategy."""
    summary: dict[str, dict[str, float]] = {}
    for strategy, result in backtests.items():
        strategy_events = events.loc[events["strategy"] == strategy] if not events.empty else events
        event_turnover = (
            float(strategy_events["turnover"].sum()) if not strategy_events.empty else 0.0
        )
        event_count = float(len(strategy_events))
        summary[strategy] = {
            "event_count": event_count,
            "event_turnover": event_turnover,
            "average_event_turnover": event_turnover / event_count if event_count else 0.0,
            "backtest_turnover": float(result.metrics["turnover"]),
        }
    return summary


def _estimate_inverse_volatility_target(
    all_returns: pd.DataFrame,
    timestamp: pd.Timestamp,
    config: DynamicRebalancingConfig,
) -> pd.Series:
    history = all_returns.loc[all_returns.index < timestamp].tail(config.lookback_periods)
    if history.empty:
        raise ValueError(f"No trailing history available before {timestamp}.")
    return inverse_volatility_weights(
        history,
        min_weight=config.min_weight,
        max_weight=config.max_weight,
    )


def _apply_return_drift(weights: pd.Series, returns: pd.Series) -> pd.Series:
    grown = weights * (1.0 + returns.reindex(weights.index).fillna(0.0))
    total = float(grown.sum())
    if total <= 0.0:
        return weights
    return (grown / total).astype(float)


def _event_payload(
    timestamp: pd.Timestamp,
    effective_timestamp: pd.Timestamp,
    strategy: str,
    reason: str,
    turnover: float,
    drift: float,
    weights: pd.Series,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "effective_timestamp": effective_timestamp.isoformat(),
        "strategy": strategy,
        "reason": reason,
        "turnover": turnover,
        "drift": drift,
    }
    for symbol, weight in weights.items():
        payload[f"weight_{symbol}"] = float(weight)
    return payload


def _validate_config(config: DynamicRebalancingConfig, asset_count: int) -> None:
    if config.transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be non-negative.")
    if config.lookback_periods <= 1:
        raise ValueError("lookback_periods must be greater than 1.")
    if config.drift_threshold <= 0.0:
        raise ValueError("drift_threshold must be positive.")
    if config.min_weight < 0.0:
        raise ValueError("min_weight must be non-negative.")
    if config.max_weight <= 0.0 or config.max_weight > 1.0:
        raise ValueError("max_weight must be in (0, 1].")
    if config.min_weight * asset_count > 1.0 or config.max_weight * asset_count < 1.0:
        raise ValueError("Weight bounds do not allow weights to sum to 1.")
    pd.Timedelta(config.rebalance_frequency)


def _period_payload(prices: pd.DataFrame) -> dict[str, Any]:
    return {
        "start_utc": prices.index.min().isoformat(),
        "end_utc": prices.index.max().isoformat(),
        "rows": int(len(prices)),
    }

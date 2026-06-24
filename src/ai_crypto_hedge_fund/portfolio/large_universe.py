"""Sparse dynamic allocation for 100+ crypto pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ai_crypto_hedge_fund.backtest import BacktestResult, backtest_returns, time_train_test_split
from ai_crypto_hedge_fund.metrics import CRYPTO_MINUTE_PERIODS_PER_YEAR, drawdown_series, returns_from_prices


@dataclass(frozen=True)
class LargeUniverseConfig:
    """Configuration for scalable large-universe portfolio experiments."""

    symbols: tuple[str, ...] | None = None
    test_size: float = 0.3
    transaction_cost_bps: float = 5.0
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR
    lookback_periods: int = 30 * 24 * 60
    momentum_periods: int = 7 * 24 * 60
    rebalance_frequency: str = "7D"
    active_count: int = 20
    max_weight: float = 0.08
    min_coverage: float = 0.95
    min_universe_size: int = 100
    target_annualized_volatility: float = 0.60
    min_gross_exposure: float = 0.25
    selection_metric: str = "sharpe_ratio"


@dataclass(frozen=True)
class LargeUniverseResult:
    """Result bundle for a large-universe sparse allocation experiment."""

    config: LargeUniverseConfig
    data_snapshot: str
    universe: list[str]
    split_timestamp: pd.Timestamp
    train_period: dict[str, Any]
    test_period: dict[str, Any]
    universe_diagnostics: dict[str, Any]
    backtests: dict[str, BacktestResult]
    equity_curves: pd.DataFrame
    drawdowns: pd.DataFrame
    selected_assets: pd.DataFrame
    event_summary: dict[str, dict[str, float]]
    selected_strategy: str
    selection_criterion: str

    def metrics_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable report payload."""
        strategies = {name: result.metrics for name, result in self.backtests.items()}
        return {
            "data_snapshot": self.data_snapshot,
            "universe": self.universe,
            "universe_size": len(self.universe),
            "config": asdict(self.config),
            "split_timestamp_utc": self.split_timestamp.isoformat(),
            "train_period": self.train_period,
            "test_period": self.test_period,
            "universe_diagnostics": self.universe_diagnostics,
            "event_summary": self.event_summary,
            "selected_strategy": self.selected_strategy,
            "selection_criterion": self.selection_criterion,
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
                    "average_active_assets": self.event_summary.get(name, {}).get(
                        "average_active_assets", 0.0
                    ),
                    "average_gross_exposure": self.event_summary.get(name, {}).get(
                        "average_gross_exposure", 0.0
                    ),
                }
                for name, metrics in strategies.items()
            ],
        }


def run_large_universe_experiment(
    prices: pd.DataFrame,
    config: LargeUniverseConfig | None = None,
    data_snapshot: str = "sample",
) -> LargeUniverseResult:
    """Run sparse momentum allocation on a large crypto universe."""
    config = config or LargeUniverseConfig()
    price_frame = select_large_universe_prices(prices, config)
    train_prices, test_prices = time_train_test_split(price_frame, test_size=config.test_size)
    split_timestamp = test_prices.index[0]
    all_returns = returns_from_prices(price_frame, dropna=False).fillna(0.0)
    train_returns = all_returns.loc[all_returns.index < split_timestamp]
    test_returns = returns_from_prices(test_prices, dropna=False).iloc[1:].fillna(0.0)
    if train_returns.empty or test_returns.empty:
        raise ValueError("Train and test partitions must produce non-empty return matrices.")

    equal_positions = pd.DataFrame(
        1.0 / price_frame.shape[1],
        index=test_returns.index,
        columns=price_frame.columns,
    )
    momentum_positions, momentum_selected = sparse_rebalance_positions(
        all_returns=all_returns,
        test_index=test_returns.index,
        config=config,
        strategy_name="top_momentum_weekly",
        risk_adjusted=False,
    )
    risk_adjusted_positions, risk_adjusted_selected = sparse_rebalance_positions(
        all_returns=all_returns,
        test_index=test_returns.index,
        config=config,
        strategy_name="risk_adjusted_momentum_weekly",
        risk_adjusted=True,
    )

    equal_backtest = backtest_returns(
        test_returns,
        positions=equal_positions,
        transaction_cost_bps=config.transaction_cost_bps,
        periods_per_year=config.periods_per_year,
    )
    backtests = {
        "large_universe_equal_weight": equal_backtest,
        "top_momentum_weekly": backtest_returns(
            test_returns,
            positions=momentum_positions,
            transaction_cost_bps=config.transaction_cost_bps,
            periods_per_year=config.periods_per_year,
            benchmark_returns=equal_backtest.returns,
        ),
        "risk_adjusted_momentum_weekly": backtest_returns(
            test_returns,
            positions=risk_adjusted_positions,
            transaction_cost_bps=config.transaction_cost_bps,
            periods_per_year=config.periods_per_year,
            benchmark_returns=equal_backtest.returns,
        ),
    }
    selected_assets = pd.concat([momentum_selected, risk_adjusted_selected], ignore_index=True)
    event_summary = summarize_selected_assets(selected_assets, backtests)
    selected_strategy = max(backtests, key=lambda name: backtests[name].metrics[config.selection_metric])
    equity_curves = pd.concat(
        [result.equity_curve.rename(method) for method, result in backtests.items()],
        axis=1,
    )
    drawdowns = pd.concat(
        [drawdown_series(result.equity_curve).rename(method) for method, result in backtests.items()],
        axis=1,
    )

    return LargeUniverseResult(
        config=config,
        data_snapshot=data_snapshot,
        universe=list(price_frame.columns),
        split_timestamp=split_timestamp,
        train_period=_period_payload(train_prices),
        test_period=_period_payload(test_prices),
        universe_diagnostics=universe_diagnostics(price_frame, all_returns, config),
        backtests=backtests,
        equity_curves=equity_curves,
        drawdowns=drawdowns,
        selected_assets=selected_assets,
        event_summary=event_summary,
        selected_strategy=selected_strategy,
        selection_criterion=f"highest out-of-sample {config.selection_metric}",
    )


def select_large_universe_prices(prices: pd.DataFrame, config: LargeUniverseConfig) -> pd.DataFrame:
    """Select available symbols for the large-universe experiment."""
    if prices.empty:
        raise ValueError("Price matrix is empty.")
    if config.symbols is None:
        frame = prices.copy()
    else:
        missing = [symbol for symbol in config.symbols if symbol not in prices.columns]
        if missing:
            raise ValueError(f"Large-universe symbols missing from prices: {missing[:10]}")
        frame = prices.loc[:, list(config.symbols)].copy()
    frame = frame.sort_index().dropna(how="all").astype(float)
    frame = frame.loc[:, frame.notna().mean() >= config.min_coverage]
    if frame.shape[1] < 2:
        raise ValueError("Large-universe experiment requires at least two assets.")
    return frame.ffill().bfill()


def sparse_rebalance_positions(
    all_returns: pd.DataFrame,
    test_index: pd.Index,
    config: LargeUniverseConfig,
    strategy_name: str,
    risk_adjusted: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build sparse top-N target positions on a fixed rebalance schedule."""
    frequency = pd.Timedelta(config.rebalance_frequency)
    event_timestamps = _scheduled_timestamps(test_index, frequency)
    target_by_timestamp: dict[pd.Timestamp, pd.Series] = {}
    selected_rows: list[dict[str, Any]] = []
    previous_weights = pd.Series(0.0, index=all_returns.columns, dtype=float)

    for event_id, timestamp in enumerate(event_timestamps, start=1):
        ranking = rank_large_universe_assets(all_returns, timestamp, config, risk_adjusted)
        selected = ranking.head(min(config.active_count, len(ranking))).copy()
        weights = allocate_sparse_weights(selected, all_returns.columns, config, timestamp, all_returns)
        turnover = float((weights - previous_weights).abs().sum())
        gross_exposure = float(weights.sum())
        for rank, (symbol, row) in enumerate(selected.iterrows(), start=1):
            selected_rows.append(
                {
                    "event_id": event_id,
                    "timestamp": timestamp.isoformat(),
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "selection_rank": rank,
                    "score": float(row["score"]),
                    "momentum": float(row["momentum"]),
                    "volatility": float(row["volatility"]),
                    "coverage": float(row["coverage"]),
                    "weight": float(weights[symbol]),
                    "gross_exposure": gross_exposure,
                    "event_turnover": turnover,
                }
            )
        target_by_timestamp[timestamp] = weights
        previous_weights = weights

    positions = pd.DataFrame(index=test_index, columns=all_returns.columns, dtype=float)
    current = pd.Series(0.0, index=all_returns.columns, dtype=float)
    event_iter = iter(sorted(target_by_timestamp))
    next_event = next(event_iter, None)
    for timestamp in test_index:
        if next_event is not None and timestamp >= next_event:
            current = target_by_timestamp[next_event]
            next_event = next(event_iter, None)
        positions.loc[timestamp] = current
    return positions.astype(float), pd.DataFrame(selected_rows)


def rank_large_universe_assets(
    all_returns: pd.DataFrame,
    timestamp: pd.Timestamp,
    config: LargeUniverseConfig,
    risk_adjusted: bool,
) -> pd.DataFrame:
    """Rank assets from trailing momentum and volatility only."""
    history = all_returns.loc[all_returns.index < timestamp].tail(config.lookback_periods)
    if len(history) < max(2, min(config.momentum_periods, config.lookback_periods) // 4):
        raise ValueError(f"Insufficient trailing history before {timestamp}.")
    momentum_window = history.tail(min(config.momentum_periods, len(history)))
    coverage = history.notna().mean()
    volatility = history.std(ddof=0).replace(0.0, np.nan)
    momentum = (1.0 + momentum_window.fillna(0.0)).prod() - 1.0
    score = momentum / volatility if risk_adjusted else momentum
    ranking = pd.DataFrame(
        {
            "coverage": coverage,
            "volatility": volatility,
            "momentum": momentum,
            "score": score.replace([np.inf, -np.inf], np.nan),
        }
    ).dropna()
    ranking = ranking.loc[ranking["coverage"] >= config.min_coverage]
    if ranking.empty:
        ranking = pd.DataFrame(
            {
                "coverage": coverage,
                "volatility": volatility.fillna(volatility.median()),
                "momentum": momentum,
                "score": momentum,
            }
        ).dropna()
    ranking = ranking.sort_values(["score", "momentum"], ascending=False)
    return ranking


def allocate_sparse_weights(
    selected: pd.DataFrame,
    columns: pd.Index,
    config: LargeUniverseConfig,
    timestamp: pd.Timestamp,
    all_returns: pd.DataFrame,
) -> pd.Series:
    """Allocate capped positive weights to selected assets with volatility targeting."""
    if selected.empty:
        raise ValueError("Cannot allocate empty selected asset set.")
    raw_scores = selected["score"].clip(lower=0.0)
    if float(raw_scores.sum()) <= 0.0:
        raw_scores = selected["momentum"].rank(method="first", ascending=True).clip(lower=1.0)
    selected_weights = _cap_and_normalize(raw_scores, config.max_weight)
    full_weights = pd.Series(0.0, index=columns, dtype=float)
    full_weights.loc[selected_weights.index] = selected_weights
    gross_exposure = volatility_target_exposure(
        weights=full_weights,
        all_returns=all_returns,
        timestamp=timestamp,
        config=config,
    )
    return full_weights * gross_exposure


def volatility_target_exposure(
    weights: pd.Series,
    all_returns: pd.DataFrame,
    timestamp: pd.Timestamp,
    config: LargeUniverseConfig,
) -> float:
    """Scale gross exposure down when trailing portfolio volatility is above target."""
    history = all_returns.loc[all_returns.index < timestamp, weights.index].tail(config.lookback_periods)
    portfolio_returns = (history.fillna(0.0) * weights).sum(axis=1)
    if len(portfolio_returns) < 2:
        return 1.0
    annualized_volatility = float(portfolio_returns.std(ddof=0) * np.sqrt(config.periods_per_year))
    if annualized_volatility <= 0.0 or annualized_volatility <= config.target_annualized_volatility:
        return 1.0
    return float(
        max(config.min_gross_exposure, config.target_annualized_volatility / annualized_volatility)
    )


def summarize_selected_assets(
    selected_assets: pd.DataFrame,
    backtests: dict[str, BacktestResult],
) -> dict[str, dict[str, float]]:
    """Summarize rebalance events and active set sizes by strategy."""
    summary: dict[str, dict[str, float]] = {}
    for strategy, result in backtests.items():
        strategy_rows = (
            selected_assets.loc[selected_assets["strategy"] == strategy]
            if not selected_assets.empty
            else selected_assets
        )
        if strategy_rows.empty:
            summary[strategy] = {
                "event_count": 0.0,
                "average_active_assets": 0.0,
                "average_gross_exposure": 1.0,
                "event_turnover": 0.0,
                "backtest_turnover": float(result.metrics["turnover"]),
            }
            continue
        grouped = strategy_rows.groupby("event_id")
        summary[strategy] = {
            "event_count": float(grouped.ngroups),
            "average_active_assets": float(grouped["symbol"].count().mean()),
            "average_gross_exposure": float(grouped["gross_exposure"].first().mean()),
            "event_turnover": float(grouped["event_turnover"].first().sum()),
            "backtest_turnover": float(result.metrics["turnover"]),
        }
    return summary


def universe_diagnostics(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: LargeUniverseConfig,
) -> dict[str, Any]:
    """Calculate compact diagnostics for the selected universe."""
    coverage = prices.notna().mean()
    volatility = returns.std(ddof=0) * np.sqrt(config.periods_per_year)
    return {
        "available_symbol_count": int(prices.shape[1]),
        "meets_100_pair_requirement": bool(prices.shape[1] >= config.min_universe_size),
        "min_coverage": float(coverage.min()),
        "median_coverage": float(coverage.median()),
        "median_annualized_volatility": float(volatility.median()),
        "max_annualized_volatility": float(volatility.max()),
    }


def _scheduled_timestamps(index: pd.Index, frequency: pd.Timedelta) -> list[pd.Timestamp]:
    if len(index) == 0:
        raise ValueError("test index must be non-empty.")
    timestamps = [index[0]]
    last = index[0]
    for timestamp in index[1:]:
        if timestamp - last >= frequency:
            timestamps.append(timestamp)
            last = timestamp
    return timestamps


def _cap_and_normalize(scores: pd.Series, max_weight: float) -> pd.Series:
    if max_weight <= 0.0 or max_weight > 1.0:
        raise ValueError("max_weight must be in (0, 1].")
    weights = scores.astype(float).clip(lower=0.0)
    if float(weights.sum()) <= 0.0:
        weights = pd.Series(1.0, index=scores.index, dtype=float)
    weights = weights / float(weights.sum())
    active = weights.copy()
    capped = pd.Series(0.0, index=weights.index, dtype=float)
    remaining = 1.0
    while not active.empty:
        proposed = active / float(active.sum()) * remaining
        over_cap = proposed > max_weight + 1e-12
        if not over_cap.any():
            capped.loc[proposed.index] = proposed
            break
        capped_symbols = proposed[over_cap].index
        capped.loc[capped_symbols] = max_weight
        remaining -= max_weight * len(capped_symbols)
        active = active.drop(index=capped_symbols)
    total = float(capped.sum())
    return capped / total if total > 0.0 else capped


def _period_payload(prices: pd.DataFrame) -> dict[str, Any]:
    return {
        "start_utc": prices.index.min().isoformat(),
        "end_utc": prices.index.max().isoformat(),
        "rows": int(len(prices)),
    }

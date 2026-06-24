"""Backtesting utilities shared by strategy and portfolio experiments."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai_crypto_hedge_fund.metrics import (
    CRYPTO_MINUTE_PERIODS_PER_YEAR,
    calculate_turnover,
    equity_curve,
    performance_summary,
    returns_from_prices,
)


@dataclass(frozen=True)
class BacktestResult:
    """Container for deterministic backtest outputs."""

    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    equity_curve: pd.Series
    positions: pd.DataFrame
    metrics: dict[str, float]


def time_train_test_split(
    data: pd.Series | pd.DataFrame,
    test_size: float | int = 0.3,
    split_at: str | pd.Timestamp | None = None,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """Split time series data into train and test partitions without shuffling."""
    if data.empty:
        raise ValueError("Cannot split empty data.")

    ordered = data.sort_index()
    if split_at is not None:
        split_timestamp = pd.Timestamp(split_at)
        train = ordered.loc[ordered.index < split_timestamp]
        test = ordered.loc[ordered.index >= split_timestamp]
    elif isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("Float test_size must be between 0 and 1.")
        split_position = int(len(ordered) * (1.0 - test_size))
        train = ordered.iloc[:split_position]
        test = ordered.iloc[split_position:]
    else:
        if test_size <= 0 or test_size >= len(ordered):
            raise ValueError("Integer test_size must be positive and smaller than data length.")
        train = ordered.iloc[:-test_size]
        test = ordered.iloc[-test_size:]

    if train.empty or test.empty:
        raise ValueError("Train and test partitions must both be non-empty.")
    return train, test


def signals_to_positions(
    signals: pd.Series | pd.DataFrame,
    allow_short: bool = False,
    lag: int = 1,
) -> pd.Series | pd.DataFrame:
    """Convert numeric trading signals to executable positions.

    Positive signals become long exposure, negative signals become short exposure when
    enabled, and zero signals become flat. The default one-period lag avoids using
    information from the return period being traded.
    """
    if lag < 0:
        raise ValueError("lag must be non-negative.")

    positions = signals.astype(float).copy()
    positions = positions.where(positions <= 0.0, 1.0)
    positions = positions.where(positions >= 0.0, -1.0 if allow_short else 0.0)
    if lag:
        positions = positions.shift(lag)
    return positions.fillna(0.0)


def backtest_returns(
    asset_returns: pd.Series | pd.DataFrame,
    positions: pd.Series | pd.DataFrame | None = None,
    signals: pd.Series | pd.DataFrame | None = None,
    transaction_cost_bps: float = 0.0,
    initial_capital: float = 1.0,
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR,
    benchmark_returns: pd.Series | None = None,
    allow_short: bool = False,
    signal_lag: int = 1,
) -> BacktestResult:
    """Backtest positions or signals against simple asset returns."""
    if positions is not None and signals is not None:
        raise ValueError("Pass either positions or signals, not both.")
    if transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be non-negative.")

    return_frame = _as_return_frame(asset_returns)
    if signals is not None:
        positions = signals_to_positions(signals, allow_short=allow_short, lag=signal_lag)
    position_frame = _align_positions(return_frame, positions)

    gross_returns = (return_frame * position_frame).sum(axis=1)
    gross_returns.name = "gross_return"
    turnover = calculate_turnover(position_frame)
    costs = turnover * (transaction_cost_bps / 10_000.0)
    costs.name = "cost"
    net_returns = gross_returns - costs
    net_returns.name = "strategy_return"
    curve = equity_curve(net_returns, initial_capital=initial_capital)
    metrics = performance_summary(
        net_returns,
        positions=position_frame,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
    )
    metrics["transaction_cost_bps"] = float(transaction_cost_bps)
    metrics["total_cost"] = float(costs.sum())

    return BacktestResult(
        returns=net_returns,
        gross_returns=gross_returns,
        costs=costs,
        turnover=turnover,
        equity_curve=curve,
        positions=position_frame,
        metrics=metrics,
    )


def backtest_prices(
    prices: pd.Series | pd.DataFrame,
    positions: pd.Series | pd.DataFrame | None = None,
    signals: pd.Series | pd.DataFrame | None = None,
    transaction_cost_bps: float = 0.0,
    initial_capital: float = 1.0,
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR,
    benchmark_returns: pd.Series | None = None,
    allow_short: bool = False,
    signal_lag: int = 1,
) -> BacktestResult:
    """Backtest positions or signals against price data."""
    returns = returns_from_prices(prices)
    return backtest_returns(
        returns,
        positions=positions,
        signals=signals,
        transaction_cost_bps=transaction_cost_bps,
        initial_capital=initial_capital,
        periods_per_year=periods_per_year,
        benchmark_returns=benchmark_returns,
        allow_short=allow_short,
        signal_lag=signal_lag,
    )


def _as_return_frame(asset_returns: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(asset_returns, pd.Series):
        name = asset_returns.name or "asset"
        frame = asset_returns.to_frame(name=name)
    else:
        frame = asset_returns.copy()
    frame = frame.sort_index().astype(float).fillna(0.0)
    return frame


def _align_positions(
    returns: pd.DataFrame,
    positions: pd.Series | pd.DataFrame | None,
) -> pd.DataFrame:
    if positions is None:
        if returns.shape[1] == 1:
            return pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
        weight = 1.0 / returns.shape[1]
        return pd.DataFrame(weight, index=returns.index, columns=returns.columns)

    if isinstance(positions, pd.Series):
        if returns.shape[1] != 1:
            raise ValueError("Series positions are only valid for single-asset returns.")
        position_frame = positions.to_frame(name=returns.columns[0])
    else:
        position_frame = positions.copy()

    missing_columns = set(returns.columns) - set(position_frame.columns)
    if missing_columns:
        raise ValueError(f"Positions are missing columns: {sorted(missing_columns)}")

    position_frame = position_frame.reindex(index=returns.index, columns=returns.columns)
    position_frame = position_frame.ffill().fillna(0.0).astype(float)
    return position_frame

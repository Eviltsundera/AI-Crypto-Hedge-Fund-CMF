"""Shared return, risk, and performance metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

CRYPTO_MINUTE_PERIODS_PER_YEAR = 365 * 24 * 60


def returns_from_prices(prices: pd.Series | pd.DataFrame, dropna: bool = True) -> pd.Series | pd.DataFrame:
    """Calculate simple period returns from prices."""
    returns = prices.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    if dropna:
        returns = returns.dropna(how="any") if isinstance(returns, pd.DataFrame) else returns.dropna()
    return returns


def equity_curve(
    returns: pd.Series,
    initial_capital: float = 1.0,
    name: str = "equity",
) -> pd.Series:
    """Build an equity curve from simple strategy returns."""
    clean_returns = returns.fillna(0.0).astype(float)
    curve = initial_capital * (1.0 + clean_returns).cumprod()
    curve.name = name
    return curve


def total_return(returns: pd.Series) -> float:
    """Calculate compounded total return from simple period returns."""
    if returns.empty:
        return 0.0
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR) -> float:
    """Calculate annualized compounded return."""
    if returns.empty:
        return 0.0
    compounded = total_return(returns)
    if compounded <= -1.0:
        return -1.0
    annualized_log_growth = math.log1p(compounded) * periods_per_year / len(returns)
    if annualized_log_growth > 709.0:
        return float("inf")
    return float(math.expm1(annualized_log_growth))


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR,
) -> float:
    """Calculate annualized volatility."""
    if len(returns) < 2:
        return 0.0
    return float(returns.fillna(0.0).std(ddof=0) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate annualized Sharpe ratio."""
    if returns.empty:
        return 0.0
    period_risk_free = risk_free_rate / periods_per_year
    excess = returns.fillna(0.0) - period_risk_free
    volatility = excess.std(ddof=0)
    mean_excess = excess.mean()
    if volatility == 0.0:
        if mean_excess > 0.0:
            return float("inf")
        if mean_excess < 0.0:
            return float("-inf")
        return 0.0
    return float(mean_excess / volatility * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown from an equity curve."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdowns = equity / running_max - 1.0
    return float(drawdowns.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Calculate drawdown series from an equity curve."""
    if equity.empty:
        return pd.Series(dtype=float, name="drawdown")
    drawdowns = equity / equity.cummax() - 1.0
    drawdowns.name = "drawdown"
    return drawdowns


def hit_rate(returns: pd.Series) -> float:
    """Share of non-zero periods with positive return."""
    non_zero = returns.dropna()
    non_zero = non_zero[non_zero != 0.0]
    if non_zero.empty:
        return 0.0
    return float((non_zero > 0.0).mean())


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate annualized Sortino ratio using downside deviation."""
    if returns.empty:
        return 0.0
    period_risk_free = risk_free_rate / periods_per_year
    excess = returns.fillna(0.0) - period_risk_free
    downside = excess[excess < 0.0]
    if downside.empty:
        return float("inf") if excess.mean() > 0.0 else 0.0
    downside_deviation = math.sqrt(float((downside**2).mean()))
    if downside_deviation == 0.0:
        return 0.0
    return float(excess.mean() / downside_deviation * math.sqrt(periods_per_year))


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR,
) -> float:
    """Calculate Calmar ratio as annualized return divided by absolute max drawdown."""
    curve = equity_curve(returns)
    drawdown = abs(max_drawdown(curve))
    if drawdown == 0.0:
        return float("inf") if total_return(returns) > 0.0 else 0.0
    return float(annualized_return(returns, periods_per_year) / drawdown)


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR as a positive loss number."""
    if returns.empty:
        return 0.0
    quantile = returns.dropna().quantile(1.0 - confidence)
    return float(max(0.0, -quantile))


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical CVaR as a positive average tail loss number."""
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0
    cutoff = clean_returns.quantile(1.0 - confidence)
    tail = clean_returns[clean_returns <= cutoff]
    if tail.empty:
        return 0.0
    return float(max(0.0, -tail.mean()))


def calculate_turnover(positions: pd.Series | pd.DataFrame) -> pd.Series:
    """Calculate per-period absolute turnover from position weights."""
    if isinstance(positions, pd.Series):
        turnover = positions.fillna(0.0).diff().abs()
        if not turnover.empty:
            turnover.iloc[0] = abs(float(positions.fillna(0.0).iloc[0]))
        turnover.name = "turnover"
        return turnover.fillna(0.0)

    clean_positions = positions.fillna(0.0)
    turnover = clean_positions.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = clean_positions.iloc[0].abs().sum()
    turnover.name = "turnover"
    return turnover.fillna(0.0)


def performance_summary(
    returns: pd.Series,
    positions: pd.Series | pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Calculate standard performance and risk metrics for strategy returns."""
    clean_returns = returns.fillna(0.0).astype(float)
    curve = equity_curve(clean_returns)
    summary: dict[str, Any] = {
        "total_return": total_return(clean_returns),
        "annualized_return": annualized_return(clean_returns, periods_per_year),
        "annualized_volatility": annualized_volatility(clean_returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(clean_returns, periods_per_year, risk_free_rate),
        "max_drawdown": max_drawdown(curve),
        "hit_rate": hit_rate(clean_returns),
        "sortino_ratio": sortino_ratio(clean_returns, periods_per_year, risk_free_rate),
        "calmar_ratio": calmar_ratio(clean_returns, periods_per_year),
        "var_95": value_at_risk(clean_returns, confidence=0.95),
        "cvar_95": conditional_value_at_risk(clean_returns, confidence=0.95),
    }

    if positions is not None:
        period_turnover = calculate_turnover(positions)
        summary["turnover"] = float(period_turnover.sum())
        summary["average_turnover"] = float(period_turnover.mean()) if not period_turnover.empty else 0.0
    else:
        summary["turnover"] = 0.0
        summary["average_turnover"] = 0.0

    if benchmark_returns is not None:
        aligned_strategy, aligned_benchmark = clean_returns.align(
            benchmark_returns.fillna(0.0).astype(float),
            join="inner",
        )
        summary["benchmark_total_return"] = total_return(aligned_benchmark)
        summary["excess_total_return"] = total_return(aligned_strategy) - total_return(aligned_benchmark)

    return summary

"""Single-asset baseline trading strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from ai_crypto_hedge_fund.backtest import BacktestResult, backtest_returns, time_train_test_split
from ai_crypto_hedge_fund.metrics import CRYPTO_MINUTE_PERIODS_PER_YEAR, drawdown_series, returns_from_prices


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for the single-asset baseline experiment."""

    symbol: str = "BTCUSDT"
    fast_window: int = 60
    slow_window: int = 360
    test_size: float = 0.3
    transaction_cost_bps: float = 5.0
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR


@dataclass(frozen=True)
class BaselineExperimentResult:
    """Result bundle for the single-asset baseline experiment."""

    config: BaselineConfig
    split_timestamp: pd.Timestamp
    train_prices: pd.Series
    test_prices: pd.Series
    buy_and_hold: BacktestResult
    moving_average: BacktestResult
    signals: pd.Series
    equity_curves: pd.DataFrame
    drawdowns: pd.DataFrame

    def metrics_payload(self, data_snapshot: str) -> dict[str, Any]:
        """Return a JSON-serializable metrics payload."""
        return {
            "data_snapshot": data_snapshot,
            "symbol": self.config.symbol,
            "config": asdict(self.config),
            "train_period": _period_payload(self.train_prices),
            "test_period": _period_payload(self.test_prices),
            "split_timestamp_utc": self.split_timestamp.isoformat(),
            "strategies": {
                "buy_and_hold": self.buy_and_hold.metrics,
                "moving_average_crossover": self.moving_average.metrics,
            },
            "comparison": {
                "moving_average_excess_total_return": (
                    self.moving_average.metrics["total_return"]
                    - self.buy_and_hold.metrics["total_return"]
                ),
                "moving_average_excess_sharpe": (
                    self.moving_average.metrics["sharpe_ratio"]
                    - self.buy_and_hold.metrics["sharpe_ratio"]
                ),
            },
        }


def moving_average_crossover_signals(
    prices: pd.Series,
    fast_window: int = 60,
    slow_window: int = 360,
) -> pd.Series:
    """Build long/flat signals from a dual moving average crossover."""
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("Moving-average windows must be positive.")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")

    clean_prices = prices.sort_index().astype(float)
    fast_average = clean_prices.rolling(fast_window, min_periods=fast_window).mean()
    slow_average = clean_prices.rolling(slow_window, min_periods=slow_window).mean()
    signals = (fast_average > slow_average).astype(float)
    signals = signals.where(slow_average.notna(), 0.0)
    signals.name = "ma_crossover_signal"
    return signals


def run_single_asset_baseline(
    prices: pd.Series | pd.DataFrame,
    config: BaselineConfig | None = None,
) -> BaselineExperimentResult:
    """Run buy-and-hold and moving-average baselines on the out-of-sample period."""
    config = config or BaselineConfig()
    price_series = _select_price_series(prices, config.symbol)
    train_prices, test_prices = time_train_test_split(price_series, test_size=config.test_size)
    test_returns = returns_from_prices(test_prices)
    if test_returns.empty:
        raise ValueError("Test period is too short to calculate returns.")

    buy_and_hold_positions = pd.Series(1.0, index=test_returns.index, name=config.symbol)
    buy_and_hold = backtest_returns(
        test_returns,
        positions=buy_and_hold_positions,
        transaction_cost_bps=config.transaction_cost_bps,
        periods_per_year=config.periods_per_year,
    )

    full_signals = moving_average_crossover_signals(
        price_series,
        fast_window=config.fast_window,
        slow_window=config.slow_window,
    )
    test_signals = full_signals.reindex(test_returns.index).fillna(0.0)
    moving_average = backtest_returns(
        test_returns,
        signals=test_signals,
        transaction_cost_bps=config.transaction_cost_bps,
        periods_per_year=config.periods_per_year,
        benchmark_returns=buy_and_hold.returns,
    )

    equity_curves = pd.concat(
        [
            buy_and_hold.equity_curve.rename("buy_and_hold"),
            moving_average.equity_curve.rename("moving_average_crossover"),
        ],
        axis=1,
    )
    drawdowns = pd.concat(
        [
            drawdown_series(buy_and_hold.equity_curve).rename("buy_and_hold"),
            drawdown_series(moving_average.equity_curve).rename("moving_average_crossover"),
        ],
        axis=1,
    )

    return BaselineExperimentResult(
        config=config,
        split_timestamp=test_prices.index[0],
        train_prices=train_prices,
        test_prices=test_prices,
        buy_and_hold=buy_and_hold,
        moving_average=moving_average,
        signals=test_signals,
        equity_curves=equity_curves,
        drawdowns=drawdowns,
    )


def _select_price_series(prices: pd.Series | pd.DataFrame, symbol: str) -> pd.Series:
    if isinstance(prices, pd.Series):
        series = prices.copy()
        series.name = series.name or symbol
    else:
        if symbol not in prices.columns:
            raise ValueError(f"Symbol {symbol!r} is not present in the price matrix.")
        series = prices[symbol].copy()
        series.name = symbol

    series = series.sort_index().dropna().astype(float)
    if series.empty:
        raise ValueError("Selected price series is empty.")
    return series


def _period_payload(prices: pd.Series) -> dict[str, Any]:
    return {
        "start_utc": prices.index.min().isoformat(),
        "end_utc": prices.index.max().isoformat(),
        "rows": int(len(prices)),
    }

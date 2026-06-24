"""Single-asset econometric, ML, and agent-enhanced model comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from ai_crypto_hedge_fund.agents import AgentDecisionConfig, AgentDecisionEngine
from ai_crypto_hedge_fund.backtest import BacktestResult, backtest_returns, time_train_test_split
from ai_crypto_hedge_fund.features import build_model_frame
from ai_crypto_hedge_fund.metrics import CRYPTO_MINUTE_PERIODS_PER_YEAR, drawdown_series, returns_from_prices
from ai_crypto_hedge_fund.strategies.baseline import (
    BaselineConfig,
    moving_average_crossover_signals,
    run_single_asset_baseline,
)


@dataclass(frozen=True)
class SingleAssetModelConfig:
    """Configuration for single-asset model comparison."""

    symbol: str = "BTCUSDT"
    fast_window: int = 60
    slow_window: int = 360
    test_size: float = 0.3
    transaction_cost_bps: float = 5.0
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR
    econometric_window: int = 240
    econometric_threshold_quantile: float = 0.65
    econometric_confirmation_window: int = 30
    econometric_confirmation_ratio: float = 0.8
    max_train_rows: int = 120_000
    random_state: int = 42
    random_forest_estimators: int = 80
    random_forest_max_depth: int = 7
    ml_probability_threshold: float = 0.54
    ml_probability_smoothing_window: int = 30
    agent_min_long_votes: int = 2
    agent_volatility_quantile: float = 0.9
    agent_max_drawdown_limit: float = -0.08


@dataclass(frozen=True)
class SingleAssetModelComparisonResult:
    """Result bundle for the single-asset model comparison."""

    config: SingleAssetModelConfig
    data_snapshot: str
    split_timestamp: pd.Timestamp
    train_period: dict[str, Any]
    test_period: dict[str, Any]
    feature_columns: list[str]
    target_description: str
    training_summary: dict[str, Any]
    signals: pd.DataFrame
    agent_rationale: pd.Series
    backtests: dict[str, BacktestResult]
    equity_curves: pd.DataFrame
    drawdowns: pd.DataFrame

    def metrics_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable metrics payload."""
        strategies = {name: result.metrics for name, result in self.backtests.items()}
        return {
            "data_snapshot": self.data_snapshot,
            "symbol": self.config.symbol,
            "config": asdict(self.config),
            "split_timestamp_utc": self.split_timestamp.isoformat(),
            "train_period": self.train_period,
            "test_period": self.test_period,
            "feature_columns": self.feature_columns,
            "target": self.target_description,
            "training_summary": self.training_summary,
            "strategies": strategies,
            "comparison_table": [
                {
                    "strategy": name,
                    "total_return": metrics["total_return"],
                    "annualized_volatility": metrics["annualized_volatility"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "hit_rate": metrics["hit_rate"],
                    "turnover": metrics["turnover"],
                }
                for name, metrics in strategies.items()
            ],
        }


def run_single_asset_model_comparison(
    prices: pd.Series | pd.DataFrame,
    config: SingleAssetModelConfig | None = None,
    data_snapshot: str = "sample",
) -> SingleAssetModelComparisonResult:
    """Run baseline, econometric, ML, and deterministic agent strategies."""
    config = config or SingleAssetModelConfig()
    price_series = _select_price_series(prices, config.symbol)
    train_prices, test_prices = time_train_test_split(price_series, test_size=config.test_size)
    split_timestamp = test_prices.index[0]
    test_returns = returns_from_prices(test_prices)
    if test_returns.empty:
        raise ValueError("Test period is too short to calculate returns.")

    baseline = run_single_asset_baseline(
        price_series,
        BaselineConfig(
            symbol=config.symbol,
            fast_window=config.fast_window,
            slow_window=config.slow_window,
            test_size=config.test_size,
            transaction_cost_bps=config.transaction_cost_bps,
            periods_per_year=config.periods_per_year,
        ),
    )

    model_frame = build_model_frame(price_series)
    train_frame = model_frame.loc[model_frame["next_timestamp"] < split_timestamp].copy()
    signal_frame = model_frame.loc[model_frame.index >= split_timestamp].copy()
    if train_frame.empty or signal_frame.empty:
        raise ValueError("Model frame did not produce non-empty train and test partitions.")

    feature_columns = [
        column for column in model_frame.columns if column not in {"next_return", "target_up", "next_timestamp"}
    ]
    train_frame_for_model = train_frame.tail(config.max_train_rows)
    x_train = train_frame_for_model[feature_columns]
    y_train = train_frame_for_model["target_up"].astype(int)

    econometric_forecast = _rolling_econometric_forecast(
        price_series,
        window=config.econometric_window,
    )
    econ_threshold = float(
        econometric_forecast.reindex(train_frame.index).dropna().quantile(
            config.econometric_threshold_quantile
        )
    )
    raw_econometric_signal = (econometric_forecast > econ_threshold).astype(float)
    econometric_signal = _confirm_binary_signal(
        raw_econometric_signal,
        window=config.econometric_confirmation_window,
        ratio=config.econometric_confirmation_ratio,
    )
    econometric_signal.name = "econometric_signal"

    classifier = RandomForestClassifier(
        n_estimators=config.random_forest_estimators,
        max_depth=config.random_forest_max_depth,
        min_samples_leaf=200,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=config.random_state,
    )
    classifier.fit(x_train, y_train)

    x_signal = signal_frame[feature_columns]
    ml_probability = pd.Series(
        classifier.predict_proba(x_signal)[:, 1],
        index=x_signal.index,
        name="ml_up_probability",
    )
    ml_probability_smoothed = ml_probability.rolling(
        config.ml_probability_smoothing_window,
        min_periods=1,
    ).mean()
    ml_probability_smoothed.name = "ml_up_probability_smoothed"
    ml_signal = (ml_probability_smoothed >= config.ml_probability_threshold).astype(float)
    ml_signal.name = "ml_signal"

    baseline_signal = moving_average_crossover_signals(
        price_series,
        fast_window=config.fast_window,
        slow_window=config.slow_window,
    ).rename("baseline_ma_signal")

    rolling_volatility = returns_from_prices(price_series, dropna=False).rolling(
        config.econometric_window,
        min_periods=config.econometric_window,
    ).std(ddof=0)
    volatility_limit = float(
        rolling_volatility.reindex(train_frame.index).dropna().quantile(config.agent_volatility_quantile)
    )
    baseline_drawdown = drawdown_series(baseline.moving_average.equity_curve)
    baseline_drawdown = baseline_drawdown.reindex(signal_frame.index).ffill().fillna(0.0)

    agent_inputs = pd.concat(
        [
            baseline_signal.reindex(signal_frame.index),
            econometric_signal.reindex(signal_frame.index),
            ml_signal.reindex(signal_frame.index),
        ],
        axis=1,
    )
    agent = AgentDecisionEngine(
        AgentDecisionConfig(
            min_long_votes=config.agent_min_long_votes,
            volatility_limit=volatility_limit,
            max_drawdown_limit=config.agent_max_drawdown_limit,
        )
    )
    agent_signal, agent_rationale = agent.decide(
        signals=agent_inputs,
        rolling_volatility=rolling_volatility,
        rolling_drawdown=baseline_drawdown,
    )

    backtests = {
        "buy_and_hold": baseline.buy_and_hold,
        "moving_average_crossover": baseline.moving_average,
        "econometric_rolling": _backtest_signal(
            test_returns,
            econometric_signal,
            config,
            benchmark_returns=baseline.buy_and_hold.returns,
        ),
        "random_forest": _backtest_signal(
            test_returns,
            ml_signal,
            config,
            benchmark_returns=baseline.buy_and_hold.returns,
        ),
        "agent_enhanced": _backtest_signal(
            test_returns,
            agent_signal,
            config,
            benchmark_returns=baseline.buy_and_hold.returns,
        ),
    }
    equity_curves = pd.concat(
        [result.equity_curve.rename(name) for name, result in backtests.items()],
        axis=1,
    )
    drawdowns = pd.concat(
        [drawdown_series(result.equity_curve).rename(name) for name, result in backtests.items()],
        axis=1,
    )

    y_train_pred = classifier.predict(x_train)
    y_test = signal_frame["target_up"].astype(int)
    y_test_pred = (ml_probability >= config.ml_probability_threshold).astype(int)
    majority_class_rate = float(max(y_train.mean(), 1.0 - y_train.mean()))
    training_summary = {
        "training_rows_used": int(len(train_frame_for_model)),
        "available_training_rows": int(len(train_frame)),
        "test_signal_rows": int(len(signal_frame)),
        "target_positive_rate_train": float(y_train.mean()),
        "target_positive_rate_test": float(y_test.mean()),
        "majority_class_rate_train": majority_class_rate,
        "random_chance_reference": 0.5,
        "random_forest_train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "random_forest_test_direction_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "random_forest_test_roc_auc": _safe_auc(y_test, ml_probability),
        "econometric_threshold": econ_threshold,
        "econometric_confirmation_window": config.econometric_confirmation_window,
        "econometric_confirmation_ratio": config.econometric_confirmation_ratio,
        "agent_volatility_limit": volatility_limit,
        "feature_importance": dict(
            sorted(
                zip(feature_columns, classifier.feature_importances_, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )[:10]
        ),
    }

    signals = pd.concat(
        [
            baseline_signal.reindex(signal_frame.index),
            econometric_signal.reindex(signal_frame.index),
            ml_signal.reindex(signal_frame.index),
            ml_probability,
            ml_probability_smoothed,
            agent_signal,
        ],
        axis=1,
    )

    return SingleAssetModelComparisonResult(
        config=config,
        data_snapshot=data_snapshot,
        split_timestamp=split_timestamp,
        train_period=_period_payload(train_prices),
        test_period=_period_payload(test_prices),
        feature_columns=feature_columns,
        target_description="next 1-minute return direction: 1 if next_return > 0 else 0",
        training_summary=training_summary,
        signals=signals,
        agent_rationale=agent_rationale,
        backtests=backtests,
        equity_curves=equity_curves,
        drawdowns=drawdowns,
    )


def _rolling_econometric_forecast(prices: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        raise ValueError("window must be greater than 1.")
    returns = returns_from_prices(prices, dropna=False).fillna(0.0)
    forecast = returns.rolling(window, min_periods=window).mean()
    forecast.name = "rolling_mean_return_forecast"
    return forecast


def _confirm_binary_signal(signal: pd.Series, window: int, ratio: float) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive.")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1].")
    confirmed = signal.rolling(window, min_periods=1).mean() >= ratio
    return confirmed.astype(float)


def _backtest_signal(
    test_returns: pd.Series,
    signal: pd.Series,
    config: SingleAssetModelConfig,
    benchmark_returns: pd.Series,
) -> BacktestResult:
    return backtest_returns(
        test_returns,
        signals=signal,
        transaction_cost_bps=config.transaction_cost_bps,
        periods_per_year=config.periods_per_year,
        benchmark_returns=benchmark_returns,
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


def _safe_auc(y_true: pd.Series, probabilities: pd.Series) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))

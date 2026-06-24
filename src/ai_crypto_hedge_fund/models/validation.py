"""Validation-tuned single-asset ML experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from ai_crypto_hedge_fund.backtest import BacktestResult, backtest_returns, time_train_test_split
from ai_crypto_hedge_fund.features import build_single_asset_feature_frame
from ai_crypto_hedge_fund.metrics import CRYPTO_MINUTE_PERIODS_PER_YEAR, drawdown_series, returns_from_prices


@dataclass(frozen=True)
class ValidationModelConfig:
    """Configuration for validation-tuned single-asset ML."""

    symbol: str = "BTCUSDT"
    test_size: float = 0.3
    validation_size: float = 0.2
    horizon_periods: int = 60
    max_train_rows: int = 160_000
    random_state: int = 42
    random_forest_estimators: int = 120
    random_forest_max_depth: int = 8
    min_samples_leaf: int = 300
    probability_smoothing_window: int = 60
    threshold_grid: tuple[float, ...] = (0.50, 0.52, 0.54, 0.56, 0.58, 0.60)
    transaction_cost_bps: float = 5.0
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR


@dataclass(frozen=True)
class ValidationModelResult:
    """Result bundle for validation-tuned ML."""

    config: ValidationModelConfig
    data_snapshot: str
    split_timestamp: pd.Timestamp
    feature_columns: list[str]
    validation_summary: dict[str, Any]
    backtests: dict[str, BacktestResult]
    equity_curves: pd.DataFrame
    drawdowns: pd.DataFrame

    def metrics_payload(self) -> dict[str, Any]:
        strategies = {name: result.metrics for name, result in self.backtests.items()}
        return {
            "data_snapshot": self.data_snapshot,
            "symbol": self.config.symbol,
            "config": asdict(self.config),
            "split_timestamp_utc": self.split_timestamp.isoformat(),
            "feature_columns": self.feature_columns,
            "target": f"future {self.config.horizon_periods}-minute return direction",
            "validation_summary": self.validation_summary,
            "strategies": strategies,
            "comparison_table": [
                {
                    "strategy": name,
                    "total_return": metrics["total_return"],
                    "annualized_volatility": metrics["annualized_volatility"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "turnover": metrics["turnover"],
                }
                for name, metrics in strategies.items()
            ],
        }


def run_validation_model_experiment(
    prices: pd.Series | pd.DataFrame,
    config: ValidationModelConfig | None = None,
    data_snapshot: str = "sample",
) -> ValidationModelResult:
    """Tune ML threshold on validation data and evaluate once on test data."""
    config = config or ValidationModelConfig()
    price_series = _select_price_series(prices, config.symbol)
    _, test_prices = time_train_test_split(price_series, test_size=config.test_size)
    split_timestamp = test_prices.index[0]
    one_minute_returns = returns_from_prices(price_series)
    test_returns = returns_from_prices(test_prices)

    model_frame = _build_horizon_model_frame(price_series, config.horizon_periods)
    train_frame = model_frame.loc[model_frame["target_timestamp"] < split_timestamp].copy()
    signal_frame = model_frame.loc[model_frame.index >= split_timestamp].copy()
    if train_frame.empty or signal_frame.empty:
        raise ValueError("Model frame did not produce non-empty train and test partitions.")

    validation_rows = int(len(train_frame) * config.validation_size)
    if validation_rows <= 0 or validation_rows >= len(train_frame):
        raise ValueError("validation_size leaves no train or validation rows.")
    fit_frame = train_frame.iloc[:-validation_rows].tail(config.max_train_rows)
    validation_frame = train_frame.iloc[-validation_rows:]

    feature_columns = [
        column for column in model_frame.columns if column not in {"future_return", "target_up", "target_timestamp"}
    ]
    classifier = RandomForestClassifier(
        n_estimators=config.random_forest_estimators,
        max_depth=config.random_forest_max_depth,
        min_samples_leaf=config.min_samples_leaf,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=config.random_state,
    )
    classifier.fit(fit_frame[feature_columns], fit_frame["target_up"].astype(int))

    validation_probability = _smoothed_probability(
        classifier,
        validation_frame[feature_columns],
        config.probability_smoothing_window,
    )
    threshold_results = []
    validation_returns = one_minute_returns.reindex(validation_frame.index).fillna(0.0)
    for threshold in config.threshold_grid:
        signal = (validation_probability >= threshold).astype(float)
        result = backtest_returns(
            validation_returns,
            signals=signal,
            transaction_cost_bps=config.transaction_cost_bps,
            periods_per_year=config.periods_per_year,
        )
        threshold_results.append(
            {
                "threshold": threshold,
                "validation_total_return": result.metrics["total_return"],
                "validation_sharpe_ratio": result.metrics["sharpe_ratio"],
                "validation_max_drawdown": result.metrics["max_drawdown"],
                "validation_turnover": result.metrics["turnover"],
            }
        )
    selected = max(threshold_results, key=lambda item: item["validation_sharpe_ratio"])
    selected_threshold = float(selected["threshold"])

    test_probability = _smoothed_probability(
        classifier,
        signal_frame[feature_columns],
        config.probability_smoothing_window,
    )
    test_signal = (test_probability >= selected_threshold).astype(float)
    strategy_returns = test_returns.reindex(signal_frame.index).fillna(0.0)
    buy_and_hold = backtest_returns(
        test_returns,
        transaction_cost_bps=config.transaction_cost_bps,
        periods_per_year=config.periods_per_year,
    )
    validation_rf = backtest_returns(
        strategy_returns,
        signals=test_signal,
        transaction_cost_bps=config.transaction_cost_bps,
        periods_per_year=config.periods_per_year,
        benchmark_returns=buy_and_hold.returns,
    )
    backtests = {
        "buy_and_hold": buy_and_hold,
        "validation_tuned_rf_60m": validation_rf,
    }
    equity_curves = pd.concat(
        [result.equity_curve.rename(name) for name, result in backtests.items()],
        axis=1,
    )
    drawdowns = pd.concat(
        [drawdown_series(result.equity_curve).rename(name) for name, result in backtests.items()],
        axis=1,
    )

    y_validation = validation_frame["target_up"].astype(int)
    y_test = signal_frame["target_up"].astype(int)
    validation_summary = {
        "fit_rows": int(len(fit_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(signal_frame)),
        "selected_threshold": selected_threshold,
        "threshold_grid_results": threshold_results,
        "validation_direction_accuracy": float(
            accuracy_score(y_validation, (validation_probability >= selected_threshold).astype(int))
        ),
        "test_direction_accuracy": float(
            accuracy_score(y_test, (test_probability >= selected_threshold).astype(int))
        ),
        "validation_roc_auc": _safe_auc(y_validation, validation_probability),
        "test_roc_auc": _safe_auc(y_test, test_probability),
    }

    return ValidationModelResult(
        config=config,
        data_snapshot=data_snapshot,
        split_timestamp=split_timestamp,
        feature_columns=feature_columns,
        validation_summary=validation_summary,
        backtests=backtests,
        equity_curves=equity_curves,
        drawdowns=drawdowns,
    )


def _build_horizon_model_frame(prices: pd.Series, horizon_periods: int) -> pd.DataFrame:
    if horizon_periods <= 0:
        raise ValueError("horizon_periods must be positive.")
    features = build_single_asset_feature_frame(prices)
    future_return = prices.shift(-horizon_periods) / prices - 1.0
    target_timestamps = pd.Series(prices.index, index=prices.index).shift(-horizon_periods)
    targets = pd.DataFrame(
        {
            "future_return": future_return,
            "target_up": (future_return > 0.0).astype(int),
            "target_timestamp": target_timestamps,
        },
        index=prices.index,
    )
    return features.join(targets, how="inner").dropna(how="any")


def _smoothed_probability(
    classifier: RandomForestClassifier,
    features: pd.DataFrame,
    smoothing_window: int,
) -> pd.Series:
    probabilities = classifier.predict_proba(features)
    if probabilities.shape[1] == 1:
        positive_probability = float(classifier.classes_[0] == 1)
        values = [positive_probability] * len(features)
    else:
        positive_index = list(classifier.classes_).index(1)
        values = probabilities[:, positive_index]
    probability = pd.Series(
        values,
        index=features.index,
        name="up_probability",
    )
    if smoothing_window <= 1:
        return probability
    return probability.rolling(smoothing_window, min_periods=1).mean()


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


def _safe_auc(y_true: pd.Series, probabilities: pd.Series) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))

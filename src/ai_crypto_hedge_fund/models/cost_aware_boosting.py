"""Cost-aware single-asset boosting experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from ai_crypto_hedge_fund.backtest import BacktestResult, backtest_returns, time_train_test_split
from ai_crypto_hedge_fund.features import build_single_asset_feature_frame
from ai_crypto_hedge_fund.metrics import CRYPTO_MINUTE_PERIODS_PER_YEAR, drawdown_series, returns_from_prices


@dataclass(frozen=True)
class CostAwareBoostingConfig:
    """Configuration for cost-aware boosted single-asset classification."""

    symbol: str = "BTCUSDT"
    test_size: float = 0.3
    validation_size: float = 0.2
    horizon_periods: int = 60
    cost_buffer: float = 0.0010
    max_train_rows: int = 180_000
    random_state: int = 42
    max_iter: int = 160
    learning_rate: float = 0.04
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 250
    l2_regularization: float = 0.02
    probability_smoothing_window: int = 60
    threshold_grid: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70)
    transaction_cost_bps: float = 5.0
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR


@dataclass(frozen=True)
class CostAwareBoostingResult:
    """Result bundle for cost-aware boosting."""

    config: CostAwareBoostingConfig
    data_snapshot: str
    split_timestamp: pd.Timestamp
    feature_columns: list[str]
    target_summary: dict[str, Any]
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
            "target": (
                f"future {self.config.horizon_periods}-minute return > "
                f"{self.config.cost_buffer}; neutral rows inside +/- buffer dropped from training"
            ),
            "target_summary": self.target_summary,
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


def run_cost_aware_boosting_experiment(
    prices: pd.Series | pd.DataFrame,
    config: CostAwareBoostingConfig | None = None,
    data_snapshot: str = "sample",
) -> CostAwareBoostingResult:
    """Train a cost-aware boosted classifier and evaluate long-only signals."""
    config = config or CostAwareBoostingConfig()
    price_series = _select_price_series(prices, config.symbol)
    _, test_prices = time_train_test_split(price_series, test_size=config.test_size)
    split_timestamp = test_prices.index[0]
    one_minute_returns = returns_from_prices(price_series)
    test_returns = returns_from_prices(test_prices)

    model_frame = _build_cost_aware_frame(price_series, config.horizon_periods, config.cost_buffer)
    train_frame = model_frame.loc[model_frame["target_timestamp"] < split_timestamp].copy()
    signal_frame = model_frame.loc[model_frame.index >= split_timestamp].copy()
    train_labeled = train_frame.loc[train_frame["target_trade"].notna()].copy()
    if train_labeled.empty or signal_frame.empty:
        raise ValueError("Cost-aware target produced empty train or signal partitions.")

    validation_rows = int(len(train_labeled) * config.validation_size)
    if validation_rows <= 0 or validation_rows >= len(train_labeled):
        raise ValueError("validation_size leaves no train or validation rows.")
    fit_frame = train_labeled.iloc[:-validation_rows].tail(config.max_train_rows)
    validation_frame = train_labeled.iloc[-validation_rows:]

    feature_columns = [
        column
        for column in model_frame.columns
        if column not in {"future_return", "target_trade", "target_timestamp"}
    ]
    classifier = HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        random_state=config.random_state,
    )
    classifier.fit(fit_frame[feature_columns], fit_frame["target_trade"].astype(int))

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
    cost_aware_boosting = backtest_returns(
        strategy_returns,
        signals=test_signal,
        transaction_cost_bps=config.transaction_cost_bps,
        periods_per_year=config.periods_per_year,
        benchmark_returns=buy_and_hold.returns,
    )
    backtests = {
        "buy_and_hold": buy_and_hold,
        "cost_aware_hist_gradient_boosting": cost_aware_boosting,
    }
    equity_curves = pd.concat(
        [result.equity_curve.rename(name) for name, result in backtests.items()],
        axis=1,
        sort=False,
    )
    drawdowns = pd.concat(
        [drawdown_series(result.equity_curve).rename(name) for name, result in backtests.items()],
        axis=1,
        sort=False,
    )

    y_validation = validation_frame["target_trade"].astype(int)
    test_labeled = signal_frame.loc[signal_frame["target_trade"].notna()].copy()
    if test_labeled.empty:
        test_accuracy = None
        test_auc = None
    else:
        labeled_probability = test_probability.reindex(test_labeled.index)
        test_accuracy = float(
            accuracy_score(test_labeled["target_trade"].astype(int), (labeled_probability >= selected_threshold).astype(int))
        )
        test_auc = _safe_auc(test_labeled["target_trade"].astype(int), labeled_probability)

    target_summary = _target_summary(model_frame, train_frame, signal_frame, config.cost_buffer)
    validation_summary = {
        "fit_rows": int(len(fit_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(signal_frame)),
        "selected_threshold": selected_threshold,
        "threshold_grid_results": threshold_results,
        "validation_direction_accuracy": float(
            accuracy_score(y_validation, (validation_probability >= selected_threshold).astype(int))
        ),
        "test_direction_accuracy_on_labeled_rows": test_accuracy,
        "validation_roc_auc": _safe_auc(y_validation, validation_probability),
        "test_roc_auc_on_labeled_rows": test_auc,
    }

    return CostAwareBoostingResult(
        config=config,
        data_snapshot=data_snapshot,
        split_timestamp=split_timestamp,
        feature_columns=feature_columns,
        target_summary=target_summary,
        validation_summary=validation_summary,
        backtests=backtests,
        equity_curves=equity_curves,
        drawdowns=drawdowns,
    )


def _build_cost_aware_frame(
    prices: pd.Series,
    horizon_periods: int,
    cost_buffer: float,
) -> pd.DataFrame:
    if horizon_periods <= 0:
        raise ValueError("horizon_periods must be positive.")
    if cost_buffer < 0.0:
        raise ValueError("cost_buffer must be non-negative.")
    features = build_single_asset_feature_frame(prices)
    future_return = prices.shift(-horizon_periods) / prices - 1.0
    target = pd.Series(pd.NA, index=prices.index, dtype="Float64")
    target.loc[future_return > cost_buffer] = 1.0
    target.loc[future_return < -cost_buffer] = 0.0
    target_timestamps = pd.Series(prices.index, index=prices.index).shift(-horizon_periods)
    targets = pd.DataFrame(
        {
            "future_return": future_return,
            "target_trade": target,
            "target_timestamp": target_timestamps,
        },
        index=prices.index,
    )
    return features.join(targets, how="inner").dropna(subset=["future_return", "target_timestamp"])


def _target_summary(
    model_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    signal_frame: pd.DataFrame,
    cost_buffer: float,
) -> dict[str, Any]:
    train_labeled = train_frame["target_trade"].dropna()
    test_labeled = signal_frame["target_trade"].dropna()
    return {
        "cost_buffer": cost_buffer,
        "all_rows": int(len(model_frame)),
        "train_rows": int(len(train_frame)),
        "train_labeled_rows": int(len(train_labeled)),
        "train_neutral_rows": int(train_frame["target_trade"].isna().sum()),
        "train_positive_rate_labeled": float(train_labeled.mean()) if not train_labeled.empty else None,
        "test_rows": int(len(signal_frame)),
        "test_labeled_rows": int(len(test_labeled)),
        "test_neutral_rows": int(signal_frame["target_trade"].isna().sum()),
        "test_positive_rate_labeled": float(test_labeled.mean()) if not test_labeled.empty else None,
    }


def _smoothed_probability(
    classifier: HistGradientBoostingClassifier,
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
    probability = pd.Series(values, index=features.index, name="positive_probability")
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

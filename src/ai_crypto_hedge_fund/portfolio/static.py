"""Static long-only portfolio construction for a small crypto universe."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ai_crypto_hedge_fund.backtest import BacktestResult, backtest_returns, time_train_test_split
from ai_crypto_hedge_fund.metrics import CRYPTO_MINUTE_PERIODS_PER_YEAR, drawdown_series, returns_from_prices


DEFAULT_STATIC_UNIVERSE = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT")


@dataclass(frozen=True)
class StaticPortfolioConfig:
    """Configuration for a static long-only portfolio comparison."""

    symbols: tuple[str, ...] = DEFAULT_STATIC_UNIVERSE
    test_size: float = 0.3
    transaction_cost_bps: float = 5.0
    periods_per_year: int = CRYPTO_MINUTE_PERIODS_PER_YEAR
    max_weight: float = 0.35
    min_weight: float = 0.0
    covariance_ridge: float = 1e-10
    selection_metric: str = "sharpe_ratio"


@dataclass(frozen=True)
class StaticPortfolioResult:
    """Result bundle for the static portfolio experiment."""

    config: StaticPortfolioConfig
    data_snapshot: str
    universe: list[str]
    split_timestamp: pd.Timestamp
    train_period: dict[str, Any]
    test_period: dict[str, Any]
    weights: pd.DataFrame
    backtests: dict[str, BacktestResult]
    equity_curves: pd.DataFrame
    drawdowns: pd.DataFrame
    diversification: dict[str, dict[str, float]]
    selected_method: str
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
            "weights": self.weights.to_dict(orient="index"),
            "diversification": self.diversification,
            "selected_method": self.selected_method,
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
                    "effective_assets": self.diversification[name]["effective_assets"],
                    "max_weight": self.diversification[name]["max_weight"],
                }
                for name, metrics in strategies.items()
            ],
        }


def run_static_portfolio_experiment(
    prices: pd.DataFrame,
    config: StaticPortfolioConfig | None = None,
    data_snapshot: str = "sample",
) -> StaticPortfolioResult:
    """Compare static allocation methods on a train/test split."""
    config = config or StaticPortfolioConfig()
    price_frame = select_static_universe(prices, config.symbols)
    _validate_config(config, price_frame.shape[1])

    train_prices, test_prices = time_train_test_split(price_frame, test_size=config.test_size)
    train_returns = returns_from_prices(train_prices)
    test_returns = returns_from_prices(test_prices)
    if train_returns.empty or test_returns.empty:
        raise ValueError("Train and test partitions must produce non-empty return matrices.")

    weights = pd.DataFrame(
        {
            "equal_weight": equal_weight_weights(price_frame.columns),
            "inverse_volatility": inverse_volatility_weights(train_returns),
            "max_sharpe_constrained": max_sharpe_weights(
                train_returns,
                min_weight=config.min_weight,
                max_weight=config.max_weight,
                covariance_ridge=config.covariance_ridge,
            ),
        }
    ).T
    weights = weights.reindex(columns=price_frame.columns)

    backtests: dict[str, BacktestResult] = {}
    equal_weight_returns: pd.Series | None = None
    for method, method_weights in weights.iterrows():
        positions = static_positions(test_returns.index, method_weights)
        benchmark = equal_weight_returns if method != "equal_weight" else None
        result = backtest_returns(
            test_returns,
            positions=positions,
            transaction_cost_bps=config.transaction_cost_bps,
            periods_per_year=config.periods_per_year,
            benchmark_returns=benchmark,
        )
        backtests[method] = result
        if method == "equal_weight":
            equal_weight_returns = result.returns

    diversification = {
        method: diversification_summary(method_weights, train_returns)
        for method, method_weights in weights.iterrows()
    }
    selected_method = select_best_method(backtests, config.selection_metric)
    equity_curves = pd.concat(
        [result.equity_curve.rename(method) for method, result in backtests.items()],
        axis=1,
    )
    drawdowns = pd.concat(
        [drawdown_series(result.equity_curve).rename(method) for method, result in backtests.items()],
        axis=1,
    )

    return StaticPortfolioResult(
        config=config,
        data_snapshot=data_snapshot,
        universe=list(price_frame.columns),
        split_timestamp=test_prices.index[0],
        train_period=_period_payload(train_prices),
        test_period=_period_payload(test_prices),
        weights=weights,
        backtests=backtests,
        equity_curves=equity_curves,
        drawdowns=drawdowns,
        diversification=diversification,
        selected_method=selected_method,
        selection_criterion=f"highest out-of-sample {config.selection_metric}",
    )


def select_static_universe(prices: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Select and validate the static portfolio universe from a price matrix."""
    if prices.empty:
        raise ValueError("Price matrix is empty.")
    missing = [symbol for symbol in symbols if symbol not in prices.columns]
    if missing:
        raise ValueError(f"Static portfolio symbols are missing from prices: {missing}")
    frame = prices.loc[:, list(symbols)].sort_index().dropna(how="any").astype(float)
    if frame.empty:
        raise ValueError("Selected static portfolio universe has no complete price rows.")
    if frame.shape[1] < 2:
        raise ValueError("Static portfolio requires at least two assets.")
    return frame


def equal_weight_weights(symbols: pd.Index | list[str] | tuple[str, ...]) -> pd.Series:
    """Return equal long-only weights."""
    index = pd.Index(symbols)
    if len(index) == 0:
        raise ValueError("Cannot allocate an empty symbol list.")
    return pd.Series(1.0 / len(index), index=index, dtype=float)


def inverse_volatility_weights(returns: pd.DataFrame) -> pd.Series:
    """Return inverse-volatility weights with equal-weight fallback."""
    clean_returns = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean_returns.empty:
        return equal_weight_weights(returns.columns)
    volatility = clean_returns.std(ddof=0).replace(0.0, np.nan)
    inverse_vol = 1.0 / volatility
    if inverse_vol.isna().all() or float(inverse_vol.sum(skipna=True)) <= 0.0:
        return equal_weight_weights(returns.columns)
    weights = inverse_vol.fillna(0.0) / float(inverse_vol.sum(skipna=True))
    return weights.reindex(returns.columns).astype(float)


def max_sharpe_weights(
    returns: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 0.35,
    covariance_ridge: float = 1e-10,
) -> pd.Series:
    """Return constrained max-Sharpe weights, falling back to inverse volatility."""
    clean_returns = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    fallback = _cap_and_normalize(inverse_volatility_weights(returns), min_weight, max_weight)
    if clean_returns.empty:
        return fallback

    asset_count = clean_returns.shape[1]
    _validate_weight_bounds(asset_count, min_weight, max_weight)
    mean_returns = clean_returns.mean().to_numpy(dtype=float)
    covariance = clean_returns.cov().to_numpy(dtype=float)
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    covariance = covariance + np.eye(asset_count) * covariance_ridge
    bounds = [(min_weight, max_weight)] * asset_count
    constraints = {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}
    initial = fallback.to_numpy(dtype=float)

    def objective(weights: np.ndarray) -> float:
        expected_return = float(weights @ mean_returns)
        variance = float(weights @ covariance @ weights)
        if variance <= 0.0:
            return 1e6
        return -expected_return / np.sqrt(variance)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12, "disp": False},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return fallback
    weights = pd.Series(result.x, index=returns.columns, dtype=float)
    return _cap_and_normalize(weights, min_weight, max_weight)


def static_positions(index: pd.Index, weights: pd.Series) -> pd.DataFrame:
    """Expand one static weight vector into a position matrix."""
    clean_weights = weights.astype(float)
    return pd.DataFrame(
        np.tile(clean_weights.to_numpy(), (len(index), 1)),
        index=index,
        columns=clean_weights.index,
    )


def diversification_summary(weights: pd.Series, train_returns: pd.DataFrame) -> dict[str, float]:
    """Calculate weight concentration and train-period correlation diagnostics."""
    clean_weights = weights.astype(float).clip(lower=0.0)
    hhi = float((clean_weights**2).sum())
    nonzero = clean_weights[clean_weights > 1e-8]
    correlation = train_returns.corr()
    if correlation.shape[0] > 1:
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        average_pairwise_correlation = float(upper.stack().mean())
    else:
        average_pairwise_correlation = 0.0
    return {
        "nonzero_assets": float(len(nonzero)),
        "max_weight": float(clean_weights.max()) if not clean_weights.empty else 0.0,
        "min_weight": float(nonzero.min()) if not nonzero.empty else 0.0,
        "weight_hhi": hhi,
        "effective_assets": float(1.0 / hhi) if hhi > 0.0 else 0.0,
        "average_pairwise_correlation_train": average_pairwise_correlation,
    }


def select_best_method(backtests: dict[str, BacktestResult], selection_metric: str) -> str:
    """Select the best portfolio method by a metric in the backtest output."""
    if not backtests:
        raise ValueError("Cannot select from an empty backtest dictionary.")
    missing = [name for name, result in backtests.items() if selection_metric not in result.metrics]
    if missing:
        raise ValueError(f"Selection metric {selection_metric!r} is missing for {missing}")
    return max(backtests, key=lambda name: backtests[name].metrics[selection_metric])


def _cap_and_normalize(weights: pd.Series, min_weight: float, max_weight: float) -> pd.Series:
    _validate_weight_bounds(len(weights), min_weight, max_weight)
    capped = weights.astype(float).clip(lower=min_weight, upper=max_weight)
    total = float(capped.sum())
    if total <= 0.0:
        capped = equal_weight_weights(weights.index)
    else:
        capped = capped / total
    if float(capped.max()) <= max_weight + 1e-10 and float(capped.min()) >= min_weight - 1e-10:
        return capped

    result = minimize(
        lambda candidate: float(np.sum((candidate - weights.to_numpy(dtype=float)) ** 2)),
        capped.to_numpy(dtype=float),
        method="SLSQP",
        bounds=[(min_weight, max_weight)] * len(weights),
        constraints={"type": "eq", "fun": lambda candidate: float(np.sum(candidate) - 1.0)},
        options={"maxiter": 200, "ftol": 1e-12, "disp": False},
    )
    if result.success and np.all(np.isfinite(result.x)):
        return pd.Series(result.x, index=weights.index, dtype=float)
    return equal_weight_weights(weights.index)


def _validate_config(config: StaticPortfolioConfig, asset_count: int) -> None:
    if config.transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be non-negative.")
    if config.covariance_ridge < 0.0:
        raise ValueError("covariance_ridge must be non-negative.")
    _validate_weight_bounds(asset_count, config.min_weight, config.max_weight)


def _validate_weight_bounds(asset_count: int, min_weight: float, max_weight: float) -> None:
    if asset_count <= 0:
        raise ValueError("asset_count must be positive.")
    if min_weight < 0.0:
        raise ValueError("min_weight must be non-negative.")
    if max_weight <= 0.0 or max_weight > 1.0:
        raise ValueError("max_weight must be in (0, 1].")
    if min_weight > max_weight:
        raise ValueError("min_weight cannot exceed max_weight.")
    if min_weight * asset_count > 1.0 or max_weight * asset_count < 1.0:
        raise ValueError("Weight bounds do not allow weights to sum to 1.")


def _period_payload(prices: pd.DataFrame) -> dict[str, Any]:
    return {
        "start_utc": prices.index.min().isoformat(),
        "end_utc": prices.index.max().isoformat(),
        "rows": int(len(prices)),
    }

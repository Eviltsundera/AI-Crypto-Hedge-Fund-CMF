"""Feature engineering for single-asset trading models."""

from __future__ import annotations

import pandas as pd

from ai_crypto_hedge_fund.metrics import returns_from_prices


def build_single_asset_feature_frame(
    prices: pd.Series,
    lag_count: int = 5,
    volatility_windows: tuple[int, ...] = (60, 240),
    momentum_windows: tuple[int, ...] = (15, 60, 360),
) -> pd.DataFrame:
    """Build lagged and rolling features available at each timestamp."""
    if lag_count <= 0:
        raise ValueError("lag_count must be positive.")

    price_series = prices.sort_index().dropna().astype(float)
    returns = returns_from_prices(price_series, dropna=False).fillna(0.0)
    frame = pd.DataFrame(index=price_series.index)
    frame["return_1"] = returns

    for lag in range(1, lag_count + 1):
        frame[f"return_lag_{lag}"] = returns.shift(lag)

    for window in volatility_windows:
        if window <= 1:
            raise ValueError("volatility windows must be greater than 1.")
        frame[f"rolling_mean_{window}"] = returns.rolling(window, min_periods=window).mean()
        frame[f"rolling_vol_{window}"] = returns.rolling(window, min_periods=window).std(ddof=0)

    for window in momentum_windows:
        if window <= 0:
            raise ValueError("momentum windows must be positive.")
        frame[f"momentum_{window}"] = price_series.pct_change(window, fill_method=None)

    if 60 in momentum_windows and 360 in momentum_windows:
        fast_average = price_series.rolling(60, min_periods=60).mean()
        slow_average = price_series.rolling(360, min_periods=360).mean()
        frame["ma_distance_60_360"] = fast_average / slow_average - 1.0

    frame = frame.replace([float("inf"), float("-inf")], pd.NA)
    return frame.dropna(how="any")


def build_next_return_targets(prices: pd.Series) -> pd.DataFrame:
    """Build next-period return and direction targets for timestamped features."""
    price_series = prices.sort_index().dropna().astype(float)
    returns = returns_from_prices(price_series, dropna=False)
    targets = pd.DataFrame(index=price_series.index)
    targets["next_return"] = returns.shift(-1)
    targets["target_up"] = (targets["next_return"] > 0.0).astype(int)
    next_timestamps = pd.Series(price_series.index, index=price_series.index).shift(-1)
    targets["next_timestamp"] = next_timestamps
    return targets.dropna(how="any")


def build_model_frame(prices: pd.Series) -> pd.DataFrame:
    """Join feature and next-return target data."""
    features = build_single_asset_feature_frame(prices)
    targets = build_next_return_targets(prices)
    return features.join(targets, how="inner")

"""Load committed sample data or ignored external snapshots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_crypto_hedge_fund.paths import data_dir, external_data_dir, sample_data_dir

FULL_SNAPSHOT_NAME = "binance_spot_1m_120_12mo"


def snapshot_processed_dir(snapshot: str = "sample", root: Path | None = None) -> Path:
    """Return the processed directory for a named snapshot."""
    if root is not None:
        return root
    if snapshot == "sample":
        return sample_data_dir()
    if snapshot == "full":
        return external_data_dir() / FULL_SNAPSHOT_NAME / "processed"
    raise ValueError(f"Unknown snapshot {snapshot!r}; expected 'sample' or 'full'.")


def load_price_matrix(snapshot: str = "sample", root: Path | None = None) -> pd.DataFrame:
    """Load a price matrix with a UTC timestamp index."""
    path = snapshot_processed_dir(snapshot=snapshot, root=root) / "prices_1m.parquet"
    prices = pd.read_parquet(path)
    prices.index = pd.to_datetime(prices.index, utc=True)
    prices.index.name = "timestamp"
    return prices


def load_return_matrix(snapshot: str = "sample", root: Path | None = None) -> pd.DataFrame:
    """Load a return matrix with a UTC timestamp index."""
    path = snapshot_processed_dir(snapshot=snapshot, root=root) / "returns_1m.parquet"
    returns = pd.read_parquet(path)
    returns.index = pd.to_datetime(returns.index, utc=True)
    returns.index.name = "timestamp"
    return returns


def load_universe(kind: str = "large") -> pd.DataFrame:
    """Load the large or small universe table from processed project data."""
    if kind not in {"large", "small"}:
        raise ValueError("kind must be 'large' or 'small'")
    return pd.read_csv(data_dir() / "processed" / f"universe_{kind}.csv")

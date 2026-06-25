# Data Layout

This project uses two data layers:

- `data/sample/` is a compact committed 1-minute OHLCV snapshot for smoke tests and quick notebook checks.
- `data/external/` is an ignored local/server directory for the full Binance spot snapshot.

The intended full research bundle is:

```text
data/external/binance_spot_1m_120_12mo/
├── archives/
├── raw/
└── processed/
    ├── prices_1m.parquet
    └── returns_1m.parquet
```

Full data can be moved between the laptop and `beleriand` with `scp` when needed. The processed
bundle is published on Yandex Disk for review, and `data/manifest.json` records its period,
symbol count, row counts, file sizes, checksums, and public URL.

Prepare the compact sample:

```bash
uv run python scripts/prepare_data.py --mode sample --start 2026-06-01 --end 2026-06-08
```

Prepare the full ignored bundle:

```bash
uv run python scripts/prepare_data.py --mode full --large-limit 120 --candidate-limit 250 --target-symbol-count 120 --min-symbol-coverage 0.95 --start 2025-06-22 --end 2026-06-22
```

## Full External Bundle

The committed repository contains only the compact sample data. The full processed 120-pair 1-minute bundle is published separately:

```text
https://disk.yandex.ru/d/Ztu0gLiKMCiiIw
```

Place downloaded files here:

```text
data/external/binance_spot_1m_120_12mo/processed/
```

Expected files:

- `prices_1m.parquet`
- `returns_1m.parquet`

Checksums are recorded in `data/manifest.json`.

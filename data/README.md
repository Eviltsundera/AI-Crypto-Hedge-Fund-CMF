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

Full data should be moved between the laptop and `beleriand` with `scp` and can be uploaded to
external object storage such as Yandex Disk for final review. The full bundle must be documented
in `data/manifest.json` with period, symbol count, row counts, file size, and checksum.

Prepare the compact sample:

```bash
uv run python scripts/prepare_data.py --mode sample --start 2026-06-01 --end 2026-06-08
```

Prepare the full ignored bundle:

```bash
uv run python scripts/prepare_data.py --mode full --large-limit 120 --start 2025-06-22 --end 2026-06-22
```

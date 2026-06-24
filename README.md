# AI Crypto Hedge Fund CMF

Reproducible MVP for an AI agent-based automated cryptocurrency trading and risk management system.

The project is structured around the assignment deliverables:

- concept presentation for an AI crypto hedge fund;
- reproducible technical implementation in one final notebook;
- modular Python package for data preparation, metrics, backtesting, models, agents, and portfolio logic;
- included data snapshot for offline reproduction;
- tests and QA notes for reproducibility.

## Quickstart

Preferred setup when `uv` is available:

```bash
uv sync
uv run pytest
```

Fallback setup when `uv` is not installed:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifest.json
├── docs/
├── notebooks/
│   └── final_solution.ipynb
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── slides/
├── src/
│   └── ai_crypto_hedge_fund/
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Reproducibility Notes

- All runtime paths should be repository-relative.
- Final notebook must run top-to-bottom from a clean kernel.
- Required market data snapshots should be stored under `data/` and documented in `data/manifest.json`.
- The repository keeps a compact 1-minute sample under `data/sample/` for smoke tests.
- The full 1-minute 120-pair research snapshot is stored under ignored `data/external/` on `beleriand`; publishing it to external storage is intentionally deferred until final packaging.
- Shared metrics and backtesting utilities live in `ai_crypto_hedge_fund.metrics` and `ai_crypto_hedge_fund.backtest`; see `docs/backtesting.md`.
- `internal_docs/` and `AGENTS.md` are intentionally ignored because they contain local planning and agent instructions.

## Remote Experiments

Heavy experiments can be run on the `beleriand` server. Use exactly:

```bash
/mnt/c/Windows/System32/OpenSSH/ssh.exe beleriand
```

Copy only reproducible outputs needed for the final notebook back into the repository.

Prepare the full ignored data bundle on `beleriand`:

```bash
uv sync
uv run python scripts/prepare_data.py --mode full --large-limit 120 --candidate-limit 250 --target-symbol-count 120 --min-symbol-coverage 0.95 --start 2025-06-22 --end 2026-06-22
```

Prepare or refresh the compact committed sample:

```bash
uv run python scripts/prepare_data.py --mode sample --start 2026-06-01 --end 2026-06-08
```

Generate the single-asset BTCUSDT baseline report:

```bash
uv run python scripts/run_baseline_strategy.py --snapshot auto
```

Generate the single-asset econometric, ML, and agent comparison:

```bash
uv run python scripts/run_single_asset_models.py --snapshot auto
```

Generate the static multi-asset portfolio comparison:

```bash
uv run python scripts/run_static_portfolio.py --snapshot auto
```

Generate the dynamic portfolio rebalancing comparison:

```bash
uv run python scripts/run_dynamic_rebalancing.py --snapshot auto
```

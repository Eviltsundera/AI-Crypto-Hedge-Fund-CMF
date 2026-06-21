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
- `internal_docs/` and `AGENTS.md` are intentionally ignored because they contain local planning and agent instructions.

## Remote Experiments

Heavy experiments can be run on the `beleriand` server. Use exactly:

```bash
/mnt/c/Windows/System32/OpenSSH/ssh.exe beleriand
```

Copy only reproducible outputs needed for the final notebook back into the repository.

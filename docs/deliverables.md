# Deliverables

This file lists the final submission artifacts and how to use them.

## Primary Deliverables

| Deliverable | Path | Purpose |
|---|---|---|
| Final technical notebook | `notebooks/final_solution.ipynb` | Single self-contained Part 2 solution |
| Concept presentation | `reports/slides/presentation_outline.md` | Part 1 presentation deck |
| Technical implementation presentation | `reports/slides/technical_implementation.pdf` | Part 2 results deck |
| Technical presentation source | `reports/slides/technical_implementation.typ` | Typst source for the Part 2 deck |
| Technical speaker notes | `reports/slides/technical_implementation_speaker_notes.md` | Speech script for the Part 2 deck |
| Public README | `README.md` | Setup, reproduction, report generation |
| QA report | `docs/qa_report.md` | Final verification results and known gaps |
| Submission checklist | `docs/submission_checklist.md` | Reviewer-facing map and commands |
| Data manifest | `data/manifest.json` | Sample/full data metadata, checksums, public URL |

## Data Deliverables

| Data | Path or URL | Notes |
|---|---|---|
| Committed sample prices | `data/sample/prices_1m.parquet` | Git-included smoke data |
| Committed sample returns | `data/sample/returns_1m.parquet` | Git-included smoke data |
| Small universe | `data/processed/universe_small.csv` | Six-asset portfolio universe |
| Large universe | `data/processed/universe_large.csv` | 120 selected USDT spot pairs |
| Full processed bundle | `https://disk.yandex.ru/d/Ztu0gLiKMCiiIw` | External full data; checksums in manifest |

## Report Artifacts

| Report | Metrics | Figure |
|---|---|---|
| Single-asset baseline | `reports/metrics/baseline_metrics.json` | `reports/figures/baseline_equity_curve.png` |
| Single-asset model comparison | `reports/metrics/single_asset_model_comparison.json` | `reports/figures/single_asset_model_comparison.png` |
| Validation-tuned ML | `reports/metrics/single_asset_validation_model.json` | `reports/figures/single_asset_validation_model.png` |
| Cost-aware boosting | `reports/metrics/single_asset_cost_aware_boosting.json` | `reports/figures/single_asset_cost_aware_boosting.png` |
| Static portfolio | `reports/metrics/static_portfolio_metrics.json` | `reports/figures/static_portfolio_equity_curve.png`; `reports/figures/static_portfolio_weights.png` |
| Dynamic rebalancing | `reports/metrics/rebalancing_metrics.json` | `reports/figures/rebalancing_comparison.png` |
| Large universe | `reports/metrics/large_universe_metrics.json`; `reports/metrics/large_universe_selected_assets.csv` | `reports/figures/large_universe_equity_curve.png` |

## Reproduction Commands

```bash
uv sync --locked
uv run pytest
DATA_MODE=sample uv run jupyter nbconvert --to notebook --execute notebooks/final_solution.ipynb --output /tmp/final_solution_qa.ipynb --ExecutePreprocessor.timeout=300
typst compile --root . reports/slides/technical_implementation.typ reports/slides/technical_implementation.pdf
```

## Full-Data Placement

After downloading the Yandex Disk bundle, place processed files here:

```text
data/external/binance_spot_1m_120_12mo/processed/
```

Expected files:

- `prices_1m.parquet`
- `returns_1m.parquet`

The notebook and scripts can then use `--snapshot full` or `DATA_MODE=full`.

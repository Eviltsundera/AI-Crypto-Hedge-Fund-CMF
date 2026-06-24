# Submission Checklist

Use this file as the reviewer-facing map for the final submission.

## Primary Artifacts

- Final notebook: `notebooks/final_solution.ipynb`
- Presentation: `reports/slides/presentation_outline.md`
- QA report: `docs/qa_report.md`
- Main project README: `README.md`
- Full data manifest and checksums: `data/manifest.json`

## Reproduce from Git-Only Data

```bash
uv sync --locked
uv run pytest
DATA_MODE=sample uv run jupyter nbconvert --to notebook --execute notebooks/final_solution.ipynb --output /tmp/final_solution_qa.ipynb --ExecutePreprocessor.timeout=300
```

## Regenerate Reports

```bash
uv run python scripts/run_baseline_strategy.py --snapshot auto
uv run python scripts/run_single_asset_models.py --snapshot auto
uv run python scripts/run_validation_model.py --snapshot auto
uv run python scripts/run_static_portfolio.py --snapshot auto
uv run python scripts/run_dynamic_rebalancing.py --snapshot auto
uv run python scripts/run_large_universe.py --snapshot auto
```

## Full Data Bundle

The full 1-minute 120-pair data bundle is not committed. It remains deferred to Task 13 for external publication.

Current server path:

```text
/home/gajnanovda/AI-Crypto-Hedge-Fund-CMF/data/external/binance_spot_1m_120_12mo/processed
```

Checksums are documented in `data/manifest.json`, `docs/large_universe.md`, and `docs/qa_report.md`.

## Expected Final Status

- Tests pass.
- Notebook executes in `DATA_MODE=sample`.
- All report JSON files validate.
- `AGENTS.md` and `internal_docs/` remain ignored.
- Local, `beleriand`, and GitHub `origin/main` point to the same commit after the final push.

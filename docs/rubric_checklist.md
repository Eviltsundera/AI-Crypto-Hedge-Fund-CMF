# Rubric Checklist

This checklist maps the assignment requirements to committed project artifacts.

## Part 1: Concept Presentation

| Requirement | Status | Evidence |
|---|---|---|
| Hedge fund model | Complete | `reports/slides/presentation_outline.md`, slide 1 |
| Risk management | Complete | `reports/slides/presentation_outline.md`, slide 3; `docs/backtesting.md` |
| Portfolio management | Complete | `reports/slides/presentation_outline.md`, slide 4; `docs/static_portfolio.md`; `docs/dynamic_rebalancing.md`; `docs/large_universe.md` |
| System architecture | Complete | `reports/slides/presentation_outline.md`, slide 2 |
| Future/live-trading scope separated from MVP | Complete | `reports/slides/presentation_outline.md`; `docs/qa_report.md` |

## Part 2: Technical Implementation

| Level | Requirement | Status | Evidence |
|---|---|---|---|
| 2.1 | Baseline strategy for one cryptocurrency | Complete | `scripts/run_baseline_strategy.py`; `docs/baseline_strategy.md`; `reports/metrics/baseline_metrics.json` |
| 2.2 | Econometric, ML, and AI-agent strategy comparison | Complete | `scripts/run_single_asset_models.py`; `src/ai_crypto_hedge_fund/models/`; `src/ai_crypto_hedge_fund/agents/`; `docs/single_asset_models.md` |
| 2.2 plus | Improved ML experiments | Complete | `scripts/run_validation_model.py`; `scripts/run_cost_aware_boosting.py`; `reports/metrics/single_asset_cost_aware_boosting.json` |
| 2.3 | Static portfolio management for 5-7 coins | Complete | `scripts/run_static_portfolio.py`; `docs/static_portfolio.md`; `reports/metrics/static_portfolio_metrics.json` |
| 2.4 | Dynamic portfolio rebalancing | Complete | `scripts/run_dynamic_rebalancing.py`; `docs/dynamic_rebalancing.md`; `reports/metrics/rebalancing_metrics.json` |
| 2.5 | Expansion to 100+ cryptocurrency pairs | Complete | `scripts/run_large_universe.py`; `docs/large_universe.md`; `reports/metrics/large_universe_metrics.json` |
| Part 2 presentation | Results presentation aligned to technical levels | Complete | `reports/slides/technical_implementation.pdf`; `reports/slides/technical_implementation_speaker_notes.md` |

## Reproducibility and Submission Requirements

| Requirement | Status | Evidence |
|---|---|---|
| One self-contained notebook | Complete | `notebooks/final_solution.ipynb` |
| Notebook runs top-to-bottom | Complete | `docs/qa_report.md` |
| Included data for offline smoke run | Complete | `data/sample/` |
| Full data bundle documented | Complete | `data/manifest.json`; `docs/large_universe.md`; `docs/qa_report.md` |
| Full data bundle externally available | Complete | Yandex Disk URL in `data/manifest.json` |
| Modular code, not only notebook code | Complete | `src/ai_crypto_hedge_fund/` |
| Out-of-sample validation | Complete | shared train/test split in scripts and reports |
| Performance metrics | Complete | `src/ai_crypto_hedge_fund/metrics.py`; `reports/metrics/` |
| Risk metrics | Complete | Sharpe, Sortino, Calmar, max drawdown, VaR/CVaR in `reports/metrics/` |
| Visualizations | Complete | `reports/figures/` |
| Public repository readiness | Complete | `docs/qa_report.md`; `docs/submission_checklist.md` |
| No mandatory network call for smoke run | Complete | `DATA_MODE=sample` notebook execution |

## Optional or Future Scope

| Item | Status | Rationale |
|---|---|---|
| Telegram integration | Future scope | Not required for MVP; monitoring and alerts are described in docs/presentation |
| Live trading | Future scope | Deliberately excluded; execution is simulated only |
| Paper trading | Future scope | Roadmap item after reproducible research MVP |
| Production order execution | Future scope | Requires exchange-specific slippage, limits, custody, and failure handling |

## Final Acceptance Checks

- `uv sync --locked`
- `uv run pytest`
- `DATA_MODE=sample uv run jupyter nbconvert --to notebook --execute notebooks/final_solution.ipynb --output /tmp/final_solution_qa.ipynb --ExecutePreprocessor.timeout=300`
- `typst compile --root . reports/slides/technical_implementation.typ reports/slides/technical_implementation.pdf`
- Validate `data/manifest.json` and all `reports/metrics/*.json`.
- Verify `AGENTS.md`, `internal_docs/`, and `data/external/` are ignored.

# Quality Assurance Report

QA date: 2026-06-25.

Primary execution environment: `beleriand`.

## Final Checks

| Check | Result |
|---|---|
| Dependency sync | Passed: `uv sync --locked` |
| Test suite | Passed: `uv run pytest`, 26 tests passed |
| Final notebook execution | Passed: `DATA_MODE=sample uv run jupyter nbconvert --to notebook --execute notebooks/final_solution.ipynb --output /tmp/final_solution_qa.ipynb --ExecutePreprocessor.timeout=300` |
| Sample data load | Passed: prices `(10080, 6)`, returns `(10079, 6)` |
| JSON validation | Passed: `data/manifest.json`, notebook JSON, and all metric JSON files |
| Presentation artifact | Present: `reports/slides/presentation_outline.md` |
| Full data documentation | Present in `data/manifest.json` and `docs/large_universe.md` |
| External data publication | Deferred to Task 13 |

## Included Artifacts

- Final notebook: `notebooks/final_solution.ipynb`
- Presentation outline: `reports/slides/presentation_outline.md`
- Public docs under `docs/`
- Committed sample data under `data/sample/`
- Full-run reports under `reports/metrics/`
- Full-run figures under `reports/figures/`

## Full Data Bundle

The full 1-minute 120-pair bundle is intentionally not committed to git. It is stored on `beleriand` under:

```text
/home/gajnanovda/AI-Crypto-Hedge-Fund-CMF/data/external/binance_spot_1m_120_12mo/processed
```

Checksums:

| File | Size Bytes | SHA256 |
|---|---:|---|
| `prices_1m.parquet` | 95258376 | `027a8a0ba6841e04e39513f8262a5d451bcb27369bb07a308e6059d617ef9dff` |
| `returns_1m.parquet` | 193142203 | `2c4b8ed154f959fe7cb598524b214fcb423acab61ef3ae3169a4c717af6fe491` |

The public URL remains `null` until the final external publication task.

## Known Gaps

- No live trading integration is included.
- Execution modeling uses simplified minute-close fills and transaction costs.
- Full external data must be supplied separately for authoritative reruns.
- The sparse large-universe momentum signal is an MVP baseline, not production alpha research.

## Reproduction Commands

```bash
uv sync --locked
uv run pytest
DATA_MODE=sample uv run jupyter nbconvert --to notebook --execute notebooks/final_solution.ipynb --output /tmp/final_solution_qa.ipynb --ExecutePreprocessor.timeout=300
```

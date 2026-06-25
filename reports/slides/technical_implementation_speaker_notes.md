# Part 2 Technical Implementation: Speaker Notes

These notes accompany `reports/slides/technical_implementation.typ`.

## Slide 1: Title

This presentation covers Part 2 of the assignment: the working technical implementation of the AI crypto hedge fund MVP. The implementation is organized around the five required levels: a single-asset baseline, single-asset econometric and ML models with an AI agent layer, a static small portfolio, a dynamically rebalanced small portfolio, and a large 100-plus-pair portfolio.

The main deliverable is a single reproducible notebook backed by modular Python code. The presentation focuses on what was implemented, how it was validated, and what the out-of-sample results show.

## Slide 2: Requirements Coverage

The codebase maps directly to the assignment structure. Level 2.1 is implemented by the baseline BTCUSDT moving-average strategy. Level 2.2 compares econometric, machine learning, and agent-enhanced strategies on the same BTCUSDT test set. Level 2.3 implements a six-asset static portfolio. Level 2.4 adds weekly and threshold-based dynamic rebalancing. Level 2.5 expands the system to 120 USDT spot pairs.

The implementation is modular under `src/ai_crypto_hedge_fund`, uses `uv` for reproducibility, includes tests, and stores metrics and figures as committed report artifacts.

## Slide 3: Data Preparation and Validation

The data pipeline uses timestamped Binance spot minute bars. There is a compact committed sample dataset for smoke tests, and the full processed 120-pair bundle is documented with checksums and an external URL.

All experiments use a chronological train/test split: roughly 70 percent train and 30 percent test. There is no random shuffling, and trading signals are shifted before returns are applied. This is important because otherwise a backtest can accidentally use future information.

The reviewer can reproduce the smoke run from git-only data and can reproduce the full run by placing the documented external bundle under `data/external`.

## Slide 4: Level 2.1 Baseline Strategy

The baseline strategy is a moving-average crossover on BTCUSDT. It uses a 60-minute fast window and a 360-minute slow window. This is intentionally simple because the baseline should be interpretable and easy to challenge.

The result is negative: buy-and-hold returned -11.26 percent in the test window, while the moving-average crossover returned -40.40 percent after transaction costs. This tells us that a classical indicator is not automatically useful at one-minute frequency. It is a baseline, not a production strategy.

## Slide 5: Level 2.2 Model and Agent Comparison

For the single-asset model comparison, every strategy uses the same backtest engine and cost model. The features include lagged returns, rolling mean and volatility, momentum, and moving-average distance.

The econometric rolling strategy and moving-average strategy overtraded and performed poorly. RandomForest performed much better on drawdown, with -6.84 percent max drawdown compared with -28.59 percent for buy-and-hold. The deterministic agent combines baseline, econometric, and ML votes with risk gates. It did not beat RandomForest in this window, but it demonstrates the required agent-based interaction.

## Slide 6: Targets, Training, and Robustness

The first ML target is next-period direction. To improve the target, the project also tests a 60-minute cost-aware target: the model predicts whether the future return is large enough to clear a transaction-cost buffer, and neutral rows are dropped.

The cost-aware boosting model returned -6.00 percent and had -8.06 percent max drawdown. This is still not production alpha, but it improved return versus buy-and-hold and materially reduced drawdown. The validation-tuned random forest did not improve results, which is also useful evidence because it shows the evaluation was not just cherry-picking.

The robustness controls are out-of-sample scoring, validation-threshold selection inside the train period, and explicit transaction costs.

## Slide 7: Level 2.3 Static Portfolio

The static portfolio level uses six popular coins: BTC, ETH, BNB, SOL, XRP, and ADA. The methods are equal weight, inverse volatility, and constrained max-Sharpe.

All methods had negative returns in the test window, but constrained max-Sharpe was best by Sharpe ratio and drawdown. It selected ETH at 35 percent, BNB at 35 percent, and XRP at 30 percent, respecting the 35 percent maximum asset weight cap.

The important point is that weights are fitted on train data and evaluated out of sample, rather than optimized directly on the test set.

## Slide 8: Level 2.4 Dynamic Rebalancing

Dynamic rebalancing tests whether an adaptive policy improves the static reference. The weekly inverse-volatility policy rebalanced 16 times. The threshold inverse-volatility policy rebalanced 5 times when passive weight drift exceeded two percentage points.

Neither dynamic policy beat the static max-Sharpe reference in this test period. This is a valid result: dynamic rebalancing adds turnover and should only be used when it improves after-cost risk-adjusted performance.

The selection criterion is highest out-of-sample Sharpe after costs, so the static reference remains selected.

## Slide 9: Level 2.5 Large Universe

The large-universe level expands the system to 120 Binance USDT spot pairs. Pair selection is based on coverage and liquidity. The large-universe strategies compare broad equal weighting with sparse weekly top-momentum and risk-adjusted momentum selection.

The 120-pair equal-weight portfolio returned 14.39 percent with a Sharpe ratio of 1.18. The sparse momentum methods were weaker in this window. This suggests that broad diversification was more reliable than simple momentum timing during the tested period.

Risk controls include active-count limits, an 8 percent maximum asset weight, and gross exposure checks.

## Slide 10: Backtesting and Risk Metrics

All strategies use the same evaluation contract: chronological splits, one-period signal lag, turnover-based transaction costs, and long-only exposure in the MVP.

The metrics include total and annualized return, annualized volatility, Sharpe, Sortino, Calmar, maximum drawdown, VaR, CVaR, hit rate, turnover, and concentration metrics. This matters because high-frequency signals can look good before costs but fail once turnover is charged.

The common backtest layer makes the five assignment levels comparable.

## Slide 11: Result Interpretation

The strongest result is not a single-asset alpha model. The strongest result is the complete reproducible pipeline and the evidence that broad diversification worked better than sparse momentum in this window.

RandomForest and cost-aware boosting reduced BTC drawdown versus buy-and-hold. The six-asset constrained max-Sharpe portfolio improved versus equal weight in a bad market. The 120-pair equal-weight portfolio produced the best overall result.

The weak areas are also clear: one-minute single-asset alpha remains noisy, simple moving averages and econometric signals overtrade, and dynamic rebalancing did not beat the static reference.

## Slide 12: Reviewer Reproduction Map

The final submission is reproducible through the notebook, modular package, tests, metrics, figures, and data manifest. The smoke run works from committed sample data. The authoritative full-data run requires the documented external 120-pair bundle.

The key commands are `uv sync --locked`, `uv run pytest`, and notebook execution with `DATA_MODE=sample`. The conclusion is that the system satisfies the technical assignment as a research MVP. Live execution is intentionally left as future scope.

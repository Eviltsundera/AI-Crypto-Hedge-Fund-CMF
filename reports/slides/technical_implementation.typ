#set page(width: 16in, height: 9in, margin: 0.55in)
#set text(font: "DejaVu Sans", size: 15.5pt, fill: rgb("#172026"))
#set par(justify: false, leading: 0.56em)

#let navy = rgb("#172026")
#let teal = rgb("#007f7a")
#let gray = rgb("#5f6b72")
#let pale = rgb("#eef6f3")
#let soft = rgb("#f8fbfa")
#let rule = rgb("#d8e2df")
#let amber = rgb("#b97918")
#let red = rgb("#b54945")
#let green = rgb("#3f8f46")

#let title-block(title, subtitle: none) = {
  text(size: 30pt, weight: "bold", fill: navy)[#title]
  if subtitle != none {
    v(0.06in)
    text(size: 13pt, fill: gray)[#subtitle]
  }
  v(0.12in)
  line(length: 100%, stroke: 1.1pt + rule)
  v(0.18in)
}

#let slide(title, subtitle: none, do-break: true, body) = {
  title-block(title, subtitle: subtitle)
  body
  place(bottom + right)[#text(size: 10pt, fill: gray)[AI Crypto Hedge Fund CMF - Part 2]]
  if do-break { pagebreak() }
}

#let card(title, body, fill: soft) = box(
  width: 100%,
  inset: 0.15in,
  radius: 4pt,
  stroke: 0.8pt + rule,
  fill: fill,
)[
  #text(size: 14pt, weight: "bold", fill: navy)[#title]
  #v(0.06in)
  #text(size: 12pt, fill: gray)[#body]
]

#let metric(label, value, tone: teal) = box(
  width: 100%,
  inset: 0.12in,
  radius: 4pt,
  stroke: 0.8pt + rule,
  fill: rgb("#ffffff"),
)[
  #text(size: 10.5pt, fill: gray)[#label]
  #v(0.03in)
  #text(size: 18pt, weight: "bold", fill: tone)[#value]
]

#let small-table(body) = text(size: 12pt)[#body]

#align(center + horizon)[
  #text(size: 40pt, weight: "bold", fill: navy)[Part 2: Technical Implementation]
  #v(0.14in)
  #text(size: 19pt, fill: gray)[AI Crypto Hedge Fund MVP: data, models, backtests, portfolios]
  #v(0.30in)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    gutter: 0.12in,
    metric("2.1", "Baseline"),
    metric("2.2", "Models + agents"),
    metric("2.3", "Static portfolio"),
    metric("2.4", "Dynamic rebalance"),
    metric("2.5", "120 pairs"),
  )
  #v(0.34in)
  #box(width: 84%, inset: 0.20in, radius: 5pt, stroke: 0.8pt + teal, fill: pale)[
    #text(size: 16pt, fill: navy)[
      One reproducible notebook and modular Python package covering data preparation,
      model validation, strategy backtesting, result visualization, and explanation.
    ]
  ]
]
#pagebreak()

#slide(
  "1. Requirements Coverage",
  subtitle: "Implementation is structured exactly by the five assignment levels",
)[
  #small-table[
    #table(
      columns: (0.55fr, 1.55fr, 1.7fr),
      inset: 7pt,
      align: (left, left, left),
      [*Level*], [*Requirement*], [*Committed evidence*],
      [2.1], [Single-crypto baseline strategy], [`scripts/run_baseline_strategy.py`, `baseline_metrics.json`],
      [2.2], [Econometric, ML, and AI-agent single-pair comparison], [`run_single_asset_models.py`, model and agent modules],
      [2.3], [5-7 coin static portfolio on historical data], [`run_static_portfolio.py`, six-asset universe],
      [2.4], [Dynamic rebalancing for small portfolio], [`run_dynamic_rebalancing.py`, rebalance events],
      [2.5], [100+ pair dynamic portfolio expansion], [`run_large_universe.py`, 120-pair universe],
    )
  ]
  #v(0.20in)
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.16in,
    card("Reproducible", [`uv`, locked dependencies, smoke data, final notebook]),
    card("Modular", [`src/ai_crypto_hedge_fund/` contains reusable code]),
    card("Validated", [Chronological train/test split, no shuffling, out-of-sample metrics]),
  )
]

#slide(
  "2. Data Preparation and Validation",
  subtitle: "All experiments share the same timestamped minute-bar data contract",
)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.24in,
    [
      #card("Data snapshots", [
        Committed sample data supports offline smoke tests. The full 1-minute 120-pair
        processed bundle is documented by manifest, checksums, and external URL.
      ])
      #v(0.16in)
      #card("Universe construction", [
        Small universe: BTC, ETH, BNB, SOL, XRP, ADA. Large universe: 120 Binance USDT
        spot pairs selected by coverage and liquidity filters.
      ])
    ],
    [
      #card("Model validation", [
        Time series split: 70% train, 30% test. No random shuffle. Signals are shifted
        one period before returns are applied.
      ])
      #v(0.16in)
      #card("Reproducibility checks", [
        `uv run pytest`, notebook execution with `DATA_MODE=sample`, JSON validation
        for manifest, notebook, and all metric reports.
      ])
    ],
  )
  #v(0.20in)
  #box(width: 100%, inset: 0.15in, radius: 4pt, fill: rgb("#fff8ec"), stroke: 0.8pt + amber)[
    #text(size: 13pt, fill: navy)[Data is not fetched implicitly during review: smoke runs use committed data; full runs use the documented external bundle.]
  ]
]

#slide(
  "3. Level 2.1: Single-Crypto Baseline",
  subtitle: "BTCUSDT moving-average crossover is the benchmark strategy",
)[
  #grid(
    columns: (0.92fr, 1.08fr),
    gutter: 0.22in,
    [
      #small-table[
        #table(
          columns: (1.25fr, 0.75fr, 0.75fr, 0.75fr),
          inset: 7pt,
          align: (left, right, right, right),
          [*Strategy*], [*Return*], [*Sharpe*], [*Max DD*],
          [Buy and hold], [-11.26%], [-0.72], [-28.59%],
          [MA crossover], [-40.40%], [-5.81], [-40.91%],
        )
      ]
      #v(0.16in)
      - Fast window: 60 minutes; slow window: 360 minutes.
      - Transaction cost: 5 bps per turnover.
      - Evaluation period: out-of-sample test only.
      - Result: baseline is simple and interpretable, but not profitable after costs.
    ],
    image("../figures/baseline_equity_curve.png", width: 100%, height: 5.1in, fit: "contain"),
  )
]

#slide(
  "4. Level 2.2: Econometric, ML, and Agent Strategy",
  subtitle: "Single-pair model comparison uses one shared backtest layer",
)[
  #grid(
    columns: (0.98fr, 1.02fr),
    gutter: 0.22in,
    [
      #small-table[
        #table(
          columns: (1.18fr, 0.72fr, 0.72fr, 0.72fr),
          inset: 6pt,
          align: (left, right, right, right),
          [*Strategy*], [*Return*], [*Sharpe*], [*Max DD*],
          [Buy and hold], [-11.26%], [-0.72], [-28.59%],
          [MA crossover], [-40.40%], [-5.81], [-40.91%],
          [Econometric], [-41.25%], [-7.22], [-42.07%],
          [RandomForest], [-6.13%], [-4.89], [-6.84%],
          [Agent enhanced], [-13.21%], [-4.39], [-13.99%],
        )
      ]
      #v(0.12in)
      #text(size: 12.5pt)[Features: lagged returns, rolling mean/volatility, momentum, MA distance.]
    ],
    image("../figures/single_asset_model_comparison.png", width: 100%, height: 5.1in, fit: "contain"),
  )
]

#slide(
  "5. Level 2.2: Targets, Training, and Robustness",
  subtitle: "The model target and validation design are explicitly documented",
)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.22in,
    [
      #card("Target variable", [
        Initial ML target: next-period return direction. Improvement experiment:
        future 60-minute return greater than a 10 bps cost buffer; neutral rows are dropped.
      ])
      #v(0.16in)
      #card("Training / testing", [
        Models train only before the split timestamp. Validation-tuned models choose
        thresholds inside the train period and evaluate the final test period once.
      ])
    ],
    [
      #small-table[
        #table(
          columns: (1.15fr, 0.72fr, 0.72fr, 0.72fr),
          inset: 6pt,
          align: (left, right, right, right),
          [*Strategy*], [*Return*], [*Sharpe*], [*Max DD*],
          [Buy and hold], [-11.26%], [-0.72], [-28.59%],
          [Cost-aware boosting], [-6.00%], [-2.99], [-8.06%],
          [Validation RF 60m], [-15.24%], [-5.47], [-15.85%],
        )
      ]
      #v(0.14in)
      - Random chance check: out-of-sample-only scoring, validation threshold grid, and cost-aware labels.
      - Retraining policy: periodic walk-forward refresh, not continuous online fitting in MVP.
    ],
  )
]

#slide(
  "6. Level 2.3: Static 6-Asset Portfolio",
  subtitle: "Historical portfolio management on BTC, ETH, BNB, SOL, XRP, ADA",
)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.22in,
    [
      #small-table[
        #table(
          columns: (1.2fr, 0.7fr, 0.7fr, 0.7fr),
          inset: 6pt,
          align: (left, right, right, right),
          [*Method*], [*Return*], [*Sharpe*], [*Max DD*],
          [Equal weight], [-19.41%], [-1.25], [-32.85%],
          [Inverse vol], [-17.28%], [-1.13], [-31.08%],
          [Max-Sharpe], [-14.97%], [-0.96], [-29.31%],
        )
      ]
      #v(0.12in)
      - Selected method: constrained max-Sharpe by out-of-sample Sharpe.
      - Selected weights: ETH 35%, BNB 35%, XRP 30%.
      - Max asset weight cap: 35%.
    ],
    image("../figures/static_portfolio_weights.png", width: 100%, height: 5.0in, fit: "contain"),
  )
]

#slide(
  "7. Level 2.4: Dynamic Rebalancing",
  subtitle: "Weekly and threshold-based policies are tested against the static reference",
)[
  #grid(
    columns: (0.96fr, 1.04fr),
    gutter: 0.22in,
    [
      #small-table[
        #table(
          columns: (1.25fr, 0.7fr, 0.7fr, 0.7fr),
          inset: 6pt,
          align: (left, right, right, right),
          [*Strategy*], [*Return*], [*Sharpe*], [*Events*],
          [Static reference], [-14.97%], [-0.96], [0],
          [Weekly inv-vol], [-18.01%], [-1.19], [16],
          [Threshold inv-vol], [-17.55%], [-1.14], [5],
        )
      ]
      #v(0.12in)
      - Logic: trailing inverse-volatility target weights.
      - Threshold policy trades when passive drift crosses 2 percentage points.
      - Selection criterion: highest out-of-sample Sharpe after costs.
    ],
    image("../figures/rebalancing_comparison.png", width: 100%, height: 5.1in, fit: "contain"),
  )
]

#slide(
  "8. Level 2.5: Large-Universe Portfolio",
  subtitle: "The system scales to 120 USDT spot pairs",
)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.22in,
    [
      #small-table[
        #table(
          columns: (1.2fr, 0.7fr, 0.7fr, 0.7fr),
          inset: 6pt,
          align: (left, right, right, right),
          [*Strategy*], [*Return*], [*Sharpe*], [*Max DD*],
          [120-pair equal weight], [14.39%], [1.18], [-29.32%],
          [Top momentum weekly], [1.49%], [0.41], [-29.80%],
          [Risk-adj. momentum], [-3.78%], [0.12], [-27.83%],
        )
      ]
      #v(0.12in)
      - Pair selection: coverage and liquidity first.
      - Signal priority: top-20 weekly momentum or risk-adjusted momentum.
      - Risk controls: active-count, max weight 8%, gross exposure floor.
    ],
    image("../figures/large_universe_equity_curve.png", width: 100%, height: 5.1in, fit: "contain"),
  )
]

#slide(
  "9. Backtesting and Risk Metrics",
  subtitle: "Every strategy uses the same cost-aware evaluation contract",
)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.24in,
    [
      #text(size: 17pt, weight: "bold")[Backtest controls]
      #v(0.08in)
      - Chronological train/test split
      - No shuffling for time series
      - Signal lag prevents lookahead
      - Turnover-based transaction costs
      - Long-only exposure in MVP
      - Same annualization convention across reports
    ],
    [
      #text(size: 17pt, weight: "bold")[Reported metrics]
      #v(0.08in)
      - Total and annualized return
      - Annualized volatility
      - Sharpe, Sortino, Calmar
      - Max drawdown
      - VaR and CVaR
      - Hit rate, turnover, cost drag
      - Effective assets and concentration
    ],
  )
  #v(0.18in)
  #card("Why this matters", [
    Comparing models without a shared cost and risk layer would overstate noisy high-frequency signals.
    The same evaluation layer makes all five assignment levels comparable.
  ], fill: pale)
]

#slide(
  "10. Result Interpretation",
  subtitle: "The MVP is reproducible; the strongest current result is diversification",
)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.24in,
    [
      #text(size: 17pt, weight: "bold")[What worked]
      #v(0.08in)
      - RandomForest and cost-aware boosting reduced BTC drawdown versus buy-and-hold.
      - Static constrained max-Sharpe improved the six-asset portfolio versus equal weight.
      - 120-pair equal weight achieved positive out-of-sample return and Sharpe.
      - Code, data manifest, reports, figures, and notebook are reproducible.
    ],
    [
      #text(size: 17pt, weight: "bold")[What did not work yet]
      #v(0.08in)
      - One-minute single-asset alpha remained noisy and cost-sensitive.
      - Moving averages and rolling econometric signals overtraded.
      - Dynamic rebalancing did not beat the static reference in the test window.
      - Sparse momentum underperformed broad diversification in the large universe.
    ],
  )
]

#slide(
  "11. Reviewer Reproduction Map",
  subtitle: "Single notebook plus committed reports and data manifest",
  do-break: false,
)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.24in,
    [
      #text(size: 17pt, weight: "bold")[Primary artifacts]
      #v(0.08in)
      - Notebook: `notebooks/final_solution.ipynb`
      - Package: `src/ai_crypto_hedge_fund/`
      - Tests: `tests/`
      - Metrics: `reports/metrics/`
      - Figures: `reports/figures/`
      - Data manifest: `data/manifest.json`
    ],
    [
      #text(size: 17pt, weight: "bold")[Validation commands]
      #v(0.08in)
      #box(inset: 0.14in, radius: 4pt, fill: soft, stroke: 0.8pt + rule)[
        #raw(
          "uv sync --locked\nuv run pytest\nDATA_MODE=sample uv run jupyter nbconvert \\\n  --to notebook --execute notebooks/final_solution.ipynb",
          block: true,
        )
      ]
      #v(0.10in)
      #text(size: 12.5pt, fill: gray)[Full 120-pair data is externally published and checksum-documented.]
    ],
  )
  #v(0.18in)
  #box(width: 100%, inset: 0.16in, radius: 4pt, fill: rgb("#fff8ec"), stroke: 0.8pt + amber)[
    #text(size: 13.5pt, fill: navy)[Conclusion: the submitted system satisfies the technical assignment as a reproducible research MVP; live execution remains future scope.]
  ]
]

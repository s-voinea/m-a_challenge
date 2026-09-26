# Extracting signal from the M&A dataset: method and audit

## 1. Review of `deeptech-ma-signals.ipynb`

Evidence is quoted from the notebook's saved outputs. Each item names the artifact, why it happens and the fix in `ma_signals/`.

| # | Finding (cell) | Why it's a problem | Fix |
|---|---|---|---|
| 1 | Data ends **2025-10-22** (cell 5). Momentum "trough 0.01 (2025-10)" (cell 30). Every 2019→2025 CAGR (cell 39) uses a 2025 that covers ~9.7 months. | Partial periods show up as crashes. The momentum trough and most of the "↓" CAGRs are at least partly calendar artifacts. | `drop_partial_tail`. The last 3 months are flagged `provisional`. |
| 2 | 2025 has 2,138 deals vs 4,241 in 2024 (cell 17). Annualised, 2025 is about 2,650, still −38%. | This mixes a real slowdown with database backfill: deals get entered after their announcement date. The export has no entry date, so the two can't be separated. | Recent months are flagged, and signals are *relative* (sector share of the market), so a uniform backfill lag largely cancels out. |
| 3 | "Intra-sector" 94–97% for every sector (cell 35). | Buyer sector = the sector of the buyer's own deals, so the diagonal is high **by construction**. It says nothing about consolidation. | `buyer_home_sector` leaves the current deal out. |
| 4 | Keyword classifier: first match wins in dict order, the buyer name is included in the text, and `property`, `\bgas\b` and `pipeline` are unguarded. Real Estate = 20.7% of deals, 44.4% unclassified (cell 22). Cybersecurity's top country is Bermuda (81%) and one deal = 78% of sector value (cells 41, 46). | "Intellectual property", "property & casualty" (insurers), "drug pipeline" and buyer names like "… Real Estate Partners" all mislabel deals. A Bermuda-dominated cybersecurity sector looks like insurance/reinsurance leakage (inference, not verified). | Scored rules, target-only text, negative phrases, and a margin requirement, followed by the weak-supervised model. |
| 5 | Sector correlations on quarterly **levels** (cell 37). | Trending series correlate spuriously. With about 31 quarters, \|r\| < 0.36 is not distinguishable from 0 at 5%. | `flow_correlation` uses monthly changes in log counts. |
| 6 | "Capital corridors" run seller country → target country (cell 29). | Capital flows buyer → seller. The seller is usually in the target's own country, so this mostly measures noise in how CIQ records seller countries. | Buyer country proxy from the listing exchange. `cross_border` is computed for listed buyers only, and that is stated. |
| 7 | Seasonality heatmap uses **mean deal value** per month (cell 43). Most sector charts use summed value. | Dominated by a few mega-deals (cell 46: top-5 deals are 25–92% of sector value). | Counts first, and value winsorised at p99. |
| 8 | Statuses are only Closed / Announced / Effective (cell 10). | Withdrawn deals are absent, so "completion rate" (cell 44) measures deal recency, not closing risk. | Documented. The completion analysis is not reused. |
| 9 | Gini on buyers (cell 31). | The sample is truncated at $50M, so the Gini describes concentration *among large deals* only. | Documented. |

## 2. Method

Runtime on 41k synthetic deals padded to realistic comment length: **35 s** end to end with `google-re2`, including 5-fold cross-validation of the classifier. With only stdlib `re` it took 393 s: backtracking through ~150 alternated keyword patterns dominates.

### 2.1 Features parsed from `comments`
- consideration (cash / stock / mixed)
- premium %
- stake % and minority flag
- tender offer, hostile, competing bid, take-private, SPAC, non-binding/LOI
- debt financing, antitrust and national-security review
- termination fee as a fraction of deal value
- financial-sponsor buyer, consortium
- bid-to-announcement lag
- buyer listing country

Why these carry signal (model-based; these are hypotheses to test, not established results for this dataset):
- A high **stock-consideration share** is associated with overvalued acquirer equity (Shleifer & Vishny 2003, "Stock market driven acquisitions").
- **Premium** and **contested share** measure how aggressively buyers compete for a sector.
- **Sponsor share** tracks private-capital deployment into the sector.

### 2.2 Sector classification (weak supervision + optional LLM on ~1% of rows)
1. Scored rules over the target name (weight 2) and comments (weight 1). The buyer name is excluded, and a label needs a margin of at least 1 over the runner-up.
2. TF-IDF (1–2 grams) + SGD logistic regression, trained on rule-labelled rows **with seed keywords masked out**. It learns contextual vocabulary and is applied where the rules abstain, with a probability threshold of 0.55.
3. `llm_labeling_queue.csv`: 400 low-confidence rows, spread across 40 TF-IDF/SVD clusters. Label them with any LLM (the prompt is included) or by hand, then re-run with `--gold`. Gold labels get 5× training weight.
   Cost (inference from the notebook's own estimate of ~13.3M tokens for the whole dataset): about 400 × ~450 tokens ≈ 0.2M tokens, i.e. **~1.5%** of labelling everything.
   The same labelled rows are the only honest accuracy estimate available. Out-of-fold agreement with the rules is only an upper bound.

### 2.3 Signals (monthly, per sector, point-in-time)
- **`flow_z`: market-neutral abnormal deal flow.** Expected count = market count × the sector's trailing 24-month share (shrunk). The z-score uses binomial variance × trailing overdispersion φ ≥ 1.
  This removes the M&A cycle, market-wide seasonality and mega-deal noise in one closed-form step.
- **`count_mom` / `value_mom`**: log EWMA(3) − log EWMA(12) on counts and on winsorised value.
- **`sponsor_z`, `stock_z`, `premium_z`, `contested_z`**: 3-month-smoothed composition shares, z-scored against the sector's own trailing 24 months.
- **`heat`**: cross-sectional mean z of flow, momentum, premium and contested share.
- **Quality flags**: `provisional` (last 3 months) and `sparse` (trailing median below 3 deals/month; aggregate these quarterly).

A test (`test_signals_are_point_in_time`) checks that appending future data never changes past signal values.

### 2.4 Validation (not done here: needs price data)
`information_coefficient(sig, returns, horizon)` computes the monthly cross-sectional Spearman IC against forward sector returns. Candidate ETFs are in `SECTOR_ETF_CANDIDATES`. Expect about 90 months × ~12 sectors.

A monthly IC is noisy: its standard error is about 1/√12 ≈ 0.29 per month. So a mean IC must be around 0.09 or higher to reach |t| ≈ 3 over 90 months, before correcting for overlapping horizons (model-based: normal approximation). Also:
- Pre-register which signal and horizon you test.
- Treat any IC below that bar as noise.
- Most ETFs listed above launched before 2018, but check each one's inception date.

## 3. What the synthetic tests do and don't show
The tests run on generated data with planted structure:
- keyword-free context words that the model must learn;
- an injected burst of about 3× AI deals over 4 months.

They show:
- the code is correct;
- the burst is recovered (flow_z ≈ 2–2.7 inside the window, ≈ 0 outside);
- there is no lookahead.

They say **nothing** about how much signal the real dataset contains. That needs the run on `trans50m_2k18.xlsx` plus §2.4.

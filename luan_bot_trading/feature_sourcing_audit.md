# Feature Sourcing Audit for PEAD v2

> **Status**: UPDATED 2026-07-23 — FMP $49/mo plan purchased and
> verified. This audit now reflects confirmed endpoint availability.
>
> **Bottom line**: FMP replaces EODHD (earnings) + adds analyst revision
> history (the #1 PEAD feature). Tiingo stays (FMP has no historical
> prices). Net cost: +$32/mo. See §7 for the confirmed data plan.
>
> Informs Path B of the strategy reset (see `strategy_v2_synthesis.md`
> Doc K update). This audit answers: what PEAD-relevant data can we
> actually source, and at what cost?

---

## 0. The problem (why we need new features)

Doc K (`phase_g_pos_vs_neg_findings.md`) revealed three layered failures
in the current Phase G model:

1. **Feature ceiling**: the 17 Sunday-safe features are 1990s-era PEAD
   predictors (SUE, earnings surprise, price momentum, volatility). Modern
   PEAD literature finds these have weak, decaying predictive power —
   PEAD has been arbitraged down over 30 years. The classifier's VAL AUC
   is barely above 0.5.

2. **Binary label throws away magnitude**: the 3 PEAD gates produce a
   binary `pead_pass` (0/1). The model can't distinguish a +12% gap
   massive-beat PEAD (61-85% precision, +14.68% drift) from a -2% gap
   bounce (6% precision, +0.31% drift). A 3-class softprob is needed.

3. **Entry timing destroys POS alpha**: entering at Open[T+1] means
   buying AFTER the market prices in the earnings reaction. For the
   (+10%, +15%] gap bucket, the gap ate +12.88% of the +14.68% drift —
   we captured only 11% of the actual move.

### The objective mismatch (user's key insight)

The Phase G backtest optimized for **PnL/Sharpe**, which conflates two
different alpha mechanisms: PEAD drift and gap mean-reversion. The system
selected for "whatever makes money," not "whatever captures PEAD." This
is why NEG_only (a mean-reversion trade with 0% PEAD recall) "won" the
backtest — it's mechanically more profitable than post-reaction PEAD
drift when you enter at Open[T+1].

**The v2 backtest objective must include a PEAD-capture metric**, not
just PnL. See §6 below.

---

## 1. Target features (from modern PEAD literature)

The features that *actually* predict modern PEAD, per post-2010 academic
and practitioner literature (Bernard & Thomas 1989 → Bartov et al. 2000
→ Livnat & Petrovits 2009 → Truong 2014 → recent work):

| # | Feature | Why it predicts PEAD | Priority |
|---|---------|---------------------|----------|
| 1 | **Analyst revision momentum** | Upgrades/downgrades around earnings signal analyst expectation shifts; pre-earnings revisions predict post-earnings drift direction | **HIGH** |
| 2 | **Short interest changes** | High short interest + positive surprise = squeeze; short interest trends signal crowd positioning | **HIGH** |
| 3 | **Options implied volatility skew** | Pre-earnings IV crush signals; put/call skew predicts directional bias | **MEDIUM** |
| 4 | **Institutional ownership flow** | 13F filings show smart-money accumulation/distribution; pre-earnings flow predicts drift | **MEDIUM** |
| 5 | **Earnings call transcript NLP** | Forward guidance tone, sentiment, hedge words — not captured by numeric surprise | **MEDIUM** |
| 6 | **Standardized unexpected fundamentals** | Revenue surprise, gross margin surprise (beyond EPS surprise alone) | **MEDIUM** |
| 7 | **Pre-earnings price drift** | "Anchoring" drift into earnings predicts post-earnings continuation | **LOW** (we have rel_ret_*) |
| 8 | **Earnings guidance revision** | Company-issued guidance changes vs prior quarter | **LOW** |

---

## 2. Tiingo audit (current subscription: paid, 10k req/hr)

### 2.1 Endpoints probed

| Endpoint | Status | Result |
|----------|--------|--------|
| `/tiingo/fundamentals/{ticker}/quarterly` | **404** | Not available on our plan (requires a higher tier or the fundamentals-only subscription) |
| `/tiingo/fundamentals/meta` | 200 | Returns 20,167 field descriptions — but `sector`, `industry`, `sicCode` are all "Field not available for free/evaluation" |
| `/tiingo/daily/{ticker}` (meta) | 200 | Returns `ticker, name, description, startDate, endDate, exchangeCode` — **no fundamental data** |
| `/iex/{ticker}` | 200 | Real-time quote only: `open, high, low, last, volume, bidPrice, askPrice` — **no analyst, institutional, or short interest data** |

### 2.2 Verdict: Tiingo is prices-only for our plan

**Tiingo provides NO PEAD-relevant fundamental data on our current plan.**
The `/fundamentals/` endpoint exists but returns 404, and the meta endpoint
shows the fundamental fields are gated behind a higher tier. Our Tiingo
subscription covers:
- Daily OHLCV prices (already used in Phase B) ✅
- PermaTicker identity + metadata (already used in Phase A) ✅
- Real-time IEX quotes (not useful for historical PEAD) ⚠️

**To get Tiingo fundamentals** (analyst estimates, financial statements,
institutional ownership): would need to upgrade to the Tiingo Fundamental
Data plan (~$30-60/mo additional). Not probed yet — the EODHD fundamentals
403 (see §3) suggests fundamentals are a separate paid add-on on both
platforms.

---

## 3. EODHD audit (current subscription: ~$20/mo, earnings calendar only)

### 3.1 Endpoints probed

| Endpoint | Status | Result |
|----------|--------|--------|
| `/api/fundamentals/{ticker}` | **403 Forbidden** | Fundamentals not in our plan |
| `/api/calendar/analyst` | **422** | "Type not found. We support only 'ipos', 'earnings', 'trends' and 'splits'" — no analyst estimates calendar |
| `/api/options/{ticker}` | **404** | Options data not available (may need different ticker format or higher plan) |
| `/api/insider-transactions/{ticker}` | **404** | Returns HTML 404 page — endpoint may be deprecated or plan-gated |
| `/api/bulk-summary/eod` | **404** | Not available on our plan |

### 3.2 Verdict: EODHD is earnings-calendar-only for our plan

**EODHD provides NO PEAD-relevant data beyond what we already use.** Our
plan is limited to:
- `/api/calendar/earnings` (already used in Phase D) ✅
- End-of-day prices (we use Tiingo instead — better quality) ⚠️

To get EODHD fundamentals, options, or insider transactions would require
upgrading to the EODHD "Fundamentals" or "All World" plan (~$60-100/mo).
The 403 on `/api/fundamentals/` confirms our plan doesn't include it.

---

## 4. Free alternatives audit

### 4.1 SEC EDGAR (FREE, no API key, unlimited)

SEC EDGAR is the most promising free source. Two high-value endpoints
were confirmed working:

#### 4.1.1 `/data/sec/submissions/CIK{cik}.json` — company filings index

- **Status**: 200 ✅
- **What it returns**: full filing history per company, including Form 4
  (insider transactions), 10-Q/10-K (financial statements), 8-K (earnings
  releases), SCHEDULE 13G (institutional ownership).
- **PEAD relevance**: Form 4 insider transactions (589 recent filings for
  AAPL alone) give us **insider buying/selling around earnings** — a known
  PEAD predictor. Accession numbers are provided for XML download.
- **Limitation**: this endpoint shows the company's OWN filings. For
  institutional OWNERSHIP of a company (13F), we need to query the
  institutions' filings and filter — more complex (see 4.1.3).

#### 4.1.2 `/data/sec/api/xbrl/companyfacts/CIK{cik}.json` — all XBRL financials

- **Status**: 200 ✅
- **What it returns**: every US-GAAP financial concept the company has
  ever reported, with full historical time series. For AAPL: **503
  us-gaap concepts**, all with quarterly history.
- **Confirmed available**: `EarningsPerShareBasic`, `Revenues`,
  `NetIncomeLoss`, `LongTermDebt`, `Assets`, `Liabilities`,
  `CommonStockSharesOutstanding`.
- **PEAD relevance**: enables **revenue surprise** and **gross margin
  surprise** features (beyond EPS surprise alone). Also enables balance
  sheet features (debt levels, asset turnover) that predict earnings
  quality.
- **CIK mapping needed**: we already have CIKs in
  `/metadata/sp400_permatickers` (from Phase A). The SEC EDGAR endpoints
  use `CIK{10-digit-padded}` format.

#### 4.1.3 13F institutional ownership (bulk)

- **Status**: bulk data exists at `sec.gov/dera/data/form-13f` ✅
- **What it requires**: downloading quarterly 13F-HR filings from all
  institutions (~5000 filers) and parsing the XML for each S&P 400
  constituent. This is a **large bulk fetch** (~5000 XML files per
  quarter, ~20KB each) but free.
- **PEAD relevance**: institutional ownership changes around earnings
  predict drift. But 13Fs are filed quarterly with a 45-day lag, so
  real-time flow isn't available — only quarterly snapshots.
- **Effort**: HIGH. Requires building a 13F parser + bulk downloader.

### 4.2 yfinance (FREE, unofficial Yahoo Finance API)

- **Status**: all endpoints returned **429 Too Many Requests** ⚠️
- yfinance uses Yahoo's unofficial API, which has aggressive rate limiting.
  Even with a 5-second delay, all calls failed.
- **Theoretically provides**: analyst recommendations, analyst price
  targets, earnings dates with estimates, institutional holders, short
  interest (via `info` dict), major holders.
- **Practical reality**: unreliable for bulk historical fetching. Yahoo
  rate-limits aggressively and the library breaks frequently when Yahoo
  changes their API. Not suitable for a 928-ticker × 15-year backfill.
- **Potential niche use**: small-scale spot checks (a handful of tickers
  at a time with long delays) — but not a reliable data pipeline source.

### 4.3 FRED (FREE, we have FRED_API_KEY)

- Already used for macro series (VIX, fed funds, yield curve, etc.)
- Found `UMCSENT` (University of Michigan Consumer Sentiment) — a market
  sentiment proxy, but monthly granularity and broad-market (not
  stock-specific). Low PEAD relevance.
- **Verdict**: no new PEAD-relevant FRED data beyond what we have.

---

## 4A. FMP audit (CONFIRMED — $49/mo plan purchased 2026-07-23)

> FMP migrated to a new "stable" API on Aug 31, 2025. All v3/v4 legacy
> endpoints return 403. The `/stable/` API is the only access path.
> Probed with real API key on 5 mid-cap tickers (AOS, EXLS, MMM, XYL, PNR).

### 4A.1 Confirmed working endpoints (mid-cap verified)

| Endpoint | Data | History depth | Mid-cap verified |
|----------|------|---------------|------------------|
| `/stable/grades?symbol={ticker}` | **Analyst upgrade/downgrade**: date, gradingCompany, previousGrade, newGrade, action (upgrade/downgrade/maintain) | **2012-2026** (14 yrs) | AOS=177, EXLS=128, MMM=432, XYL=236, PNR=312 entries |
| `/stable/earnings?symbol={ticker}&includeReportTimes=true` | EPS actual/estimated, revenue actual/estimated, **time ("bmo"/"amc")**, periodEnding, fiscalPeriod ("Q3"), fiscalYear, confirmed | **1985-2026** (41 yrs) | AOS=164, EXLS=90, MMM=165, XYL=65, PNR=164 entries |
| `/stable/analyst-estimates?symbol={ticker}&period=quarter` | Quarterly consensus estimates: epsAvg/High/Low, revenueAvg/High/Low, numAnalysts | ~10 future quarters (forward estimates) | All 5 mid-caps return 200 |
| `/stable/analyst-estimates?symbol={ticker}&period=annual` | Annual consensus estimates (same fields, annual) | ~5 past + 5 future | ✅ |
| `/stable/profile?symbol={ticker}` | Company profile: sector, industry, CIK, ISIN, marketCap, beta | Snapshot | ✅ |
| `/stable/quote?symbol={ticker}` | Real-time quote: price, volume, dayHigh/Low, 50/200-day MA | Snapshot (today only) | ✅ |
| `/stable/price-target-consensus?symbol={ticker}` | Analyst target price consensus (high/low/median/consensus) | Snapshot | ✅ |
| `/stable/price-target-summary?symbol={ticker}` | Target counts by period (lastMonth/Quarter/Year/allTime) | Snapshot | ✅ |

### 4A.2 NOT available on FMP stable API

| Endpoint | Result | Impact |
|----------|--------|--------|
| Historical daily OHLCV (`/stable/historical-price-*`) | **404** — not on stable API; legacy v3 returns 403 | **Cannot replace Tiingo for prices** |
| Institutional holders / 13F (`/stable/institutional-holder`) | **404** — endpoint not found | No institutional flow data |
| Insider transactions / Form 4 (`/stable/insider-trading`) | **404** — endpoint not found | No insider data from FMP |
| Short interest (`/stable/short-interest`) | **404** — endpoint not found | No short interest from FMP |
| Earnings call transcripts (`/stable/earning-call-transcript`) | **402 Premium** (higher tier than $49) | No transcript NLP |

### 4A.3 The `grades` endpoint — the #1 PEAD feature (confirmed)

The analyst upgrade/downgrade history is the highest-value data we found.
Sample entry for AAPL:
```json
{
  "symbol": "AAPL",
  "date": "2026-07-17",
  "gradingCompany": "HSBC",
  "previousGrade": "Hold",
  "newGrade": "Buy",
  "action": "upgrade"
}
```

- **14 years of history** (2012-02-08 -> 2026-07-23 for AAPL)
- **Daily granularity** — can compute revision momentum in any pre-earnings window
- **111 unique grading companies** (Morgan Stanley, HSBC, Keybanc, Jefferies, etc.)
- **Action distribution** (AAPL): 87 upgrades, 122 downgrades, 1566 maintains
- **Mid-cap coverage confirmed**: AOS has 177 entries, EXLS 128, MMM 432

### 4A.4 The `earnings` endpoint — fully replaces EODHD (confirmed)

With `includeReportTimes=true`, FMP returns **strictly more data** than EODHD:

| Field | EODHD | FMP |
|-------|-------|-----|
| Report date | `report_date` | `date` |
| EPS actual | `actual` | `epsActual` |
| EPS estimate | `estimate` | `epsEstimated` |
| Revenue actual | ❌ | `revenueActual` (BONUS) |
| Revenue estimate | ❌ | `revenueEstimated` (BONUS) |
| BMO/AMC timing | `"BeforeMarket"`/`"AfterMarket"` (CamelCase) | `"bmo"`/`"amc"` (clean) |
| Fiscal period end | `fiscal_period_end` | `periodEnding` |
| Fiscal period label | ❌ | `fiscalPeriod` ("Q3") (BONUS) |
| Fiscal year | ❌ | `fiscalYear` (2026) (BONUS) |
| Confirmed flag | ❌ | `confirmed` (BONUS) |
| Currency | `currency` | ❌ (assume USD for S&P 400) |
| Surprise % | `percent` | ❌ (derivable) |
| Difference | `difference` | ❌ (derivable) |
| History depth | 15 years | **41 years** (1985-present) |

The `difference` and `percent` fields are trivially derivable
(`actual - estimate` and `(actual-estimate)/estimate`). The clean
`"bmo"`/`"amc"` format eliminates the CamelCase parsing bug we fixed
in Phase E (the `is_bmo` feature).

### 4A.5 The `analyst-estimates` endpoint — estimate trajectory (confirmed)

Quarterly consensus estimates give us the **estimate trajectory** — how
the consensus moved over time. This is the raw input for analyst revision
momentum features.

**Caveat**: the endpoint returns ~10 future quarters of forward estimates.
Historical estimate revisions (how the consensus for Q3 2024 changed over
the 3 months before the earnings date) may need to be derived from the
`grades` endpoint instead, or may require a higher FMP tier. Needs
verification during pipeline build.

---

## 5. Feature-by-feature availability matrix (CONFIRMED)

| # | Feature | Source | Cost | Status |
|---|---------|--------|------|--------|
| 1 | **Analyst revision momentum** | **FMP `/stable/grades`** | $49/mo | ✅ **CONFIRMED** — 14 yrs, daily, 111 firms, mid-caps verified |
| 2 | **Short interest changes** | FINRA (free, needs registration) or Polygon ($29-199/mo) | Free or $$ | ❌ Not on FMP; FINRA API needs auth token |
| 3 | **Options IV skew** | Polygon ($29-199/mo) | $$ | ❌ Not on FMP; needs specialist vendor |
| 4 | **Institutional ownership flow** | SEC EDGAR 13F (free, bulk parser needed) | Free | ❌ Not on FMP stable API; SEC EDGAR is the free fallback |
| 5 | **Earnings call transcript NLP** | SEC 8-K (free) or FMP (higher tier) | Free or $$$ | ❌ FMP $49 returns 402 for transcripts; SEC 8-K is free but heavy parsing |
| 6 | **Revenue surprise** | **FMP `/stable/earnings`** (revenueActual/Estimated) | $49/mo | ✅ **CONFIRMED** — bonus from earnings endpoint |
| 7 | **Pre-earnings price drift** | Tiingo (already have) | €30/mo | ✅ Already computed (rel_ret_*) |
| 8 | **Earnings BMO/AMC timing** | **FMP `/stable/earnings?includeReportTimes=true`** | $49/mo | ✅ **CONFIRMED** — `time: "bmo"/"amc"`, cleaner than EODHD |
| 9 | **EPS surprise + SUE** | **FMP `/stable/earnings`** | $49/mo | ✅ **CONFIRMED** — replaces EODHD; 41 yrs of history |
| 10 | **Quarterly consensus estimate trajectory** | **FMP `/stable/analyst-estimates?period=quarter`** | $49/mo | ✅ **CONFIRMED** — forward estimates; historical revision trajectory needs `grades` endpoint |

### 5.1 What FMP gives us (confirmed, $49/mo)

- **Analyst revision history** (`grades`) — the #1 PEAD feature, 14 yrs, daily
- **Full earnings data** (`earnings` with `includeReportTimes`) — replaces EODHD
- **Revenue estimates** — bonus for revenue surprise features
- **Quarterly consensus estimates** (`analyst-estimates`) — estimate trajectories
- **Company profiles** (`profile`) — sector, industry, CIK

### 5.2 What FMP does NOT give us

- **Historical daily prices** — cannot replace Tiingo (404 on all price endpoints)
- **Short interest** — endpoint not found on stable API
- **Institutional 13F** — endpoint not found on stable API
- **Insider Form 4** — endpoint not found on stable API
- **Options IV** — FMP doesn't do options
- **Earnings transcripts** — 402 (higher tier than $49)

### 5.3 What still needs a separate source

| Feature | Best source | Cost | Priority |
|---------|-------------|------|----------|
| Historical OHLCV prices | **Tiingo** (keep) | €30/mo | CRITICAL — cannot replace |
| Short interest | FINRA (register) or Polygon | Free or $29-199/mo | MEDIUM — defer |
| Options IV skew | Polygon | $29-199/mo | LOW — defer |
| Institutional 13F | SEC EDGAR (free, bulk parser) | Free | LOW — defer |
| Insider Form 4 | SEC EDGAR (free, confirmed working) | Free | LOW — defer |

---

## 6. The objective mismatch fix (user's key insight)

Before sourcing any new features, we must fix the backtest objective.
The Phase G backtest optimized for PnL/Sharpe, which conflated PEAD drift
with gap mean-reversion. The v2 backtest MUST separate these:

### 6.1 New metrics required

| Metric | What it measures | Formula |
|--------|-----------------|--------|
| **PEAD capture rate** | % of accepted trades that are true PEAD events | `n_pead_accepted / n_trades` |
| **PEAD drift contribution** | How much of the trade PnL is PEAD drift vs gap | `closeT_pnl / entry_pnl` (or the `diff` column from Doc K §6) |
| **Gap-reversion contribution** | How much of the PnL is the gap itself | `entry_pnl - closeT_pnl` |
| **PEAD-weighted PnL** | PnL weighted by whether the trade captured true PEAD | `Σ(pead_flag * entry_pnl) / n_trades` |

### 6.2 The dual-objective backtest

The v2 backtest should report BOTH:
- **Raw PnL/Sharpe** (the old metric — for live-trading viability)
- **PEAD capture rate + PEAD drift contribution** (the new metric — for
  model-quality evaluation)

A model that achieves Sharpe +1.5 with 80% PEAD capture is strictly
better than one that achieves Sharpe +1.5 with 10% PEAD capture (the
latter is a mean-reversion trade, not PEAD). The old backtest couldn't
distinguish these.

### 6.3 Pre-gap entry testing

For the POS side (where real PEAD lives), the v2 backtest should test
**Close[T] entry** (pre-gap) for high-conviction picks. This directly
addresses Doc K §6.2: the (+10%, +15%] bucket had +14.68% closeT drift
but only +1.60% entry PnL because the gap ate 89% of the move. Entering
at Close[T] captures the full +14.68%.

**Risk**: Close[T] entry means buying BEFORE the earnings announcement,
which requires the Sunday classifier to be precise enough to bet
pre-earnings. This is a higher bar than the current Open[T+1] entry.

---

## 7. Recommendation

### 7.1 What to do NOW (zero cost, weeks of effort)

**Phase H-1: SEC EDGAR XBRL features** (revenue + gross margin surprise)
- Source: `data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` (free)
- We already have CIKs in `/metadata/sp400_permatickers`
- New features: `revenue_surprise_pct`, `gross_margin_surprise`,
  `revenue_sue_score`, `debt_to_assets_change`
- Effort: ~2-3 days to build the fetcher + feature computation
- Expected lift: MODERATE. Revenue surprises are a known PEAD predictor
  and complement EPS surprise. Won't be a game-changer alone, but it's
  free and tests whether new fundamentals move the needle.

**Phase H-2: SEC EDGAR Form 4 insider signals**
- Source: `data.sec.gov/submissions/CIK{cik}.json` + Form 4 XML
- New features: `insider_buy_days_pre_earnings`,
  `insider_net_buy_volume_30d`, `insider_buy_intensity`
- Effort: ~3-5 days (Form 4 XML parsing is fiddly)
- Expected lift: LOW-MODERATE. Insider buying pre-earnings is predictive
  but rare (most companies have minimal insider activity).

### 7.2 What to do NEXT (requires paid upgrade, ~$30-60/mo)

**Phase H-3: Upgrade Tiingo to Fundamentals plan**
- Unlocks: analyst estimates (for revision momentum), full financial
  statements, institutional ownership snapshots
- This is the **highest-value upgrade** — analyst revision momentum is
  the #1 most-cited modern PEAD predictor
- Cost: ~$30-60/mo additional on top of current Tiingo
- Effort: ~1 week to build the fetcher + features once access is confirmed
- Expected lift: HIGH. Analyst revisions are the strongest post-2010
  PEAD predictor in the literature.

**Decision point**: after Phase H-1 + H-2, if the new SEC EDGAR features
don't lift VAL AUC above 0.6, we should upgrade Tiingo and pursue H-3.
If the SEC EDGAR features DO lift AUC, we can build on that first.

### 7.3 What to DEFER (high effort or low ROI for now)

- **Options IV skew** (Phase H-4): requires paid EODHD/Tiingo upgrade +
  options chain parsing. High effort. Defer until the fundamental
  features prove the concept.
- **Institutional 13F flow** (Phase H-5): free but requires building a
  bulk 13F parser (~5000 XML files per quarter). High engineering effort
  for quarterly-granularity data. Defer.
- **Earnings call NLP** (Phase H-6): free via SEC 8-K but requires an
  NLP pipeline (transcript extraction + sentiment model). Very high
  effort. Defer to v3.

---

## 8. Proposed v2 architecture (CONFIRMED)

```
Data layer:
  /sp400/{permaTicker}         — Tiingo prices (KEEP, irreplaceable)
  /earnings/fmp                — FMP earnings (NEW, replaces /earnings/raw)
  /analyst/grades/{permaTicker} — FMP analyst grades (NEW)
  /analyst/estimates/{permaTicker} — FMP quarterly estimates (NEW)
  /macros/fred_*                — FRED macro data (KEEP)
  /metadata/sp400_permatickers — Tiingo identity (KEEP)

Feature layer:
  Sunday-safe 17 (existing) + revision_momentum_30d/60d/90d
  + revenue_surprise_pct + revenue_sue_score
  + n_analysts_covering + last_action_days_before_earnings

Model layer:
  3-class XGBClassifier (multi:softprob)
    Target: {no PEAD, small PEAD, large PEAD}
    Based on car_60d_pass1 magnitude thresholds (not just binary gate)

Backtest layer:
  Dual-objective: PnL/Sharpe + PEAD capture rate
  Entry: test BOTH Open[T+1] (current) and Close[T] (pre-gap)
  Eval: nested CV with PEAD-capture metric alongside Sharpe

Live layer:
  Paper-trade the current NEG_only baseline (fold #5) in parallel
  Deploy v2 only if PEAD capture rate > 40% AND Sharpe > 1.0
```

---

## 9. Immediate next steps (CONFIRMED, ordered)

1. **Cancel EODHD subscription** — FMP `/stable/earnings` fully replaces
   it with better data (BMO/AMC, revenue, fiscal labels, 41-yr history).
   Do this BEFORE the next billing cycle.

2. **Build FMP earnings fetcher** (`01_data/06b_fmp_earnings_gathering.py`)
   - Fetch `/stable/earnings?symbol={ticker}&includeReportTimes=true`
     for all 928 permaTickers (via canonical_ticker join)
   - Store under `/earnings/fmp` (new path; keep `/earnings/raw` as backup)
   - Derive `difference = epsActual - epsEstimated` and
     `percent = (epsActual - epsEstimated) / epsEstimated` ourselves
   - Map `time` field ("bmo"/"amc") directly to `is_bmo` (1 if "bmo")

3. **Build FMP grades fetcher** (`01_data/07_fmp_grades_gathering.py`)
   - Fetch `/stable/grades?symbol={ticker}` for all 928 permaTickers
   - Store under `/analyst/grades/{permaTicker}`
   - 14 years of daily-granularity analyst actions (upgrade/downgrade/maintain)

4. **Build FMP estimates fetcher** (`01_data/08_fmp_estimates_gathering.py`)
   - Fetch `/stable/analyst-estimates?symbol={ticker}&period=quarter`
   - Store under `/analyst/estimates/{permaTicker}`
   - Forward estimate trajectories

5. **Add analyst revision momentum features** to
   `02_features/02_build_feature_matrix.py`:
   - `revision_momentum_30d`, `revision_momentum_60d`,
     `revision_momentum_90d` (net upgrades minus downgrades pre-earnings)
   - `n_analysts_covering`, `last_action_days_before_earnings`
   - `revenue_surprise_pct`, `revenue_sue_score`
   - These join on `(permaTicker, report_date)`

6. **Retrain with expanded feature set** and measure:
   - Does VAL AUC lift above 0.55 (current baseline)?
   - Does the PEAD capture rate improve?
   - Use the dual-objective backtest (§6)

7. **In parallel**: paper-trade the NEG_only baseline (`05_live/`)
   to accumulate fold #5 forward-looking data.

---

End of Feature Sourcing Audit.

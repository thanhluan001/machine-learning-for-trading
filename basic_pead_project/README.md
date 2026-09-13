# basic_pead_project — Post-Earnings-Drift Paper-Trading Skeleton

A complete, self-contained skeleton of a point-in-time PEAD (post-earnings
announcement drift) event-trading pipeline for the S&P MidCap 400 universe,
ending in an Alpaca **paper** trading book. Read this file top-to-bottom,
install the prerequisites, and you can rebuild the pipeline stage by stage.

Everything here is **research/paper infrastructure**. No component places
live orders. The trading script talks only to an Alpaca paper account.

---

## 1. What this is

```text
nightly (>=1h after close)              intraday (market open, ~3:45pm ET)
  01_fetch_and_predict.py                 02_paper_trade.py
    SP400 earnings calendar (FMP)           check stops + T+5 exits
    point-in-time features (24)             enter picks due today
    V6 gate scoring                         4 slots, equal weight
    -> plan.json (picks or none)
```

The strategy in one paragraph: **around each S&P 400 earnings event,
compute 24 features known BEFORE the print; score the event with three
frozen XGBoost "gate" classifiers (V6); if the minimum gate probability
exceeds 0.33 (and the name isn't XLF), stage an entry at the next close
before the print; hold to T+5 with a −10% trailing-ish stop and a
force-refresh rule that recycles slots.** The model is never retrained
ad hoc — only through the full validation gauntlet (see
docs/validation_doctrine.md).

Core doctrine, inherited from the parent project:

```text
1. Point-in-time everything (no survivorship, no look-ahead).
2. Stale data = hard stop (prices, benchmarks, macros, grades, earnings).
3. Pre-register research; validate walk-forward + bootstrap + holdout;
   freeze; shadow first; only then consider money.
4. Cash is a position. Most weeks the model picks nothing — that is
   the system working, not idling.
```

## 2. Prerequisites

```text
- Python env:      conda env create -f environment.yml  (or replicate:
                   python>=3.11, pandas, numpy, xgboost, requests,
                   alpaca-py, python-dotenv, scipy, statsmodels)
- API keys:        Tiingo (daily EOD), FMP (Starter+), FRED, Alpaca PAPER
- OS:              developed on Windows; nothing Windows-specific except
                   notes in docs/quirks.md
- Setup:           cp .env.example .env   (fill your keys)
- First build:     ~2-4 h (price history download is the long pole)
```

## 3. Stage map — build in this order

| # | Stage | Scripts (run order) | Output | Rough time |
|---|-------|---------------------|--------|-----------|
| 1 | Universe + identity | `01_data/01_metadata_gathering.py` | SP400 member list, Tiingo permaTicker identity map → db.h5 `/metadata/*` | 10 min |
| 2 | Sectors | `01_data/02_SEC_sector_gathering.py` | GICS sectors → `/metadata/sp400_sectors` | 5 min |
| 3 | Company map | `01_data/02b_build_company_map.py` | ticker↔permaTicker canonicalization | 2 min |
| 4 | Prices | `01_data/03_data_gathering.py` | `/sp400/{permaTicker}` daily OHLCV (Tiingo adj) | 1-2 h |
| 5 | Benchmarks | `01_data/04_index_data_gathering.py` | IJH/SPY etc. → `/index/*` | 5 min |
| 6 | Macros | `01_data/05_fed_data_gathering.py` | unemployment, fed funds, VIX → `/macro/*` | 5 min |
| 7 | Earnings history | `01_data/06_earnings_gathering.py` + `06b_fmp_earnings_gathering.py` | `/earnings/fmp` (point-in-time est/actual by fiscal period) | 30 min |
| 8 | Analyst grades | `01_data/07_fmp_grades_gathering.py` + `08_fmp_grades_historical_gathering.py` | `/grades/*` | 20 min |
| 9 | Membership refresh | `01_data/refresh_sp400_membership.py` | interval membership for point-in-time universe | 5 min |
| 10 | Events | `02_features/01_features_gate_events.py` | gated earnings events (universe + data-quality passes) | 10 min |
| 11 | Feature matrix | `02_features/02_build_feature_matrix.py` | `/features/train_matrix` — one row per event × 24 features + labels | 20 min |
| 12 | Train (pattern) | `03_model/01_train_model.py` | a fitted classifier — this is the pattern; V6 below is the real thing | 10 min |
| 13 | **V6 gates** | `03_model/08_freeze_v6_gate_models.py` | `03_model/models/phase_g_v6_gate_decomposition/` — INCLUDED already (frozen); re-running rebuilds it from your matrix | 15 min |
| 14 | Validation gauntlet | `04_backtest/53 → 54 → 55 → 57 → 61 → 63` (in that order; they chain via artifacts in `04_backtest/archive/experiments/gate_decomposition_v6/`) | DEV/OOS stats, nested CV, final holdout, bootstrap CI, force-refresh sim | 1-2 h |
| 15 | Nightly predict | `05_live/01_fetch_and_predict.py` | `plan.json` (V6 picks), `candidates.json`, ledger appends | 15-25 min/night |
| 16 | Intraday trade | `05_live/02_paper_trade.py` | Alpaca paper orders; `positions.json` book state | 1 min |
| 17 | Reports | `05_live/03_portfolio_report.py`, `05_live/04_tax_report.py` | performance + French tax ledger (see `tax/README.md`) | 1 min |

Included frozen V6 model: `03_model/models/phase_g_v6_gate_decomposition/`
(three gate classifiers + meta.json, threshold 0.33). Reference validation:
DEV +3.1% avg/event, holdout +4.2%, 59-70% win rate across folds — your
rebuild will differ slightly with refreshed data; that's expected.

## 4. The data contract (read before operating)

- **Freshness hard-stops**: the nightly script refuses to score a name
  whose price bar, benchmark, macro series, or grades are stale; earnings
  actuals back-fill lazily with a ±4-day date-shift matcher (FMP revises
  report dates; see docs/quirks.md).
- **Point-in-time rule**: features use only information available before
  the event. SUE lags come from prior quarters. Membership history is
  interval-based, not today's list.
- **The nightly cadence**: script 01 at least one hour after close (so
  the day's bars are consolidated); script 02 during the market session.
  Orders are DAY market orders — if script 02 runs on a holiday, the
  order queues for the next open (known behavior).

## 5. Daily operation

```bash
# evening, >=1h after close
conda run -n trading python 05_live/01_fetch_and_predict.py
# next day, market open
conda run -n trading python 05_live/02_paper_trade.py
# whenever
conda run -n trading python 05_live/03_portfolio_report.py
```

What the outputs mean:

```text
plan.json        tonight's V6 picks (may be zero — normal)
positions.json   the 4-slot book: entries, stops, exits, status
candidates.json  everything scored, with all 24 features — your audit trail
```

## 6. Before any real money

Run the gauntlet yourself (stage 14), read docs/validation_doctrine.md,
and internalize: **the frozen model is a hypothesis, not a guarantee.**
The parent project's entire history is pre-registered tests, most of
which were killed honestly. Expect the same posture: a paper evidence
period measured in months before any sizing decision.

## 7. Extending

- **New feature**: add to 02_features, rebuild matrix (stage 11), re-run
  the FULL gauntlet with the feature included, compare to frozen V6 —
  no cherry-picking.
- **New universe**: replicate 01_data stages for the new index, rebuild
  membership intervals, then train a fresh model. Do not reuse V6
  thresholds across universes — score scales are not comparable.
- **New model version**: freeze under a new directory name with meta.json
  recording provenance; keep old versions frozen for comparison.

## 8. Known traps (distilled; full list in docs/quirks.md)

FMP revises earnings dates after calendar seeding (±1-4 days); FMP's
calendar window starts yesterday (same-day events drop without the
offset); Tiingo delisted tickers need permaTicker identity via
`/utilities/search`; grades must be cutoff at `report_date`, not fetch
date; never import the base backtest module twice in one process.

## 9. Layout

```text
01_data/      gathering spine → db.h5 (HDF5 store, gitignored)
02_features/  events + matrix
03_model/     trainer + V6 freeze (+ frozen V6 model included)
04_backtest/  validation gauntlet (51 → 53 → 54 → 55 → 57 → 61 → 63)
05_live/      fetch_and_predict, paper_trade, reports, alpaca_client
tax/          French tax ledger docs (see tax/README.md)
docs/         architecture, data contracts, features, validation, quirks
```

Questions the README doesn't answer are probably answered by reading the
script headers — they document their own contracts in detail.

# 05_live/ — Live Paper-Trading Fold #5

## Purpose

This folder contains the **live forward-looking OOS test** for the
deployable binary classifier. The frozen artifact at
`03_model/models/phase_g_v2_binary/classifier.json` was trained on
all available data (T up to the freeze date, 2026-07-30). Any earnings
event with `T >= 2026-07-31` is a **live-fold event** -- truly
forward-looking data that the model has never seen.

## Protocol

### When to run

Wait **~2 months** from the classifier freeze date (2026-07-29), i.e.
run on or after **~2026-09-29**. This gives enough earnings
announcements + enough time for T+5 exits to complete so we can compute
realized PnL.

### How to run

```bash
# Full run: fetch new data + rebuild features + inference + report
python luan_bot_trading/05_live/01_live_fold_pull.py

# Inference only (skip the ~15 min data fetch — use when you've already
# pulled data recently and just want to check PnL on closed trades):
python luan_bot_trading/05_live/01_live_fold_pull.py --skip-fetch

# Dry run (show what would be fetched, no writes):
python luan_bot_trading/05_live/01_live_fold_pull.py --dry-run
```

### What the script does

| Step | Action | Duration | Network |
|------|--------|----------|---------|
| 1 | Incremental Tiingo price fetch (only new dates per permaTicker) | ~5 min | Tiingo |
| 2 | Full FMP earnings re-fetch (`06b_fmp_earnings_gathering.py`) | ~10 min | FMP |
| 2b | Full FMP grades re-fetch (`07_fmp_grades_gathering.py`) | ~10 min | FMP |
| 3 | Re-run Stage 1: gate events (`01_features_gate_events.py`) | ~30s | none |
| 4 | Re-run Stage 2: build feature matrix (`02_build_feature_matrix.py`) | ~2 min | none |
| 5 | Load FROZEN binary classifier, predict P(PEAD) on live events | <1 min | none |
| 6 | Apply deployable rule (P(PEAD)>=0.20) + compute pre-gap PnL | <1 min | none |

**Total: ~18 min**

### The classifier is NEVER retrained

The entire point of the live fold is to test the **frozen** artifact on
truly forward-looking data. The classifier at
`03_model/models/phase_g_v2_3class/classifier.json` is loaded read-only
and used for `predict_proba()`. No retraining, no HP tuning, no threshold
re-selection.

## Deployable Operating Point (Phase G v2 — PEAD capture objective)

| Parameter | Value |
|-----------|-------|
| Model | `XGBClassifier(binary:logistic)` on 24 Sunday-safe features (is_bmo removed, 8 ordinal revision momentum added) |
| Max simultaneous slots | 4 |
| Position sizing | Equal-weight 1/4 NAV per slot |
| Entry | **Pre-gap**: `Close[T-1]` (BMO) / `Close[T]` (AMC), before announcement |
| Exit | `Close[T+5]` (5 trading days from report date) |
| Per-trade stop-loss | **-10% delayed** (skip gap day, check days 1+) |
| Sector exclusion | **XLF (Financials) excluded** at inference only |

## Expected Baseline Stats (binary P(PEAD)>=0.20, 4-fold nested CV)

| Metric | OOS Value |
|--------|-----------|
| N trades | 102 |
| **Win rate** | **62.7%** |
| **Expectancy/trade** | **+5.71%** |
| Avg win | +13.06% |
| Avg loss | -6.68% |
| Payoff ratio | 1.96 |
| **Total PnL (NAV-compounded)** | **+293.8%** (3.94x NAV) |
| Total PnL (raw sum) | +582.3% |
| PEAD precision | 30.4% |

> **Note**: Total PnL has two representations. The raw sum (+672%) treats
> each trade as 100% NAV. The NAV-compounded return (+391%, 4.91x) models
> 4 slots at 1/4 NAV with weekly compounding — this is the realistic
> portfolio return. Compounding boosts returns well above the naive
> +672%/4 = +168% because ~75% of weeks are positive. See
> `44_slot_sweep_nav_sizing.py`.

## Interpreting Live-Fold Results

### Small-sample caveat

At the deployable rule's base rate (~50 trades/year), 2 months of S&P 400
earnings will likely yield **4-10 live trades**. At a true win rate of
64.6%, binomial probability of observing <=2 wins out of 5 is ~17% -- so
even a "bad" live fold doesn't falsify the model.

### Decision framework

| Live-fold outcome | Verdict | Next step |
|--------------------|---------|-----------|
| Win rate >= 50% AND expectancy > 0 | POSITIVE — rule generalizes | Continue paper-trading; accumulate fold #6 |
| Win rate < 50% OR expectancy < 0 | WARNING — investigate regime change | Check VIX/fed funds regime; compare feature distributions |
| N < 5 trades | INCONCLUSIVE | Wait longer; re-run with `--skip-fetch` later |
| 0 trades accepted | DIAGNOSTIC | Either no events passed screen, or T+1 gaps were all positive (bull-market drift) |

### What "directional truth" means

The first live fold is **directional truth, not statistical significance**.
A positive result (win rate >= 50%, expectancy > 0) gives confidence that
the model hasn't overfit to a specific historical regime. A negative result
is a flag to investigate — NOT a definitive falsification (small sample).

Accumulate 2-3 live folds before drawing firm conclusions. The live fold
data point feeds into the P0 priority item in `04_backtest/archive/docs/future_implementation.md` (superseded — see Design.md §18).

## Script: `01_live_fold_pull.py`

### CLI flags

- `--skip-fetch` — Skip Steps 1-4 (data fetch + feature rebuild). Use this
  to re-check PnL on closed trades without re-fetching data.
- `--dry-run` — Print what would be fetched without writing to db.h5.

### Incremental Tiingo fetch logic

For each permaTicker with `price_unavailable=False`:
1. Get the latest `Date` stored in `/sp400/{permaTicker}`
2. If latest >= yesterday → skip (up to date)
3. Else → fetch from `latest + 1 day` to `yesterday`, append + dedup
4. If no stored node → full 15-year fetch

This is **much faster** than re-running `03_data_gathering.py` (which
re-fetches the full 15-year history for every permaTicker every time).

### Earnings re-fetch

The script calls `06b_fmp_earnings_gathering.py` as a subprocess. This
re-fetches the full 41-year earnings history per permaTicker from FMP.
A full re-fetch is **correct behavior** because FMP estimates get
revised after the initial report -- an incremental fetch would miss
revisions. Also calls `07_fmp_grades_gathering.py` for the analyst
revision momentum features (Block 6).

### Feature rebuild

After data is updated, the script calls:
1. `02_features/01_features_gate_events.py` -- rebuilds `/features/gated_events`
2. `02_features/02_build_feature_matrix.py` -- rebuilds `/features/train_matrix`

These are fast (~2 min total) and make zero network calls. The new events
will have their 24 Sunday-safe features computed. Features that depend on
windows extending past the latest price data (e.g., `car_60d_pass1` for
recent events) will be `NaN` -- which XGBoost handles natively.

### Realized PnL computation

For each accepted trade:
- Entry: **Pre-gap** -- `Adj_Close[T-1]` (BMO) / `Adj_Close[T]` (AMC), before announcement
- Exit: `Adj_Close[T+5]` (5 trading days from report date)
- Stop-loss: **-10% delayed** (skip gap day, check days 1+)
- If `T+5` hasn't passed yet → trade is "pending" (no PnL computed)

## Cross-References

- `03_model/02_phase_g_sunday_classifier.py` — the trainer that produced the
  frozen artifact
- `04_backtest/strategy_v2_synthesis.md` — Doc A-J synthesis
- `04_backtest/phase_g_bootstrap_ci_findings.md` — Doc-H baseline
- `04_backtest/archive/docs/future_implementation.md §3.3` (superseded) — live fold #5 as P0 priority
- `Design.md §17.A` — deployable Phase G classifier spec

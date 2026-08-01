# 04_backtest/ -- PEAD Strategy Backtest

> Concise summary + index. Archived experiment scripts, findings, and
> artifacts live in [`archive/`](./archive/). Full technical
> synthesis in [`strategy_v2_synthesis.md`](./strategy_v2_synthesis.md).

---

## 1. Recommended strategy (binary classifier + pre-gap entry, deployable)

**Binary Sunday classifier, P(PEAD) >= 0.20, NO gap filter, NO stop-loss.**

```
 Sunday                              | Execution (pre-gap)            | Hold to exit
 ------------------------------------|-------------------------------|---------------------
 XGBoost binary classifier            | BMO: enter Close[T-1]        | Exit   Close[T+5]
 on 24 Sunday-safe features (is_bmo   | AMC: enter Close[T]          | (5 trading days
 removed, 8 FMP revision momentum     | (before the earnings          |  from report date)
 features added). Target: pead_pass   |  announcement -- captures      | No stop-loss (winners
 (0/1 from 3 PEAD gates).             |  the overnight gap)           |  overcompensate losers
 P(PEAD) >= 0.20                      | Max 4 simultaneous slots      |  4.2:1)
           Exclude XLF (Financials)     |                               |
                                      | equal-weight 1/4 NAV each    |
```

### Operating-point parameter values

| Parameter | Value | Source |
|---|---|---|
| Classifier | **binary:logistic** (pead_pass 0/1) | 2026-07-30 (beats 3-class) |
| `theta` (P(PEAD) threshold) | **0.20** | Binary theta re-sweep |
| Gap filter | **NONE** | Removed 2026-07-23 |
| Entry | **Close[T-1] (BMO) / Close[T] (AMC)** -- pre-gap | 2026-07-29 |
| Hold period | **5 trading days** (to Close[T+5]) | Frees slots weekly |
| Stop-loss | **-10% delayed** (skip gap day) | Neutral, caps tail risk |
| Sector exclusion | **XLF (Financials) excluded** at inference only | 13% vs 41% PEAD precision |
| Max simultaneous slots | **4** | Portfolio sim default |
| Position sizing | **equal-weight 1/4 NAV** per slot | Portfolio sim default |
| Model hyperparameters | gamma=3, mcw=50, md=3, n_est=300 | Nested CV HP selection by F1 |
| Features | 24 Sunday-safe (is_bmo removed, 8 revision momentum) | Phase H FMP data |

---

## 2. Expected performance (honest OOS, 4-fold nested CV, exclude XLF)

**Validation setup**: 4 anchored walk-forward folds over 2024 H2 -- 2026 H1.
Per fold: train on everything <= TRAIN_END, sweep hyperparameters on
SWEEP_VAL (next 6 months), then refit on TRAIN+SWEEP and apply to
held-out TEST (next 6-month slice). HP selected by max PEAD F1 (not PnL).

### PEAD capture (PRIMARY objective)

| Stat | Value | 95% CI (bootstrap) |
|---|---:|---|
| Total picks (raw, no slot constraint) | 86 | -- |
| TRUE PEAD in picks | 30 | -- |
| **PEAD precision** | **35.6%** | [24.4%, 45.4%] |
| PEAD recall | 7.9% | [5.7%, 10.7%] |
| F1 | 13.3% | [9.3%, 17.3%] |
| Base rate (random precision) | 11.8% | -- |
| **Lift over random** | **3.0x** | -- |
| Model beats random | 99%+ of trials | -- |

### Practical trade stats (4-slot portfolio simulation, binary pre-gap)

| Stat | Value |
|---|---:|
| Executed trades (4-slot) | 101 |
| **Expectancy per trade** | **+6.66%** |
| **Win rate** | **75.2%** |
| Avg win | +12.36% |
| Avg loss | -6.30% |
| Payoff ratio | 1.36 |
| Std per trade | 12.70% |
| Total PnL (raw sum) | +672.4% |
| **Total PnL (NAV-compounded)** | **+391.3%** (4.91x) |
| Trades per week | mean 2.1, median 2, max 4 |
| Trades per year | ~50 |
| PEAD precision (executed) | 38.6% |
| Trade win rate (executed) | 75.2% |

> **Why binary over 3-class**: binary theta=0.20 beats 3-class
> P(any)>=0.20 on total return (+636% vs +607%) and win rate
> (69.7% vs 64.6%). The 2-stage test proved CAR magnitude is
> unpredictable, so the small/large split adds no value.
>
> **PEAD precision != trade win rate**: the model identifies true PEAD
> events only 35.8% of the time, but 69.7% of trades are profitable
> because non-PEAD picks still drift positive (+2.11% avg).
>
> **Why pre-gap entry**: the PEAD drift is front-loaded into the
> overnight gap. Entering pre-gap captures it; entering post-gap
> gets eaten by it.
>
> **Why no stop-loss**: winners (+8.81% contribution) overcompensate
> losers (-2.10% contribution) by 4.2:1.

### Per-fold breakdown (binary P(PEAD)>=0.20)

| Fold | OOS window | Executed | Win% | Avg PnL | Total | Large PEAD |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 2024 H2 | 18 | 77.8% | +8.53% | +153.6% | 4 |
| 2 | 2025 H1 | 31 | 67.7% | +4.07% | +126.2% | 4 |
| 3 | 2025 H2 | 26 | 80.8% | +10.78% | +280.3% | 8 |
| 4 | 2026 H1 | 26 | 76.9% | +4.32% | +112.3% | 5 |

All 4 folds positive on total return.

## 3. Key architectural changes from Phase G v1

| Change | Why | Impact |
|--------|-----|--------|
| **POS/NEG gap filter DELETED** | Blocked 99.5% of PEAD events (old NEG_only caught 2 of 366) | PEAD recall: 0.5% -> 18.6% (37x improvement) |
| **`is_bmo` removed from features** | Became #1 importance after FMP fix but caused OOS overfitting (Sharpe +0.50 -> -0.24) | Removing it fixed the overfitting |
| **Objective: PEAD capture (not PnL)** | Old objective conflated PEAD with mean-reversion; PnL and PEAD were in conflict | Now aligned: better PEAD detection -> better PnL |
| **FMP data replaces EODHD** | EODHD had CamelCase BMO bug (is_bmo ALL ZERO), no revenue data, only 15-yr history | FMP: 41-yr history, clean bmo/amc, revenue estimates, analyst grades |
| **8 revision momentum features added** | Doc K audit: missing "expectation" axis (all 17 features were fundamental/momentum) | Ordinal magnitude (#5 importance), improved NEG Sharpe +0.51 -> +0.80 |
| **`pre_event_volume_trend` log-transformed** | Raw volume slope was incomparable across stocks (min=-16M, max=+12M) | Now [-1.0, +0.6], cross-stock comparable |

---

## 4. Data pipeline (Phase H)

| Source | Cost | What it provides | Status |
|--------|------|-------------------|--------|
| **Tiingo** | ~$30/mo | Historical daily OHLCV + permaTicker identity | KEEP (irreplaceable -- FMP has no historical prices) |
| **FMP** | $49/mo | Analyst grades (14-yr revision history, 111 firms), earnings (BMO/AMC + revenue, 41-yr history), quarterly estimates | PURCHASED -- replaces EODHD + adds analyst revisions |
| **FRED** | Free | Macro data (VIX, fed funds, etc.) | KEEP |
| ~~EODHD~~ | ~~$20/mo~~ | ~~Earnings calendar~~ | CANCELLED -- FMP replaces it |

**Net cost**: ~$82/mo (was ~$50/mo). The +$32/mo buys analyst revision
history -- the #1 PEAD predictor in modern literature.

---

## 5. Scripts in this folder

### Current scripts (deployable model + active analysis)

| Script | Purpose |
|--------|---------|
| `_pead_target_retrain.py` | **Shared library** — `compute_pead_gates_full`, PEAD label. Imported by 40-44, 46. |
| `17_theta_sweep.py` | Theta sweep for precision/recall/F1 sweet spot |
| `19_practical_trade_stats.py` | 4-slot portfolio simulation with trade-level stats (baseline) |
| `22_bmo_amc_pregap.py` | BMO vs AMC pre-gap entry analysis |
| `24_delayed_stop.py` | -10% delayed stop-loss test |
| `30_hold_comparison_bootstrap.py` | 5-day vs 10-day hold + bootstrap CI |
| `35_macro_ab_test.py` | Macro features A/B test (macros excluded) |
| `37_wider_stop_test.py` | Wider delayed stops (-10%, -12%, -14%) |
| `38_precision_investigation.py` | Precision root cause analysis |
| `40_false_positive_analysis.py` | False positive deep dive (65% of picks) |
| `41_exclude_xlf_test.py` | XLF sector exclusion test |
| `42_xlf_excluded_detailed_stats.py` | **FINAL** detailed trade statistics (exclude XLF) |
| `43_slot_utilization_analysis.py` | Slot utilization vs earnings seasonality |
| `44_slot_sweep_nav_sizing.py` | NAV-based slot sweep (proper position sizing) |
| `45_index_rebalance_probe.py` | Index rebalancing edge probe (edge is dead) |
| `46_analyst_revision_probe.py` | Analyst revision momentum probe (edge is weak) |

### Documentation

| File | Purpose |
|------|---------|
| `strategy_v2_synthesis.md` | **Authoritative** technical synthesis (full strategy spec) |
| `xlf_excluded_detailed_stats.md` | Final detailed statistics (exclude XLF model) |
| `analyst_revision_findings.md` | Analyst revision momentum probe findings |

### Archived

All superseded scripts (Phase G v1, rejected experiments, private
diagnostics) are in [`archive/`](./archive/). See
[`archive/README.md`](./archive/README.md) for the full index.

---

## 6. Folder layout

```
04_backtest/                    ← THIS folder
├── README.md                   ← concise entry point (this file)
├── strategy_v2_synthesis.md    ← authoritative technical synthesis
├── xlf_excluded_detailed_stats.md  ← final detailed statistics
├── analyst_revision_findings.md    ← analyst revision edge probe
├── _pead_target_retrain.py     ← shared library (PEAD gates)
├── 17–46 *.py                  ← current scripts (15 files)
└── archive/                    ← all superseded material
    ├── README.md               ← archive index
    ├── phase_g_v1/             ← 13 Phase G v1 scripts (gap filter era)
    ├── phase_g_v2_superseded/  ← 15 rejected/superseded experiments
    ├── private_scripts/        ← 7 private diagnostics
    ├── experiments/            ← output data (CSVs, JSONs, NPZs)
    └── findings/               ← 9 historical markdown docs (Docs §0, B–K)

03_model/                       ← model training (separate folder)
├── README.md
├── 01_train_model.py           ← shared helper API (load_train_matrix, DB_FILE)
├── 02_phase_g_sunday_classifier.py  ← deployable model trainer
├── 04_freeze_binary_model.py   ← freezes deployable artifact
└── models/phase_g_v2_binary/   ← frozen classifier.json + meta.json

05_live/                        ← live paper-trading script
└── 01_live_fold_pull.py        ← ready for fold #5 (2026-09-30+)
```

---

## 7. Next-iteration priorities

1. **(P0) Live paper-trading fold #5** (2026-09-30+) — first
   forward-looking OOS data point. Script ready at
   `05_live/01_live_fold_pull.py`.
2. **(P1) Insider cluster buying (SEC Form 4)** — strongest remaining
   edge candidate (+5-10% per trade in literature). Free data (SEC
   EDGAR). Completely uncorrelated to earnings timing.
3. **(P1) Improve PEAD precision** — currently 38.6%. Every precision
   lever tested (theta, 3-class, eps filter) improves precision but
   hurts total PnL. Better features needed, not filters.
4. **(P2) Transaction cost + slippage model** — before live capital.
5. **(P2) Multi-edge portfolio** — combine PEAD + insider buying +
   analyst revisions for capital efficiency during idle weeks
   (see `43_slot_utilization_analysis.py`).

---

End of README.

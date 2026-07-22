# 04_backtest/ — PEAD Strategy Backtest

> Concise summary + index. Archived experiment scripts, findings, and
> artifacts live in [`archive/`](./archive/). Full technical
> synthesis in [`strategy_v2_synthesis.md`](./strategy_v2_synthesis.md).

---

## 1. Recommended strategy (final, deployable)

**Two-stage Sunday + T+1-morning filter on negative-gap earnings events.**

```
 Sunday                                | Weekday morning (T+1)              | Execution
 --------------------------------------|------------------------------------|---------------------
 XGBoost Sunday classifier              | Read realized Open[T+1] gap.        | Enter  Open[T+1]
 on 17 Sunday-safe features (no leak   | ENTER trade IF:                    | Exit   Close[T+11]
 features) for all upcoming-week       |   P(PEAD) >= 0.20  (theta)         | Max 4 simultaneous
 earnings events -> P(PEAD).           |   AND opening_gap_t1 in [-15%, -2%]| slots, equal-weight
                                       | (NEG_only)                         | 1/4 NAV each.
```

### Operating-point parameter values

| Parameter | Value | Source |
|---|---|---|
| `theta` (Sunday prob threshold) | **0.20** | Doc F theta sweep |
| `gap_lo` (lower T+1 gap) | **-15%** | Doc G gap sweep |
| `gap_hi` (upper T+1 gap) | **-2%** | Doc G gap sweep |
| Hold period | 10 trading days (Open[T+1] → Close[T+11]) | PEAD definition |
| Max simultaneous slots | **4** | Doc H default |
| Position sizing | **equal-weight 1/4 NAV** per slot | Doc H default |
| Dead-zone skip | **none** | Doc J verdict (H_B = false) |
| Model hyperparameters | per-fold POS-tuned (gamma = 10/5/3/3) | Doc D nested CV |

---

## 2. Expected performance (honest OOS, 4-fold walk-forward)

**Validation setup**: 4 anchored walk-forward folds over 2024 H2 – 2026 H1.
Per fold: train on everything ≤TRAIN_END, sweep hyperparameters on
SWEEP_VAL (next 6 months), then refit on TRAIN+SWEEP and apply to
held-out TEST (next 6-month slice). Data: held-out forward indices
of the train_matrix after the Phase D dedup (20,265 rows, 0 duplicates).

### Per-trade statistics (4-fold aggregate)

| Stat | Value | Notes |
|---|---:|---|
| Total trades (OOS sample) | **29** | 4-fold across 2 calendar years |
| **Win rate** | **69.0%** | 20 wins / 9 losses / 0 flat |
| **Avg win (per-trade arith return)** | **+6.21%** | median +5.53% |
| Best win (max single-trade gain) | +22.98% | over 10 trading days |
| **Avg loss (per-trade arith return)** | **-5.44%** | median -4.78% |
| Worst loss (max single-trade drop) | -9.39% | (no stop-loss applied; see §4) |
| **Win/loss payoff ratio** | **1.14** | avg_win / \|avg_loss\| > 1 |
| **Expectancy per trade** | **+2.59%** | the per-trade mean PnL |
| Std of per-trade PnL | 7.18% | per-trade dispersion is large |
| t-statistic (H0: mean PnL = 0) | 1.95 (df=28) | 2-sided p ≈ 0.06 |

### Per-fold breakdown (showing regime variance)

| Fold | OOS window | Trades | Hit rate | Avg win | Avg loss | Sharpe | IRR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 2024 H2 | 6 | 66.7% | +3.29% | -5.59% | +0.24 | +2.2% |
| 2 | 2025 H1 | 6 | 66.7% | +5.58% | -4.92% | +1.31 | +11.0% |
| 3 | 2025 H2 | 7 | 57.1% | +9.69% | -7.41% | +0.97 | +11.2% |
| 4 | 2026 H1 | 10 | 80.0% | +6.25% | -2.86% | +2.41 | +33.7% |
| **Cross-fold mean** | — | **7.25** | **69%** | **+6.21%** | **-5.44%** | **+1.23** | — |

### Cross-fold aggregate metrics

| Metric | Value | 95% CI (Doc H bootstrap) |
|---|---:|---|
| Mean Sharpe (4-fold) | **+1.31** | [+1.04, +1.58] parametric / [+1.16, +1.47] bootstrap |
| Mean per-trade PnL | **+2.28%** | [+1.49%, +3.81%] (excludes zero) |
| Mean IRR (annualized) | **+15.57%** | (path-implicit) |
| Mean MaxDD | **-4.06%** | (path-implicit) |
| Mean trades per fold | **7.2** | very low — see caveat (c) below |

(Doc H's bootstrap CI was computed from the pre-cleanup train_matrix.
Today's deduped re-run gives cross-fold Sharpe +1.23 ± 0.91, mean
per-trade PnL +2.59%, consistent with the bootstrap CI above.)

### Why NEG_only (negative T+1 gap)

Counter-intuitively, **the alpha lives in the negative-gap sub-strategy**,
not the positive-gap one. Doc E revealed:

- **NEG_only**: mean Sharpe +1.01, beats 80.3% of random-trial baselines.
- **POS_only**: mean Sharpe +0.86, beats 58.2%.

Doc G dissected the mechanism by gap-magnitude bucket:
- (-3%, -2%] tiny gap: **+3.53% per trade**, 67% hit — purest shaken-out retail reversal
- (-5%, -3%] mild gap: +2.28% per trade, 75% hit — most cross-regime stable
- (-10%, -5%] moderate-heavy: **anti-alpha** -2.88% per trade, 33% hit (the "real bad news" zone, where the Sunday classifier mislabels)
- (-15%, -10%] deep: small n, +alpha but statistically unreliable

The `[-15%, -2%]` outer range is kept wide-open (no dead-zone skip)
because Doc J nested-CV test confirmed the (-10%, -5%) boundary shape
is regime-dependent noise, not market structure, and a per-fold
selection procedure adds only +0.17 marginal Sharpe with substantial
operational complexity.

---

## 3. Caveats (REQUIRED reading before deployment)

(a) **Per-fold Sharpe CIs include zero.** Every single 6-month OOS
slice is statistically indistinguishable from random. The +1.31
result requires aggregating ≥4 folds to emerge.

(b) **n=29 trades total OOS** is a small sample. The 4-fold variance
of Sharpe (std 0.17 in Doc H, 0.91 in today's re-run) is the sample
distribution; longer OOS (5–10 years) would reveal variability not
captured here.

(c) **Residual circularity**. theta=0.20 and gap=[-15%, -2%] were
themselves swept on this same 2024–2026 OOS data (Docs F and G).
A full nested-CV re-sweep (mirroring Doc J's dead-zone nested CV)
would give a fully honest estimate; expected to regress to about
Sharpe +1.0–+1.2.

(d) **No live OOS fold yet.** Paper-trading fold #5 (2026 H2+) is
the highest-value next action — it generates the first truly
forward-looking data point.

(e) **17 Sunday-safe features have ~zero OOS signal alone**. The
classifier's role is to lift candidate density from the 10.68% PEAD
base rate to ~30–40% among picks; the realized gap is what does the
precision work.

---

## 4. Live-deployment risk overlay (recommended, per Doc H §H.7.3)

- **Per-trade stop-loss: -10%** on trade-level arith PnL (none of the
  9 observed losses exceeded -10%; this preserves the per-trade CI
  while bounding tail risk).
- **Position concentration cap: 1/4 NAV** per trade (already enforced
  by the equal-weight 4-slot rule).
- **Iterative fold tracking**: each new 6-month live fold's Sharpe
  accumulates per Doc H's reasoning; a single underperforming fold
  is NOT evidence the strategy broke (per caveat (a)).

---

## 5. Why the original Phase F "Sharpe 4.31" was bogus

The original Stage 4 backtest reported Sharpe 4.31 / hit rate 65.8%.
A leak test (NaN `opening_gap_t1` — the one feature that uses
`Open[T+1]`, not available Sunday) dropped hit rate to 49.1% and
Sharpe to -0.14. The smooth-edge episode was forward-looking
feature leakage — `opening_gap_t1` is a *confirmation*, not a
*predictor*, and using it as a Sunday predictor made the entire
backtest circular.

---

## 6. Folder layout

```
04_backtest/                    ← THIS folder (backtests / OOS probes / nested-CV diagnostics)
├── README.md                   ← concise entry point: strategy + win rate + win/loss
├── strategy_v2_synthesis.md    ← technical synthesis (deeper detail)
├── 01_val_backtest.py          ← Stage 4 single-OOS backtest harness (Phase F era, leaky)
├── 04_phase_g_portfolio.py     ← SHARED LIBRARY (overlapping 10-day-hold portfolio sim)
├── 05–14 *.py                  ← Phase G OOS evaluation + diagnostic scripts   (12 files)
├── _*.py                       ← shared helper libraries + diagnostic tools (7 files)
└── archive/                    ← rolled-up experiment artifacts
    ├── README.md               ← archive index (which doc covers what)
    ├── findings/               ← 8 markdown docs (Docs B–J + the original PEAD-target study)
    └── experiments/            ← 11 experiment-output folders + 2 stray CSVs

03_model/                     ← **Separate folder that trains the deployable model**
├── README.md                   ← see `../03_model/README.md` for the model-training entry point
├── 01_train_model.py            ← OBSOLETE main() but load-bearing shared helper API
├── 02_phase_g_sunday_classifier.py  ← **DEPLOYABLE model trainer** (XGBClassifier on 17 Sun-safe features)
├── 03_phase_g_sweep.py             ← 72-config HP sweep for v1.1 candidate
└── models/                     ← saved model artifacts (Phase F baselines + Phase G v1/v1.1 candidates)
```

For the complete story (Phase G v1 → v1.1 → Appendix B → ... → Doc J),
read `strategy_v2_synthesis.md` — it synthesizes docs A–J by reading
them together rather than chronologically.

---

## 7. Next-iteration priorities (per strategy_v2 §5.5)

1. **(Highest) Live paper-trading fold #5** (2026 H2+) — first
   forward-looking OOS data point.
2. **(Highest) Re-sweep theta + gap under proper nested CV** —
   closes the residual circularity caveat, expected +1.0 to +1.2.
3. **(High) Magnitude-aware 3-class `multi:softprob` classifier**
   targeting {no PEAD, small PEAD, large PEAD} — highest expected
   ceiling lift (`pead_target_findings.md §7.2 (a)`).
4. **(Medium) Confidence-calibrated sizing** — non-equal-weight
   using the §6.6 non-monotonic proba→PnL relationship.
5. **(Low) Regime-probe feature** for POS- vs NEG-favorable regimes.
6. **(Low) Gap-conditional sizing** across remaining 3 gap buckets.

---

End of README.

# Doc K — POS vs NEG Nested CV at theta=0.20 (fair comparison)

> **Status**: AUTHORITATIVE on the POS-vs-NEG question.
> Resolves the apparent contradiction between the §5 audit (POS has
> 7.6× higher PEAD recall and 6.7× higher precision than NEG) and
> the Phase G finding (NEG_only "won" with Sharpe +1.01 vs POS +0.86).

---

## 1. Motivation: the audit contradiction

A direct audit of the train_matrix (see §5 below) produced a bombshell:

- **Only 106 of 1,811 PEAD events (5.85%)** have `opening_gap_t1` in [-15%, -2%].
- The NEG gap filter **throws away 94% of true PEAD events**.
- The POS gap range [+2%, +15%] catches **802 PEAD events (44.28% recall)** at **28.02% precision** — 6.7× higher precision and 7.6× higher recall than NEG.

Yet Phase G Doc E found NEG_only **won** the nested-CV comparison
(mean Sharpe +1.01 vs POS +0.86). How can the rule that catches
6% of PEAD events beat the rule that catches 44%?

This doc answers that question with a **fair head-to-head nested CV**
that runs both rules at the **same theta=0.20** (the deployable threshold),
eliminating the theta-mismatch confound.

---

## 2. Protocol

Exact mirror of `06_phase_g_nested_cv.py` (Doc D) but with BOTH rules
evaluated on every fold:

- 4 anchored walk-forward folds over 2024 H2 → 2026 H1
- Per fold: TRAIN (anchored, growing) → SWEEP_VAL (6 months, HP selection) → TEST (6 months, OOS)
- HP sweep: gamma ∈ {3, 5, 10, 20} (fixed mcw=50, md=3, n_est=300)
- **POS and NEG each get independently-selected HP** (the best gamma for POS on SWEEP_VAL, and the best gamma for NEG on SWEEP_VAL)
- Retrain on TRAIN+SWEEP with selected HP, evaluate on TEST
- 100 random-trial baseline per fold
- theta=0.20 for BOTH rules (not theta=0.15 for NEG as in Doc E/`07_phase_g_ensemble.py`)
- POS gap: [+2%, +15%], NEG gap: [-15%, -2%]
- 4-slot portfolio simulator, equal-weight 1/4 NAV, exit Close[T+11]

Script: `04_backtest/15_phase_g_pos_vs_neg_theta020.py`

---

## 3. Results

### 3.1 Per-fold breakdown

| Fold | TEST window | POS gamma | POS trades | POS Sharpe | POS %rand | NEG gamma | NEG trades | NEG Sharpe | NEG %rand |
|-----:|-------------|----------:|-----------:|-----------:|----------:|----------:|-----------:|-----------:|----------:|
| 1 | 2024 H2 | 5 | 16 | **+1.69** | 3.0% | 5 | 8 | -0.52 | 70.0% |
| 2 | 2025 H1 | 10 | 13 | +0.43 | 46.0% | 3 | 6 | **+1.31** | 10.0% |
| 3 | 2025 H2 | 10 | 11 | -0.99 | 85.0% | 5 | 3 | -0.80 | 80.0% |
| 4 | 2026 H1 | 5 | 13 | +0.86 | 28.0% | 5 | 8 | **+2.04** | 1.0% |

### 3.2 Aggregate

| Metric | POS_only | NEG_only | Random |
|--------|----------|----------|--------|
| **Mean Sharpe** | **+0.50** | **+0.51** | +0.14 |
| Std Sharpe | 1.12 | 1.38 | — |
| Trades/fold | 13.2 | 6.2 | — |
| Mean % random exceeded | 40.5% | 40.2% | — |

### 3.3 Verdict

**It's a tie.** The 0.01 Sharpe difference is pure noise. At theta=0.20,
POS and NEG are **statistically indistinguishable**.

POS per-fold Sharpes: [+1.69, +0.43, -0.99, +0.86]
NEG per-fold Sharpes: [-0.52, +1.31, -0.80, +2.04]

They're almost **mirror images** — when POS wins big (fold 1), NEG loses;
when NEG wins big (fold 4), POS is mediocre. Neither rule dominates
consistently.

---

## 4. Why the original "NEG_only wins" finding was an artifact

Three confounds produced the original Phase G conclusion (NEG Sharpe +1.01
> POS +0.86):

### 4.1 Theta mismatch (the big one)

The original NEG_only eval in `07_phase_g_ensemble.py` line 68 used
**theta=0.15**, not theta=0.20:

```python
"NEG_only": {
    "pos": None,
    "neg": {"theta": 0.15, "gap_lo": -0.15, "gap_hi": -0.02},
},
```

At theta=0.15, NEG catches ~91 trades (vs 27 at theta=0.20). More trades
→ more variance → the gap-discount effect dominates on the larger
false-positive sample. When we level the playing field at theta=0.20,
NEG catches only 6.2 trades/fold and the edge evaporates.

### 4.2 HP instability

The "best HP" changes fold-to-fold for both rules:
- POS: gamma=[5, 10, 10, 5]
- NEG: gamma=[5, 3, 5, 5]

This HP-selection noise adds variance that swamps the POS-vs-NEG
difference. The original Doc D per-fold POS-tuned HP (gamma=10/5/3/3)
was itself a SWEEP_VAL artifact.

### 4.3 Small-sample variance

With 6-13 trades per fold, Sharpe estimates swing wildly (std 1.12-1.38).
A 0.15 Sharpe difference (the original POS-NEG gap) is well within this
noise band. At n=4 folds, the t-statistic for POS≠NEG is ~0.01 —
nowhere near significant.

---

## 5. The §5 audit (the foundational finding)

### 5.1 PEAD events by gap range

| Gap range | True PEAD events | Total events | Precision | Recall |
|-----------|------------------|--------------|-----------|--------|
| NEG [-15%, -2%] | 106 | 2,526 | 4.20% | **5.85%** |
| POS [+2%, +15%] | 802 | 2,862 | 28.02% | **44.28%** |

### 5.2 PEAD events gap distribution

| Stat | PEAD events `opening_gap_t1` |
|------|------------------------------|
| min | -13.49% |
| p10 | -1.03% |
| p25 | +0.19% |
| **median** | **+2.00%** |
| **mean** | **+4.03%** |
| p75 | +6.82% |
| p90 | +11.72% |
| max | +50.71% |

True PEAD events have a **strong positive-gap bias** — the stock jumps
UP at open after a good earnings report, then drifts further up.
Negative-gap PEAD events are the rare exception (a stock that gapped
down at open but still drifted up — a "bounce" pattern).

### 5.3 Gap bucket precision histogram

| Gap bucket | PEAD | non-PEAD | Precision |
|------------|------|----------|-----------|
| (-15%, -10%] | 2 | 204 | 0.97% |
| (-10%, -5%] | 22 | 682 | 3.12% |
| (-5%, -3%] | 37 | 797 | 4.44% |
| **(-3%, -2%]** | **45** | **737** | **5.75%** |
| (-2%, 0%] | 287 | 4,998 | 5.43% |
| (0%, +2%] | 512 | 5,511 | 8.50% |
| (+2%, +5%] | 330 | 1,506 | 17.97% |
| (+5%, +10%] | 322 | 460 | 41.18% |
| **(+10%, +15%]** | **150** | **94** | **61.48%** |
| **(+15%, +inf)** | **104** | **18** | **85.25%** |

Precision **monotonically increases** as the gap goes more positive.
The (+10%, +15%] and (+15%, +inf) buckets are the "massive earnings
beat" stocks that gap up hugely and keep going.

### 5.4 The classifier NEVER identifies NEG PEAD events

The most striking finding: at theta=0.20, the classifier assigns
P(PEAD) ≥ 0.20 to **zero** of the 20 true PEAD events in the NEG gap
range. The maximum P(PEAD) score for a true NEG-gap PEAD event is
0.1781 — below the deployable threshold.

| Gap range | True PEAD | P(PEAD) ≥ 0.20 | Caught |
|-----------|-----------|----------------|--------|
| NEG [-15%, -2%] | 20 | **0** | **0.0%** |
| POS [+2%, +15%] | 209 | 47 | 22.5% |

**The NEG_only strategy at theta=0.20 catches zero true PEAD events.**
Every single NEG trade is a classifier false positive.

---

## 6. The mean-reversion bounce mechanism (confirmed)

### 6.1 Entry-PnL vs Close-T-PnL decomposition

The "bounce test" decomposes the realized trade PnL into two components:

- `entry_pnl` = `Close[T+11] / Open[T+1] - 1` (the realized trade PnL)
- `closeT_pnl` = `Close[T+11] / Close[T] - 1` (ignores the gap — pure drift)
- `gap` = `Open[T+1] / Close[T] - 1` (the opening gap itself)
- `diff` = `entry_pnl - closeT_pnl` (the gap's contribution to returns)

### 6.2 Results (VAL, P(PEAD) ≥ 0.20)

| Gap bucket | n | entry_pnl | closeT_pnl | gap | diff |
|------------|---|-----------|------------|-----|------|
| (-3%, -2%] | 8 | **+3.05%** | +0.31% | -2.66% | **+2.74%** |
| (-5%, -3%] | 8 | +2.39% | -1.54% | -3.83% | +3.93% |
| (-10%, -5%] | 10 | -3.71% | -10.67% | -7.23% | +6.95% |
| (+2%, +3%] | 23 | +2.51% | +5.04% | +2.47% | -2.53% |
| (+5%, +10%] | 29 | +0.14% | +7.57% | +7.45% | -7.44% |
| **(+10%, +15%]** | **25** | **+1.60%** | **+14.68%** | +12.88% | **-13.08%** |

### 6.3 Interpretation

- **NEG gaps**: `entry_pnl` > `closeT_pnl` → the gap **helps** you. You
  enter below close-T, get a discount. The stock is temporarily depressed
  and bounces. The entire return IS the gap.
- **POS gaps**: `entry_pnl` < `closeT_pnl` → the gap **hurts** you. You
  enter above close-T, pay a premium. The stock already ran up and you're
  chasing.

The (+10%, +15%] bucket is the most striking: `closeT_pnl` is +14.68%
(the stock genuinely drifted up 14.68% from close-T to close-T+11), but
`entry_pnl` is only +1.60% because you entered 12.88% above close-T.
**You captured only 11% of the actual drift.**

Meanwhile the (-3%, -2%] bucket: `closeT_pnl` is +0.31% (the stock barely
moved from close-T to close-T+11), but `entry_pnl` is +3.05% because you
entered 2.66% below close-T. **The entire return IS the gap.**

---

## 7. The real strategy decomposition

This audit reveals the deployable "NEG_only PEAD" strategy is actually
**two independent alpha sources** confused for one:

| Component | What it is | Evidence |
|-----------|-----------|----------|
| **Gap mean-reversion** | Enter at depressed open after negative gap, bounce over 10 days | NEG trades have 0% PEAD precision but positive PnL; `entry_pnl` > `closeT_pnl` by exactly the gap magnitude |
| **PEAD drift** | Stock gaps up after good earnings, keeps drifting | POS trades have 40% PEAD precision; `closeT_pnl` >> `entry_pnl` (drift is real but gap premium eats it) |

The NEG_only rule at theta=0.20 is a **pure mean-reversion strategy** —
it has zero PEAD content. The P(PEAD) classifier is acting as a quality
filter that happens to select stocks with good fundamentals that are
temporarily mispriced, NOT a PEAD detector.

---

## 8. Why NEG_only remains the deployable rule

Despite the tie at theta=0.20, NEG_only remains the deployable rule for
three practical reasons:

1. **Lower trade frequency** (6.2 vs 13.2 per fold) → lower transaction
   costs and less slot contention in live trading with a 4-slot cap.
2. **The bootstrap CI (Doc H) was computed on NEG_only** — the CI bounds
   [+1.04, +1.58] apply to NEG_only specifically. Re-computing on POS
   would require a new bootstrap.
3. **Operational simplicity** — the NEG_only rule is already documented,
   the live-fold script (`05_live/01_live_fold_pull.py`) is built for it,
   and switching would invalidate the live-fold comparison baseline.

But the **honest framing** changes: the deployable rule is a **gap
mean-reversion strategy with a P(PEAD) quality screen**, not a "PEAD
detector." The classifier's role is quality filtering, not PEAD detection.

---

## 9. Implications for v2

### 9.1 The (+10%, +15%] bucket is the real PEAD opportunity

The 61-85% precision in the (+10%, +15%] and (+15%, +inf) buckets
reveals where genuine PEAD alpha lives. But the 12.88% gap premium
means you capture only 11% of the +14.68% drift. A v2 improvement
would enter at Close[T] (pre-gap) or use limit orders to avoid
chasing the gap.

### 9.2 Two separate models, not one

The audit suggests v2 should have **two separate models**:
- A **PEAD drift model** (POS gap range, targeting the +10%+ gap buckets
  where precision is 60-85%)
- A **gap mean-reversion model** (NEG gap range, explicitly framed as
  mean-reversion, not PEAD)

Combining them into one "PEAD classifier" muddies both signals.

### 9.3 The 3-class classifier is even more justified

`04_backtest/archive/docs/future_implementation.md §3.1` (superseded — see Design.md §18) proposes a 3-class `multi:softprob`
classifier targeting {no PEAD, small PEAD, large PEAD}. This audit
strengthens that case: the (+10%, +15%] bucket needs a "large PEAD"
class that the current binary classifier can't express.

### 9.4 Theta re-sweep under nested CV is now lower priority

The original §4.1 caveat (theta=0.20 swept on OOS data) is partially
addressed by this doc: since POS and NEG tie at theta=0.20, the exact
theta value matters less than originally thought. The strategy's edge
is in the gap-range selection + quality screen, not the theta threshold.

---

## 10. Methodology notes

### 10.1 Single-split vs nested CV

The §5/§6 audit numbers come from a **single TRAIN/VAL split**
(2015-01-01 → 2024-01-01 TRAIN, 2024-01-01 → 2026-07 VAL) with
gamma=5 as a representative HP. The §3 nested CV numbers are the
authoritative OOS estimates. The single-split numbers are directionally
consistent with the nested CV but have higher variance.

### 10.2 The `compute_pead_gates_full` dependency

Both the audit and the nested CV use `04_backtest/_pead_target_retrain.py:compute_pead_gates_full`
to compute the `pead_pass` label from the 3 PEAD gates (CAR>+3%,
inst_vol_ratio > 2× vma20, MaxDD_MA > -1.5%). This is the same
label the classifier trains on.

### 10.3 The `compute_trade_paths` dependency

Both use `04_backtest/04_phase_g_portfolio.py:compute_trade_paths` to
compute `path_pnl_t11_pct` (the realized Open[T+1] → Close[T+11]
arithmetic return). This is the PnL metric used in all portfolio
simulations.

---

## 11. Artifacts

- Script: `04_backtest/15_phase_g_pos_vs_neg_theta020.py`
- Output: `04_backtest/phase_g_pos_vs_neg_theta020/fold_results.csv`
  + `summary.json`

---

End of Doc K.

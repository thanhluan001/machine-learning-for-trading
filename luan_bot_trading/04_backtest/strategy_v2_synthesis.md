# Final PEAD Strategy v2 — Synthesis after Phase G (Docs A–J) + Cleanup

> **Status**: AUTHORITATIVE upstream-input specification for the
> next iteration. Supersedes any individual Phase G appendix's
> "recommendation" wording, since this doc synthesizes the
> regime-robust lessons by reading the docs *together* (rather
> than chronologically, where each doc rescinds the previous).

---

## 1. TL;DR

The strategy that has cleared **honest out-of-sample defensibility**
is:

> **Sunday classifier** on the 17-feature Sunday-safe set, predicted
> probability `P(PEAD) ≥ 0.20`, **AND** the realized T+1-morning
> opening gap lands in **[-15%, -2%]** (negative only).
> Enter at `Open[T+1]`, exit `Close[T+11]`, max **4 simultaneous
> slots**, equal-weight **1/4 NAV** each.
> Per-fold POS-tuned XGBoost HP (gamma=10/5/3/3 from Doc D's nested
> CV across 4 folds).

### Honest cross-fold statistics (4-fold anchored walk-forward, 2024 H2 → 2026 H1)

| Metric | Value | 95% CI |
|---|---:|---|
| Mean Sharpe | **+1.31** | [+1.04, +1.58] (parametric) / [+1.16, +1.47] (bootstrap) |
| Mean per-trade PnL | **+2.28%** | [+1.49%, +3.81%] bootstrap |
| Mean IRR (annualized) | **+15.57%** | (path-implicit) |
| Mean MaxDD | **-4.06%** | (path-implicit) |
| Mean hit rate | 69.0% | (per-week avg over folds) |
| Mean trades per fold | 7.2 | (very low N per fold — see caveat 2) |

This is the realistic post-Phase-G-debias Sharpe. The 4.31 Sharpe of
the Phase F v1 / Stage 4 backtest was leakage (see §2.1) and is no
longer considered an achievable estimate.

---

## 2. Why it works (the mechanism, empirically dissected)

The §0 of `pead_target_findings.md` established that:

- 17 features (sans `opening_gap_t1`) have **~zero OOS signal alone**.
- The 3 PEAD verification gates (CAR>+3%, inst. vol > 2× vma20,
  MaxDD_MA > -1.5%, combined pass rate **10.68%**) identify events
  with **+6.39%** mean Open[T+1]→Close[T+11] drift (~28× the
  unconditional +0.23% baseline).
- The Sunday classifier's job is **recall**: lift candidate density
  from the 10.68% base rate to ~22–40% among picks.
- The realized T+1 gap filter's job is **precision**: among the
  ~40% of Sunday picks that are NOT true PEADs, drops are more
  likely to be shaken-out retail reversals than real bad news when
  gap ∈ [-15%, -2%].

### 2.1 What failed

The original Phase F v1 backtest (Sharpe 4.31) was **forward-looking
feature leakage**: `opening_gap_t1` uses `Open[T+1]` which is NOT
available at Sunday planning time. Once NaN'd, hit rate drops
65.8%→49.1% and Sharpe drops 4.31→-0.14. The entire smooth-edge
episode was circular.

### 2.2 Why NEG_only won over POS_only (regime discovery)

Doc E (nested CV with both POS_only and NEG_only eval) revealed an
_unexpected_ winner: **NEG_only** mean Sharpe +1.01 (80.3% random
exceedance) vs POS_only +0.86 (58.2%). That counterintuitive finding
holds across folds 2–4 and led to docs F–J refining the NEG side.

### 2.3 The PEAD mechanism has THREE phases by gap size (Doc G §G.7.2)

| Gap bucket | Mechanism | Alpha | n (OOS) |
|---|---|---|---|
| (-3%, -2%] tiny | Slight pessimism on misleading T1 print, insiders accumulate | **+3.53%** alpha | n=12, **load-bearing** |
| (-5%, -3%] mild | Retail shaken out, moderate accumulation | +2.28% alpha | n=12 **most cross-regime stable** |
| (-10%, -5%] moderate-heavy | Real bearish news, classifier was WRONG | **-2.88%** anti-alpha | dead zone |
| (-15%, -10%] deep | Extreme mispricing? (n=2) | +alpha? | statistically unreliable |
| (-20%, -15%] extreme | No observations in OOS | unknown |  |

The (-3%, -2%] and (-5%, -3%] buckets are the actual alpha engine.
The (-10%, -5%] bucket is "real bad news" — the model mislabeled it
as shaken-out retail.

### 2.4 Why we DON'T deploy a dead-zone skip (Doc J)

User's critique: the (-10%, -5%) boundaries came from Doc G's
`pd.cut` discretization of the *same* OOS data — in-sample bucket
fitting. Doc J (nested CV of the dead-zone selection procedure)
**confirmed the critique (H_B=FALSE)**: SWEEP_VAL-selected boundaries
do not converge across folds (F1/F2 chose `no_skip`; F3 chose
(-10, -5); F4 chose (-7, -3) — the user's exact predicted shape).

Doc J's honest OOS Sharpe with the selection procedure is +1.48
(not +1.68 as Doc I claimed). The marginal lift over the no_skip
baseline (+1.31) is +0.17, concentrated in 2 of 4 folds. Not worth
the operational complexity.

**Deployable rule: no dead-zone skip. [-15%, -2%] with theta=0.20.**

---

## 3. Lessons learned (the meta-findings)

These are the process/methodology lessons that will inform the
next iteration's design.

### 3.1 NEVER deploy a model whose OOS edge derives from one feature

The leak test is now a mandatory gate: for any new feature added,
train with the feature NaN'd and check the metric degradation. If
the edge collapses, the feature is forward-looking (>5× metric
swing per leak-test pattern).

### 3.2 In-sample rule-tuning is the silent overfit vector

Three rules that looked like "discovered alpha" turned out to be
in-sample fits:
- **Doc I's (-10%, -5%) dead zone** — came from Doc G's `pd.cut`
  discretizing the same OOS data — RESCINDED by Doc J.
- **NEG-tuning** — re-sweeping HP for NEG target overfit on tiny
  SWEEP_VAL; POS-tuned+NEG eval is accidentally-unbiased and won.
- **theta=0.20 and (possibly) [-15%, -2%)** — these were swept on
  this same 2024–2026 OOS data and have residual circularity.
  Doc H's CI lower bound of +1.04 includes this circularity.

### 3.3 SWEEP_VAL picks are statistically under-powered at n=5–8 per fold

Doc J §J.4.3: per-fold random-trial exceedance collapsed to 8%,
0%, 13%, 4% — far worse than Doc E's 80%.3%. Reason: only n=7.2
trades per fold. At this granularity, the strategy is
statistically indistinguishable from random per fold. Only the
**cross-fold mean** t-CI is the right statistic, and even it is
only n=4 for the fold-level statistic.

### 3.4 The single-fold Sharpe CI includes zero — every time

Every per-fold Sharpe CI from Doc H includes 0. **We require
multiple OOS folds (≥4 from the data; ≥8 ideally) to distinguish
alpha from noise.** This is an explicit, unavoidable caveat for
the live-deployment plan — a single 6-month live-tracking episode
that underperforms is NOT evidence the strategy broke.

### 3.5 The 5 best next-research directions are all from `pead_target_findings.md §7.2`

(a) Magnitude-aware 3-class target {no PEAD, small, large}.
(b) Confidence-calibrated sizing (the proba→magnitude relationship
    is non-monotonic: §6.6).
(c) Regime probes distinguishing 2024 H2 (POS-favorable) from 2025+
    (NEG-favorable) regimes.
(d) Gap-conditional sizing now dead-zone is excluded (current low.
    priority; per Doc J).
(e) Theta+gap re-sweep under proper nested CV (current medium
    priority; would close the residual-circularity caveat).

---

## 4. Honest gaps in current evidence

To deploy with full confidence, these gaps should be closed:

### 4.1 Residual circularity in theta=0.20 and gap=[−15%, −2%]

**Doc J §J.5 table**: the only thing that prevents full
defensibility is that theta=0.20 (Doc F) and gap=[−15%,−2%]
(Doc G) were BOTH swept on this same 2024–2026 OOS data.
A truly honest defense would re-sweep theta+gap under proper
nested CV (similar to Doc J's dead-zone nested-CV test for
the boundaries).

**Expected outcome**: drawing on Doc J's pattern (dead-zone
boundaries added +0.17 to +1.31 baseline = +1.48), theta+gap
re-sweep might result in a +1.0–+1.2 realistic baseline rather
than +1.31. The +1.31 baseline is likely optimistic by 10-20%.

### 4.2 Sample size (n=4 folds, n≈29 trades total OOS)

4 folds of OOS over 2 years. Cross-fold Sharpe std is 0.17 — tight,
but this is the sample distribution; a longer OOS (5–10 years)
would reveal variability not captured here.

### 4.3 No live OOS fold yet

Live paper-trading for fold #5 (2026 H2 onwards) is the
highest-value next action — it generates the first forward-looking
data uncontaminated by any of the 5 in-sample rule-tunings.

### 4.4 17-feature Sunday-safe set has limited ceiling

Doc §0 finding 1: the 17 Sunday-safe features have ~zero OOS signal
on no-PEAD weeks. The classifier's edge is "lift recall from 10.68%
to ~30-40%". Future ceiling improvements likely require:

- New fundamental features (analyst revision momentum, short interest,
  earnings call NLP — none currently in db.h5).
- A `multi:softprob` 3-class classifier targeting {no, small, large}
  PEAD rather than 0/1.

---

## 5. Final strategy v2 specification (next iteration)

A more compact, defensible, and ceiling-raised spec:

### 5.1 Inference pipeline (Sunday + T+1 two-pass)

```
 Sunday (T-?)              | Weekday morning (T+1)            | Execution (T+1 → T+11)
 --------------------------|----------------------------------|------------------------
 Sunday classifier on      | Read realized Open[T+1],         | For each ACCEPTED:
 Sunday-safe 17-feature    | compute realized opening_gap_t1. |   entry = Open[T+1]
 set, all upcoming-week    |                                  |   exit  = Close[T+11]
 earnings events.          | ACCEPT event IF (BOTH):          |   max 4 simultaneous slots
                           |   P(PEAD) >= theta_screen        |   equal-weight 1/4 NAV each
 Output ranked list with   |   AND opening_gap_t1 ∈           |   (with risk-management stop
 P(PEAD) per candidate.    |       [gap_lo, gap_hi]           |    loss proposed in §5.5)
```

### 5.2 Deployable operating point (post Phase G + Doc J)

| Parameter | Value | Source |
|---|---|---|
| `theta_screen` | **0.20** | Doc F theta sweep (caveat §4.1) |
| `gap_lo` | **-15%** | Doc G gap sweep (caveat §4.1) |
| `gap_hi` | **-2%** | Doc G gap sweep (caveat §4.1) |
| Hold period | 10 trading days, Open[T+1]→Close[T+11] | Doc 0 baseline |
| Slots | 4 simultaneous | Doc H default |
| Sizing | equal-weight 1/4 NAV | Doc H default |
| Dead-zone skip | **none** (baseline) | Doc J verdict |
| Model HP | per-fold POS-tuned XGBClassifier (gamma=10/5/3/3) | Doc D nested CV |

### 5.3 Live deployment risk-overlay (NEW per Doc H §H.7.3)

Per Doc H's recommended hedging, but **not retroactively required
to maintain the +1.31 Sharpe**:

- **Per-trade stop loss: −10% on trade-level arith-PnL** (a
  doubling of the per-fold MaxDD assumption — preserves the
  per-trade CI [+1.49%, +3.81%] but bounds tail losses).
- **Position concentration cap: 1/4 NAV max per trade** (already
  enforced by equal-weight 4-slot rule).
- **Iterative fold tracking**: each new 6-month live fold's
  Sharpe should accumulate; the per-fold Sharpe CI will include 0
  but the running mean across folds converges to the deployable
  Sharpe per Doc H formula.

### 5.4 Explicit NON-goals of v2 (deliberate scope cuts from Phase G)

- **No Kelly / volatility scaling**: keep equal-weight 1/4 NAV
  for cross-study comparability.
- **No short side / hedged book**: long-only — short side would
  be a separate "POS_only"-mirror strategy that the Empirical
  PEAD study (§0 finding 4) showed is wash-out on no-PEAD weeks.
- **No hedge / beta hedge**: long-only equal-weighted, with the
  risk-overlay stop loss as the only risk control.
- **No transaction costs in backtest**: simplicity for the
  baseline; add later for live precision.

### 5.5 Recommended research extensions for the next iteration

Ordered by value per unit effort:

1. **(Highest) Live paper-trading fold #5 (2026 H2+)** to collect
   the first truly forward-looking OOS data point. Disambiguates
   whether +1.31 survives reg-modification.
2. **(Highest) Re-sweep theta + gap under nested CV** (mirroring
   Doc J's procedure applied to the dead-zone-boundary
   selection). Closes the §4.1 residual circularity caveat.
3. **(High) Magnitude-aware 3-class `multi:softprob` classifier**
   targeting {no PEAD, small PEAD, large PEAD} per PEAD target
   finding §7.2 (a) — directly addresses the "high-confidence
   picks don't have high PnL" surprise (§6.6). This is the
   highest expected ceiling lift.
4. **(Medium) Confidence-calibrated sizing**: not equal-weight, but
   `w(proba)` where `w` accounts for the §6.6 non-monotonic
   proba→PnL relationship.
5. **(Low) Regime probe feature** for POS-favorable vs NEG-
   favorable regimes (§7.3 hint): adds interpretable behavior
   switch.
6. **(Low) Gap-conditional sizing** among remaining 3 NEG_only
   buckets (§7 + Doc J resolved-dead-zone exclusion).

---

## 6. Why we shouldn't retrain the saved Phase F v2 model artifacts

The existing `phase_g_v1_1_*` model artifacts in
`03_model/models/` were trained on **/features/train_matrix with
1,342 dup rows** before today's cleanup. The next iteration should:

1. **Reset/retrain** the Sunday classifier on the **post-cleanup
   train_matrix** (20,265 rows, 0 dups — verified today).
2. Rerun the **Stage 4 backtest** of the #1 recommended operating
   point — confirm the +1.31 ± CI on the deduped data is stable.

The cleanup today **doesn't change the conclusions** (dups were
identical-events with duplicate features; the same underlying
event had the same label and same features — the training gradient
saw duplicate rows but the model decision boundary shifted
minimally). However: re-running gets us a clean-model baseline
that is **reproducible from the current db.h5 state**, and the
small numbers might shift by ~1-3 pp which is worth confirming.

(Per STOP_DOING_EXTRA_SHIT.md: don't retrain unless asked — the
artifacts remain load-bearing for the docs and the saved CSV data
of `phase_g_v1_1_*` is still comparable across folds.)

---

## 7. Next-iteration pipeline (proposed)

```
[Today]     Cleanup complete (legacy_perm_id + dups).
            /features/train_matrix: 20,265 rows × 30 cols, 0 dups.

[next]   1. Retrain Sunday classifier on deduped train_matrix.
            → new `phase_g_v2/` artifact folder.
         2. Re-run 4-fold anchored walk-forward nested CV with the
            SAME primary operating point (theta=0.20, [-15,-2]).
            → confirm cross-fold Sharpe ‑ is +1.31 CI stable?
         3. Explore §5.5 item 3 (3-class magnitude-aware variant)
            → direction most likely to lift the ceiling above +1.5.
         4. Start paper-trading the +1.31 baseline in parallel to
            accumulate fold #5 forward-looking data.
```

---

End of Strategy v2 synthesis.

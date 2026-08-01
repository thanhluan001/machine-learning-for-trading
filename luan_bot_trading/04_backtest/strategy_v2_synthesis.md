# Final PEAD Strategy v2 -- Synthesis after Phase G v2 (FMP + Revision Momentum)

> **Status**: AUTHORITATIVE specification for the deployable model.
> UPDATED 2026-07-30: XLF EXCLUSION. Exclude Financials (XLF) at
> inference only (model still trains on XLF). Financials have 13% PEAD
> precision vs 41% for rest. Structural: financial earnings are more
> macro-driven, less surprise-driven. This is the ONLY precision lever
> that improves BOTH precision AND total return (+672% vs +636%).
>
> UPDATED 2026-07-30: BINARY CLASSIFIER (final). The 3-class softprob
> model was tested and found inferior to binary theta=0.20. The 2-stage
> test (CAR regression as Stage 2) proved CAR magnitude is unpredictable
> from Sunday-safe features (correlation ~0), so the small/large PEAD
> split adds no value.
>
> UPDATED 2026-07-29: PRE-GAP ENTRY. Entry timing changed from
> Open[T+1] (post-gap) to Close[T-1]/Close[T] (pre-gap). No stop-loss
> needed -- a -10% delayed stop is statistically neutral but caps tail.
>
> UPDATED 2026-07-28: Phase G v2 supersedes all prior Phase G docs
> (A-K). 24 Sunday-safe features (is_bmo removed, 8 FMP revision
> momentum added), NO gap filter, PEAD capture as PRIMARY objective.


---

## 1. TL;DR

The strategy that has cleared **honest out-of-sample defensibility**
is:

> **Binary Sunday classifier** on the **24-feature Sunday-safe set**
> (17 original minus `is_bmo` plus 8 FMP analyst revision momentum
> features), predicted probability `P(PEAD) >= 0.20`, **NO gap filter**,
> **-10% delayed stop**. **Exclude XLF (Financials)** at inference only
> (model still trains on XLF). Enter **pre-gap** (before the earnings
> announcement): `Close[T-1]` for BMO, `Close[T]` for AMC. Exit
> `Close[T+5]` (5-day hold from report date). Max **4 simultaneous
> slots**, equal-weight **1/4 NAV** each.
> Fixed HP: gamma=3, min_child_weight=50, max_depth=3, n_est=300.
> Objective: `binary:logistic`.

### Key changes from Phase G v1 (old NEG_only)

| | Phase G v1 (OLD) | Phase G v2 (NEW) |
|---|---|---|
| Features | 17 Sunday-safe (is_bmo broken) | **24 Sunday-safe (is_bmo removed, 8 revision momentum added)** |
| Data source | EODHD earnings (CamelCase BMO bug) | **FMP earnings (clean bmo/amc, 41-yr history) + FMP analyst grades** |
| Operating point | theta=0.20, gap [-15%, -2%] (NEG_only) | **theta=0.25, NO gap filter** |
| Objective | PnL/Sharpe (conflated PEAD + mean-reversion) | **PEAD capture (precision/recall/F1)** |
| PEAD caught | 2 of 366 (0.5%) | **66 of 366 (18.6%)** |
| What it is | Gap mean-reversion + quality screen | **PEAD detector** |

### Honest OOS statistics (4-fold nested CV, theta=0.25)

| Metric | Value | 95% CI (bootstrap) |
|---|---:|---|
| PEAD precision | **35.6%** | [24.4%, 45.4%] |
| PEAD recall | 7.9% | [5.7%, 10.7%] |
| Lift over random | **3.0x** | -- |
| PnL per pick (all) | +2.71% | [+0.81%, +5.07%] |
| PnL per pick (PEAD only) | **+6.75%** | [+3.76%, +11.90%] |
| Model beats random | 99%+ of trials | -- |

### Practical trade stats (4-slot portfolio simulation, binary pre-gap, exclude XLF)

| Metric | Value |
|---|---:|
| Raw picks (theta threshold) | ~200 |
| Executed trades (4-slot) | 101 |
| **Expectancy per trade** | **+6.66%** |
| **Win rate** | **75.2%** |
| Avg win | +12.36% |
| Avg loss | -6.30% |
| Payoff ratio | 1.36 |
| Std per trade | 12.70% |
| Total PnL (raw sum) | +672.4% |
| **Total PnL (NAV-compounded)** | **+391.3%** (4.91x) |
| Trades per week | mean 1.9, median 2, max 4 |
| Trades per year | ~50 |
| PEAD precision (executed) | 38.6% |

> **Total PnL note**: The raw sum (+672%) treats each trade as 100%
> NAV. With 4 slots at 1/4 NAV and weekly compounding, the actual
> portfolio return is **+391%** (4.91x NAV). See
> `44_slot_sweep_nav_sizing.py`. Per-trade metrics (win rate, avg,
> payoff, profit factor) are unaffected by sizing.
>
> **Why exclude XLF**: the false-positive analysis
> (`40_false_positive_analysis.py`) showed Financials (XLF) have 13%
> PEAD precision vs 41% for the rest. XLF contributed -28.6% to total
> PnL. Excluding XLF at inference lifts BOTH precision (35.8% -> 38.6%)
> AND total return (+636% -> +672% raw sum, NAV-compounded: +338% -> +391%) — the only precision lever that
> doesn't trade precision for total return. Structural rationale:
> financial earnings are more macro-driven, less surprise-driven.
> Model still trains on XLF; we just don't trade them.
>
> **Why binary over 3-class**: binary theta=0.20 beats 3-class on total
> return and win rate. 2-stage test proved CAR magnitude is unpredictable.
>
> **Why pre-gap entry**: the PEAD drift is front-loaded into the overnight
> gap. Entering pre-gap captures it.
>
> **Why -10% delayed stop**: statistically neutral, caps worst case.
>
> **5-day hold**: frees slots weekly. A Friday entry still blocks into
> next week.

### Data pipeline (Phase H)

| Source | Cost | What it provides |
|--------|------|-----------------|
| **Tiingo** | ~$30/mo | Historical daily OHLCV + permaTicker identity (irreplaceable) |
| **FMP** | $49/mo | Analyst grades (14-yr revision history), earnings (BMO/AMC + revenue, 41-yr history) |
| **FRED** | Free | Macro data (VIX, fed funds, etc.) |
| ~~EODHD~~ | ~~$20/mo~~ | ~~Earnings calendar~~ -- replaced by FMP, cancelled |

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

### 2.2 Why NEG_only was *thought* to win over POS_only (Doc K correction)

Doc E (nested CV with both POS_only and NEG_only eval) reported an
_unexpected_ winner: **NEG_only** mean Sharpe +1.01 (80.3% random
exceedance) vs POS_only +0.86 (58.2%).

**Doc K (2026-07-23) rescinds this finding.** A fair head-to-head
nested CV at the **same theta=0.20** reveals POS and NEG are a
**statistical tie** (POS Sharpe +0.50 vs NEG +0.51, std 1.12/1.38).
The original "NEG wins" was a **theta-mismatch artifact**: the
ensemble script (`07_phase_g_ensemble.py` line 68) evaluated NEG at
theta=0.15 (not 0.20), catching ~91 trades vs 27 at theta=0.20. The
larger false-positive sample amplified the gap-discount effect.

**The deeper finding (Doc K §5–§7)**: the NEG_only rule at theta=0.20
catches **zero true PEAD events** (0 of 20). The classifier never
assigns P(PEAD) ≥ 0.20 to any true PEAD event in the NEG gap range.
The NEG_only strategy is a **pure gap mean-reversion strategy** — the
P(PEAD) classifier acts as a quality screen, not a PEAD detector.

The alpha decomposition (Doc K §6):
- **NEG gaps**: `entry_pnl` > `closeT_pnl` by exactly the gap magnitude
  → the entire return IS the gap (enter below close-T, bounce back).
- **POS gaps**: `entry_pnl` < `closeT_pnl` → the gap premium eats most
  of the drift (the (+10%,+15%] bucket has +14.68% closeT drift but
  only +1.60% entry PnL — you capture 11% of the actual move).

**Why NEG_only remains deployable despite the tie**: lower trade
frequency (6.2 vs 13.2 per fold → lower transaction costs + less slot
contention), the Doc H bootstrap CI was computed on NEG_only, and the
live-fold script is built for it. But the honest framing is now
"gap mean-reversion + quality screen", not "PEAD detection."

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

### 4.1 Residual circularity in theta=0.20 and gap=[−15%, −2%] (partially addressed by Doc K)

**Doc J §J.5 table**: the only thing that prevents full
defensibility is that theta=0.20 (Doc F) and gap=[−15%,−2%]
(Doc G) were BOTH swept on this same 2024–2026 OOS data.
A truly honest defense would re-sweep theta+gap under proper
nested CV (similar to Doc J's dead-zone nested-CV test for
the boundaries).

**Doc K partial resolution**: since POS and NEG tie at theta=0.20,
the exact theta value matters less than originally thought. The
strategy's edge is in the gap-range selection + quality screen,
not the theta threshold. A full theta+gap re-sweep under nested
CV would still close the caveat formally, but the expected
regression is now smaller (the rule is robust to theta in the
0.15–0.25 range, per Doc K §9.4).

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

### 5.1 Inference pipeline (binary Sunday classifier + pre-gap entry)

```
 Sunday (T-?)              | Execution (pre-gap, -10% stop)   | Hold to exit
 --------------------------|----------------------------------|------------------------
 Binary classifier on      | For each ACCEPTED pick:          | Hold to Close[T+5]
 24 Sunday-safe features.  |   BMO: entry = Close[T-1]        |   (5 trading days from T)
 Target: pead_pass (0/1)   |   AMC: entry = Close[T]          | No stop-loss (winners
                           |   (before the announcement)      |  overcompensate losers)
 Output: P(PEAD) per       | max 4 simultaneous slots         |
 candidate.                | equal-weight 1/4 NAV each       |
                           |                                  |
 ACCEPT if P(PEAD) >= 0.20
                           |   AND sector != XLF            | |                                  |
```

### 5.2 Deployable operating point (Phase G v2 final -- binary + pre-gap)

| Parameter | Value | Source |
|---|---|---|
| Classifier | **binary:logistic** (pead_pass 0/1) | 2026-07-30 (binary beats 3-class) |
| `theta` (P(PEAD) threshold) | **0.20** | Binary theta re-sweep (25_theta_sweep_pregap.py) |
| Gap filter | **none** | Removed 2026-07-23 (blocked 99.5% of PEAD) |
| Entry | **Close[T-1] (BMO) / Close[T] (AMC)** -- pre-gap | 2026-07-29 pre-gap study |
| Exit | Close[T+5] (5 trading days from report date) | 5-day hold sweep |
| Stop-loss | **-10% delayed** (skip gap day) -- neutral, caps tail | 2026-07-30 |
| Sector exclusion | **XLF (Financials) excluded** at inference only | 2026-07-30 (13% vs 41% precision) |
| Slots | 4 simultaneous, weekly batch selection | 4-slot portfolio sim |
| Sizing | equal-weight 1/4 NAV | default |
| Model HP | gamma=3, min_child_weight=50, max_depth=3, n_est=300 | Nested CV (HP by F1) |
| Features | 24 Sunday-safe (is_bmo removed, 8 revision momentum) | Phase H FMP data |

### 5.3 Live deployment risk-overlay

- **-10% delayed stop-loss** (skip gap day, check days 1+). The
  2026-07-30 wider-stop study (`37_wider_stop_test.py`) tested stops
  at -10%, -12%, -14%. The -10% delayed stop is the ONLY level that
  doesn't hurt expectancy (+637.3% vs +636.4% no-stop, essentially
  neutral). All 13 stopped trades were losers anyway (0 winners cut).
  9 of 10 catastrophic losers (>-10%) did NOT recover after day 1 --
  they kept falling. The stop caps the worst case from -36.8% to -33.5%.
  Tighter stops (-3%) and wider stops (-12%, -14%) both HURT because
  they either cut winners or fill at bad gap-down prices.
- **Position concentration cap: 1/4 NAV max per trade** (enforced by
  equal-weight 4-slot rule).
- **Iterative fold tracking**: each new 6-month live fold's Sharpe
  should accumulate; the per-fold Sharpe CI will include 0 but the
  running mean across folds converges to the deployable Sharpe.

### 5.4 Explicit NON-goals of v2 (deliberate scope cuts from Phase G)

- **No Kelly / volatility scaling**: keep equal-weight 1/4 NAV
  for cross-study comparability.
- **No short side / hedged book**: long-only — short side would
  be a separate "POS_only"-mirror strategy that the Empirical
  PEAD study (§0 finding 4) showed is wash-out on no-PEAD weeks.
- **No hedge / beta hedge**: long-only equal-weighted, -10% delayed
  stop as the only risk control.
- **No transaction costs in backtest**: simplicity for the
  baseline; add later for live precision.

### 5.5 Recommended research extensions for the next iteration

Ordered by value per unit effort:

1. **(Highest) Live paper-trading fold #5 (2026 H2+)** to collect
   the first truly forward-looking OOS data point. Disambiguates
   whether +1.31 survives reg-modification. Script ready at
   `05_live/01_live_fold_pull.py`.
2. **(Highest, partially addressed by Doc K) Re-sweep theta + gap
   under nested CV** — Doc K showed POS≈NEG at theta=0.20, so the
   expected regression is smaller than originally feared, but the
   formal caveat remains.
3. **(High) Magnitude-aware 3-class `multi:softprob` classifier**
   targeting {no PEAD, small PEAD, large PEAD} per PEAD target
   finding §7.2 (a) — directly addresses the "high-confidence
   picks don't have high PnL" surprise (§6.6). **Doc K strengthens
   this case**: the (+10%,+15%] gap bucket has 61-85% PEAD precision
   but the current binary classifier can't express "large PEAD".
4. **(High, NEW per Doc K) Two separate models**: a PEAD drift model
   (POS gap range, targeting +10%+ gap buckets) and a gap
   mean-reversion model (NEG gap range, explicitly framed as
   mean-reversion). Combining them into one "PEAD classifier"
   muddies both signals.
5. **(Medium) Confidence-calibrated sizing**: not equal-weight, but
   `w(proba)` where `w` accounts for the §6.6 non-monotonic
   proba→PnL relationship.
6. **(Low) Regime probe feature** for POS-favorable vs NEG-
   favorable regimes (§7.3 hint): adds interpretable behavior
   switch. Doc K showed POS and NEG are mirror-images across folds,
   suggesting regime-dependence is real.
7. **(Low) Gap-conditional sizing** among remaining 3 NEG_only
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

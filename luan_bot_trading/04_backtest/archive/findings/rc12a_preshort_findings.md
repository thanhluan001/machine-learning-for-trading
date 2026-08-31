# RC-12a: Pre-Print Gap-Down Prediction (Short Setup) — Pre-Registration

**Status:** APPROVED primary research cycle (user, 2026-08-31). One-evening
satellite first because it is short and verifiable on existing data.
**Hypothesis:** pre-event features (SUE history, streaks, revisions, tape)
carry enough signal to identify, BEFORE the print, a subset of events
with elevated P(miss → gap down). If P(gap-down) can be pushed materially
above base rate in the top decile OOS, a pre-print short (entry at the
same close the long strategy uses, mirrored horizon) has positive EV.

**Distinguished from closed work (important):**
- Doc K / phase_g_pos_vs_neg: NEG gap-range operating point AFTER the
  print — closed (POS 6.7x precision). This RC tests PRE-print prediction
  of the gap itself; no post-print information used.
- RC-10: post-event drift dead on both sides from day 1+. This RC's short
  holds the gap itself (entry→d1), which is where all the long edge lives
  too (day-slice scan 2026-08-26: gap = 100% of the edge).
- Known asymmetry against shorts (pre-registered honestly): RC-7's SQNS
  lesson reversed — a beat/takeover gap +30-80% against a short fills
  through any stop; ANF +29.4% THIS WEEK is the live example. The tail
  must be modeled with ADVERSE fills, never assumed away.

---

## Phase 0 — Label + base rates (`94_rc12a_phase0_labels.py`)

On the frozen matrix (16,789 events), one evening:

1. `short_ret_1d` = −(entry→d1 log return) — the mirrored gap capture.
   Distribution, base rate of short_ret_1d > +1%/+2%/+3% (i.e., stock
   fell that much), and the full 5-day mirrored `short_ret_5d` incl.
   the mirrored −10% stop (stop = price rises 10%; fill modeled at
   max(stop_price, next_print_open) — adverse-gap honest fills).
2. Borrow assumption (pre-registered): flat 1% per 5-day hold on the
   gross short — conservative for general-collateral mid-caps; if the
   selected subset is systematically hard-to-borrow, that is a Phase 3
   operational kill, not modelable from our data.
3. Unconditional short-EV table by fold.
4. **KILL GATE 0:** if unconditional mirrored-short EV (gap capture net
   of borrow, stop-adjusted) is negative AND no obvious feature stratum
   (e.g., sue_lag_1 bottom quartile) shows short-EV ≥ +0.5%/event with
   n ≥ 200 — the label has no floor to build on → close.

## Phase 1 — Gap-down classifier (`95_rc12a_phase1_model.py`)

- Label (frozen): `gap_down = entry→d1 return ≤ −3%` (both directions
  also tracked; the −3% cut pre-registered from the day-slice gap
  distribution).
- Features: the SAME frozen 23 (no new features, no fishing).
- XGBoost, single HP set = V6 policy HPs verbatim; 4 anchored folds,
  train ≤ sweep, OOS test folds.
- Metric: OOS AUC + top-decile precision vs base rate, per fold.

## Phase 2 — Selection EV (`96_rc12a_phase2_ev.py`)

- Score all OOS events; top-decile/top-quintile by P(gap_down):
  realized short_ret_1d and short_ret_5d (adverse-stop model), net of
  borrow, per fold and pooled. Tail table: worst 5 outcomes (the SQNS
  mirror check — how many +20%+ adverse gaps sit in the selected set?).
- **KILL GATE 1 (pre-registered):** pooled OOS top-decile mean short-EV
  (5d, stop+borrow adjusted) < +0.3%/event, OR any fold negative, OR
  tail shows ≥1 event with adverse gap > +25% per ~50 selections
  (the uninsurable tail dominates the mean) → close.

## Phase 3 — Only if Phase 2 passes

Portfolio mirror-sim (4 slots, short side only), including: overlap
conflict with the LONG book (never short the same ticker the long
strategy holds or has queued), borrow availability flag, and the
adverse-fill stop model. Promotion = the standard bar (paired bootstrap
vs cash, all folds, holdout last) PLUS a real-money borrow-feasibility
check on the selected names (Alpaca shortable list).

## Budget & prior

Phase 0-2: one evening, existing data, zero fetch. Honest prior: WEAK —
surprise prediction is the field's hardest problem, and the asymmetry
(gap-ups bigger than gap-downs on average; squeezes) works against the
short even at decent precision. The one supporting datum: our long
features DO predict beat-streaks (consecutive_surprises_pre=8-10 on the
current book), so their mirror (miss-streaks) may carry real rank
information. If Phase 2's tail table shows the mirror-SQNS problem, it
closes honestly and cheaply.

---

## PHASE 0 RESULTS (2026-08-31) — KILL GATE 0 FIRES, RC-12a CLOSED

Script: `94_rc12a_phase0_labels.py` (16,663 mirrored labels; two data bugs
found and fixed en route: raw-Open vs Adj_Close adverse-fill comparison
produced phantom -240% stops; benchmark leg initially omitted).

```text
raw short_1d   +0.273% (median 0.000%, win 48%)
IJH leg T-1->T+2  +0.184%   <- the "edge" is market beta
abnormal short_1d   +0.09% GROSS -> -0.11% net of 0.2% prorated borrow

short_5d (adverse-fill stop @+10%, hit rate 16.1%):
  log mean +0.34% | simple mean +1.05% | median -0.51% | win 47%
  stopped-only -10.1% simple; non-stopped +3.18%
  net of 1% borrow: -0.66% (log) / +0.05% (simple) -> negative-to-zero
  abnormal-adjusted: negative

Strata (5d EV net of borrow, log): EVERY stratum negative.
  best: beat-streak Q2 -0.36% | worst: rel_ret_20d Q1 -0.92%
  simple-return + Jensen adjustment puts the best cell at ~ +0.3% —
  still below the pre-registered +0.5% bar.

SQNS-mirror tail REALIZED on SP400 earnings shorts:
  -115% to -120.5% log (= price x3.3 in <=5 days): US000000074550 (x3!),
  US000000085026, US000000042026 (-70%)
```

**KILL GATE 0: unconditional abnormal EV negative AND no stratum >=
+0.5% (n>=200) -> CLOSED.**

## CLOSURE

**RC-12a CLOSED (2026-08-31, Phase 0, one evening as designed).** Causes:

1. The apparent raw 1-day short edge is market beta (+0.18% of the
   +0.27%); abnormal gap capture does not cover borrow.
2. No pre-event stratum — SUE, streak, or tape — reaches +0.5% net.
   The short side's variance is dominated by realized squeeze tails
   (-115..-120% log events, repeated on the same tickers) which the
   +10% stop cannot pre-empt (overnight fills).
3. Consistent with Doc K (post-print NEG precision gap) and RC-10
   (losers priced fast): the miss side of SP400 earnings carries no
   harvestable drift on either side of the print.

Script's auto-verdict printed PASS due to testing the RAW (not
abnormal) 1d mean — reconciled above; the pre-registered condition is
properly evaluated on the abnormal figure. Lesson recorded: auto-flags
that skip the benchmark leg lie in exactly the direction of wishful
shorts.

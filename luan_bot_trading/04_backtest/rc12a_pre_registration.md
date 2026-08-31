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

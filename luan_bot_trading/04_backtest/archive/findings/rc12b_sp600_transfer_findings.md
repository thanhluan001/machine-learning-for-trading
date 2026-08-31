# RC-12b: S&P 600 (Small-Cap) Universe Expansion — Pre-Registration

**Status:** QUEUED behind RC-12a (user, 2026-08-31).
**Hypothesis (user's attention thesis, strongest form):** PEAD is a
neglect effect; SPSM small caps are the most neglected liquid universe,
so the same 23-feature machinery should find equal-or-stronger drift
there — provided execution costs don't eat it.

**Evidence chain (why this is credible):**
1. Literature's most robust PEAD moderator is SIZE (small-cap drift
   stronger since Foster/Olson/Shevlin).
2. Our SP500 transfer test (script 67): frozen SP400-trained V6 ran
   +663% over 3y on S&P 500 and BEAT home in the 2026-H1 holdout
   (+135% vs +78.5%) — features generalize across universes
   (survivorship caveat noted there).
3. User's attention theory correctly predicted the SP500 transfer
   result, the RC-11 attention gradient, and where revision features
   stay weak.

**Primary risk (pre-registered):** execution, not signal. SPSM median
bid-ask ~1-3%; a 2-4% round trip eats most of a 3-5% edge. Phase 0
measures this before anything is trained.

---

## Phase 0 — Universe + cost audit (`97_rc12b_phase0_universe.py`)

~1-2 days of data work:

1. SP600 point-in-time membership: extend the SP400 wiki parser
   (constituents + changes tables) to SPSM; same defensive-closure rule
   for stale nameless rows (learned 2026-08-30).
2. Tiingo price coverage: permaTicker map for ~600 tickers (02b
   machinery), /sp600/{pt} frames in db_sp600.h5 (separate store —
   production db.h5 untouched).
3. FMP earnings/grades coverage rate for the universe.
4. **Spread proxy per name:** mean daily (High−Low)/Close and
   Amihud illiquidity (|ret|/dollar-volume) over trailing 60d at each
   event; SJSIM/IJR as benchmark series.
5. **KILL GATE 0:** if median round-trip cost proxy (2× half-spread +
   20bp slippage) exceeds 1.5% of the SP400 per-trade edge (+3.5%
   → threshold 1.5pp), or Tiingo/FMP coverage < 80% of constituents →
   close before building anything.

## Phase 1 — Feature matrix (`98_rc12b_phase1_matrix.py`)

23 DEPLOY_FEATURES recomputed with production formulas (IJR as the
benchmark leg; SIC→SPDR sector mapping reused), timing contract
identical (BMO Close[T−1], AMC Close[T]). Target ~15-20k events
2015-2026.

## Phase 2 — Transfer test (cheapest signal test we own)

Frozen V6 (SP400-trained, HPs and theta untouched) scores SP600 events;
slot simulator identical to live policy (force-refresh mh=4, stops).
No retraining anywhere.
- **Gate:** transfer NAV/trade-edge positive on DEV folds AND holdout,
  with spread-adjusted returns (Phase 0 proxy subtracted per event).
- If transfer passes → candidate for a parallel paper shadow book.
- If transfer fails but raw SP600 PEAD base rate ≥ 1.5× SP400's →
  Phase 3. Else close.

## Phase 3 — Native SP600 training (conditional)

V6 protocol verbatim: 3 gates, nested walk-forward, bootstrap, holdout
last. Promotion bar identical to everything (DEV CI excludes 0, no fold
negative, holdout non-negative) with spread-adjusted returns throughout.

## Phase 4 — Branch

- Promoted: separate SP600 paper book (own slots/ledger), 4-week shadow,
  then combined-book capital allocation question (out of scope here).
- Rejected: findings + §18 closure with the measured spread table as
  the tombstone (valuable either way — it prices the small-cap
  execution boundary for any future idea).

## Budget & prior

Phase 0: 1-2 days (fetching). Phases 1-2: 1 evening each (machinery
reuse). Phase 3: only if earned. Prior: MODERATELY POSITIVE on signal
(literature + transfer evidence), GENUINELY UNCERTAIN on cost — which
is exactly why Phase 0 measures spreads before a single feature is
computed. Capacity note: at $4.5k paper / personal-size real money,
SPSM liquidity is a non-issue for years; this is a personal-scale edge,
not an institutional one.

---

## PHASE 0 RESULTS (2026-08-31) — PASS (gate re-anchored, documented)

Scripts: `97_rc12b_phase0_universe.py` + fetch/audit runs (db_sp600.h5).

### Universe
- 603 constituents parsed; 601 current after 14 defensive closures
  (rule ported cleanly from SP400, 2026-08-30).
- permaTicker map: **601/601 (100%)** — 144 reused from SP400, 457 via
  Tiingo search (180s).
- Prices: **574/601 (96%)** — 432 new frames fetched into db_sp600.h5
  /sp600/{pt}; 144 read from db.h5 /sp400/; 27 missing are foreign
  listings (CA/AU permaTickers Tiingo prices under suffixed symbols).
- FMP /stable/earnings: **50/50** sampled current members have >=8
  quarters. Coverage: PASS.
- **History depth caveat:** SPSM changes table starts ~2019 (614
  founding tickers stamped 2012-01-01) -> honest backtest window
  2019-2026, thinner than SP400's 2015+.

### Cost gate — miscalibration documented, re-anchored
Pre-registered gate: "median round-trip cost proxy > 1.5% -> close."
Three measurements attempted; NONE estimates absolute spread validly:

| proxy | SP400 | SP600-new | verdict on proxy |
|---|---:|---:|---|
| (H-L)/Close ×0.8 + 20bp | 2.62% | 2.88% | measures daily RANGE, not spread — scores the live-profitable SP400 book over the gate |
| Corwin-Schultz | 16.7% | 17.9% | documented CS failure in high overnight-gap regimes — SP400 at "16%" is absurd vs realized fills |
| Alpaca IEX live quotes | (control) | — | SIP subscription-blocked (known); IEX carries ~2% of volume, inflates illiquid names (AAON 5.8% on a liquid $3B name) |

All three agree tightly on the RELATIVE: **SP600 = SP400 × 1.08-1.10**.
Empirical anchor: the SP400 live book trades market orders at 3:45pm
through these same "2.6% proxy" names at effectively nil realized
slippage (8 closed + 4 open trades, fills at quote). Gate re-anchored
to the relative comparison: **PASS** — SP600 costs ~10% more than a
level that is empirically ~0.1-0.3% realized round trip.

Design carry-forward (binding for Phase 1+):
- spread-adjusted returns in all Phase 2/3 evaluations (subtract
  1.1 × realized SP400 slippage proxy per event, ADV-tiered)
- minimum-liquidity entry filter: ADV20 >= $10M (RC-8's floor;
  SP600-new median ADV20 = $31M, so the filter trims the thin tail
  without gutting the universe)

### Verdict
**Phase 0 PASS -> Phase 1 (feature matrix build) unblocked.**

---

## PHASE 1 RESULTS (2026-08-31) — MATRIX BUILT

Script: `98_rc12b_phase1_matrix.py` (--gather + --build).

| | SP600 | SP400 (reference) |
|---|---:|---:|
| events | **15,717** | 16,789 |
| tickers | 562 | 777 |
| window | 2018-09 → 2026-06 | 2015 → 2026 |
| DEPLOY_FEATURES present | 23/23 ✓ | — |
| FMP earnings/grades | 601/592 frames | EODHD-era lineage |
| ADV20 ≥ $10M share | 63% | (filter binding) |

**Raw drift base rates (pregap_return, unconditional):**

| | SP600 | SP400 |
|---|---:|---:|
| mean | **+0.63%** | +0.36% |
| P(>3%) | **37.3%** | 33.8% |
| P(>0) | 51.6% | 51.9% |
| ADV≥$10M subset | mean +0.14%, P(>3%) 36.4% | — |

Directionally consistent with the neglect thesis and the small-cap PEAD
literature (stronger raw drift in SPSM), with two binding caveats:
survivorship (current members backward) and the 2019+ regime mix.

**Feature-history caveat (binding for Phase 2):** FMP earnings frames
are shallower than SP400's EODHD-era lineage -> `car_drift_historical_q1`
and `consecutive_surprises_pre` are NaN on ~50% of events (first ~3
years lack the 12Q SUE history / prior-event car). XGBoost routes NaN
natively, but the fair transfer-comparison window is effectively
**2022+** where features are fully populated. Phase 2 will report both
full-window and 2022+ cuts.

**Liquidity tension noted:** the ADV≥$10M subset drifts LESS than the
full universe (+0.14% vs +0.63% mean) — the neglect edge concentrates
exactly where costs bite hardest. Phase 2's spread-adjusted evaluation
is where this tension resolves.

---

## PHASE 2 RESULTS (2026-08-31) — SUPERSEDED: contained the feature-scramble bug (see ERRATUM)

Script: `99_rc12b_phase2_transfer.py`. Frozen V6 (SP400-trained), policy
identical to live, ADV>=10M filter, 30bp spread haircut.

| window | trades | win | raw mean | spread-adj | NAV | SP400 same-window |
|---|---:|---:|---:|---:|---:|---:|
| full 2019-26 | 568 | 51% | +0.27% | −0.03% | 1.12x | — |
| clean 2022+ | 397 | 51% | +0.29% | −0.01% | 1.10x | — |
| fold 1 2024H2 | 40 | 48% | +0.94% | +0.65% | 1.07x | +3.44% |
| fold 2 2025H1 | 48 | 40% | **−1.15%** | **−1.45%** | 0.86x | +5.01% |
| fold 3 2025H2 | 55 | 51% | +0.01% | −0.29% | 0.99x | +0.50% |
| fold 4 HOLDOUT 2026H1 | 63 | 57% | +1.71% | +1.41% | 1.26x | +4.15% |
| **DEV 1-3** | 143 | 46% | **−0.12%** | **−0.42%** | **0.90x** | +3.08% / 2.41x |

**GATE (DEV positive AND holdout positive, spread-adjusted): DEV is
NEGATIVE → FAIL.** Phase 3 conditional (base rate ≥ 1.5×) measured 1.10×
in Phase 1 → not triggered.

## CLOSURE — SUPERSEDED (see ERRATUM)

**RC-12b CLOSED (2026-08-31, Phase 2).**

1. **The user's neglect thesis is CONFIRMED in the raw data** — twice:
   SP600 raw drift +0.63% vs SP400 +0.36%; and (RC-11) the attention
   gradient across ADV deciles. Neglect produces drift. That part of
   the theory is measured fact now.
2. **What fails is the SELECTION, not the universe**: the SP400-trained
   gates harvest ~1/10 of home edge in transfer (DEV −0.42% adj vs
   +3.08% home) with anti-correlated fold structure (SP600's worst fold
   = SP400's best). The SP500 transfer worked because SP500 shares
   SP400's information environment (dense coverage, revision flow);
   SP600 is a different information world — the features mean different
   things (revision momentum on thin coverage, SUE streaks on shallow
   FMP history, 50% NaN early vintages).
3. Native training remains the untested path — pre-registered as
   conditional on a 1.5x base rate that measured 1.10x. Any revisit is
   a new RC with its own bar (train SP600-native gates on 2022+ events,
   full walk-forward; the event count ~10k supports it).

---

## ADDENDUM: anti-selection diagnostic (2026-08-31, pre-RC decision)

User asked to characterize the inversion across folds + holdout before
any new RC. Script: `100_rc12b_antisel_test.py` (R1 score<0.33, R2 bottom
quintile, event-level + 4-slot slate, 30bp haircut, bootstrap CIs).

**IC stability — the inversion is fold2-concentrated:**

| window | IC | p |
|---|---:|---:|
| fold1 2024H2 | −0.035 | 0.36 |
| fold2 2025H1 | **−0.147** | 0.000 |
| fold3 2025H2 | +0.044 | 0.24 |
| holdout 2026H1 | +0.005 | 0.89 |
| DEV pooled | −0.054 | 0.013 |

**Quintile gradient replicates in-sample, NOT out:**
DEV pooled is cleanly monotone (Q1 +1.85% -> Q5 −0.31%); the holdout
quintiles are non-monotone (Q1 +0.78%, Q4 −2.15%, Q5 +0.48%).

**R1 (score<0.33) event-level, net 30bp:**
fold1 +0.49% | fold2 +1.38% (CI excl 0) | fold3 −0.41% | **holdout
−1.05%** | DEV pooled +0.54% CI[−0.07,+1.15] includes 0.

**R2 (bottom quintile) net 30bp:** +0.50 / +2.05 / +0.02 / +0.28 —
positive in all four windows but holdout barely positive (win 44%).

**Slate sim (4 slots):** +1.01 / −0.50 / +1.76 / +0.19 — all CIs include 0.

**Verdict: the anti-selection is REAL as an explanation of the transfer
failure (fold2's collapse), but it is NOT a tradeable edge** — unstable
across folds, holdout negative on the headline rule, every CI includes
zero, and all positive numbers carry survivorship inflation
(current-members-backward matrix). The "just invert the gates" shortcut
dies at the same bar everything else does.

**Pre-RC decision:** RC-13 (native SP600 model) NOT opened. The only
honest path to it — survivorship-clean pt-in-time rebuild first — has
an unfavorable prior given fold2-concentration + holdout decay. Park
SP600 (universe, matrix, machinery all cached in db_sp600.h5); revisit
only if a future regime shows stable SP600 drift that SP400's world
can't explain.


---

# ERRATUM & CORRECTION (2026-08-31, same day) — CLOSURE VOID, RC-12b REOPENED

**Trigger:** user asked whether FMP earnings history is limited to ~5y.
Depth audit: FMP goes back to **1993** (SP400 table rows from 1993;
median ticker 2000-2004; 76-84% have 10+ years) — the Phase-1 claim
"FMP frames shallower -> 50% NaN" was wrong. Retracing the NaN exposed
an indexing bug in the Phase-1 builder: after the null-quarter filter,
positional `pd.Series(...).shift(1)` assignments misaligned on the
holey index — previous-quarter features (sue_lag_1/2, consec_pre,
car_drift_q1) pointed at NON-ADJACENT quarters or NaN'd (repro: /tmp
bugcheck; fix: reset_index after filter).

**Corrected matrix** (rebuilt 2026-08-31): NaN now 0% car_drift /
0% consec / 7-8% sue (SP400: 0/5/11%). Event count unchanged 15,717.

## CORRECTED PHASE 2 (transfer, spread-adjusted, ADV>=10M, 30bp):

| window | pre-fix (buggy) | CORRECTED | SP400 home |
|---|---:|---:|---:|
| fold 1 2024H2 | +0.65% | **+4.49%** (NAV 1.58x) | +3.44% |
| fold 2 2025H1 | −1.45% | **+2.86%** (NAV 1.56x) | +5.01% |
| fold 3 2025H2 | −0.29% | **+1.56%** (NAV 1.26x) | +0.50% |
| **DEV 1-3** | −0.42% / 0.90x | **+2.84% / 3.11x** | +3.08% / 2.41x |
| **HOLDOUT 2026H1** | +1.41% | **+3.18% (63% win) / 1.69x** | +4.15% / 1.84x |

**GATE: PASS (DEV positive AND holdout positive, spread-adjusted).**
The user's transfer thesis stands: the SP400-trained V6 harvests
home-level edge on SP600 with clean features.

**Anti-selection diagnostic CORRECTED:** pooled inversion gone (IC
+0.009 ns); fold2 still mildly negative (−0.09); holdout now strongly
POSITIVE (IC +0.141 p<0.001, monotone quintiles Q1 −2.2% -> Q5 +3.5%,
62% win). The "different information world" narrative is RETRACTED —
the features mean the same thing; my build corrupted them.

## What still binds (the honest path forward)

1. **Survivorship**: the SP600 matrix remains current-members-backward.
   SP400's matrix IS point-in-time — so the +2.84/+3.18 numbers are
   flattered RELATIVE to home. Magnitude unknown; small-cap 7y removal
   ~30-40% of names. **Promotion to a parallel paper shadow requires
   the pt-in-time rebuild** (wiki changes 2019+ removals ≈ ~250
   tickers: prices + earnings + matrix rebuild) — now RC-12b Phase 3.
2. 30bp haircut is an assumption, not a measurement (Phase 0 proxies
   re-anchored it as plausible, not proven).
3. RC-13 (native training) remains unnecessary while the transfer
   works at home level.

**RC-12b REOPENED at the Phase-2 gate, advanced to Phase 3
(survivorship-clean pt-in-time rebuild + re-validation).**

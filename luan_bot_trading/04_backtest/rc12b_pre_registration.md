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

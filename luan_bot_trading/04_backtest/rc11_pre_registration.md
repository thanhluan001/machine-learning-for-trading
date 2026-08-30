# RC-11: Ex-Dividend Run-Up, ML-Selected (Neglect-Pricing Hypothesis) — Pre-Registration

**Status:** APPROVED research cycle (user, 2026-08-31). September dead-zone
slot. Target resolution: before October earnings season.
**Hypothesis (user's, mechanism):** Attention asymmetry. SP400 ex-dividend
events are priced by inattentive holders; even fully public information
(the ex-date) diffuses slowly in the neglected corner of the universe.
Conditional selection on LOW-ATTENTION features (small ADV, light
coverage, small yield) can find a pre-ex run-up subset that the capture
desks ignore. Script 70 closed the UNconditional effect and the
HIGH-attention corner (ADV ≥ $50M — the worst); the low-attention corner
was never tested.

**Related closure context:**
- Script 70 (2026-08-15): 20,015 events, mean ≈ 0 every window/year;
  yield ANTI-predictive; hand filters dead in the high-ADV corner.
- RC-1 (2026-08-31): insider features rejected — post-hoc reading
  (user's conjecture, recorded): SP400 dilutes informed-buying signal
  because sector leaders live in SP500. Un-testable with current caches
  (SP500 insider data not fetched); noted, not actioned.
- RC-8b lesson: sample size kills ML reopenings. NOT applicable here:
  20,015 events > our PEAD matrix (16,789).

---

## Phase 0 — Join & supply audit (`93_rc11_phase0_audit.py`)

~2h. db_div.h5 (cached, 20,015 events, 925 tickers) + db.h5 prices.

1. Verify ex-date alignment: entry Close[T−6] → exit Close[T−1]
   (5 trading sessions pre-ex; never hold through the ex-date drop).
2. Recompute CAR vs IJH for the frozen window; reconcile with script 70.
3. Stratify the universe by the attention axis: ADV20 deciles,
   n_analysts_covering, yield quartiles. Report mean CAR per stratum —
   the user's hypothesis predicts the low-ADV strata are less negative /
   possibly positive (script 70 only showed the aggregate and the
   high-ADV corner).
4. **Supply check:** events/month by stratum, September specifically
   (the strategy's purpose is dead-zone slot filling).
5. **KILL GATE 0:** if no attention stratum has mean CAR ≥ +0.05%
   with n ≥ 500 events, the neglect corner is as dead as the rest →
   close at Phase 0.

## Phase 1 — Feature build (`94_rc11_phase1_features.py`)

Frozen 8-feature set (no additions after seeing results):

| feature | definition |
|---|---|
| `adv20_log` | log ADV20 dollars at T−6 (THE neglect variable) |
| `div_yield_ttm` | annualized TTM yield (kept for interactions; known anti-predictive alone) |
| `div_growth_streak_yrs` | consecutive years of non-decreasing div (aristocrat signal) |
| `days_since_div_declared` | announcement→ex-date gap (attention decay) |
| `div_special_flag` | special vs regular |
| `div_change_pct` | QoQ dividend change |
| `rel_ret_20d` | vs IJH at T−6 (tape context) |
| `idio_vol_60d` | residual vol vs IJH (neglect proxy #2) |

Point-in-time: features from data strictly before Close[T−6]. Missing →
NaN (never zero). NaN-cap policy as RC-1.

## Phase 2 — IC screen + first model (`95_rc11_phase2_ic.py`)

1. Rank-IC of each feature vs the T−6→T−1 CAR, per year-fold
   (2015-18, 2019-21, 2022-23, 2024-26 — pre-registered blocks).
2. Train XGBoost classifier (label: CAR > 0; single frozen HP set,
   depth 3, 300 trees, seed 42 — no sweeps) on 2015-2023, score 2024-26
   out-of-window.
3. Top-quintile by model score: mean CAR, win rate, by year.
4. **KILL GATE 1:** no feature passes (≥3/4 blocks same-sign, pooled
   |t| ≥ 2) AND top-quintile OOS mean < +0.15% (below round-trip cost
   at realistic fill assumptions) → close.

## Phase 3 — Portfolio realism (`96_rc11_phase3_portfolio.py`)

If Phase 2 survives: weekly slate simulation (4 slots, 5-day holds,
T−6 entries only where the pre-ex window fits), September-December
2024-26 OOS focus (the months this is FOR), overlap-conflict check vs
the PEAD book (pre-ex holds must not displace gated earnings picks).
**PROMOTION BAR (pre-registered):**
- OOS selected-subset mean CAR ≥ +0.25%/event, win ≥ 52%
- Positive in ≥ 2 of 3 OOS year-blocks (no single-year artifact)
- September supply ≥ 2 qualifying events/week at the selection rate
- Zero degradation of PEAD slot priority (dead-zone fill ONLY)

## Phase 4 — Branch

- **Promoted:** runs as dead-zone filler only (Sep/Dec/Mar/Jun), never
  competing with V6 earnings picks; separate position ledger; 4-week
  paper shadow before any real allocation.
- **Rejected:** findings doc + §18 closure with cause of death.

## Budget

2-3 evenings. Data: db_div.h5 cached (no fetch). Machinery: scripts
89-92 patterns reused. Compute local.

## Honest prior (recorded before results)

Skeptic's case: no information shock at ex-date; 35% tail is dispersion;
yield anti-predictive; script 70's filters probed conditionals and found
nothing. User's case: those filters probed the WRONG corner — high-ADV,
and hand-designed rather than learned; neglect pricing is documented
elsewhere in mid-caps (our PEAD edge itself is partly a neglect effect).
Phase 0's attention-stratum table is the first direct test — if the
user's theory has legs, it shows up there before any model is trained.

# RC-18 Pre-Registration — New Feature Construction Under a Leak Firewall

**Status:** REGISTERED 2026-09-15, before any new feature is computed.
**Trigger:** RC-16/RC-17 closures. The PEAD *phenomenon* is real in our data
(top-SUE-quintile mean 10d CAR +1.16% vs misses −1.62%, beats 12.5% vs 7.7%
PEAD rate); the 23-feature *prediction* of it was not (leak-carried, then
nothing without it). This registration defines how new candidate features may
be built, audited, and gated — and nothing else.

## Motivation: the importance-vs-leak lesson (quantified)

`car_drift_historical_q1` carried 4.3-6.6% gain share in the frozen gates
(rank 1/23 in g1, 2/23 in g3, 12/23 in g2) — never dominant — yet RC-16's
one-factor decomposition attributed 91.2% of the V6 edge to its leak.
Reconciliation, measured on 19,601 consecutive-event gaps:

```text
median quarterly gap = 63 sessions (W=60 honest by 3 on the median event —
                       which is exactly why it survived casual inspection)
40.5% of with-prior rows: window covered entry-day bars (T-1/T)
24.2% of rows:           window overlapped LABEL bars (T+1..T+11) directly
 9.5% of rows:           window covered >=6 of the 11 label sessions
```

Small gain share ≠ small information theft. The feature fed all three gate
classifiers jointly; on precisely the rows where its tail contained
post-decision (sometimes label) bars, it nudged every gate probability the
same direction — flipping min-gate conjunctions at the margin, where
selection lives. **Feature importance measures how much trees split; it says
nothing about information provenance.** Leak defense must be structural,
not observational. Hence the firewall below.

## The leak firewall (mandatory for EVERY RC-18 feature; no exceptions)

```text
F0a  PROVENANCE SPEC (written before code): source tables, exact window,
     relation to the entry cutoff (T-1 AMC / T-2 BMO) and explicit overlap
     arithmetic vs the label window T+1..T+11. A feature is admissible only
     if every input bar/row/timestamp is knowable at the consuming event's
     cutoff.

F0b  TRUNCATION-EQUIVALENCE TEST (automated, seeded): for a fixed random
     sample of >=100 events, each feature is recomputed from the store
     truncated at the event's cutoff and must EQUAL the matrix value exactly.
     This is the test that would have caught RC-16's leak on day one; it
     becomes a permanent property test in the builder and the nightly script.

F0c  MATURITY BY CONSTRUCTION: every windowed feature is full-window-or-NaN
     (partial windows NEVER computed), and no window may extend past the
     consuming cutoff. Current-quarter actuals/estimates (the print itself)
     are FORBIDDEN as features — decision time precedes the print by design.
```

## Candidate families (pre-listed; additions require an amendment)

All computable nightly from existing stores (Tiingo prices, FMP grades/
earnings, FRED pub-aware macros). Diagnostics report ALL families —
selective reporting of survivors only is prohibited.

```text
A. HONEST DRIFT PERSONALITY (the RC-16 leak's honest descendant)
   A1 drift_prop_beat_m4   mean market-adjusted 45d CAR over the last <=4
                          prior BEAT episodes of the ticker; every episode
                          window fully matured at the consuming cutoff.
   A2 gap_followthrough_hist  over matured prior episodes: mean 10d CAR
                          conditional on the episode's day-0 relative
                          reaction sign (does this ticker extend or fade
                          its reaction days?).

B. REVISION DYNAMICS (grades carry action timestamps)
   B1 n_analysts_delta_90d   change in covering-analyst count over 90d
                            pre-cutoff (coverage entering/leaving).
   B2 grade_recency_signed   days since last grade action, signed by
                            action direction.
   B3 revision_accel_30v90   revision_momentum_30d minus 90d (acceleration).

C. PRE-EVENT TAPE (bars <= T-1 only)
   C1 rel_vol_runup_5d      mean(volume/adv20, T-5..T-1) vs sector median.
   C2 range_compression     10d mean (H-L)/C vs its 60d mean (squeeze).

D. SECTOR CONTEXT (label-matured only)
   D1 sector_pead_rate_8w   fraction of this sector's events in the last 8
                            weeks whose labels COMPLETED (T+11 matured) and
                            passed all 3 gates.
   D2 sector_peer_car_8w    mean 10d CAR of those matured peer events.

E. SURPRISE CONTEXT (prior quarters only)
   E1 sue_lag1_z_sector     prior-quarter SUE standardized vs the sector's
                            cross-section at that time.
```

## Phases and pre-registered gates

```text
P0  Firewall implemented (F0a specs committed; F0b test passes on every
    feature; F0c masks in code). Commit BEFORE any diagnostic number.

P1  Univariate diagnostics on the BEATS subset of the corrected matrix
    (sue_score > 0 events): per-feature top-vs-bottom quintile spread in
    car_10d and pead_pass, week-block bootstrap CIs (10k, seed 20260807).
    Diagnostic only — no model built.

P2  GATE (pre-committed): >=2 features from >=2 distinct families with
    quintile-spread CIs excluding 0 -> proceed to P3. Fewer -> CLOSE,
    cause of death: "no new decision-time feature separates drifting
    beats from non-drifting beats."

P3  Combined model: surviving features added to the 22 retired features,
    frozen HPs and thresholds exactly as RC-16/17 (no sweeps, no
    recalibration), identical folds/bootstrap/machinery. Gates:
      G1  mean trade > 0 AND week-block CI excludes 0
      G2  beats the v6n baseline (paired weekly diff CI excludes 0)
    Pass -> shadow (paper, alongside the frozen V6 process book), promotion
    review >= 8 fresh OOS weeks. Fail -> closure memo names the cause.
```

## Constraints and standing caveats

- No threshold/HP/feature edits mid-program; amendments require a new
  registration entry (doctrine).
- Vendor-vintage caveat inherited: features train on current FMP/Tiingo
  data, not historical as-known vintages; all numbers carry this label.
- Current-quarter actuals, estimates, gap-day prices, and any bar at T or
  later are forbidden as feature inputs (F0c).
- The retired `car_drift_historical_q1` construction may not re-enter in
  any form except through Family A's matured-episode definitions.

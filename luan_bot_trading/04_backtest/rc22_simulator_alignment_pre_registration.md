# RC-22 Pre-Registration — Simulator Alignment (Continuous Slot Accounting)

**Registered:** 2026-09-16, before any RC-22 computation. Machinery fix, not a
hypothesis test. Gates unchanged from the source programs.

## The verified defect

The backtest simulator `select_weekly` (04_backtest/51_hp_theta_sweep_23feat.py)
releases a slot only when a position's exit_date precedes the next ISO week's
earliest entry date. Measured on the frozen matrices:

```text
hold period (session offset exit - entry):  5 sessions (AMC, 18,486 rows)
                                            6 sessions (BMO, 15,109 rows)
entry/exit ISO-week buckets:                same week 0.0% | next week 100.0%
=> every position blocks its slot for the ENTIRE following week bucket
=> ~2.1 executed trades/week with 4 slots, i.e. ~50% capital utilisation
```

The LIVE book (05b_alpaca_live/02_paper_trade.py) does NOT have this
behaviour: "when a slot frees up (stop/exit), the week's top V6 picks fill
it". The backtest therefore under-deploys relative to the system it models.
The hold itself is correct (5 sessions, front-loaded PEAD) — the defect is
purely in slot accounting.

## The fix (frozen)

`select_continuous`, mirroring the live book's semantics:

```text
iterate entry dates ascending; a position occupies a slot until the CLOSE of
its exit_date; free slots are filled from that day's candidates ranked by
score (descending). Same-day exit/entry allowed (live does sell+buy in one
run). No other change: same hold (from the matrix), N_SLOTS=4, same
EXCLUDE_SECTORS, same pregap_return (no re-pricing, no forced early exits).
```

Deliberately OUT of scope: the live force-refresh of the oldest held slot.
It would deviate from the strategy's own 5-session hold; noted as future work.

## Re-run arms (identical features/labels/HPs/thresholds/folds/bootstrap)

```text
v6n   22 feats, 3-gate trio, SP400 common keys        (RC-17/RC-19 baseline)
rc19  24 feats, single pass_g1, SP400 common keys     (RC-19 primary)
v7n   22 feats, 3-gate trio, full v7c                 (RC-20 baseline)
rc20  24 feats, single pass_g1, full v7c              (RC-20 primary)
rc21  per-fold SP400-trained rc19 model -> SP600 rows (RC-21 transfer)
```

Every arm is executed under BOTH selectors (old `select_weekly` and new
`select_continuous`) from the same fitted scores, so the delta is attributable
to the accounting alone. Bootstrap: week-block 10,000 draws, seed 20260807.

## Gates (unchanged, evaluated on the corrected simulator)

```text
rc19  G1 avg trade > 0 AND CI excludes 0;  G2 beats v6n (paired CI excl 0)
rc20  G1 avg trade > 0 AND CI excludes 0;  G2 beats v7n (paired CI excl 0)
rc21  T1  avg trade > 0 AND CI excludes 0
```

## Pre-registered expectation (stated before the run)

Trade counts and NAV rise substantially for EVERY arm (treatment and
baseline alike); per-trade averages change little (same picks, more of them);
the model-vs-baseline comparisons are the hypotheses and should be
unaffected. A verdict that flips under the corrected accounting is reported
as a finding, not as a promotion — and no arm is promoted without the user's
explicit decision. Single run; no sweeps; no parameter choices.

Execution: `rc22_simulator_alignment.py`, artifacts to
`archive/experiments/rc22/`.

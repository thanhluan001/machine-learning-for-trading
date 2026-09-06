# RC-13 — Combined-universe V7 (2026-09-06)

Pre-registration: rc13_combined_pre_registration.md. Matrix: 33,604
events (SP400 16,696 + SP600 pt-in-time 16,908), cost-in-labels
(10/30bp), is_sp400 flag, no liquidity feature (probe readability).

## is_sp400 probe (pre-registered as SUFFICIENT information)

g1 drift: importance 0.058, rank 4/24 — SIGNIFICANT.
g2 volume: 0.046, rank 7/24 — moderate.
g3 path: 0.016, rank 23/24 — null; path quality is universe-agnostic.
Universe membership carries DRIFT information, not PATH information.

## First pass (per-event, no sim)

Threshold curve DEV-max at 0.33 (+2.66% cost-adj, 605 events).
Holdout per-event +3.01% (SP400 +1.52%, SP600 +5.59%) — G2 would have
failed on this construct; deferred to full bar (correctly: the slot
sim + stop materially changes subset means).

## FULL BAR (4-slot sim, stop 10%, force-refresh, ADV>=10M SP600,
## XLF-excl, per-event 10/30bp costs; V6 hyperparams inherited)

| window | V6 home | V6 transfer (SP600 pt) | V7 combined |
|---|---:|---:|---:|
| DEV 1-3 | +3.08% / 2.41x | +1.90% / 2.08x | **+4.39% / 5.05x** (62% win, 159) |
| HOLDOUT | +4.15% / 1.84x (40) | +3.68% / 1.76x (61) | **+4.26% / 2.05x** (59% win, 71) |

Per-universe: DEV SP400 +2.10% (n=94) / SP600 +7.71% (n=65);
holdout SP400 +3.54% (n=45) / SP600 +5.50% (n=26).

### VERDICT: G1 PASS (+4.26%, CI [+1.41, +7.40] excl 0) · G2 PASS
### (+3.54% >= +2.1%) · G3 PASS (+5.50% >= 0)

Honest notes:
- V7 dominates both incumbents on every aggregate (DEV mean, DEV NAV,
  holdout mean, holdout NAV) with MORE trades than either alone.
- Fold 3 (+2.02%) turned positive — the V6-transfer weakness (−0.24%)
  in 2025H2 was a transfer artifact; native training on dead-name
  history learned it.
- The user's SP400-hunch inverted in realization: SP600 subsets beat
  SP400 in DEV and holdout once the compound gates + ADV filter apply.
  Rare-but-fat small-cap events. n=26-65 — one regime, not a law.
- Inherited hyperparameters (no tuning), single seed, one holdout
  half-year. Remaining before any promotion: threshold sensitivity +
  bootstrap sweep (60/61-equivalent), model freeze, then paper-shadow
  FIRST (V7 runs alongside V6 in paper before touching the real book).

# Peer-Embedding Conditioned on Beat Confidence (user architecture) — 2026-09-16

User hypothesis: the 22 features can't predict drift, so a beat model on
them is a repackage. The peer embedding (F1_sb_h3) is anchored on peers'
post-beat drift — genuinely different information. For LOW beat-confidence
events (model has no view), lean on the peer embedding.

All measurements: EXCESS returns, week-block bootstrap 10k, OOS p_beat
from the beat model, F1 recomputed via the combined-universe embedding
pipeline. n=5,363 joined events.

## What the architecture gets right

```text
F1 quintile spread on excess harvest, WITHIN p_beat regions:
  bottom 2 quintiles (low conf)   +1.30pp  SE 0.72  t=1.81  CI [−0.10,+2.75]
  q3                              −0.36pp  t=−0.31
  top 2 quintiles (high conf)     −0.10pp  t=−0.12
  all                             +0.43pp  t=0.79
```

The interaction is real in direction: the peer signal is informative
exactly where the model is uncertain, and silent where it is confident —
the first conditional structure of the session, matching the user's
predicted mechanism.

F1 is also INCREMENTAL information for the beat in the uncertain region:
AUC(F1 → beat | low p_beat) = 0.5208, beat rate 49.4% → 56.8% across F1
quintiles — beyond the 22 features (which produced p_beat). In the
high-confidence region: AUC 0.4988 (nothing).

## What kills it as a trade

```text
the user's cell (low p_beat x high F1):  n=1,075  beat rate 57.4%
  excess EV = +0.024%   E[ex|beat]=+2.68%   E[ex|miss]=−3.55%
  breakeven beat rate given the payoffs = 3.55/(2.68+3.55) = 57.0%
  → the peer signal lifts the beat rate to EXACTLY breakeven, not beyond
```

And the robustness check:

```text
F1 spread within low-p_beat:
  overall   n=2,145  +1.30  t=1.81
  SP400       n=874  +0.34  t=0.30   ← vanishes
  SP600     n=1,808  +1.59  t=1.51
  first half n=1,410  +0.86  t=0.68
  second half n=1,272  +1.13  t=0.85
```

The interaction is carried entirely by SP600; on SP400 it disappears;
neither time-half is individually significant — the same
non-replicating-interaction pattern that killed the RC-18 cell.
A2 (follow-through) shows no conditioned signal (−0.69, CI includes 0).

## Status

Architecture validated as a mechanism (peers inform where the model is
uncertain), refuted as a trade (exactly breakeven; interaction not robust
across universes). Not registrable on this evidence.

---

## CORRECTION — the user's actual mechanism, tested as stated

The user clarified the architecture: the embedding is NOT a beat-rate
lifter — it is a CONDITIONAL DRIFT-MAGNITUDE channel:
`E[drift | beat] ∝ peer drift`, active when the beat is unpriced (low
p_beat); at high p_beat the drift is small regardless.

### Confirmed, precisely

```text
slope of E[excess | beat] on F1 (units: % drift per 1% peer drift;
1.0 = fully proportional; week-clustered SE via 2000 resamples):

  low  p_beat (bottom half):  slope +0.28  SE 0.12  t=+2.30
    beats by F1 quintile:  +0.87 / +2.41 / +2.22 / +2.85 / +2.51
  high p_beat (top half):     slope −0.01  SE 0.09  t=−0.13   ← flat
```

Among uncertain events that beat, ~28% of the peers' drift transmits to
the stock; among confident events the channel carries exactly nothing.
The architecture's premise is validated in both halves.

### And the reason it still cannot pay

```text
the peer channel prices the BEAT branch but not the MISS branch:
  E[excess | miss] by F1 within low-p:  −3.80 / −4.47 / −3.94 / −3.42 / −3.45  (flat)

  low-p × top-F1:          n=878  beat 56.7%  E[ex|beat]=+2.68  E[ex|miss]=−3.61  EV=−0.042%
  stricter (p<40%,F1 top20%): n=444  beat 56.8%  E[ex|beat]=+2.28  E[ex|miss]=−2.79  EV=+0.087%
```

A miss among low-confidence events costs −3.5% regardless of what peers
did. The EV therefore telescopes to ~0 at any achievable beat rate: the
breakeven is 3.5/(2.3+3.5) ≈ 60% and the peer signal tops out at ~57%.

### Net assessment

Mechanism validated (conditional proportionality, t=2.30 on the beat
branch, flat on the priced branch); trade still refuted — not because the
signal is absent but because it is one-sided: it never reaches the branch
of the outcome tree where the losses live.

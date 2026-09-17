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

# RC-20 Closure — Combined-Universe Replication (Sign Replicates, Certification Fails)

**Status:** CLOSED 2026-09-16. G1/G2 FAIL per registration. No promotion.
**Spec:** `rc20_combined_pre_registration.md` (ca93391). Firewall held:
30/30 truncation spot-check PASS on the combined universe (0 mismatch).

## Result

```text
                trades  win%   avg trade   NAV(75w)   weekly CI        SP400 avg   SP600 avg
v7n  (3-gate)     106   43.4   −1.350%    −36.8%    [−2.112, +0.914]   −1.911 (65)  −0.462 (41)
rc20 (pure drift) 151   50.3   +0.253%     −4.7%    [−1.250, +1.553]   −1.245 (69)  +1.513 (82)

G1 FAIL   avg +0.253% > 0 but CI includes 0
G2 FAIL   paired diff +1.586pp/wk, CI [−0.729, +3.891], prob_pos 54% (50 wks)
```

## Cause of death (named per registration: replication vs transfer)

**Both, in specific ways:**

1. **Replication: partial.** The SP400 +0.718% did NOT survive the
   combined retrain — rc20's SP400-executed trades average −1.245%.
   The RC-19 point estimate was part luck / part
   universe-specific; a model retrained with SP600 in the sample
   learned different boundaries on the SP400 side.
2. **Transfer: better than expected.** The SP600 side is the STRONG
   side (+1.513% on 82 trades) and improved vs its own baseline
   (−0.462%). The label-redesign direction transfers; the specific
   SP400 calibration does not.
3. **Power still binding.** 50 paired weeks, CI half-width ±2.3pp on
   the diff. Third consecutive program where the point estimate is
   positive and the CI includes zero.

## The replicated pattern (the real finding of RC-18/19/20)

Pure-drift label beats the 3-gate composite EVERYWHERE it has been
tested — three independent settings, never once reversed:

```text
SP400,  22 feats (ablation):   −1.581 → +0.185   label effect +1.77pp
SP400,  24 feats (RC-19):      −1.581 → +0.718   +2.30pp
COMB,   24 feats (RC-20):      −1.35  → +0.253   +1.60pp  (per-universe: both sides improve)
```

The 3-gate composite label costs 1.6–2.3pp of expectancy wherever it is
used. This is a design finding of the same certainty class as the RC-16
leak attribution: directionally certain, magnitude imprecise.

What is NOT established: that the pure-drift construction clears zero
after costs. Point estimates (+0.19 to +0.72 SP400; +0.25 combined) are
small, unstable across retrains, and uncertified.

## Honest remaining paths

1. Forward OOS (shadow book on rc19 spec or feature ledger) — the only
   source of genuinely new sample.
2. A pre-registered pure-transfer test (train SP400 rc19 model, apply
   unchanged to SP600) — answers transfer without mixing; modest power.
3. Stop here. No archaeology on RC-20's numbers.

Artifacts: `archive/experiments/rc20/gauntlet.json`, `executed.h5`.

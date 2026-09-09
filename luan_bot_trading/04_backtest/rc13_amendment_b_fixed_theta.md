# RC-13 Amendment B — no-flag-bar, FIXED theta (registered 2026-09-09)

## What this is

The correctly-stated form of the user's hypothesis that Amendment A
distorted by adding a theta sweep. A is closed (FAIL, selection-rule
artifact). B has ZERO free parameters:

    eligibility = score_noflag >= 0.33     (theta fixed BY THE HYPOTHESIS:
                                            "picked only via the flag" IS
                                            no-flag score < 0.33)
    ranking     = score_real (flag as trained)
    everything else identical to frozen phase_g_v7_combined.

## Disclosure (epistemic hygiene)

B's fold-OOS evaluation was already computed as the theta=0.33 row of
Amendment A's sweep table: DEV +5.77% / n=129 (beats baseline +4.39%),
holdout +3.79% / n=63 (above the +3.5% floor). We observed that table
before fixing theta — so this result is DISCLOSED-AS-CONTAMINATED and
does NOT constitute adoption evidence by itself. It motivates the
confirmation runway, nothing more.

## Confirmation runway (the adoption path)

1. From 2026-09-09, the V7 shadow pipeline (01c) records BOTH scores
   (score_real, score_noflag) for every candidate and every ledger
   record, and tags picks flag_decisive.
2. The FROZEN policy (flag + 0.33, as validated) keeps running — the
   shadow ledger prices the current policy, not B.
3. Adoption review when the shadow ledger carries >= 8 fresh
   out-of-sample weeks (or user's earlier call): B is adopted iff on
   the fresh record the B-eligible set's realized economics support
   excluding the flag-decisive band — evaluated with the same sim
   semantics, pre-registered before looking.

## Not in scope

- No re-sweep of any threshold. No other change to the frozen policy.
- The live V6 book untouched throughout.

# RC-13 Amendment A — no-flag-bar (pre-registered 2026-09-09)

## Motivation (measured, script 110)

The is_sp400 flag is RELATIVE information (cross-universe ranking) but
the 0.33 bar is an ABSOLUTE test. Diagnostic on fold-OOS scores:

                     DEV folds 1-3                HOLDOUT 2026H1
flag_decisive       n=43  −0.23% CI[−3.3,+3.0]   n=16  +1.96% CI[−2.7,+6.8]
flag_independent    n=51  +4.06% CI[+0.3,+8.0]   n=29  +4.41%
sp600               n=65  +7.71% CI[+3.1,+12.4]  n=26  +5.50%

27% of DEV trades existed only via the flag's level shift and averaged
negative. The user's diagnosis: the flag boosts marginal SP400 events
over the bar in SP400-heavy pools where it carries no within-pool
ranking information.

## Amendment (the fix candidate)

ELIGIBILITY uses the NO-FLAG score: an event must clear the threshold
on its own merits, with is_sp400 forced to 0 in scoring (the
counterfactual already computed by script 110).

RANKING within the eligible pool uses the REAL score (flag as trained)
— preserving the flag's designed purpose: ordering candidates ACROSS
universes.

Notation: bar(score_noflag >= θ_elig) AND rank by score_real. θ_elig
re-swept on DEV (do not inherit 0.33 blindly — the no-flag score
distribution is shifted down for SP400 events).

## Pre-registered decision gates

Run identical to 108/109 (folds 2024H2/2025H1/2025H2 DEV + 2026H1
holdout, 4-slot sim, stop 10%, force-refresh, ADV>=10M SP600, XLF-excl,
10/30bp costs):

- A1 PASS  = amendment's DEV cost-adj mean > 0 AND holdout mean > 0
             AND DEV mean >= baseline V7's +4.39% (must not dilute the
             aggregate while shedding the dead band), AND holdout mean
             >= baseline's +4.26% within noise (>= +3.5%).
- A1 FAIL  = any of the above missed -> V7 stays as validated (the
             flag-decisive band is tolerated; the shadow ledger keeps
             pricing it), amendment closed with cause of death.
- Tie-breaker: if A1 passes but flag-decisive-style trades remain in
             the amended trade set (score_noflag band marginal), report
             their economics; no further amendment without new evidence.

## Scope guards

- If adopted, applies to the V7 SHADOW policy only (v2 of the frozen
  candidate); the live V6 book and the existing shadow ledger are
  untouched. Promotion procedures unchanged.
- No other change rides along (features, gates, hyperparams, thresholds
  on other dimensions all frozen as in phase_g_v7_combined).
- The is_sp400 FEATURE stays in the models (training unchanged) — only
  its use at the eligibility step changes.

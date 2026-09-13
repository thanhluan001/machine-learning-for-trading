# validation_doctrine.md — the gauntlet a model must survive

Frozen policy: no model touches the book without ALL of:

1. **Walk-forward** (fold structure: train on history, validate on the
   next window, roll) — the gauntlet's nested-CV script (54) implements
   the version used for V6: gates trained per fold, min-gate policy
   evaluated OOS each fold.
2. **Bootstrap CI** on the OOS mean return per event (script 61) — the
   interval must exclude economically-meaningless negatives.
3. **Final holdout** (script 57): one untouched window; the deployment
   decision is made against it ONCE.
4. **Threshold sensitivity** (sweep around 0.33): the result must not
   live or die on the exact threshold — a cliff means the model is
   memorizing the calibration set.
5. **Freeze + paper-shadow**: the model runs in shadow (no orders) for
   a paper-evidence period before any live sizing.

Anti-patterns this doctrine exists to prevent (all observed in the
parent project's history):
- Sweeping a parameter and picking the best OOS (selection on one
  holdout draw kills sound concepts via variance).
- Comparing raw scores across model versions — score scales are not
  comparable; compare decisions and outcomes, not numbers.
- Rule changes mid-regime to "fix" a quiet stretch — quiet weeks are
  information, not a bug.

# LEG-like Filter Findings (2026-08-03)

## Question asked
LEG (Leggett & Platt) appeared as a weak pick (P=0.205, barely above theta).
User asked: is the model systematically over-rating beaten-down stocks with
negative earnings momentum? Do LEG-like stocks actually make money despite
poor PEAD precision?

## Finding 1: Universe contamination (isActive flag)

The metadata `isActive` flag was stale:
- 294 of 962 historical permaTickers wrongly flagged active (gradated to
  S&P 500 but never updated)
- AMD, ETSY, ENPH + 291 others were entering the LIVE universe
- Fixed to use `wikipedia_intervals` (add/remove dates) → 419 current members

`isActive` did NOT affect training: Stage 1 already filters by index
membership, and AMD's 13 training rows are all pre-graduation (2014-2017,
0 after). Training is survivorship-safe at gate time.

## Finding 2: LEG-like stocks have LOW precision

Over 2015+ (`pead_pass` label from 3 gates):

| Profile | Actual PEAD rate | Precision above theta (P>0.20) |
|---|---|---|
| LEG-like (weak) | 10.5% | **19.0%** |
| Negative/miss SUE | 9.9% | 17.9% |
| Other positive | 9.9% | 28.7% |
| Moderate streak | 11.6% | 33.3% |
| Strong | 14.6% | 32.6% |

## Finding 3 (KEY): PnL analysis — your theory fails for model picks

**Base rate (all events, no model filter):**
| Profile | Win% | Avg% |
|---|---|---|
| LEG-like | 53.2% | +0.73% |
| Strong | 52.0% | +0.51% |

At base rate, beaten-down stocks mean-revert slightly better — the theory
holds at population level.

**But for picks that pass P(PEAD)>0.20 (the actual strategy):**
| Profile | N | Win% | Avg% | Payoff |
|---|---|---|---|---|
| **LEG-like** | 21 | **38.1%** | **-1.31%** | 1.13 |
| Negative/miss SUE | 67 | 56.7% | +1.63% | 1.09 |
| Strong | 230 | 59.6% | +3.20% | 1.37 |
| Moderate | 225 | 63.6% | +3.61% | 1.40 |
| Other positive | 247 | 66.0% | +4.08% | 1.38 |

**Critical insight:** The model REVERSE-SELECTS within beaten-down stocks.
Its high-confidence LEG-like picks LOSE money (-1.31%, 38% win) while its
strong-momentum picks are genuine winners (+3.2% to +4.1%, 60-66% win).

User's theory ("LEG-like may become winners more often") is FALSE for the
specific trades the model selects. Cheap stocks mean-revert on average, but
the model's LEG-like selections go the wrong way.

## Decision: Add LEG-like exclusion filter

Drop any pick where:
- `sue_lag_1 < -0.5` (prior quarter miss)
- `consecutive_surprises_pre == 0` (no beat streak)
- `rel_ret_20d < -0.05` (oversold)

Applied AFTER theta and XLF, BEFORE top-4 selection. Eliminates the -1.31%
bucket, keeping only the +3.2% to +4.1% profiles.

## Implementation
`05b_alpaca_live/01_fetch_and_predict.py` — step 6c `_is_leg_like()`
Deployment filter doc: `05b_alpaca_live/README.md` §Deployment filters

## Impact on current picks (2026-08-03)
0 dropped this week — all 4 picks (ROKU, SITM, TWLO, EXEL) are strong-momentum.
Filter is a safety net for future weeks where a weak stock sneaks past theta.

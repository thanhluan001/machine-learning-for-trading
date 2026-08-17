# Relative Capex Warning — Promising Context, Not a Portfolio Tilt

**Script:** `04_backtest/77_megatrend_relative_capex_warning.py`
**Date:** 2026-08-16 | **Design ref:** RC-4 Phase 2c
**Data:** FMP `/stable/cash-flow-statement`, point-in-time aligned by
`acceptedDate`/`filingDate`; 18 bellwether companies; cached in `db_capex.h5`.

## Revised hypothesis

Absolute capex is insufficient: a dying theme can spend more dollars while
losing its share of the limited capital pool. The tested quantity is:

```text
theme_capex_share = TTM capex(theme) / TTM capex(all tracked themes)
```

Warnings are defined ex ante as either a >=20% relative-share decline or a
>=5 percentage-point share decline over four quarters. This is a warning-
context study, not a portfolio tilt backtest.

## User scale hypothesis: confirmed

| TTM capex | 2019 | 2022 | 2024 | 2026 H1 |
|---|---:|---:|---:|---:|
| AI/hyperscale | $72B | $160B | $242B | $574B |
| Crypto panel | ~$0B | $1B | $24B | $1B |

The current panel implies AI/crypto ratios of approximately **174x in 2022,
10x in 2024, and 401x in 2026 H1**. The crypto 2024 spike is a panel warning:
a few companies can distort a small theme aggregate, so production use needs
a broader, stable universe and concentration controls.

## Relative shares

| | AI | Clean energy | Crypto | AI/clean |
|---|---:|---:|---:|---:|
| 2019 | 84.6% | 15.2% | 0.1% | 5.6x |
| 2020 | 91.3% | 8.2% | 0.5% | 11.2x |
| 2021 | 90.5% | 7.2% | 2.4% | 12.6x |
| 2022 | 91.4% | 7.3% | 1.3% | 12.5x |
| 2023 | 90.1% | 9.1% | 0.8% | 9.9x |
| 2024 | 91.3% | 5.8% | 2.9% | 15.7x |
| 2025 | 88.8% | 2.9% | 8.2% | 30.2x |
| 2026 H1 | 95.5% | 2.3% | 2.2% | 41.2x |

## Clean-energy natural experiment

This confirms the user's relative-capex insight more strongly than the prior
absolute-capex test:

```text
                         clean share     4q share change     ICLN vs 2020 peak
2019-12                       15.2%             +0.1pp                 —
2020-06                       10.0%             -5.1pp                 —
2020-12                        8.2%             -1.8pp                  0%
2021-09                        7.1%             -0.5pp                -24%
2021-12                        7.2%             +0.1pp                -24%
2022-12                        7.3%             -0.1pp                -28%
2023-12                        9.1%             +0.7pp                -43%
2024-12                        5.8%             -1.0pp                -58%
2025-12                        2.9%             -0.4pp                -38%
```

The first major relative-capex break occurred by **2020-06**: clean-energy
share fell 5.1pp / 33.6% while ICLN did not peak until 2020-12. The formal
three-month price failure began 2021-09, giving approximately **14 months of
lead time** from the initial relative-capex warning.

This is materially different from the absolute-capex conclusion. Clean-energy
absolute capex rose into 2023, but its share of the capital pool had already
fallen sharply before the price peak. Relative allocation captured loss of
sponsorship earlier.

## Why this is not yet a trading rule

- Warnings are often **too early**. AI's last warning preceded a later price
  failure by 33 months; crypto warnings can be 10–27 months early.
- Six-month returns after warnings were still positive in this small sample:
  AI +27.5%, clean energy +34.1%, crypto +12.3%. A warning would have reduced
  exposure too early and missed rebounds. This is context, not an automatic
  exit.
- The denominator is only 18 hand-selected bellwethers. The crypto series is
  especially unstable because a few companies dominate it. A broad, stable
  panel is required before any threshold can be trusted.
- The study uses accounting capex, which is slow and can represent expansion
  into future oversupply. Relative capex measures sponsorship/competition,
  not immediate shareholder returns.

## Verdict

**Relative capex: promising as a warning-context feature; rejected as a direct
portfolio tilt for now.** The clean-energy case validates the mechanism: the
relative allocation shift led price deterioration, while absolute capex did
not. However, the signal is too early and noisy to replace price exits.

Recommended interpretation:

```text
relative capex deterioration = downgrade conviction / investigate rotation
price breadth deterioration  = actionable trend stress
both together               = stronger warning candidate
```

## Next validation step

Expand to a broader, stable point-in-time panel and test the joint warning:

```text
capex share down >= threshold AND price breadth < 50%
```

Evaluate lead time, false-warning rate, and forward drawdown by regime. Keep
it as a monthly Script 74 context block unless the joint signal demonstrates
incremental warning value without causing premature de-risking.

No PEAD model, portfolio position, or production threshold was changed.

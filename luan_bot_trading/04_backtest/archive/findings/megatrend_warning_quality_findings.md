# Megatrend Multi-Source Warning Quality — Diagnostic Complete, No Automatic Exit

**Script:** `04_backtest/78_megatrend_warning_quality.py`
**Date:** 2026-08-16 | **Design ref:** RC-4
**Status:** Research diagnostic only. No portfolio allocation changed.

## Channels and contracts

| Channel | Definition | Intended role |
|---|---|---|
| Price | Proxy below 10m mean and negative 3m return | Timely market confirmation |
| Relative capex | Point-in-time TTM theme share down >=20% YoY or >=5pp | Early sponsorship warning |
| Insider | Material Form 4 net selling over trailing 90d | Corporate-behavior confirmation |
| News | >=2 negative operational keyword articles, negative > positive, trailing 90d | Operational confirmation |

Capex uses FMP `acceptedDate`/`filingDate`; news uses `publishedDate`; insider
uses Form 4 `filingDate`. No fiscal-period or transaction-date look-ahead.

## Coverage

```text
                 capex months   insider months   news months
AI/hyperscale          140             0              7
clean_energy           140           140            133
crypto                 140             0             88
```

Insider coverage is absent for AI/crypto because the panel is not mapped to the
S&P 400 insider cache. News coverage is highly uneven and the keyword screen
is preliminary; it must not be interpreted as a validated sentiment model.

## Warning counts

```text
                 price   capex   insider   news
AI/hyperscale       20       3       0       6
clean_energy        53      15     116      23
crypto              54      44       0      11
```

Price failures were defined separately as three consecutive month-ends below
the 10m mean. Relative capex warnings preceded several failures, especially:

- Clean energy failure 2021-09: last capex warning ~14 months earlier.
- Clean energy failure 2023-04: last capex warning ~33 months earlier.
- Crypto failure 2021-07: capex warning ~10 months earlier.
- Crypto failure 2025-10: capex warning ~27 months earlier.
- AI failures: capex warnings were 33–69 months early for the available
  episodes; too early for direct timing.

## Ablation diagnostic

Six-month proxy returns after warning observations:

```text
price                         n=121   mean +1.4%   median +1.4%   win 53.7%
capex                          n=56   mean +18.9%  median +12.2%  win 71.4%
price + capex                  n=23   mean +11.7%  median +11.5%  win 73.9%
price + capex + insider         n=5   mean +31.3%  median +33.4%  win 80.0%
price + capex + news            n=4   mean +21.4%  median +29.7%  win 75.0%
all four                        n=4   mean +21.4%  median +29.7%  win 75.0%
```

These positive post-warning returns are NOT evidence that warnings are
profitable. They demonstrate that capex is frequently **too early**: reducing
exposure immediately after a capex-share warning would miss subsequent
recoveries. The joint sample sizes are too small for statistical conclusions.

## Current context at the latest cached month

```text
AI/hyperscale: price false, capex false, insider unavailable, news true
clean energy : price true,  capex false, insider true, news true
crypto       : price true,  capex true,  insider unavailable, news true
```

This is a context report, not an instruction to sell. The AI news flag is not
sufficient evidence by itself; clean-energy and crypto deserve monitoring,
but the production breadth indicator remains the formal warning output.

## Verdict

**The architecture is conceptually sound, but no automatic exit is approved.**

```text
relative capex = early capital-sponsorship context
price breadth  = market-timing confirmation
insider/news   = confirmation where coverage is adequate
```

Relative capex improves the explanation of *why* a theme may be losing support,
but it is too early to time exits. News keyword classification is not yet
strong enough to count as independent confirmation. Insider coverage is too
sparse outside the S&P 400-linked names.

## Next permitted research step

If RC-4 is reopened, use a broader stable capex panel and a manually reviewed
news taxonomy. Test whether:

```text
price breadth < 50% AND relative capex share down
```

reduces false re-entry during bear rallies without causing excessive early
exits. Add insider/news only after proving coverage and timestamp quality.

Until then, Script 74 remains the only deployed megatrend component: monthly
breadth reporting, zero capital, no automatic de-risking.

# RC-4 Partial Exposure Cycle 1 — Preliminary Result

**Script:** `04_backtest/82_megatrend_partial_exposure_cycle1.py`
**Date:** 2026-08-17
**Status:** Research cycle in progress; no operational allocation rule approved.

## Research premise

The Phase-3 failure should not be interpreted as "2022 was a recession and the
portfolio should have gone to cash." 2022 was a non-recessionary market-
decision regime in which competing themes were being repriced. The proposed
hypothesis is:

```text
remain invested during theme discovery;
rotate gradually toward sustained price support;
reduce absolute exposure only under a separate recession condition.
```

This is different from the failed binary policy:

```text
broad trend break -> full exit;
bear-rally recovery -> full re-entry.
```

## Fixed Cycle-1 contract

Three theme proxy groups are used:

```text
AI/hyperscale: SMH
clean_energy: 50/50 ICLN + TAN returns when available
crypto:       50/50 MSTR + COIN returns when available
```

All variants remain invested at 100% unless the separate recession overlay is
used. Theme weights have a 10% floor, 70% cap, and maximum 10 percentage-point
movement per month.

| Variant | Theme target | Absolute exposure |
|---|---|---|
| Equal theme | 1/3 each | 100% |
| Price rotate | rank target 60/30/10 | 100% |
| Price + capex | 50% price target + 50% lagged point-in-time capex target | 100% |
| Price + capex + recession | same as above | 50% after Sahm-style trigger |

The capex target is clipped and bounded. It is a sponsorship prior, not an
exit trigger. The recession trigger is:

```text
3-month average unemployment - trailing 12-month minimum >= 0.50pp
```

The signal is observed at month-end and applies to the following month. 2022
is not treated as a recession merely because theme prices are below trend.

## Preliminary gross results

Sample: 2014-02 through 2026-07, monthly, no transaction costs or slippage.
SPY benchmark was corrected before interpreting results; the initial script
run had incorrectly compared against the average theme return.

| Variant | Total | Annualized | Max DD | 2020 | 2022 |
|---|---:|---:|---:|---:|---:|
| Equal theme | +1,061% | +21.7% | -51.0% | +141.7% | -45.2% |
| Price rotate | +1,209% | +22.8% | -50.4% | +123.6% | -42.0% |
| Price + capex | +1,894% | +27.1% | -43.4% | +102.9% | -36.9% |
| Price + capex + recession | +1,174% | +22.6% | -43.4% | +41.2% | -36.9% |
| SPY same window | +418% | +14.1% | -23.9% | +18.4% | -18.2% |

These are **not acceptance results**. The theme proxies are volatile and the
sample begins in 2014; gross returns are especially sensitive to proxy choice,
crypto availability, and the absence of costs.

## What the preliminary result says

### 1. The user's 2022 reasoning is plausible

All three always-invested variants stayed exposed through 2022. Gradual theme
rotation reduced the loss compared with equal themes:

```text
Equal theme:       -45.2%
Price rotation:    -42.0%
Price + capex:     -36.9%
```

This is directionally consistent with the hypothesis that the correct response
to an undecided market is not necessarily to abandon absolute exposure. It is
not evidence that the allocation rule is ready.

### 2. Capex blend helped in this sample, but this is not yet causal evidence

The price-plus-capex variant had the best gross result and lower drawdown than
equal-theme and price-only variants. Its latest allocation is approximately:

```text
AI/hyperscale: 69%
clean_energy:  21%
crypto:        11%
```

This is deliberately below the raw 2026 capex shares (95% / 4% / 1%) because
the test imposes a 70% theme cap and a 10% floor. The result may reflect
beneficial exposure to the long AI trend, not a generally valid capex edge.
The clean-energy natural experiment still shows that capex cannot time a theme
death; this cycle uses it only as a bounded prior.

### 3. The recession overlay hurt the rapid recovery

The Sahm-style overlay reduced 2020 return from +102.9% to +41.2% for the
price-plus-capex variant. It also triggered in August 2024, a labor-market
warning without an official recession. This demonstrates that:

```text
recession detector != crash detector
```

A recession overlay may reduce risk, but it can materially sacrifice recovery
capture and can produce false positives. It is not approved.

### 4. Partial rotation is not the same as partial absolute de-risking

Cycle 1 currently tests partial **theme** exposure while remaining 100%
invested. This directly tests the user's "market is deciding" hypothesis.
It has not yet tested a gradual absolute exposure ladder such as 100/75/50%
outside a recession overlay. That is a separate sensitivity and must not be
quietly selected after seeing 2022.

## Data and implementation caveats

- The three-theme proxy panel is narrower than the six-theme operational
  dashboard.
- Crypto has limited COIN history; MSTR provides earlier proxy history.
- The point-in-time capex join uses accepted/filing availability dates.
- The FRED unemployment series is point-in-time as cached, but the Sahm-style
  threshold is only a recession-warning proxy.
- No costs, slippage, taxes, ETF spreads, or implementation capacity are
  modeled.
- This is one expanding sample, not walk-forward validation.

## Next validation gate

Before any operational change, pre-register and run:

1. walk-forward evaluation with parameters fixed before each test window;
2. equal-theme versus price-only versus capex-blend ablation;
3. bounded grid of theme floor/cap/monthly step, including 90/5/5-like
   sponsorship targets only as a pre-registered sensitivity;
4. gradual absolute exposure ladder, tested separately from theme rotation;
5. explicit 2008/2020/2022/2024 regime table where proxy coverage permits;
6. transaction-cost and turnover sensitivity;
7. bootstrap confidence intervals and out-of-sample stability.

Until that gate clears:

```text
Operational watcher: monthly manual panel report
Research cycle:      partial theme exposure only
Core allocation:     unchanged
Automatic orders:    none
```

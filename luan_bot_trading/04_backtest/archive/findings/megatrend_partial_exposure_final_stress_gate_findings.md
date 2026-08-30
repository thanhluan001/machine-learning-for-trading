# RC-4 Final Benchmark/Stress Gate — Partial Theme Exposure

**Script:** `04_backtest/archive/megatrend_wave_2026/85_megatrend_partial_exposure_final_stress_gate.py`
**Date:** 2026-08-17
**Status:** Research result; operational promotion rejected.

## Test

The four fixed shortlist configurations were compared against SPY and static
60/40 at 50bp and 100bp turnover costs. The gate also tested:

- normal point-in-time capex;
- capex delayed 3, 6, and 12 months;
- capex unavailable/missing;
- rolling 36-month excess-return stability;
- 12-month moving-block bootstrap of excess return;
- leave-one-theme-out stress for the strongest apparent D configuration.

No new parameters were searched.

## Normal sample, 50bp costs

| Config | Rotation | Cap | Step | Total | SPY total | Excess return | Max DD | Rolling 36m excess positive |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | price | 50% | 10% | +1,342% | +418% | +225% | -46.4% | 63.5% |
| B | price | 50% | 15% | +1,419% | +418% | +244% | -46.4% | 64.3% |
| C | price+capex | 50% | 10% | +1,385% | +418% | +235% | -45.4% | 65.2% |
| D | price+capex | 70% | 10% | +1,655% | +418% | +294% | -43.3% | 84.3% |

SPY maximum drawdown in this same sample was approximately -23.9%.

## Findings

### 1. Return advantage is real in-sample, but risk is not core-book safe

All four variants beat SPY on cumulative return in the 2014–2026 proxy sample.
However, all four had materially larger maximum drawdowns:

```text
Active variants: approximately -43% to -46%
SPY:                         approximately -24%
```

This is not a small implementation discrepancy. The variants are volatile
theme portfolios. They cannot yet be described as a lower-risk replacement or
automatic overlay for the user's approximately 90% core book.

They may be suitable for a separately sized high-risk carve-out only after
additional risk-budget and live-observation decisions, neither of which is
approved here.

### 2. D remains strongest in the normal sample, but is not promoted

D (price+capex, 70% cap, 10% step) has the strongest historical return and
lowest maximum drawdown among the four candidates. Its excess-return bootstrap
probability against SPY at 50bp was approximately 97.1%.

The bootstrap interval still crossed negative excess return for the active
strategy in the reported resamples. It is not proof of future superiority.
The advantage may reflect the long AI/hyperscale period and the proxy panel.

### 3. Capex delay stress is encouraging but incomplete

At 50bp, D's SPY-relative cumulative excess return was approximately:

```text
normal capex:  +294.5%
3-month delay: +285.5%
6-month delay: +285.7%
12-month delay:+297.6%
missing capex:+207.9%
```

The result does not depend on exact filing timing in this sample. But missing
capex reduces the result materially and worsens active maximum drawdown toward
-49%. The correct operational design must treat missing capex as missing
information, not as a reason to infer a favorable or unfavorable theme weight.

### 4. Leave-one-theme-out exposes AI dependence

D was tested at 50bp with each theme removed:

| Removed theme | Active total | SPY total | Excess | Max DD | Rolling 36m excess positive |
|---|---:|---:|---:|---:|---:|
| AI/hyperscale | +231% | +400% | -24% | -51.5% | 43.1% |
| clean energy | +2,652% | +400% | +560% | -52.9% | 100.0% |
| crypto | +1,264% | +400% | +203% | -32.4% | 84.5% |

The AI omission is the decisive stress result: D no longer beats SPY and has
worse drawdown. The headline outperformance therefore depends materially on
the AI trend. That is not necessarily invalid—the strategy is intended to
capture the supported market theme—but it means the result is not a generic,
all-regime megatrend law.

### 5. Static 60/40 is not a fair return competitor, but is a risk reference

The variants greatly exceed static 60/40 return in this sample, but static
60/40 has much lower drawdown. It is useful as a risk-budget reference, not as
a direct theme-return benchmark.

## Final gate status

```text
Return versus SPY:             positive in sample
Drawdown versus SPY:           failed — materially worse
Cost sensitivity:              survived 50/100bp
Capex timing delay:            broadly survived
Missing capex:                 material degradation
Leave-one-AI-out:              failed return and stability
Safe core-book overlay:        rejected
Automatic allocation:          rejected
Manual watcher:                unchanged
```

## Decision

The partial-exposure hypothesis remains intellectually and empirically
interesting:

```text
stay invested during theme discovery;
rotate gradually toward sustained support;
do not use capex as an emergency exit.
```

But the tested implementation is a volatile thematic portfolio, not yet a
validated core-book governor. The correct deployment decision is to keep the
monthly panel operational and manual, with no automatic partial allocation.

A future cycle would need to start from a declared risk budget, not from the
highest-return D configuration. It would need to test:

- a fixed small carve-out sized against its -50% stress case;
- a broader theme universe so AI dependence is measurable;
- benchmark-relative downside utility rather than total return alone;
- live paper observation of monthly weights;
- explicit whether the user wants risk reduction or maximum theme capture.

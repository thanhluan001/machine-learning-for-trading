# Megatrend Warning Quality v2 — Panel Split Useful; Confirmations Not Ready

**Script:** `04_backtest/79_megatrend_warning_quality_v2.py`
**Date:** 2026-08-16
**Status:** Diagnostic only. No allocation or PEAD policy changed.

## Purpose

This second study targeted Phase 3's failure mode:

```text
trend break -> temporary bear rally -> re-entry -> renewed decline
```

It split the breadth signal into equity, theme, and cross-asset panels and
added relative capex, full-market insider, and operational-news confirmations.

## Current panel reading

```text
equity breadth:      88% (15/17) above 10-month mean
theme breadth:       33% (1/3; AI above, clean energy and crypto below)
cross-asset breadth: 25%
```

This is a useful distinction. The equity trend remains broad while leadership
is concentrated in AI/hyperscale and several themes lag. That is rotation and
narrowing, not yet a systemic equity stress signal.

## Signal counts, 2015–2026 monthly panel

```text
price stress:          29
relative-capex:         3
insider net-selling:  128
news pressure:          7
confirmed stress:       26
```

## Findings

### 1. Panel separation is useful

The three-panel output is more informative than one blended 26-asset breadth
number:

- equity breadth answers whether the broad equity trend is weakening;
- theme breadth identifies leadership rotation;
- cross-asset breadth indicates whether bonds/gold/defensive assets confirm a
  broad risk-off regime.

This survives as a reporting improvement.

### 2. Insider definition is not discriminating yet

Full-market coverage was corrected: AI and crypto mega-cap leaders are now
included via direct FMP ticker panels. However, the simple flag "net material
selling in trailing 90d < 0" fires in 128 months. That is too frequent and
likely reflects persistent executive selling, compensation/liquidity behavior,
and panel-size effects rather than a discrete warning.

Required next correction before using it:

- normalize dollars by theme market capitalization or company size;
- separate number of selling insiders from dollar volume;
- require a cross-company selling cluster;
- compare selling rate against that theme's own historical baseline;
- exclude scheduled/known plan sales where possible.

No insider confirmation claim is accepted from v2.

### 3. News definition is not ready

The keyword classifier is transparent but coverage is uneven. AI has only a
small historical FMP response window in the current cache, while other themes
have much more coverage. Article counts are not comparable across themes.

Before use, news needs:

- complete historical pagination validation;
- syndicated-article deduplication by company/day/event;
- a reviewed operational taxonomy;
- source-quality weighting (SEC/company release above opinion article);
- event-rate normalization by company and theme.

No news confirmation claim is accepted from v2.

### 4. Capex remains early context

Relative capex is point-in-time and correctly measured, but it often warns many
months before price failure. It should describe capital sponsorship and
rotation, not trigger immediate exit.

### 5. No automatic re-entry rule approved

The v2 recovery diagnostic is not a sufficient out-of-sample test for an
allocation rule. Additional confirmation channels cannot be used to block or
allow re-entry until the insider/news definitions are normalized and the
historical episode construction is independently audited.

## Verdict

```text
ADOPT now:     split equity/theme/cross-asset breadth reporting
CONTEXT only:  relative capex
RESEARCH:      normalized full-market insider signal
RESEARCH:      audited operational-news signal
NOT approved:  automatic exit, automatic re-entry, or allocation tilt
```

`05c_megatrend_watcher/monthly_panel_report.py` is the only deployed
megatrend component: manual monthly panel reporting with zero capital at risk.

## Panelized dashboard implemented (2026-08-16)

The operational watcher now reports separate panels rather than one blended breadth number (`05c_megatrend_watcher/monthly_panel_report.py`):

```text
Equity breadth:       15/17 = 88% HEALTHY
Theme breadth:         3/6  = 50% NARROWING
Cross-asset breadth:   1/4  = 25%
```

Current equity active: SPY, QQQ, IWM, IJR, EFA, EEM, XLB, XLE, XLF, XLI,
XLK, XLP, XLRE, XLV, XLY. Laggards: XLC and XLU.

Current theme active: AI/hyperscale, biotech, metals. Laggards: clean energy,
crypto, uranium. Current cross-asset active: SHY only; AGG, GLD, TLT are below
10-month means.

Fundamental context is printed but not acted upon: AI has 95.0% of tracked TTM
capex ($574B), clean energy 4.3% ($26B), crypto 0.7% ($5B). Full-market
insider and news panels are included; their current negative flags remain
informational because the insider measure is not baseline-normalized and the
news classifier is keyword-based.

The operational dashboard logs separate panels, rosters, capex, insider, and
news context in `05c_megatrend_watcher/logs/megatrend_breadth_log.json`. It is
run manually at month-end and remains a warning report, not an allocation
engine.

## Step 3 normalization milestone (2026-08-16)

Script `80_megatrend_normalize_insider_news.py` establishes a separate,
point-in-time normalization layer. It does not change Script 74 and does not
create an approved warning rule.

### Insider audit and normalization

```text
raw Form 4 rows:              60,993
exact P/S open-market rows:   32,570
after transaction dedup:      32,480
non-open-market rows excluded:28,423
```

The eligible set is restricted to exact `P-Purchase` and `S-Sale` Form 4
codes. Awards, gifts, option/exempt/derivative and other transaction codes are
not treated as buying or selling. The output reports independent seller and
buyer counts, cross-company breadth, raw dollar flow for audit, and each
company's trailing-90d sell-dollar percentile against its own strictly prior
monthly history. Raw dollars are not used as a cross-company warning because
there is no point-in-time market-cap series and the cache contains vendor/entity
outliers.

The candidate diagnostic requires at least two companies with unusually high
within-company selling percentiles and at least two independent sellers. In
the latest available month (August 2026), no theme passes this rule.

### Operational-news audit and normalization

```text
raw articles:             25,957
articles after dedup:     25,954
classified operational events: 3,875
missing text rows:             0
```

The fixed taxonomy is demand/orders, guidance/financial, capex/capacity,
supply/pricing, labor/operations, and balance-sheet/solvency. Counts are
company-day based, with raw article/company-day denominators, text coverage,
negative/positive event rates, and within-company historical percentiles.
Source provenance is explicitly unavailable in the cache and is not inferred.

Latest-month result:

```text
AI/hyperscale: historical news coverage gate FAILS (cache starts 2026)
clean_energy:  coverage adequate; normalized warning candidate FALSE
crypto:        coverage adequate; normalized warning candidate FALSE
```

The candidate rule requires adequate historical coverage, at least two
companies with unusually high negative company-day counts, and negative event
breadth exceeding positive event breadth. Across the full diagnostic panel,
there are 9 candidate month-theme observations, but this is **not** predictive
validation and does not justify deployment.

### Artifacts

```text
04_backtest/80_megatrend_normalize_insider_news.py
04_backtest/archive/experiments/rc4_normalized_insider_news_monthly.csv
04_backtest/archive/experiments/rc4_normalized_insider_news_summary.json
```

Next gate: independently join these normalized month-end observations to the
fixed equity-stress and recovery episodes, measure false-reentry reduction and
missed recoveries, and keep the result research-only unless it clears the
pre-registered episode bar. The manual `05c_megatrend_watcher/monthly_panel_report.py`
remains the sole deployed component.

## Step 4 false-reentry test (2026-08-16)

Script `81_megatrend_false_reentry_test.py` performed the fixed episode test.
The first run was rejected because the 17-name equity panel was historically
incomplete (only SPY/QQQ existed in the cache in 2008). The corrected run uses
an expanding, category-fixed panel and requires at least five names with valid
10-month means. It also excludes the current partial month and prevents one
recovery from being reused by overlapping stress onsets.

### Corrected result

```text
complete non-overlapping recovery episodes: 9
baseline allowed-reentry relapse rate:       33.3%
insider-filter blocked episodes:              3
insider-filter blocked relapse rate:         33.3%
insider-filter blocked positive recoveries:  66.7%
news-filter blocked episodes:                  0
```

Episode observations included 2020 and 2022, but only five valid historical
equity names were available in those years. 2008 is explicitly marked
`NO_VALID_STRESS` under the five-name coverage requirement; it is not reported
as a zero-relapse success. This means the result is underpowered and should not
be read as a general historical validation.

The normalized insider filter did not reduce relapse and blocked profitable
recoveries. The normalized news candidate produced no recovery blocks. The
channels therefore **fail the timing-confirmation test** and are not promoted
to Script 74, automatic exits, or automatic re-entry logic.

### Status after Step 4

```text
ADOPT:       panelized monthly dashboard (Script 74)
CONTEXT:     relative capex; normalized insider/news diagnostics
CLOSED:      insider/news as current false-reentry filters
NOT tested:  causal predictive value; PEAD features (separate RC-1 cycle)
```

Artifacts:

```text
04_backtest/81_megatrend_false_reentry_test.py
04_backtest/archive/experiments/rc4_false_reentry_episodes.csv
04_backtest/archive/experiments/rc4_false_reentry_summary.json
```

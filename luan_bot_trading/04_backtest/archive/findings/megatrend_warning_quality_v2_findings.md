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

Script 74 remains the only deployed megatrend component: monthly breadth
reporting with zero capital at risk.

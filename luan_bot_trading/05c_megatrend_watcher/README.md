# 05c_megatrend_watcher/ — Manual Monthly Core-Book Watcher

## Purpose

This is the operational RC-4 megatrend component. It is a **manual monthly
warning and context report** for the user's approximately 90% core book:
real estate, S&P 500 exposure, and blue-chip holdings.

It is not part of the 5–8% PEAD sleeve, does not select PEAD trades, and does
not submit orders or modify positions automatically.

## Run cadence

Run once at month-end, after the market close, when the monthly bar is complete:

```bash
conda run -n trading --no-capture-output python \
  luan_bot_trading/05c_megatrend_watcher/monthly_panel_report.py
```

The report refreshes Tiingo prices, computes the panels, prints the current
reading, and appends the result to:

```text
05c_megatrend_watcher/logs/megatrend_breadth_log.json
```

Run it manually. There is no scheduler, daemon, broker connection, or
automatic de-risking.

## Panels

### 1. Equity breadth — primary timing panel

Fixed category universe:

```text
SPY, QQQ, IWM, IJR, EFA, EEM
XLB, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY, XLC, XLE
```

Each asset is compared with its own 10-month moving average.

```text
>=75%   HEALTHY
50–75%  NARROWING
<50%    STRESS
<25%    CRISIS
```

These are judgment bands, not automatic trading instructions.

### 2. Theme breadth — leadership and rotation

Current operational theme proxies:

```text
AI/hyperscale   SMH
clean_energy    ICLN, TAN
crypto          MSTR, COIN
biotech         XBI, IBB
metals          GDX, LIT
uranium         URA
```

A weak theme is a rotation observation, not by itself a core-book sell signal.

### 3. Cross-asset context

```text
AGG, TLT, GLD, SHY
```

This provides risk-on/risk-off context. It does not override equity breadth.

### 4. Relative-capex context

Point-in-time TTM capex shares are shown for the AI/hyperscale, clean-energy,
and crypto panels. Capex describes capital sponsorship and rotation; it is not
an exit or position-sizing signal.

### 5. Insider and news context

The report includes full-market theme-company Form 4 and FMP operational-news
context. These are descriptive diagnostics only. The normalized research
versions were tested and did not pass the false-re-entry timing test.

## Manual interpretation

Use this sequence:

```text
1. Read equity breadth first.
2. Identify which equity sectors or regions are lagging.
3. Check whether theme weakness is isolated rotation or broad deterioration.
4. Use cross-asset, capex, insider, and news fields as context only.
5. Record any core-book decision manually outside this script.
```

The watcher does not tell the operator to buy or sell. It supplies a repeatable
monthly state report for judgment.

## Operational boundaries

```text
PEAD model:             untouched
Alpaca paper book:      untouched
Core allocation:        manual only
Frequency:              monthly
Orders:                 none
Automatic exits:        none
Automatic re-entry:     none
```

## Provenance

The research and validation history remains in:

```text
04_backtest/archive/findings/megatrend_phase1_findings.md
04_backtest/archive/findings/megatrend_phase2_findings.md
04_backtest/archive/findings/megatrend_phase2b2c_findings.md
04_backtest/archive/findings/megatrend_phase3_findings.md
04_backtest/archive/findings/megatrend_warning_quality_findings.md
04_backtest/archive/findings/megatrend_warning_quality_v2_findings.md
```

The operational script was promoted only after the strategy, selection, capex
tilt, and insider/news timing-filter paths were rejected. The surviving role is
the panelized monthly watcher.

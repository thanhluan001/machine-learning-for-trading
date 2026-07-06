# Database Layout — `db.h5`

> Status: reference doc. Snapshot of the HDF5 organization the pipeline
> writes into `luan_bot_trading/data/db.h5`.

## Diagram

```
db.h5
│
├── /sp400/{canonical_ticker}           # 15y adjusted daily OHLCV per COMPANY
│                                          (one node per company; canonical ticker)
│
├── /macros/
│   ├── {TICKER}                        # sector ETF + market index OHLCV
│   │                                      (IJH, IJJ, IJK, IJS, XLB, XLF, XLRE,
│   │                                       XLU + XLK, XLI, XLY, XLP, XLV, XLE,
│   │                                       XLC, SPY, VIXY)
│   └── fred_{name}                     # 6 FRED macro series
│                                          (fed_funds_rate, yield_curve_spread,
│                                           vix, wti_oil, unemployment, cpi)
│
├── /earnings/
│   └── raw                             # raw EODHD earnings rows, long-form
│                                          one row per (canonical_ticker, report_date)
│
└── /metadata/
    ├── sp400                           # per-TICKER view (aliases preserved)
    │                                      ticker, name, gics_sector,
    │                                      gics_sub_industry, intervals (JSON),
    │                                      sic, index_ref,
    │                                      cik, canonical_ticker   ← added by 02b
    │
    └── sp400_companies                 # per-COMPANY view (one row per CIK)
                                           canonical_ticker, cik, aliases (JSON),
                                           name, sic, index_ref,
                                           combined_intervals (JSON),
                                           per_ticker_intervals (JSON),
                                           price_unavailable
```

## Node-by-node detail

### `/sp400/{canonical_ticker}` — written by `03_data_gathering.py`

- One node per **company** (keyed by `canonical_ticker`, not per ticker symbol).
- 15 years of Tiingo adjusted daily OHLCV.
- Schema: `Date, Open, High, Low, Close, Volume` (all from Tiingo's `adj*` columns — split/dividend consistent).
- Aliases of the same company (rebrands) share one node under the canonical
  ticker; Tiingo's retro-adjusted history spans the alias periods seamlessly.
- Companies flagged `price_unavailable=True` in `/metadata/sp400_companies`
  have **no node written** (skipped + logged by `03`).

### `/macros/{TICKER}` — written by `04_index_data_gathering.py`

- Sector ETFs and market indices used for relative-return and market-adjusted
  CAR calculations (per `features.md` Block 3 and `Design.md` §14).
- Current universe: `IJH, IJJ, IJK, IJS, XLB, XLF, XLRE, XLU` (mapped via
  `index_ref` per `SIC_code_to_index.md`) + broad SPDRs `XLK, XLI, XLY, XLP,
  XLV, XLE, XLC, SPY, VIXY`.
- Same OHLCV schema as `/sp400/*`.

### `/macros/fred_{name}` — written by `05_fed_data_gathering.py`

- 6 FRED macro series for `features.md` Block 4 (Macro Environment):
  | Series code | Stored name | Feature |
  |---|---|---|
  | `DFF` | `fred_fed_funds_rate` | `fed_funds_rate` |
  | `T10Y2Y` | `fred_yield_curve_spread` | `yield_curve_spread` |
  | `VIXCLS` | `fred_vix` | `vix_close` |
  | `DCOILWTICO` | `fred_wti_oil` | (regime proxy) |
  | `UNRATE` | `fred_unemployment` | (regime proxy) |
  | `CPIAUCSL` | `fed_cpi` | (regime proxy) |

### `/earnings/raw` — written by `06_earnings_gathering.py`

- Long-form table, one row per `(canonical_ticker, report_date)`.
- Source: EODHD `/api/calendar/earnings`. 15-year depth, matching `/sp400`.
- Schema (verified against live API response):

  | Column | Source / derivation | Notes |
  |---|---|---|
  | `report_date` | EODHD `report_date` | announcement date `T` (PEAD event time) |
  | `fiscal_period_end` | EODHD `date` | fiscal quarter end (NOT the event date) |
  | `code` | EODHD `code` | alias EODHD stored the row under (e.g. `AAXN.US`) |
  | `canonical_ticker` | company map | our canonical ticker for the company |
  | `cik` | company map | SEC CIK (10-digit string, may be `None` for singletons) |
  | `actual` | EODHD `actual` | reported EPS |
  | `estimate` | EODHD `estimate` | pre-report consensus EPS (historical, not forward) |
  | `difference` | EODHD `difference` | `actual − estimate` |
  | `percent` | EODHD `percent` | surprise % → maps directly to `eps_surprise_pct` |
  | `before_after_market` | EODHD `before_after_market` | `"Bmo"` / `"AfterMarket"` → `is_bmo` |
  | `currency` | EODHD `currency` | usually `USD` |

- **Deduplicated** by `(canonical_ticker, report_date)` to handle rebrand
  overlaps (e.g. `AAXN` + `AXON` rows for the same announcement).
- EODHD's `/api/calendar/trends` is NOT used (forward-looking only; cannot
  backfill historical training estimates).
- SUE / `consecutive_surprises` / `sue_acceleration` / `sue_lag_1` / `sue_lag_2`
  / `is_bmo` encoding are computed downstream by the feature builder
  (see `features.md`), not stored here.

### `/metadata/sp400` — written by `01`, extended by `02`, extended by `02b`

- Per-**ticker** view (one row per ticker symbol, including removed constituents).
- Survivorship-safe: pre-2012 `added` dates backfilled to `2012-01-01` for all
  historical tickers, not just current ones.
- Schema:

  | Column | Source | Description |
  |---|---|---|
  | `ticker` | `01` Wikipedia | ticker symbol (this alias) |
  | `name` | `01` Wikipedia | company name from current constituents table |
  | `gics_sector` | `01` Wikipedia | GICS sector (historical, may be sparse pre-2012) |
  | `gics_sub_industry` | `01` Wikipedia | GICS sub-industry |
  | `intervals` | `01` Wikipedia changes table | JSON list of `{"added", "removed"}` memberships (multi-interval captures "boomerang" stocks) |
  | `sic` | `02` SEC EDGAR | SEC SIC code (3-tier lookup: ticker.txt → DERA historical → `SIC_PATCH`) |
  | `index_ref` | `02` derived | sector index ticker from SIC per `SIC_code_to_index.md` |
  | `cik` | `02b` derived | SEC CIK or `None` (no-CIK → singleton) |
  | `canonical_ticker` | `02b` derived | the canonical this alias belongs to |

### `/metadata/sp400_companies` — written by `02b_build_company_map.py`

- Per-**company** view (one row per CIK; ticker aliases collapsed).
- This is the table the feature builder and `03_data_gathering.py` iterate over.
- Schema:

  | Column | Type | Description |
  |---|---|---|
  | `canonical_ticker` | str | Ticker chosen for Tiingo/EODHO fetch (priority: active+Tiingo-verified → most-recent-removed+Tiingo-verified → latest-ticker) |
  | `cik` | str or `None` | SEC CIK (10-digit), or `None` for singletons |
  | `aliases` | JSON `list[str]` | All tickers in this company group, ordered (canonical first, then Tiingo-verified fallbacks, then unverified) |
  | `name` | str | best-available name (canonical's row, fallback to any alias) |
  | `sic` | str | SIC code (from canonical ticker's row) |
  | `index_ref` | str | sector index reference |
  | `combined_intervals` | JSON `list[{"added","removed"}]` | merged S&P 400 membership spans across aliases (overlapping/abutting merged; gaps > 7 days kept as separate spans) |
  | `per_ticker_intervals` | JSON `dict[ticker -> list[{"added","removed"}]]` | audit trail of original per-ticker intervals |
  | `price_unavailable` | bool | `True` if no alias is fetchable from Tiingo (skipped by `03` and `06`) |

## Producer-consumer map

| Node | Producer | Consumer(s) |
|---|---|---|
| `/metadata/sp400` | `01` (create), `02` (cols), `02b` (cols) | `02`, `02b`, `06` (read aliases), audit |
| `/metadata/sp400_companies` | `02b` | `03` (iterate per company), `06` (iterate per company), feature builder |
| `/sp400/{canonical}` | `03` | feature builder (price/OHLCV) |
| `/macros/{TICKER}` | `04` | feature builder (relative returns, sector-adjusted) |
| `/macros/fred_{name}` | `05` | feature builder (Block 4 macro features) |
| `/earnings/raw` | `06` | feature builder (one row per earnings event) |

## Write-safety pattern (all producers)

All writes use `pd.HDFStore(DB_FILE, mode="a")` + `store.remove(key)` to overwrite
only the target node — **never** `mode="w"`, which would truncate the entire
`db.h5` and wipe every group at once. (This bug class was debugged in earlier
versions; the fix is now enforced across all pipeline scripts.)

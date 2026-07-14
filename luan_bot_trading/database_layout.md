# Database Layout — `db.h5`

> Status: reference doc. Snapshot of the HDF5 organization the pipeline
> writes into `luan_bot_trading/01_data/db.h5`.
>
> Phase A update (`02b_build_company_map.py` rewrite, 2026-07-13):
> `/metadata/sp400_companies` (per-CIK view) is **REMOVED**. Replaced by
> `/metadata/sp400_perm_ids` (per-`perm_id` view, interval-forked,
> point-in-time-CIK anchored). See `01_data/merger_identity_patch.md`
> for the design.

## Diagram

```
db.h5
│
├── /sp400/{canonical_ticker}           # 15y adjusted daily OHLCV per PERM_ID
│                                          (one node per perm_id; keyed by canonical alias)
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
│                                          one row per (perm_id-at-write-time, report_date)
│
├── /features/
│   └── gated_events                    # Stage-1-gated earnings events
│                                          (Schema not yet migrated to perm_id; see Phase B-E)
│
└── /metadata/
    ├── sp400                           # per-TICKER view (aliases preserved)
    │                                      ticker, name, gics_sector,
    │                                      gics_sub_industry, intervals (JSON),
    │                                      sic, index_ref,
    │                                      cik, canonical_ticker   ← legacy (deprecated; see cik_at_added / perm_id)
    │                                      cik_at_added, perm_id   ← Phase A audit cols (02b)
    │
    └── sp400_perm_ids                  # PERM_ID view (one row per tradable-asset track)
                                         perm_id, cik, canonical_ticker, aliases (JSON),
                                         name, sic, index_ref,
                                         combined_intervals (JSON),
                                         per_ticker_intervals (JSON),
                                         price_unavailable
                                          (REPLACES legacy /metadata/sp400_companies)
```

## Node-by-node detail

### `/sp400/{canonical_ticker}` — written by `03_data_gathering.py`

- One node per **perm_id**, keyed by `canonical_ticker` (an alias the perm_id
  is currently fetchable under on EODHD).
- 15 years of EODHD adjusted daily OHLCV (we locally derive adj-OHLCV from
  `close`/`adjusted_close` ratio; raw EODHD only returns `adjusted_close`).
- Schema (11 cols): `Date, Open, High, Low, Close, Volume, Adj_Open, Adj_High,
  Adj_Low, Adj_Close, Adj_Volume`.
- Aliases of the same perm_id (rebrands across disjoint intervals) share one
  node under the canonical ticker; EODHD's retro-adjusted history spans the
  alias periods seamlessly.
- Perm_ids flagged `price_unavailable=True` in `/metadata/sp400_perm_ids` have
  **no node written** (skipped + logged by `03`).
- Note: as of Phase A completion, the existing `/sp400/*` nodes were built
  off the now-deprecated `/metadata/sp400_companies` table; Phase B will
  rerun `03` against `/metadata/sp400_perm_ids`.

### `/macros/{TICKER}` — written by `04_index_data_gathering.py`

- Sector ETFs and market indices used for relative-return and market-adjusted
  CAR calculations (per `features.md` Block 3 and `Design.md` §14).
- Current universe: `IJH, IJJ, IJK, IJS, XLB, XLF, XLRE, XLU` (mapped via
  `index_ref` per `01_data/SIC_code_to_index.md`) + broad SPDRs `XLK, XLI, XLY, XLP,
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

- Long-form table, one row per `(canonical_ticker-at-write-time, report_date)`.
- Source: EODHD `/api/calendar/earnings`. 15-year depth, matching `/sp400`.
- Schema (verified against live API response):

  | Column | Source / derivation | Notes |
  |---|---|---|
  | `report_date` | EODHD `report_date` | announcement date `T` (PEAD event time) |
  | `fiscal_period_end` | EODHD `date` | fiscal quarter end (NOT the event date) |
  | `code` | EODHD `code` | alias EODHD stored the row under (e.g. `AAXN.US`) |
  | `canonical_ticker` | perm_id derived | canonical alias for the perm_id |
  | `cik` | perm_id derived | SEC CIK (10-digit string, may be `None` for `__nocik_*`) |
  | `actual` | EODHD `actual` | reported EPS |
  | `estimate` | EODHD `estimate` | pre-report consensus EPS (historical, not forward) |
  | `difference` | EODHD `difference` | `actual - estimate` |
  | `percent` | EODHD `percent` | surprise % → maps directly to `eps_surprise_pct` |
  | `before_after_market` | EODHD `before_after_market` | `"Bmo"` / `"AfterMarket"` → `is_bmo` |
  | `currency` | EODHD `currency` | usually `USD` |

- **Deduplicated at write-time** by `(perm_id, fiscal_period_end)` per Phase A
  user-selected dedup rule "latest-name-change wins":
  prefer canonical alias (active SEC ticker), tiebreak by latest `report_date`,
  tiebreak by lexicographic `code`.
  This is the earliest possible layer and makes `/earnings/raw` the single
  source of truth — no downstream dedup needed.
- EODHD's `/api/calendar/trends` is NOT used (forward-looking only; cannot
  backfill historical training estimates).
- SUE / `consecutive_surprises` / `sue_acceleration` / `sue_lag_1` / `sue_lag_2`
  / `is_bmo` encoding are computed downstream by the feature builder
  (see `features.md`), not stored here.
- **Note on Phase A historical data**: the current `/earnings/raw` was built
  with the old `canonical_ticker` schema and will need to be re-fetched in
  Phase D to use `perm_id` (with the new dedup rule applied at write time).

### `/features/gated_events` — written by `02_features/01_features_gate_events.py`

- Stage-1 gating output: 21,853 gated earnings events (from 44,637 raw rows).
- Schema (will be migrated to `perm_id` in Phase B-E):
  - `canonical_ticker`, `cik`, `report_date`, `added`, `removed`, `calendar_week_group`
- Built off the pre-Phase-A `/earnings/raw`; will need re-gating in Phase D once
  `/earnings/raw` is re-fetched with the new dedup-at-write-time rule.

### `/metadata/sp400` — written by `01`, extended by `02`, extended by `02b` (Phase A)

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
  | `index_ref` | `02` derived | sector index ticker from SIC per `01_data/SIC_code_to_index.md` |
  | `cik` | `02b` (old approach, deprecated) | SEC CIK resolved prior to Phase A (present-day lookup). Superseded by `cik_at_added`. Kept for backwards compat during staged Phase B-E refactor. |
  | `canonical_ticker` | `02b` (old approach, deprecated) | Old canonical ticker; kept for backwards compat. Use `perm_id` for perm anchoring. |
  | `cik_at_added` (Phase A) | `02b` Phase A derived | Point-in-time CIK at the ticker's first Wikipedia-interval added year (via DERA `sub_{YYYY}.txt` snapshots with year-walkback/forward fallback). This is the canonical-for-this-ticker CIK used by Phase A's fork algorithm. |
  | `perm_id` (Phase A) | `02b` Phase A derived | Comma-joined list of perm_ids this ticker contributed entries to (a single ticker can fork across multiple perm_ids if its CIK differs across Wiki intervals — e.g. the ACI data spans two different historical companies). |

### `/metadata/sp400_perm_ids` — written by `02b_build_company_map.py` (Phase A)

- Per-**perm_id** view. Replaces the deprecated `/metadata/sp400_companies`
  table (which was purged by Phase A's `02b`).
- This is the table the feature builder and `03_data_gathering.py` will iterate
  over from Phase B onward.
- `perm_id` rule (per `01_data/merger_identity_patch.md`):
  - `perm_id = f"{cik_at_added}_{start_ticker}"` where `cik_at_added` is the
    point-in-time CIK at the perm_id's first interval's Wikipedia-added year.
  - If no CIK resolves: `perm_id = f"__nocik_{start_ticker}"` (4 such cases in v1).
  - **Same-CIK intervals across different ticker symbols (rebrands / renames) →
    MERGE into one perm_id (multi-alias)**; the interval-extend rule fires
    regardless of interval overlap. Wikipedia's "both tickers live at once during
    the rebrand transition day" data is overlap-shaped but represents the same
    legal entity (CIK anchored).
  - **Different-CIK intervals on the same ticker symbol (M&A where the survivor
    keeps the target's ticker symbol) → forked into separate perm_ids** — e.g.
    old Coherent Inc. (`0000021510_COHR`) vs new Coherent Corp / ex-II-VI
    (`0000820318_IIVI` with `COHR` alias).
  - Suffix `#N` is appended on rare same-CIK-same-ticker collisions (e.g.
    `0001336917_UAA` + `0001336917_UAA#2`). Phase A v1 has 0 such cases.
  - **Post-Wikipedia active SEC aliases** (acquirer-rebrand cases like
    `AAXN + AXON`, `IIVI + COHR`, `BRKS + AZTA`, `ERI + CZR`, `HTA + HR`,
    `CHK + EXE`, `SAI + LDOS`) are added in a Step 4a post-processing pass
    with safety guards to avoid re-introducing the survivor-CIK collision bug
    in the opposite direction (contaminating a historical track with a
    different live entity's price series).
- Schema:

  | Column | Type | Description |
  |---|---|---|
  | `perm_id` | str | Primary key. `f"{cik}_{start_ticker}"` or `f"__nocik_{start_ticker}"`. |
  | `cik` | str or `None` | Point-in-time CIK at perm_id's first interval's added year. May be `None` for `__nocik_*` perm_ids (4 cases). |
  | `canonical_ticker` | str | Alias chosen for EODHD price fetch. Priority: alias whose active-SEC CIK matches our `cik` (with latest-added tiebreak); fall back to most-recent latest-added alias. |
  | `aliases` | JSON `list[str]` | All ticker symbols on this perm_id's track, in insertion order (canonical not necessarily first). |
  | `name` | str | Best-available name across aliases. |
  | `sic` | str | SIC code (from first alias's row in `/metadata/sp400`). |
  | `index_ref` | str | Sector index reference (per `/metadata/sp400`). |
  | `combined_intervals` | JSON `list[{"added", "removed"}]` | Merged S&P 400 membership spans across aliases (overlapping/abutting/interleaved-open merged; gaps > 7 days kept as separate spans). |
  | `per_ticker_intervals` | JSON `dict[ticker -> list[{"added", "removed"}]]` | Audit trail of original per-ticker intervals. The union of `aliases` exactly equals the keys here. |
  | `price_unavailable` | bool | `True` if EODHD probe failed for ALL aliases (canonical + fallback reprobe). Skipped by `03` and `06`. |

### ~~`/metadata/sp400_companies`~~ — REMOVED in Phase A

- Was a per-CIK view with one row per CIK. Replaced by
  `/metadata/sp400_perm_ids` because per-CIK collapsing incorrectly merged
  different companies' pre-merger histories when SEC retroactively consolidated
  a target's CIK into the acquirer's CIK post-merger (the "Survivor-CIK
  Collision Bug" — see `01_data/merger_identity_patch.md` for analysis).
- Purged from `db.h5` by `02b_build_company_map.py` to prevent downstream
  stages from accidentally reading stale data during the Phase B-E refactor.

## Producer-consumer map

| Node | Producer | Consumer(s) |
|---|---|---|
| `/metadata/sp400` | `01` (create), `02` (cols), `02b` (Phase A cols) | `02`, `02b`, `06` (read aliases), audit |
| `/metadata/sp400_perm_ids` | `02b` (Phase A) | `03` (iterate per perm_id — Phase B), `06` (iterate per perm_id — Phase D), feature builder, Phase B-E |
| ~~`/metadata/sp400_companies`~~ | (REMOVED in Phase A) | — |
| `/sp400/{canonical}` | `03` | feature builder (price/OHLCV) |
| `/macros/{TICKER}` | `04` | feature builder (relative returns, sector-adjusted) |
| `/macros/fred_{name}` | `05` | feature builder (Block 4 macro features) |
| `/earnings/raw` | `06` | feature builder (one row per earnings event) |
| `/features/gated_events` | `02_features/01_features_gate_events.py` | feature builder (Stage 2) |

## Write-safety pattern (all producers)

All writes use `pd.HDFStore(DB_FILE, mode="a")` + `store.remove(key)` to
overwrite only the target node — **never** `mode="w"`, which would truncate
the entire `db.h5` and wipe every group at once. (This bug class was debugged
in earlier versions; the fix is now enforced across all pipeline scripts.)

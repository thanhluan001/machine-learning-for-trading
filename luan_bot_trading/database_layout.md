# Database Layout — `db.h5`

> **⛔ DEPRECATED IDENTITY LAYER (2026-07-14):** All references to
> `perm_id` / `/metadata/sp400_perm_ids` / `/sp400/{canonical_ticker}` /
> alias-concatenation in this document describe the **OBSOLETE Phase A
> identity architecture**. The pipeline is migrating to **Tiingo
> `permaTicker`** as the primary entity identifier. See
> [`01_data/tiingo_permaTicker_audit.md`](tiingo_permaTicker_audit.md)
> for the live probe-based evidence and the new identity design.
>
> Concrete deprecations each forward stage must address:
> - **Primary key** changes from `perm_id` (synthetic
>   `f"{cik}_{start_ticker}"`) to Tiingo `permaTicker` (e.g. `US000000001291`)
> - **Storage key** changes from `/sp400/{canonical_ticker}` (12 perm_id
>   collisions on this key — see merger_identity_patch §7.7) to
>   `/sp400/{permaTicker}` (clean, permaTicker-keyed, no collisions)
> - **Alias-concatenation** (Phase B's union-alias + sort + dedup fix,
>   documented as `/sp400/{canonical}` Phase B alias-concat behavior in
>   the section below) is **eliminated entirely**: one
>   `/tiingo/daily/{permaTicker}/prices` fetch per permaTicker returns the
>   full history with rebrand-back-merge done server-side by Tiingo.
> - **Dedup key** for `/earnings/raw` changes from `(perm_id,
>   fiscal_period_end)` to `(permaTicker, fiscal_period_end)` -- no
>   cik/canonical-ticker collision means to disambiguate in the
>   first place, so the §7.7 disambiguation rule falls away entirely.
> - **Phase A `02b_build_company_map.py`** is rewritten: the
>   Wikipedia+DERA+CIK-synthesis algorithm (~1000 LOC) collapses to a Ti
>   search lookup per Wikipedia (~200 LOC). See the audit doc.
>
> This file remains a useful reference for the **non-identity** schema
> (macros, FRED, earnings schema columns, gated_events schema, write-safety
> pattern) but every identity-related assertion below is stale.

> Status: reference doc. Snapshot of the HDF5 organization the pipeline
> writes into `luan_bot_trading/01_data/db.h5`.
>
> Phase A update (`02b_build_company_map.py` rewrite, 2026-07-13):
> `/metadata/sp400_companies` (per-CIK view) is **REMOVED**. Replaced by
> `/metadata/sp400_perm_ids` (per-`perm_id` view, interval-forked,
> point-in-time-CIK anchored). See `01_data/merger_identity_patch.md`
> for the design. ⛔ Both this update and the table it produces are
> deprecated by the permaTicker migration (see banner above).

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
│   └── gated_events                    # Stage-1-gated earnings events (Phase E)
│                                          perm_id, canonical_ticker, cik, report_date,
│                                          added, removed, calendar_week_group  (7 cols)
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

> ⛔ **DEPRECATED.** Replaced by `/sp400/{permaTicker}` keyed by Tiingo
> permaTicker (see banner at file top). The alias-concatenation machinery
> documented below is OBSOLETE -- permaTicker-keyed Tiingo `/prices`
> fetches return the full rebrand-covered history server-side.

- One node per **perm_id**, keyed by `canonical_ticker` (an alias the perm_id
  is currently or recently fetchable under on EODHD).
- 15 years of EODHD adjusted daily OHLCV (we locally derive adj-OHLCV from
  `close`/`adjusted_close` ratio; raw EODHD only returns `adjusted_close`).
- **ALIAS CONCATENATION (Phase B implementation detail)**: For multi-alias
  perm_ids (rebrand merges + post-Wiki acquirer-rebrand extensions), `03`
  fetches EVERY alias on EODHD and concatenates the responses on `Date`
  (dedup-keep-last on overlap days, sort by Date). This is REQUIRED because
  EODHD does NOT retro-relabel rebrands: when a company rebrands (e.g.
  CHK -> EXE), EODHD keeps the old ticker series as a dead series ending at
  the rebrand day and starts a fresh series under the new ticker -- they
  are NOT concatenated server-side. A single-alias fetch would lose the
  pre- OR post-rebrand segment.
  Empirical validation: `CHK + EXE` concatenated = 2645 rows 2016-2026 (vs
  2201 CHK-only or 1360 EXE-only); `SYNH + INCR` = 2934 rows; `ESV + VAL`
  = 3461 rows; `FTR + FYBR` = 3531 rows; `LNW + SGMS + LAWIL` = 3846
  rows (LAWIL alone is 1 row -- canonical-only fetch would be useless).
  ⛔ This alias-concat approach is the source of the Phase B Class U/V
  contamination (`phase_b_contamination_audit.md`). permaTicker-keyed
  fetching eliminates it.
- Schema (11 cols): `Date, Open, High, Low, Close, Volume, Adj_Open, Adj_High,
  Adj_Low, Adj_Close, Adj_Volume`.
  ⛔ Under the permaTicker migration, the schema becomes the richer 13 cols
  from Tiingo `/prices`: `Date, Open, High, Low, Close, Volume, AdjOpen,
  AdjHigh, AdjLow, AdjClose, AdjVolume, DivCash, SplitFactor`.
- Perm_ids flagged `price_unavailable=True` in `/metadata/sp400_perm_ids` have
  **no node written** (skipped + logged by `03`); any stale node left by the
  pre-Phase-A schema is purged in the per-perm_id step.
- **Phase B stale-node cleanup pass**: After the per-perm_id loop, `03` purges
  `/sp400/{TICKER}` nodes whose TICKER is no longer canonical for any perm_id
  (leftover from pre-Phase-A canonical selection that picked a ticker now
  categorized as a non-canonical alias; e.g. CHK, DV, POL, SGMS, ENR, FBHS,
  FTR, LNW, MODG, SAI, SYNH, UA, UNIT, WTW, ZI).

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

### `/earnings/raw` — written by `06_earnings_gathering.py` (Phase D rewrite)

> ⚠️ **PK deprecation:** `(perm_id, fiscal_period_end)` dedup key is
> being replaced by `(permaTicker, fiscal_period_end)`. See file-top
> banner. The `perm_id` / `canonical_ticker` / `cik` columns in the
> table below would become `permaTicker` (and possibly retain `cik` as
> informational/audit). The Phase D rewrite iteration source changes
> from `/metadata/sp400_perm_ids` to `/metadata/sp400_permatickers` (next
> formatting pass TBD).

- Long-form table, one row per **(perm_id, fiscal_period_end)**.
- Source: EODHD `/api/calendar/earnings`. 15-year depth, matching `/sp400`.
  ⛔ During permaTicker migration it remains EODHD (Tiingo has no
  earnings-calendar endpoint), but the dedup key switches from
  `(perm_id, fiscal_period_end)` to `(permaTicker, fiscal_period_end)`.
- Schema (verified against live API response):

  | Column | Source / derivation | Notes |
  |---|---|---|
  | `report_date` | EODHD `report_date` | announcement date `T` (PEAD event time) |
  | `fiscal_period_end` | EODHD `date` | fiscal quarter end (NOT the event date) |
  | `code` | EODHD `code` | alias EODHD stored the row under (e.g. `AAXN.US`) |
  | `perm_id` | perm_id derived | candidate-tradable-asset-track anchor (`f"{cik}_{start_ticker}"`; primary). Phase D's primary key. |
  | `canonical_ticker` | perm_id derived | perm_id's canonical alias (informational). Joins to `/sp400/{canonical_ticker}` for price data. |
  | `cik` | perm_id derived | SEC CIK (10-digit string, may be `None` for `__nocik_*`) |
  | `actual` | EODHD `actual` | reported EPS |
  | `estimate` | EODHD `estimate` | pre-report consensus EPS (historical, not forward) |
  | `difference` | EODHD `difference` | `actual - estimate` |
  | `percent` | EODHD `percent` | surprise % → maps directly to `eps_surprise_pct` |
  | `before_after_market` | EODHD `before_after_market` | `"Bmo"` / `"AfterMarket"` → `is_bmo` |
  | `currency` | EODHD `currency` | usually `USD` |

- **Deduplicated at write-time** by `(perm_id, fiscal_period_end)` per Phase A
  user-selected dedup rule "latest-name-change wins" (see
  `merger_identity_patch.md` §7.7):
  - Tiebreak 1 (primary): prefer row whose `code` is the perm_id's
    canonical alias's EODHD code (`canonical_ticker + '.US'`).
  - Tiebreak 2: latest `report_date`.
  - Tiebreak 3: lexicographic `code`.
  This is the earliest possible layer and makes `/earnings/raw` the single
  source of truth — no downstream dedup needed. Keying by `perm_id` (not
  `canonical_ticker`) sidesteps the §7.7 dedup-collision problem on the 12
  canonical-sharing perm_id pairs.
- EODHD's `/api/calendar/trends` is NOT used (forward-looking only; cannot
  backfill historical training estimates).
- SUE / `consecutive_surprises` / `sue_acceleration` / `sue_lag_1` / `sue_lag_2`
  / `is_bmo` encoding are computed downstream by the feature builder
  (see `features.md`), not stored here.
- Phase D rewrite replaced v1's `(canonical_ticker, report_date)` dedup with
  `(perm_id, fiscal_period_end)` — Phase D was the first re-fetch after Phase A.
- Null estimate convention: when EODHD returns `estimate` as null, it sets
  `difference = 0.0`. The feature builder keeps EODHD's `0.0` in the rolling
  `sue_score` denominator (the `Option B` decision in `features.md` §0).

### `/features/gated_events` — written by `02_features/01_features_gate_events.py` (Phase E rewrite)

> ⚠️ **PK deprecation:** `perm_id` primary key in the schema below is
> replaced by `permaTicker`. The §7.7 disambiguation rule (LIVE-wins-overlap
> for the 12 canonical-collision pairs) is **deleted entirely** under
> permaTicker keying (no canonical_ticker collisions because permaTicker
> IS the storage key). `canonical_ticker` column demoted to informational /
> audit join key.

- Stage-1 gating output: 21,269 gated earnings events (from 44,897 raw rows).
- Phase E re-keying: per-perm_id iteration from `/metadata/sp400_perm_ids`;
  events keyed by `perm_id` (NOT `canonical_ticker`) to avoid the §7.7
  collision-dedup problem on the 12 canonical-sharing perm_id pairs.
  ⛔ DEPRECATED: replaces by per-permaTicker iteration from
  `/metadata/sp400_permatickers`.
- §7.7 disambiguation APPLIED AT THE GATE: for LOSER perm_id events whose
  `report_date` falls in the OVERLAP ZONE shared with the WINNER perm_id
  (same canonical_ticker), the row is NaN-dropped at the gate. Empirical
  impact: 105 events dropped (0.23% of raw events). 7 of 12 pairs have
  overlap (COHR 26, LDOS 24, AZTA 16, AXON 15, VAL 14, CZR 9, EXE 1);
  5 pairs (ACI, AM, CC, JEF, HR) have NO overlap -> no drop.
- Schema (7 columns):

  | Column | Type | Description |
  |---|---|---|
  | `perm_id` | str | Phase-A perm_id anchor (`{cik}_{start_ticker}`); PRIMARY row key |
  | `canonical_ticker` | str | perm_id's canonical alias; identifies the /sp400/{canon} node to load for prices |
  | `cik` | str | SEC CIK (audit; may be `None` for `__nocik_*`) |
  | `report_date` | datetime | earnings announcement date T (PEAD event time) |
  | `added` | datetime | interval.added (audit only) |
  | `removed` | datetime | interval.removed or today (audit only) |
  | `calendar_week_group` | str | ISO week `YYYY-Www` (LTR group anchor) |

- v1 stored schema (6 cols, no `perm_id`) superseded; this Phase E version is
  the canonical truth for Stage 2.

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

> ⚠️ **ENTIRE TABLE IS OBSOLETE.** Being renamed to
> `/metadata/sp400_permatickers` with a new schema sourced from Tiingo
> search lookups (`permaTicker, name, ticker [current], isActive,
> countryCode` plus the existing Wikipedia `intervals`, SEC EDGAR `sic`,
> `index_ref`). The ticker-based `perm_id = f"{cik}_{start_ticker}"` field
> is replaced by `permaTicker` (e.g. `US000000001291` for Enovis Corp).

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

> ⚠️ See file-top banner for the permaTicker migration deltas. The `/sp400/{canonical}`
> path becomes `/sp400/{permaTicker}`; `/metadata/sp400_perm_ids` becomes
> `/metadata/sp400_permatickers`.

| Node | Producer | Consumer(s) |
|---|---|---|
| `/metadata/sp400` | `01` (create), `02` (cols), `02b` (Phase A cols, ⚠️ deprecated cols) | `02`, `02b` (OLD), `06` (read aliases), audit |
| `/metadata/sp400_perm_ids` | `02b` (Phase A, ⚠️ DEPRECATED) | (NEW: replaced by `/metadata/sp400_permatickers` produced by `02b` rewrite) |
| ~~`/metadata/sp400_companies`~~ | (REMOVED in Phase A) | — |
| `/sp400/{canonical}` | `03` (⚠️ path key becomes `/sp400/{permaTicker}`) | feature builder (price/OHLCV) |
| `/macros/{TICKER}` | `04` | feature builder (relative returns, sector-adjusted) |
| `/macros/fred_{name}` | `05` | feature builder (Block 4 macro features) |
| `/earnings/raw` | `06` (⚠️ key by `permaTicker` not `perm_id`) | feature builder (one row per earnings event) |
| `/features/gated_events` | `02_features/01_features_gate_events.py` (⚠️ key by `permaTicker`) | feature builder (Stage 2) |

## Write-safety pattern (all producers)

All writes use `pd.HDFStore(DB_FILE, mode="a")` + `store.remove(key)` to
overwrite only the target node — **never** `mode="w"`, which would truncate
the entire `db.h5` and wipe every group at once. (This bug class was debugged
in earlier versions; the fix is now enforced across all pipeline scripts.)

# Database Layout — `db.h5`

> **Authoritative reference** for the current state of `luan_bot_trading/01_data/db.h5`.
> Snapshot: post Phase-A permaTicker migration + dedup-and-cleanup-2024
> (2026-07-22). The OLD `perm_id`-based identity layer documented in
> earlier versions of this file is OBSOLETE and removed below.
> See `01_data/phase_a_b_migration_report.md` for the migration history.

## Headline facts

| Property | Value |
|---|---:|
| File | `luan_bot_trading/01_data/db.h5` |
| Size | 704 MB |
| HDF5 top-level groups | `/sp400`, `/macros`, `/earnings`, `/features`, `/metadata` |
| Total keys | 956 |
| Price-data nodes in `/sp400/` | 928 (one per Tiingo `permaTicker`) |
| Total rows in `/sp400/` | 2,752,968 |
| Writes converge to | `(permaTicker, report_date)` keying — **0 dups** at every stage |
| Primary identifier | Tiingo `permaTicker` (e.g. `US000000001291` for Enovis Corp) |

## Diagram

```
db.h5
│
├── /sp400/{permaTicker}                # 15y adjusted OHLCV from Tiingo
│                                          (one node per permaTicker)
│
├── /analyst/
│   └── grades/{permaTicker}            # FMP analyst upgrade/downgrade history (per-ticker)
├── /macros/
│   ├── {TICKER}                        # sector ETF + broad SPDR index OHLCV
│   │                                      (17 keys: IJH, IJJ, IJK, IJS,
│   │                                       XLB, XLF, XLRE, XLU, XLK, XLI,
│   │                                       XLY, XLP, XLV, XLE, XLC, SPY, VIXY)
│   └── fred_{name}                     # 6 FRED macro series
│                                          (fred_fed_funds_rate, fred_yield_curve_spread,
│                                           fred_vix_close, fred_wti_oil,
│                                           fred_unemployment_rate, fred_cpi)
│
├── /earnings/
│   └── fmp                             # FMP earnings (replaces EODHD /earnings/raw)
│                                          one row per (permaTicker, report_date)  [DE-DUPED]
│
├── /features/
│   ├── gated_events                    # Stage-1-gated earnings events (Phase E v2)
│   └── train_matrix                    # Stage-2 feature matrix (Phase E v2)
│
└── /metadata/
    ├── sp400                           # per-TICKER view (993 Wikipedia tickers,
    │                                      with legacy `perm_id`/`canonical_ticker` audit cols)
    └── sp400_permatickers              # per-PERMATICKER view (962 rows, PRIMARY identity table)
```

## Node-by-node detail

---

### `/sp400/{permaTicker}` — written by `01_data/03_data_gathering.py`

- One node per **Tiingo `permaTicker`** (identity-stable across rebrands, delistings, spin-offs, bankruptcy reorgs). The Tiingo server rebrand-back-merges the price history automatically under a single `permaTicker`, so we make ONE fetch per permaTicker — NO local alias-concatenation (the old EODHD-era machinery is gone).
- 15 years of Tiingo daily adjusted OHLCV. All adjusted columns are pre-computed by Tiingo (split + dividend back-adjusted).
- Stored schema (**11 cols**): `Date, Open, High, Low, Close, Volume, Adj_Open, Adj_High, Adj_Low, Adj_Close, Adj_Volume`. (Tiingo also returns `divCash` and `splitFactor` but they're dropped at write-time — not used downstream.)
- PermaTickers flagged `price_unavailable=True` in `/metadata/sp400_permatickers` have no node written (skipped + logged by `03`); stale nodes left by pre-Phase-A schema are also purged in the per-permaTicker step.
- Write pattern: `pd.HDFStore(DB, mode='a')` + `store.remove(key)` if key exists + `store.put(h5_path, data, format="table", data_columns=["Date"])` (the `data_columns=["Date"]` enables on-disk date-index lookup).

---

### `/macros/{TICKER}` — written by `01_data/04_index_data_gathering.py`

- Sector ETFs and broad SPDRs used for relative-return and market-adjusted CAR calculations (per `features.md` Block 3 and `Design.md §14`).
- Current universe (**17 keys**):
  - S&P 400 MidCap ETFs (mapped via `index_ref` per `01_data/SIC_code_to_index.md`): `IJH, IJJ, IJK, IJS, XLB, XLF, XLRE, XLU`
  - Broad SPDRs / sentiment: `XLK, XLI, XLY, XLP, XLV, XLE, XLC, SPY, VIXY`
- Schema (**6 cols**): `Date, Open, High, Low, Close, Volume` (NOT adjusted — sector ETFs have minimal corporate-action need; this is the raw Tiingo `/prices` shape).
- Caveats:
  - `XLC` starts 2018 (455 fewer rows than the SPY/IJH baseline of 3768 rows).
  - `XLRE` starts 2019 (1069 fewer rows).

---

### `/macros/fred_{name}` — written by `01_data/05_fed_data_gathering.py`

- 6 FRED macro series for `features.md` Block 4 (Macro Environment). All series are stored as 2-column DataFrames: `Date, {feature_column_name}`.

| Stored key | Feature column | FRED series code | Used by |
|---|---|---|---|
| `/macros/fred_fed_funds_rate` | `fed_funds_rate` | `DFF` | macro feature |
| `/macros/fred_yield_curve_spread` | `yield_curve_spread` | `T10Y2Y` | macro feature |
| `/macros/fred_vix_close` | `vix_close` | `VIXCLS` | macro feature |
| `/macros/fred_wti_oil` | `wti_oil` | `DCOILWTICO` | regime proxy |
| `/macros/fred_unemployment_rate` | `unemployment_rate` | `UNRATE` | regime proxy |
| `/macros/fred_cpi` | `cpi` | `CPIAUCSL` | regime proxy |

(Note: the actual stored keys are `fred_vix_close` and `fred_unemployment_rate` — `_close`/`_rate` suffixes — not the older `fred_vix` / `fred_unemployment` naming.)

---

### `/earnings/fmp` -- written by `01_data/06b_fmp_earnings_gathering.py`

- Source: FMP `/stable/earnings?includeReportTimes=true` (replaces EODHD, cancelled 2026-07-23).
- FMP $49/mo plan. Returns 41 years of history (1985-present) vs EODHD's 15 years.
- `includeReportTimes=true` unlocks BMO/AMC timing, fiscalPeriod, fiscalYear, confirmed flag.
- Dedup key: `(permaTicker, report_date)` -- keep latest `lastUpdated`.

**Schema** (18 cols):

| Column | Source | Description |
|--------|--------|-------------|
| `permaTicker` | `/metadata/sp400_permatickers` | PRIMARY key |
| `canonical_ticker` | FMP `symbol` | ticker used for API call |
| `cik` | `/metadata/sp400_permatickers` | SEC CIK |
| `report_date` | FMP `date` | announcement date T |
| `period_ending` | FMP `periodEnding` | fiscal quarter end |
| `fiscal_period` | FMP `fiscalPeriod` | "Q1".."Q4" |
| `fiscal_year` | FMP `fiscalYear` | e.g. 2026 |
| `eps_actual` | FMP `epsActual` | reported EPS |
| `eps_estimated` | FMP `epsEstimated` | consensus EPS estimate |
| `eps_difference` | derived | `eps_actual - eps_estimated` |
| `eps_surprise_pct` | derived | `(eps_actual - eps_estimated) / abs(eps_estimated) * 100` |
| `revenue_actual` | FMP `revenueActual` | reported revenue (EODHD did not have this) |
| `revenue_estimated` | FMP `revenueEstimated` | consensus revenue estimate |
| `revenue_difference` | derived | `revenue_actual - revenue_estimated` |
| `revenue_surprise_pct` | derived | `(rev_actual - rev_est) / abs(rev_est) * 100` |
| `before_after_market` | FMP `time` | "bmo" / "amc" (clean lowercase, no CamelCase) |
| `confirmed` | FMP `confirmed` | whether earnings date is confirmed |
| `last_updated` | FMP `lastUpdated` | data freshness timestamp |

- Current state: **87,609 rows x 18 cols, 839 distinct permaTickers, 0 dups**.
- BMO/AMC coverage: **94.1%** (was ~41% with EODHD due to CamelCase parsing bug).
- EPS actual coverage: 99.1%. Revenue actual coverage: 97.9%.

### `/analyst/grades/{permaTicker}` -- written by `01_data/07_fmp_grades_gathering.py`

- Source: FMP `/stable/grades?symbol={ticker}`. FMP $49/mo plan.
- 14 years of daily-granularity analyst upgrade/downgrade history (2012-present).
- 807 nodes stored (of 928 fetchable permaTickers; 121 have zero analyst coverage).
- Dedup key: `(permaTicker, date, grading_company, action)`.

**Schema** (7 cols per node):

| Column | Description |
|--------|-------------|
| `symbol` | ticker used for API call |
| `permaTicker` | PRIMARY key |
| `date` | date of analyst action |
| `grading_company` | analyst firm name (e.g. "Morgan Stanley") |
| `previous_grade` | previous rating (e.g. "Hold") |
| `new_grade` | new rating (e.g. "Buy") |
| `action` | "upgrade" / "downgrade" / "maintain" |

- 211 unique grading firms across all nodes.
- Action distribution (first 50 nodes): maintain 9296, downgrade 1791, upgrade 1606.


---

### `/features/gated_events` — written by `02_features/01_features_gate_events.py` (Phase E v2)

- Stage-1 gating output. Current state: **20,375 rows × 7 cols**, **0 dup groups** (today's GME multi-interval-overlap-fix applied at the gate-step: 19 events collapsed).
- Gating rule per Wikipedia interval: `[added + 90d, removed]` (90-day = 1-quarter post-addition buffer; asymmetric, forward-only). Each row's `added` is the permaTicker's earliest-`added` interval in the dedup fix.
- Schema (**7 cols**):

  | Column | Type | Description |
  |---|---|---|
  | `permaTicker` | str | PRIMARY row key (Tiingo identity-stable) |
  | `canonical_ticker` | str | informational; for join-back to original EODHD calendar lookups |
  | `cik` | str | informational / audit (may be `None`) |
  | `report_date` | datetime | earnings announcement date `T` (PEAD event time) |
  | `added` | datetime | interval.added (earliest of the permaTicker's Wikipedia intervals) |
  | `removed` | datetime | interval.removed or today |
  | `calendar_week_group` | str | ISO week `YYYY-Www` (listwise LTR group anchor) |

- §7.7 disambiguation rule: **REMOVED** under permaTicker keying (no canonical_ticker collisions because permaTicker IS the storage key).
- Multi-Wikipedia-interval overlap fix: events appearing in MULTIPLE interval windows for the same permaTicker (e.g. **GME**, has intervals `{[2016-04-22, today]}` AND `{[2021-08-04, today]}` — both open-ended) get emitted twice by the gate loop; the dedup-by-earliest-added pass after the loop keeps 1 row per `(permaTicker, report_date)`. Audit counter `n_events_dup_collapsed` in the gating report.

---

### `/features/train_matrix` — written by `02_features/02_build_feature_matrix.py` (Phase E v2)

- Per-event feature matrix consumed by `03_model/02_phase_g_sunday_classifier.py` and downstream `04_backtest/*` scripts.
- Current state: **20,299 rows × 38 cols**, **0 dup groups** (today's full re-run off /earnings/fmp).
- Schema: 8 identity + audit cols + 22 feature cols. (Old Phase E v2 had 38 cols incl. `car_10d` and `car_60d_pass1` as label candidates; the column count is still 30 today.)
- 110 T-match failures logged (mostly US000000001364 — Coherent legacy LEH/COHR; price history ends 2022-07-01 so post-2022 events fail T-match and are dropped).

  | Column | Type | Description |
  |---|---|---|
  | `permaTicker` | str | PRIMARY row key |
  | `canonical_ticker` | str | informational |
  | `cik` | str | informational / audit |
  | `report_date` | datetime | earnings announcement date `T` |
  | `T` | datetime | the trading-day matching report_date (the event day; **T-match failures dropped**) |
  | `calendar_week_group` | str | ISO week `YYYY-Www` (LTR group anchor) |
  | `added` | datetime | gated_events.added (earliest Wikipedia interval) |
  | `car_10d` | float | arith CAR Open[T+1]→Close[T+11] — **target label candidates** |
  | `car_60d_pass1` | float | 60-day arith CAR (pass-1 oracle, post-event drift) — **turns the 3 PEAD gates** |
  | `is_bmo` | bool | "Before Market Open" earnings (EODHD `BeforeMarket`/`AfterMarket` CamelCase match) |
  | `volume_vma20_ratio_pre_event` | float | Block 2 (Day-T features) |
  | `suv_day_1` | float | Block 2 |
  | `pre_event_idiosyncratic_vol` | float | Block 1 (pre-event features; Sunday-safe) |
  | `opening_gap_t1` | float | realized Open[T+1] / Close[T] gap — **LEAK FEATURE** (forward-looking, not Sunday-safe; used only as inference-time confirmation by Phase G) |
  | `intraday_range_t` | float | Block 2 |
  | `pre_event_volume_trend` | float | Block 1 (Sunday-safe) |
  | `rel_ret_{3,5,10,20,30}d` | float | Block 1 (Sunday-safe relative returns vs. IJH) |
  | `sector_adjusted_ret_20d` | float | Block 1 |
  | `sue_score` | float | Block 1 |
  | `eps_surprise_pct` | float | Block 1 — **capped at ±300%** |
  | `consecutive_surprises` | float | Block 1 (100% coverage) |
  | `sue_acceleration` | float | Block 1 |
  | `sue_lag_1`, `sue_lag_2` | float | Block 1 |
  | `car_drift_historical_q1` | float | Block 1 (Q1 drift prior to event) |
  | `sue_abs_x_inverse_vol` | float | Block 1 (interaction feature) |

- The **17 Sunday-safe features** (sans `opening_gap_t1`, `intraday_range_t`, `volume_vma20_ratio_pre_event`, `suv_day_1`) are the set the Sunday classifier is trained on. See `03_model/02_phase_g_sunday_classifier.py` for `SUNDAY_SAFE_FEATURES`.

---

### `/metadata/sp400` — written by `01_metadata_gathering.py` (Step 1), extended by `02_SEC_sector_gathering.py` (Step 2), extended by `02b_build_company_map.py` (Phase A audit cols)

- Per-**ticker** view (one row per ticker symbol on Wikipedia, including removed constituents). 993 rows × 11 cols.
- Used as the seed input to Phase A's `02b_build_company_map.py` to derive the permaTicker table.
- Schema:

  | Column | Source | Notes |
  |---|---|---|
  | `ticker` | `01` Wikipedia | ticker symbol (this alias) |
  | `name` | `01` Wikipedia | company name from current constituents table |
  | `gics_sector` | `01` Wikipedia | GICS sector (historical, may be sparse pre-2012) |
  | `gics_sub_industry` | `01` Wikipedia | GICS sub-industry |
  | `intervals` | `01` Wikipedia changes table | JSON list of `{"added", "removed"}` memberships (multi-interval captures boomerang re-additions) |
  | `sic` | `02` SEC EDGAR | SEC SIC code (3-tier lookup: ticker.txt → DERA historical → `SIC_PATCH`) |
  | `index_ref` | `02` derived | sector index ticker from SIC per `01_data/SIC_code_to_index.md` |
  | `cik` | `02b` legacy | point-in-time-present CIK (preserved for backwards compat with pre-Phase-A code). |
  | `canonical_ticker` | `02b` legacy | legacy canonical ticker (audit; superseded by `permaTicker`). Kept for Phase D migration audit. |
  | `cik_at_added` | `02b` Phase A | point-in-time CIK at the ticker's first Wikipedia-interval added year (via DERA `sub_{YYYY}.txt` snapshots w/ year-walkback). Used by Phase A's fork algorithm. |
  | `perm_id` | `02b` Phase A | legacy Phase-A synthetic perm_id (`f"{cik}_{start_ticker}"` or `f"__nocik_{start_ticker}"`). Superseded by Tiingo `permaTicker`; kept for audit only. |

---

### `/metadata/sp400_permatickers` — written by `01_data/02b_build_company_map.py` (Phase A permaTicker rewrite)

- Per-**permaTicker** view. **This is the PRIMARY identity table** the rest of the pipeline iterates over (`03_data_gathering.py`, `06_earnings_gathering.py`, `02_features/01_features_gate_events.py`, `02_features/02_build_feature_matrix.py`).
- Current state: **962 rows × 10 cols** (was 11 until today's cleanup dropped `legacy_perm_id`).
- 928 rows have `/sp400/{permaTicker}` price nodes written; 34 are `price_unavailable=True` (Tiingo probe returned no data; skipped).
- 697 `isActive=True` (Tiingo says still trading today), 265 `isActive=False` (delisted/merged/defunct).

  | Column | Type | Description |
  |---|---|---|
  | `permaTicker` | str | PRIMARY key. Tiingo identity-stable ID (e.g. `US000000001291` for Enovis Corp). Survives rebrands, delistings, spin-offs, bankruptcy reorgs. |
  | `canonical_ticker` | str | the `ticker` selected by Phase A's disambiguator (chosen first search hit per Wikipedia interval; verified by physical-row-count + isActive fallback). Used as the join-back for EODHD earnings / feature builder audit. |
  | `name` | str | best-available company name from Tiingo search response |
  | `isActive` | bool | `True` if Tiingo says this permaTicker is actively trading as of Phase A probe |
  | `openfigi` | str | OpenFIGI composite identifier from Tiingo search (defensive redundancy anchor) |
  | `cik` | str | informational CIK (the most recent for the permaTicker's audit) |
  | `sic` | str | SEC SIC code (itinherited from `/metadata/sp400` row of the chosen canonical ticker) |
  | `index_ref` | str | sector index ticker reference (per `SIC_code_to_index.md`) |
  | `wikipedia_intervals` | str (JSON) | JSON list of `{"added", "removed"|None}` S&P 400 membership spans across aliases (note: GME has 2 open-ended intervals — handled by dedup-by-earliest-added pass in Stage 1) |
  | `price_unavailable` | bool | `True` if Tiingo `/prices` probe returned no price rows for this permaTicker. `03_data_gathering.py` SKIPS writing a `/sp400/{permaTicker}` node for these. |

- Process Meta: produced by Tiingo search-lookup + `/prices` probe per Wikipedia ticker (and per interval when ambiguous). Disambiguator uses physical row count as primary tiebreaker + `isActive=False` preference for past-closed Wikipedia intervals. See `01_data/tiingo_permaTicker_audit.md` for live-probe evidence.

---

### ~~`/metadata/sp400_companies`~~ — REMOVED in Phase A
### ~~`/metadata/sp400_perm_ids`~~ — PURGED by `Phase A` permaTicker rewrite

Both legacy per-CIK / per-perm_id tables were purged from `db.h5` by `02b_build_company_map.py` to prevent downstream stages from accidentally reading stale data during the Phase B–E refactor.

---

## Producer-consumer map

| Node | Producer | Consumer(s) |
|---|---|---|
| `/metadata/sp400` | `01_metadata_gathering.py` (create), `02_SEC_sector_gathering.py` (cols), `02b_build_company_map.py` (audit cols) | `02b_build_company_map.py` (seed input to derive permatable) |
| `/metadata/sp400_permatickers` | `02b_build_company_map.py` | `03_data_gathering.py`, `06_earnings_gathering.py`, `02_features/01_features_gate_events.py`, `02_features/02_build_feature_matrix.py` |
| `/sp400/{permaTicker}` | `03_data_gathering.py` (one Tiingo `/prices` fetch per permaTicker) | `02_features/02_build_feature_matrix.py` (price data for feature computation), `04_backtest/*` (price data for backtest) |
| `/macros/{TICKER}` | `04_index_data_gathering.py` | `02_features/02_build_feature_matrix.py` (relative returns, sector-adjusted) |
| `/macros/fred_{name}` | `05_fed_data_gathering.py` | `02_features/02_build_feature_matrix.py` (Block-4 macro features) |
| `/earnings/fmp` | `06b_fmp_earnings_gathering.py` (write) | `02_features/01_features_gate_events.py` (Stage 1 gating) + `02_features/02_build_feature_matrix.py` (Block 1 features) |
| `/analyst/grades/{pt}` | `07_fmp_grades_gathering.py` (write) | `02_features/02_build_feature_matrix.py` (Block 6 revision momentum) |
| `/features/gated_events` | `02_features/01_features_gate_events.py` | `02_features/02_build_feature_matrix.py` (Stage 2) |
| `/features/train_matrix` | `02_features/02_build_feature_matrix.py` | `03_model/01_train_model.py` (helpers; main OBSOLETE), `03_model/02_phase_g_sunday_classifier.py` (deployable model trainer), `04_backtest/*` (OOS backtests) |
| Model artifacts (`03_model/models/*`) | `03_model/01_train_model.py` (OBSOLETE), `03_model/02_phase_g_sunday_classifier.py` (deployable), `03_model/03_phase_g_sweep.py` (v1.1 sweep) | `04_backtest/*` (load pre-trained model artifacts) |

## Write-safety pattern (all producers)

All DB writes use `pd.HDFStore(DB_FILE, mode="a")` + `store.remove(key)` to overwrite only the target node — **NEVER `mode="w"`**, which would truncate the entire `db.h5` and wipe every group at once. This was debugged in earlier versions of the pipeline and is now enforced across all pipeline scripts including the cleanup script (`cleanup_phase_d_2024_post_doc_j.py`).

For per-permaTicker node writes (`03_data_gathering.py`), the pattern uses `data_columns=["Date"]` so the node supports on-disk date-indexing (the feature builder uses this for date-range scans).

## Dedup policy (post 2024-07-21 cleanup)

Default rule going forward: **`(permaTicker, report_date)` is a primary key at every layer** (`/earnings/fmp` -> `/features/gated_events` -> `/features/train_matrix`). All three stages were re-run today against the deduped DB; **every table currently has 0 dup groups**. Any future re-run of `01_data/06b_fmp_earnings_gathering.py` should also enforce this dedup at write-time.

For multi-interval permaTickers (like GME), the dedup passes twice — once in the gate loop (where emissions can overlap interval boundaries) and once at train_matrix construction (the same dups propagate through Stage 2 unless Stage 1 already de-duplicated them, which it now does).

---

End of database layout reference.

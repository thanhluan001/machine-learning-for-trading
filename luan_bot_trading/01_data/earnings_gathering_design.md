# Earnings Gathering (EODHD) Design

> **⛔ IDENTITY MIGRATION NOTICE (2026-07-14).** Phase D's iteration source
> (`/metadata/sp400_perm_ids`) and its dedup key (`perm_id,
> fiscal_period_end`) are migrating to Tiingo `permaTicker`. The EODHD
> `/api/calendar/earnings` endpoint RETAINS the load-bearing role (Tiingo
> has no equivalent calendar endpoint), but ALL `perm_id` references below
> must be read as `permaTicker`. See
> [`01_data/tiingo_permaTicker_audit.md`](tiingo_permaTicker_audit.md)
> for the new identity model.

> Status: **Design approved, not yet implemented.** Created for future reference.
> Replaces the yahooquery POC (`06_fetch_earnings_poc.py`).
>
> **Phase D migration note (2026-07-14):** This design doc was written pre-Phase-A
> against the now-deleted `/metadata/sp400_companies` (per-CIK collapsing). It
> remains here for **architectural context** only. The live implementation is
> `06_earnings_gathering.py` (Phase D rewrite) which:
>   - Reads `/metadata/sp400_perm_ids` (POINT-IN-TIME CIK anchored, interval-forked).
>   - Iterates per **perm_id** (not per company / canonical).
>   - Dedup key is `(perm_id, fiscal_period_end)` (NOT `(canonical_ticker,
>     report_date)`) with tiebreaks: prefer canonical-alias code, then latest
>     `report_date`, then lexicographic `code`. See `database_layout.md`
>     `/earnings/raw` section for the live Phase D output schema, and
>     `merger_identity_patch.md` §7.7 for why keying by `canonical_ticker` was
>     wrong (12 perm_id pairs collide on canonical_ticker -> cross-perm_id
>     dedup lose events).
>
>   ⛔ All three points above (`/metadata/sp400_perm_ids`, `perm_id` primary
>   key, dedup key composition) are POST-PERMATICKER-MIGRATION targets:
>   `/metadata/sp400_perm_ids` -> `/metadata/sp400_permatickers`, `perm_id`
>   -> `permaTicker`, dedup key `(permaTicker, fiscal_period_end)`. The
>   collision-dedup problem (§7.7) dissolves because permaTicker IS the
>   storage key.

## Motivation

The feature matrix needs **one row per earnings event** per company (`features.md`,
`company_merge_design.md` §"Earnings Alignment"). The yahooquery POC
(`06_fetch_earnings_poc.py`) fetched only ~1 year of data and had reliability
issues. EODHD's `Calendar & News API` subscription provides **full historical
earnings** with actual EPS, estimated EPS, and report timing — sufficient for the
entire 15-year training window and forward inference.

## API credentials

- Provider: EOD Historical Data (`https://eodhd.com`)
- Subscription: paid plan, **100,000 calls/day, 1,000 calls/minute**
- API key read from `.env` as `EODHD_API_KEY`

### Endpoint used (only one)

```
https://eodhd.com/api/calendar/earnings
```

Parameters (verified against live response, not just docs text):

| Param | Required | Type | Description |
|---|---|---|---|
| `api_token` | yes | string | API key |
| `symbols` | yes (for our use) | string | Comma-separated `{TICKER}.{EXCHANGE}` (e.g. `AAPL.US,MSFT.US`). |
| `from` | **yes** | date | Start (YYYY-MM-DD). Docs say "ignored when symbols set" — **false**, in practice `symbols=` without `from`/`to` returns an empty `earnings` array. |
| `to` | **yes** | date | End (YYYY-MM-DD). Same caveat as `from`. |
| `fmt` | no | `json`/`csv` | Use `json`. |

Response shape (dict, not list):

```json
{
  "type": "Earnings",
  "description": "Historical and upcoming Earnings",
  "symbols": "AAPL.US",
  "earnings": [
    {"code":"AAPL.US","report_date":"2023-02-02","date":"2022-12-31",
     "before_after_market":"AfterMarket","currency":"USD",
     "actual":1.88,"estimate":1.95,"difference":-0.07,"percent":-3.5897}
  ]
}
```

Per-event fields:

| Field | Meaning |
|---|---|
| `code` | ticker with exchange suffix — the alias EODHD stored the row under |
| `report_date` | announcement date (this is `T` in PEAD terms — event time) |
| `date` | fiscal period end (quarter ending) — NOT the announcement date |
| `before_after_market` | `"Bmo"` (before market open) or `"AfterMarket"` (after close) → `is_bmo` |
| `currency` | EPS currency |
| `actual` | reported EPS |
| `estimate` | pre-report consensus EPS — this is the historical estimate that existed before the announcement, NOT a forward trend |
| `difference` | actual − estimate |
| `percent` | surprise % = difference / estimate × 100 (= `eps_surprise_pct` in `features.md`) |

### Endpoint NOT used (out of scope)

`https://eodhd.com/api/calendar/trends` is **forward-looking only**:

- All rows returned have `period` ∈ `{+1q, +1y}` with `date` ∈ the future
- It does NOT contain historical analyst estimates that existed at the time
  of past announcements (e.g. Q2 2016 estimate before Q2 2016 report)
- Cannot backfill training features for past quarters

The `earnings` endpoint already gives the pre-report consensus `estimate`
*per past event*, which is what we need for training features. The `trends`
endpoint might be useful for the Sunday **inference** path (forward consensus
for upcoming earnings), but its marginal predictive contribution is
unproven and `features.md` does not depend on it.

**Trends is fully out of scope for this step.**

## Pipeline Position

```
01_metadata_gathering.py
02_SEC_sector_gathering.py
02b_build_company_map.py
03_data_gathering.py
04_index_data_gathering.py
05_fed_data_gathering.py
06_earnings_gathering.py        <-- NEW (replaces 06_fetch_earnings_poc.py)
```

Inputs:
- `/metadata/sp400_companies` (canonical_ticker, aliases, price_unavailable) from `02b`
- `START_DATE = today - 15y` (matches Tiingo depth in `03_data_gathering.py`)
- `END_DATE = today - 1d` (matches Tiingo end date)

Outputs:
- `/earnings/raw`              raw per-event rows from EODHD, with the original
                              `ticker` (alias) preserved so the feature builder
                              can re-derive canonical alignment later.
- Optional `/earnings/calendar` (TBD by feature builder; not written here)

## Files Affected

| File | Change |
|---|---|
| `06_earnings_gathering.py` | **NEW** — EODHD fetcher + storage |
| `06_fetch_earnings_poc.py` | **DELETED** — yahooquery POC, replaced |
| `.env` | Add `EODHD_API_KEY=...` (manual, by user) |
| `luan_bot_trading/Design.md` | Update §11 (earnings protocol) to reflect EODHD + company-level storage |

## Algorithm

### Step 1 — Load company universe

Read `/metadata/sp400_companies`. Use the `aliases` field (JSON list) per
company. Skip companies with `price_unavailable=True` per `02b`'s flag
(no price history → no earnings worth modeling — keeps the feature matrix
clean; this matches the `03_data_gathering.py` skip pattern).

### Step 2 — Per-company fetch (1 call per company, with date window)

For each company:

1. Build the `symbols=` argument from its aliases. Map each alias to
   `{TICKER}.US`. For US S&P 400 mid-caps the `.US` exchange suffix is
   always correct.
2. Send a single GET to `https://eodhd.com/api/calendar/earnings` with
   `symbols={a1.US,a2.US,...}`, `from=START_DATE`, `to=END_DATE`,
   `api_token=KEY`, `fmt=json`. **`from`/`to` are required** — without them
   the endpoint returns an empty `earnings` array even when `symbols=` is
   set (verified against the live API).
3. Read `body['earnings']` (response is a dict, NOT a list).
4. Safety-filter rows with `report_date` inside `[START_DATE, END_DATE]`
   (EODHD occasionally returns out-of-window rows).
5. Per row, store both the **original alias ticker** (raw `code` returned
   by EODHD) and the **canonical ticker** (from the company row), and the
   **CIK** (from the company row).

### Step 3 — Storage

Append all per-company rows into one long-form DataFrame and write under
`/earnings/raw` (overwriting prior runs via the `HDFStore('a')` +
`store.remove('/earnings/raw')` pattern — never `mode='w'` on the whole DB).

Schema (raw EODHD rows, one per event):

| Column | Source | Notes |
|---|---|---|
| `report_date` | EODHD `report_date` | announcement date `T` (YYYY-MM-DD) |
| `fiscal_period_end` | EODHD `date` | fiscal quarter end (NOT the event date) |
| `code` | EODHD `code` | the alias ticker EODHD stored the row under (e.g. `AAXN.US`) |
| `canonical_ticker` | our company map | our canonical for the company |
| `cik` | our company map | SEC CIK (10-digit string, may be None for singletons) |
| `actual` | EODHD `actual` | reported EPS (may be null pre-event / upcoming) |
| `estimate` | EODHD `estimate` | pre-report consensus EPS |
| `difference` | EODHD `difference` | actual − estimate |
| `percent` | EODHD `percent` | `eps_surprise_pct` directly |
| `before_after_market` | EODHD `before_after_market` | raw timing flag; feature builder encodes `is_bmo` |
| `currency` | EODHD `currency` | usually `USD` |

`is_bmo` encoding rule (applied by the feature builder): `1` iff
`before_after_market == "Bmo"` (case-sensitive exact match), `0` otherwise.

SUE / `consecutive_surprises` / `sue_acceleration` / `sue_lag_1` /
`sue_lag_2` / `rev_growth_yoy` are **NOT** computed in this script — they
belong to the feature builder (`features.md` blueprint). This layer is
faithful EODHD storage only.

**`rev_growth_yoy`**: the `earnings` endpoint does NOT include revenue.
`features.md` Block 1 lists `rev_growth_yoy` but it cannot be sourced from
this endpoint. It is deferred out of the current feature builder (will be
re-evaluated later via the EODHD fundamentals API if it materially improves
the model).

### Step 4 — Checkpoint

No checkpoint needed — limits are 100k/day / 1000/min, full run is ~930 calls
finishing in minutes. Simplifies the script and removes state management
headaches (no `earnings_offset.txt`). If the run fails halfway, re-run the
whole thing (idempotent; `store.remove` + put pattern).

### Step 5 — Deduplicate per canonical

After unioning all per-company rows, deduplicate by
`(canonical_ticker, report_date)` to handle rebrand-transition overlap:
same company could have rows under both the old alias (e.g. `AAXN`) and the
new alias (`AXON`) for the same `report_date` if EODHD stores overlap.
Keep the row whose `code` matches the alias that was the active SEC ticker
on that `report_date` (best-effort using the company's `combined_intervals`);
fallback: keep the first occurrence. This yields exactly one row per
event per company, which is the unit the feature builder groups on.

### Step 5 — Audit output

Print:
- Total companies: M
- Companies skipped (price_unavailable=True): K
- Companies fetched: M-K
- Total events retrieved: N (sum across all companies)
- Coverage: events received for X / (M-K) companies
- Companies with zero events (EODHD has no history): listed separately, logged
  as `earnings_unavailable`. The feature builder treats these as companies
  with 0 earnings rows (no training contribution).

## Q6 — Companies with no EODHD data

If a company's EODHD call returns zero rows: log + skip. No `earnings/calendar`
row is written for that company. The feature builder naturally discounts such
companies to 0 rows in the training matrix.

## Decisions Logged

- **Q1 (replace vs new)**: option (b) — new file `06_earnings_gathering.py`,
  delete `06_fetch_earnings_poc.py`. Confirmed.
- **Q2 (per-company vs per-ticker)**: option (a) — fetch by `symbols=` with
  all aliases of a company on one call; merge into one company-level timeline.
  Confirmed.
- **Q3 (batching)**: 1 call per company with comma-separated aliases. No
  checkpoint needed given 100k/day, 1000/min limits. Confirmed (limits given).
- **Q4 (storage schema)**: option (a) — store raw EODHD rows with
  `canonical_ticker` and `cik` joined in; SUE / `consecutive_surprises` /
  `is_bmo` encoding deferred to the feature builder. Confirmed.
- **Q5 (history depth)**: same as Tiingo (`today - 15y` to `today - 1d`).
  Confirmed.
- **Q6 (EODHD-gap companies)**: log + skip; no row written. Confirmed.
- **Trends endpoint**: confirmed UNUSED — it returns only forward-looking
  estimates (`+1q`, `+1y` with future `date`), cannot backfill historical
  estimates for past training events. Confirmed out of scope.
- **`from`/`to` required when `symbols=` set**: docs say ignored; in practice
  required (returns empty without). Confirmed via live probe.
- **`report_date` vs `date`**: `report_date` is the announcement date (`T`);
  `date` is the fiscal period end. We use `report_date` for event timing and
  window filtering.
- **Deduplication per `(canonical_ticker, report_date)`**: applied during
  storage to handle rebrand-transition overlap. Confirmed (Q3 of new round).
- **`is_bmo` encoding**: feature builder maps `before_after_market == "Bmo"`
  → 1, else 0 (case-sensitive exact match). Confirmed (Q4 of new round).
- Ticker format: append `.US` to aliases before building the `symbols=` param.
- `rev_growth_yoy` deferred: `earnings` endpoint does not include revenue.
- No `.env` modification by this script — `EODHD_API_KEY` must be added by
  user (already done in this case).

## Open Items Out of Scope

- EODHD `/api/calendar/trends`: forward-looking only; out of scope here.
  May be revisited when the Sunday inference pipeline is built.
- `/earnings/calendar` derived view (canonical-keyed, post-merge): the
  feature builder reads `/earnings/raw` directly; no separate denormalized
  view is needed.
- `rev_growth_yoy`: not sourceable from the `earnings` endpoint; deferred.
  EODHD fundamentals API (`/api/fundamentals/{TICKER.US}`) could provide
  historical income-statement revenue if needed later.
- Non-US exchange suffix mapping: not needed (entire universe is `.US`).
- `report_date`/`fiscal_period_end` timezone: EODHD returns date strings,
  not timestamps; no timezone normalization needed.

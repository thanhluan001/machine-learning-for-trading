# Company Merge & Canonical Ticker Design

> Status: **Design approved, not yet implemented.** Created for future reference.
> Owner of this design: agreed in conversation on 2026-07-03.

## Earnings Alignment (downstream consumer)

The feature builder (post `02b`, post `06_fetch_earnings_poc.py`) is the consumer of this merge:

1. Read `/metadata/sp400_companies` and `aliases` per company.
2. For each company (CIK), gather **all earnings dates** stored under any of its `aliases` in `/earnings/calendar` (from `06_fetch_earnings_poc.py`) and collapse them into one earnings timeline for the company.
3. Use the `canonical_ticker`'s price series (`/sp400/{canonical_ticker}`) as the OHLCV source for all feature calculations — including pre-rebrand earnings events. Tiingo's retro-adjusted price history under the canonical ticker already spans the aliases' periods.
4. Use `combined_intervals` to **gate** each earnings event: only events where the earnings date falls within a membership span become training rows (per the 90-day exclusion buffer in `Design.md`).
5. For each surviving earnings event, build one feature row per `features.md`.

Result: one **company** in `/metadata/sp400_companies` → many **earnings events** → many **feature rows** in the final matrix.

## Motivation

The PEAD feature matrix has one row per **company**, not per ticker symbol. Companies change ticker symbols over time (rebrands, mergers, bankruptcy-Q suffixes) while staying in the S&P 400 under continuous membership. Treating each ticker symbol as a separate company:

- loses pre-rebrand price history (e.g. `TASER` → `AAXN` → `AXON`),
- creates phantom "still in the index" rows (e.g. `AAXN` has `removed=null` because the rename to `AXON` isn't an S&P 400 changes-table event),
- drops training rows for tickers not available on the data source (e.g. `APY` not on Tiingo, even though its successor `CHX` is),
- dilutes membership intervals across aliases of the same company (e.g. AXON-Axon-AAXN all filings same SEC CIK).

## Goal

The feature matrix for the PEAD bot has **one row per earnings event**, not one row per company (see `features.md`). Each row encodes macros, recent price movement, sector/relative returns, and catalyst fundamentals to predict post-earnings drift (CAR) for that specific earnings event.

This step (`02b_build_company_map.py`) produces a **company-level view** of the S&P 400 historical universe so that the feature builder can correctly attribute earnings events across ticker rebrands. Concretely:

1. Each row of `/metadata/sp400_companies` represents one company (identified by SEC CIK).
2. The company's S&P 400 membership is a single merged interval span (collapsed from all its ticker aliases).
3. A single **canonical ticker** is chosen for fetching price data; that ticker is verified to exist on Tiingo.
4. The data-fetcher (`03_data_gathering.py`) iterates per company, fetching price history under the canonical ticker (with fallback to aliases if needed).
5. The feature builder then, for each company (CIK), pulls **all earnings dates** of that company and creates **one feature row per earnings date** — using the canonical ticker's price history (which Tiingo retro-adjusts across rebrands) and the merged membership interval for gating.

So a single company will have **many rows** in the final feature matrix (one per earnings event over its 15-year S&P 400 membership window). The CIK merge is the mechanism that prevents the same earnings event from being fragmented across ticker aliases, and prevents pre-rebrand earnings events from being lost (e.g. AXON-era price history on Tiingo covers the old TASER/AAXN period, so historical earnings events under those tickers can still be aligned to the canonical CIK's price series).

## Canonical Anchor: SEC CIK

SEC CIK (Central Index Key) is the stable company identifier that survives:

- ticker renames (`TASER` → `AAXN` → `AXON` — all CIK 0000894288),
- corporate rebrands (`APY` Apergy → `CHX` ChampionX — same CIK 0001788401),
- bankruptcy + Q-suffix (`ASNA` → `ASNAQ` — same CIK 0000796371),
- delistings (CIK persists; SEC EDGAR filings remain accessible under that CIK indefinitely).

This makes CIK the right anchor for "same company, different tickers".

### Scope caveat (Q1 — spinoffs)

Spinoffs create a **new** CIK for the spun-off entity, even though it shares lineage with the parent. v1 treats each CIK as one company (spinoff → new company). Justification: the spun-off entity has its own distinct price history starting at the spinoff date, so treating it as a new company is the correct modeling choice. Chaining CIKs across spinoff boundaries (via SEC `predecessor`/`successor` fields) is intentionally **out of scope** for v1.

## Pipeline Position

```
01_metadata_gathering.py        # writes /metadata/sp400 (per-ticker intervals from Wikipedia)
02_SEC_sector_gathering.py     # adds sic + index_ref columns; refactored to expose CIK history map
02b_build_company_map.py  <-- NEW
                                # builds /metadata/sp400_companies (per-CIK merged view)
03_data_gathering.py            # iterates per company, fetches price under canonical ticker
04_index_data_gathering.py
05_fed_data_gathering.py
06_fetch_earnings_poc.py
```

## Inputs

`02b_build_company_map.py` reads:

1. `/metadata/sp400` from `db.h5`
   - Columns: `ticker`, `name`, `intervals` (JSON list of `{"added","removed"}`), `sic`, `index_ref`
2. **Ticker → CIK history map** (refactored utility in `02_SEC_sector_gathering.py`):
   - `build_ticker_to_cik_history()` returns `{ticker: CIK}` built from:
     - Current `sec_cache/ticker.txt` (active registrants)
     - All cached DERA `sec_cache/dera/sub_{year}.txt` files (2010–2025), where ticker is extracted from the `instance` column's leading token (`instance.split("-")[0].upper()`)
   - Union across all years → near-complete 15-year `ticker → CIK` map
3. `KNOWN_RENAMES` dict (seeded `{}` initially) — residual ticker → canonical mappings for cases where CIK lookup fails. Filled in iteratively after audit runs reveal unresolvable cases.

## Refactor to `02_SEC_sector_gathering.py`

- Extract the DERA `sub.txt` parsing logic into a reusable function `build_ticker_to_cik_history() -> dict[str, str]`.
- The existing `build_sic_map()` flow is **unchanged** — no behavioral change to SIC outputs. All merge logic lives in `02b`.
- `02b` imports the new utility to get the CIK map.

## Algorithm

### Step 1 — Build ticker → CIK map

For every ticker in `/metadata/sp400`:

1. Look up CIK in `build_ticker_to_cik_history()` (current `ticker.txt` ∪ DERA historical).
2. Tickers with no CIK found → `cik = None`; becomes its own canonical singleton company.

### Step 2 — Group tickers by CIK

- All tickers with the same non-null CIK → one company group.
- Tickers with `cik = None` → singleton groups (one ticker per company).

### Step 3 — Choose canonical ticker per group (priority)

1. Ticker that is in the current `ticker.txt` AND returns 200 on Tiingo `/daily/{ticker}` → canonical.
2. Else, ticker with the most-recent `removed` date in its intervals AND returns 200 on Tiingo → canonical.
3. Else, the most-recently-added ticker (highest `added` date) regardless of Tiingo — flagged `price_unavailable = True`.

Aliases are stored as an ordered list: canonical first, then other Tiingo-verified fallbacks, then unverified tickers.

### Step 4 — Merge intervals (option i: single span, gaps preserved per Q4-yes vote)

For each company group:

1. Collect all `intervals` arrays across all aliases.
2. Sort by `added` date.
3. Walk through and merge overlapping or **abutting** intervals (a gap of ≤7 days counts as abutting → merge into one span).
4. A real gap (company left index, then re-entered under a different alias later) → kept as a separate span.
5. Result `combined_intervals`: list of `{"added": date, "removed": date|None}` spans — usually 1 span per company.
6. Also retain `per_ticker_intervals` as JSON `{ticker: [{added, removed}, ...]}` for audit.

### Step 5 — Write outputs to `db.h5`

All writes use the safe `HDFStore(mode='a')` + `store.remove(node)` pattern (never `mode='w'` — that bug class is already fixed in this codebase).

**Updated `/metadata/sp400`** — add per-ticker columns:

- `cik` — SEC CIK or `None`
- `canonical_ticker` — the canonical ticker this alias belongs to

**New `/metadata/sp400_companies`** — one row per company (CIK):

| Column | Type | Description |
|---|---|---|
| `canonical_ticker` | str | Ticker chosen for Tiingo price fetch |
| `cik` | str | SEC CIK (or `None` for singletons) |
| `aliases` | JSON list[str] | All tickers in this company group, canonical first |
| `name` | str | Best available name (from current constituents if possible, else last-known) |
| `sic` | str | SIC code (taken from canonical ticker's row in `/metadata/sp400`) |
| `index_ref` | str | Sector index reference for the canonical ticker |
| `combined_intervals` | JSON list[{"added","removed"}] | Merged S&P 400 membership spans |
| `per_ticker_intervals` | JSON dict[ticker → list[{"added","removed"}]] | Audit trail of original per-ticker intervals |
| `price_unavailable` | bool | `True` if no alias in the group is fetchable from Tiingo |

### Step 6 — Reset checkpoint

- Write `0` to `stock_offset.txt`. The next `03_data_gathering.py` run will iterate the new per-company space.

## `03_data_gathering.py` Changes (after 02b ships)

1. **Iterate `/metadata/sp400_companies`** (not `/metadata/sp400`). Checkpoint advances per company, not per ticker.
2. **For each company row**:
   - If `price_unavailable == True`: **skip entirely + log** (one print line). Do not create an empty `/sp400/{canonical}` placeholder. (Q1-yes: keep v1 simple; revisit if many of these pile up.)
   - Else, try `aliases` in priority order on Tiingo → store the first non-empty response under `/sp400/{canonical_ticker}`.
3. **Batch size**: 45 companies per run (still under Tiingo's 50 req/hr free tier).
4. Existing Tiingo fetch + storage logic (full-history refetch on any gap) preserved.

## Audit Output (end of `02b_build_company_map.py` run)

Print to console for visibility:

- Total tickers in `/metadata/sp400`: N
- Tickers with CIK found: X / N
- Companies (groups): M (~930–950 expected after merging)
- Companies merged from multiple aliases: K (count of non-singleton groups)
- **List of merged companies** with their aliases and chosen canonical (eyeball AAXN→AXON, APY→CHX, etc.)
- **Companies with `price_unavailable=True`** (no Tiingo-available ticker): listed separately. These become candidates for entries in `KNOWN_RENAMES` on the next iteration.

## Known Cases (expected handling)

| Tickers | CIK | Expected canonical | Notes |
|---|---|---|---|
| TASER, AAXN, AXON | 0000894288 | AXON | TASER is *not* currently in `/metadata/sp400` (parser blind to pre-2012 rebrand). Only AAXN + AXON will merge. TASER silent loss ok for v1. |
| APY, CHX | 0001788401 | CHX | APY in `/metadata` but not on Tiingo; CHX active. Merge gives `/sp400/CHX` with full Apergy→ChampionX history. |
| ASNA, ASNAQ | 0000796371 | depends on Tiingo | If both delisted and not on Tiingo, `price_unavailable=True`. Membership interval still useful for gating. |
| GTM, ZI | (same CIK) | ZI (if on Tiingo) | ZoomInfo rebrand. |
| FTR, FYBR | (same CIK) | FYBR (if on Tiingo) | Frontier rebrand. |
| UA, UAA | (same CIK) | UAA (if on Tiingo) | Under Armour share-class variants — may need separate handling if they're truly different share classes (different CUSIPs but same CIK). Flag for audit. |
| ATGE | (CIK or singleton) | ATGE | Live NYSE listing but not on Tiingo (data gap). If CIK has no aliases, `price_unavailable=True`. |

## Files Affected

| File | Change |
|---|---|
| `02b_build_company_map.py` | **NEW** — implements the merge algorithm and writes both outputs |
| `02_SEC_sector_gathering.py` | Refactor only: extract `build_ticker_to_cik_history()` utility. No SIC behavior change. |
| `03_data_gathering.py` | Switch to per-company iteration + alias fallback; reset checkpoint to per-company index space |
| `Design.md` | Update to reflect company-level (not ticker-level) feature matrix and the new `02b` step |
| `features.md` | Note that feature rows are per-company (canonical ticker), not per ticker symbol |

## Open Items Out of Scope for v1

- TASER-era capture (pre-2012 rebrand with no changes-table row): not recovered. Only aliases already in `/metadata/sp400` are merged.
- Spinoff CIK chaining (Q1-a): each CIK = one company.
- Q-suffix-bankruptcy delisted price history: same CIK used; if no alias is on Tiingo, row is `price_unavailable=True`.
- Bulk pre-emptive Tiingo availability caching: redundancy with Step 3's per-ticker checks; not worth a separate cache layer yet.

## Decisions Logged

- **Q1 (spinoff boundary)**: option (a) — each CIK = one company. Confirmed.
- **Q2 (no-CIK fallback)**: option (a) + small `KNOWN_RENAMES` dict for residuals. Confirmed.
- **Q3 (canonical selection priority)**: active+Tiingo → most-recent-removed+Tiingo → latest-ticker-regardless. Confirmed.
- **Q4 (interval merge style)**: option (i) — single merged span per company, gaps preserved if real (>7 days). Audit `per_ticker_intervals` retained. Confirmed.
- **Q5 (new file)**: `02b_build_company_map.py` produces both `/metadata/sp400` (with `cik`/`canonical_ticker` cols) and new `/metadata/sp400_companies`. Confirmed.
- **Q6 (checkpoint reset)**: option (a) — reset `stock_offset.txt` to 0 when switching to per-company iteration. Confirmed.
- **price_unavailable companies**: skip entirely + log; no empty placeholder nodes in `/sp400`. Confirmed.
- **CIK refactor scope in `02`**: pure refactor, no SIC behavior change. All merge logic in `02b`. Confirmed.

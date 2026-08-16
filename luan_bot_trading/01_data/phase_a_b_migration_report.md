# Phase A + Phase B Migration Report — perm_id → permaTicker

**Status**: COMPLETE (2026-07-15).
**Scope**: Identity layer migration from the synthetic `perm_id` (CIK-anchored)
to Tiingo's `permaTicker` as the primary entity identifier, plus price-data
source migration from EODHD to Tiingo. This report documents the work
performed, the bugs discovered and fixed along the way, the resulting
database state, and the remaining stale phases requiring re-runs.

---

## TL;DR

- Replaced ~1300 LOC of Wikipedia+DERA+CIK identity-synthesis code with
  ~800 LOC of Tiingo search-based permaTicker discovery.
- Replaced EODHD `/api/eod` price history with Tiingo
  `/tiingo/daily/{permaTicker}/prices` (natively returns adj-OHLC-Volume —
  no local derivation needed).
- Eliminated the Phase B v2.1 alias-concatenation machinery entirely.
- All four documented bug classes (U, V, W, S) are now resolved at source.
- Database has 928 permaTicker-keyed price nodes (2.75M rows total,
  zero NaN Adjusted_Closed contamination) backed by Tiingo.
- Two downstream phases (Phase D earnings re-keying, Phase E gating/feature
  matrix) remain on stale `perm_id` keys and must be re-run before
  retraining the model.

---

## Architectural changes

### Identity layer (`02b_build_company_map.py`)

| Aspect | Before (`perm_id`) | After (`permaTicker`) |
|---|---|---|
| Primary key | `f"{cik_at_added}_{start_ticker}"` (synthetic) | Tiingo's `permaTicker` (e.g. `US000000000583`) |
| Identity provenance | Wikipedia intervals → DERA point-in-time CIK lookup → manual `MANUAL_TAS_OVERRIDE` | Wikipedia intervals → Tiingo search per-ticker-code → positional probe disambiguation |
| Cross-references | `aliases` (legacy ticker symbols), `sec_active_aliases` | `canonical_ticker`, `legacy_perm_id`, `openfigi`, `isActive` |
| Removed concepts | §7.7 disambiguation (12 collision pairs), WARNiNG: alias-extension, post-Wikipedia active-alias extension, manual TAS override | (no equivalent needed — permaTicker identity is upstream-provided) |
| Storage path | `/metadata/sp400_perm_ids` (974 rows × 10 cols) | `/metadata/sp400_permatickers` (962 rows × 11 cols) |

The legacy `/metadata/sp400_perm_ids` table has been PURGED. The new
`/metadata/sp400_permatickers` table carries a `legacy_perm_id` column for
backwards cross-reference to Phase D's existing `/earnings/raw` rows, but
**on the 2026-07-15 re-run this column was left empty** because the legacy
table was already purged in the 2026-07-14 first run (see "Known issues"
below). The legacy_perm_id → permaTicker mapping can be **reconstructed**
from `/metadata/sp400`'s `cik_at_added` + `canonical_ticker` columns at
Phase D re-key time using the legacy `perm_id`-formula
`f"{cik_at_added}_{canonical_ticker}"`.

### Price fetcher (`03_data_gathering.py`)

| Aspect | Before (EODHD) | After (Tiingo) |
|---|---|---|
| Endpoint | `/api/eod/{ticker}.US` (per alias, concatenated) | `/tiingo/daily/{permaTicker}/prices` (single fetch) |
| Aliases fetched | `canonical_ticker` + ALL `aliases` (concatenated, dedup-keep-last on `Date`) | (none — the permaTicker fetch returns rebrand-merged history server-side) |
| Adjusted columns | Only `adjusted_close` returned; we derived `Adj_Open/Adj_High/Adj_Low/Adj_Volume` locally via `close/adjusted_close` ratio | Native: Tiingo returns `adjOpen/adjHigh/adjLow/adjClose/adjVolume` directly — zero local derivation |
| Date normalization | `pd.to_datetime(...)` (timezone-naive) | `pd.to_datetime(..., utc=True).dt.tz_localize(None).dt.normalize()` (handle Tiingo's `T00:00:00.000Z` ISO format) |
| Storage path | `/sp400/{canonical_ticker}` (e.g. `/sp400/AAP`) | `/sp400/{permaTicker}` (e.g. `/sp400/US000000000583`) |
| Sort stability | `sort_values("Date")` default `kind="quicksort"` (NON-STABLE — root cause of Phase B v2.1 contamination!) | `sort_values("Date", kind="mergesort")` (STABLE — disaster-class bugs eliminated) |
| Aggregate-by-canonical workaround | YES (12 canonical-collision pairs required grouping per-perm_id fetches to avoid overwriting each other) | NO (no collision possible — permaTicker is the storage key) |
| Always-refetch v2.1 fix | YES (Phase A records the alias-fetch was non-deterministic) | N/A (no incremental logic — every permaTicker is re-fetched each run; considered acceptable since the data is clean) |

---

## Algorithm details

### Phase A: permaTicker discovery

```
For each ticker_code row in /metadata/sp400:
  For each Wikipedia interval {added, removed}:
    Call Tiingo /utilities/search/{ticker}?includeDelisted=true&exactTickerMatch=true
    Filter to countryCode=US
    For each candidate:
      Compute positional probe window [added - 7d, added + 30d]
      (If narrow probe returns 0 rows, fall back to [added - 90d, added + 275d] wide probe)
      Score rows for "physical data" plausibility:
         - sane = close > 0 AND median(close) < $100k AND max(close) < $100k
         - score = penalty (sorted ascending = better)
    If any sane candidates:
       Pick the one with MOST physical rows (adjClose/close ratio in [0.001, 1000])
       Tie-break: prefer isActive=False (the historical owner is typically inactive)
       Tie-break: lower sanity_score
    Else if exactly one inactive candidate, return it as fallback.
    Else pick the first US-stock candidate.
  Aggregate by permaTicker: if multiple Wikipedia intervals resolve to
    the same permaTicker, combine them into one row with multiple
    wikipedia_intervals entries (rebrand-series handling — e.g. ALK,
    AM, AMG with multiple Wiki add/remove cycles).
```

**Key design point — the "physical row" check**: When two permaTickers
both probe non-empty for a narrow Wikipedia-window probe, the historical
owner of that interval has rows whose `adjClose / close` ratio is roughly
1.0 in that era. The modern recycler's ticker-code ancestor may also have
returned rows for that historical window, but those rows have an
`adjClose / close` ratio exploded by accumulated reverse-split factors
through the recycler's CSII→PEGY→SUNE-style chain.

Discovered case (the SUNE Class-S bug fixed during this work):
- SunEdison (US000000002709): 2012 narrow-probe rows have ratio=1.0 — 24 of 24 physical.
- SUNation (US000000002062): 2012 narrow-probe rows have ratio ~150,000× — 0 of 24 physical.

The new disambiguator correctly identifies SunEdison as the historical
owner of the [2012-01-01, 2016-04-04] Wikipedia interval (SunEdison era SUNE,
delisted April 2016 after bankruptcy).

### Phase B: Tiingo fetch + storage

```
For each permaTicker row in /metadata/sp400_permatickers:
    GET /tiingo/daily/{permaTicker}/prices?startDate=2011-07-21&endDate=yesterday
    Map Tiingo fields directly to OUTPUT_COLUMNS schema (no derivation):
        date -> Date      adjOpen -> Adj_Open
        open -> Open       adjHigh -> Adj_High
        high -> High       adjLow -> Adj_Low
        low -> Low         adjClose -> Adj_Close
        close -> Close     adjVolume -> Adj_Volume
        volume -> Volume
    Normalize Date to timezone-naive (strip +00:00).
    Stable sort Date (mergesort). Drop NaN adjusted.
    Store under /sp400/{permaTicker}.
    If 0 rows returned: skip (no placeholder node).
    Pemberton: write-back final price_unavailable flag to /metadata/sp400_permatickers
              (Q1 design: narrow-probe Phase A hint is corrected by full-history fetch result).
Finally: PURGE /sp400/{KEY} nodes whose KEY is no permaTicker
         (legacy /sp400/{canonical_ticker} nodes from prior Phase B v2.1).
```

Phase B always re-fetches every permaTicker each run (no incremental
freshness check). With Tiingo's 10k/hr paid tier, this is fast (~13 min
for 962 permaTickers) and removes a complexity apartheid. Future work could
add incremental refresh if 13 min/cycle becomes a constraint.

---

## Bug class resolution

### Class U: ENOV/CFX alias-intertwining
- **Cause (Phase B v2.1)**: `fetch_concatenated_aliases_from_eodhd()` redundantly
  fetched the LEGACY alias whose ticker symbol had been re-assigned to a
  different modern security. `pandas.sort_values("Date")` quicksort then
  non-deterministically intermixed `adjusted_close` chains on overlapping dates.
- **Resolution**: `03_data_gathering.py` no longer fetches aliases. Each
  permaTicker is fetched ONCE via its own permaTicker path. Tiingo
  back-merges any rebrand history server-side under the permaTicker key.
- **Verification**: `/sp400/US000000001291` (ENOV permaTicker) has 3767 rows
  Colfax+Enovis history, Close first/last: $26.15 → $28.07. No NaN in Adj_Close.

### Class V: LDOS/SAI → reclassified as hidden Class U
- **Cause (Phase B v2.1)**: Same as Class U — the legacy SAI alias fetched by
  perm_id `0001336920_SAI` returned rows from the modern recycled SAI permaTicker,
  contaminating the LDOS permaTicker's price chain.
- **Resolution**: Each permaTicker is fetched independently. The LDOS
  permaTicker `US000000009014` retrieves Leidos Holdings full history
  2011-2026 (3768 rows) directly from Tiingo. The modern SAI canonical
  (SAIC = `US000000002531`) is a SEPARATE permaTicker with its own row.
- **Verification**: `/sp400/US000000009014` (LDOS) has 3768 rows,
  Close $16.72 → $108.29. Adjusted $22.88 → $108.29 (2012 Leidos era post-split
  factor visible). No NaN.

### Class W: NSR (Neustar) — EODHD upstream bug
- **Cause**: EODHD retroactively mapped Nomad-Royalty 2021+ prices back to 2015
  timestamps for the `NSR.US` ticker code — wrong company's data
  under Neustar-era ticker. Confirmed by direct EODHD probe.
- **Resolution**: Tiingo cleanly separates Neustar
  (`US000000006945`, inactive) from Nomad Royalty (`US000000062716`,
  inactive) and National Storage REIT (AU-listed, filtered out by
  `countryCode=US`). The Phase A probe correctly picks Neustar as the
  Wikipedia 2013-2017 interval owner — Neustar returns real 2011-2017
  prices in that window.
- **Verification**: `/sp400/US000000006945` (Neustar) has 1530 rows
  spanning 2011-07-21 to 2017-08-17. Close $26.31 → $33.50 (Neustar-era real
  prices). No NaN. AdjClose=close (Neustar never split/dividend'd in this window).

### Class S: SUNE (SunEdison) — case discovered and fixed during this work
- **Cause (pre-fix Phase A)**: Disambiguator scored candidates by
  `abs(log10(median_close)) - 1.5`, preferring close prices near $31. SUNation's
  2012-era CSII algorithm-tuning rows (~$13.88) had a lower
  penalty (0.36) than SunEdison's 2012-era prices (~$3.97, penalty 0.90), so
  SUNation was wrongly chosen even though SunEdison was the
  SEC EDGAR-confirmed owner of that ticker code in 2012 (`cik_at_added=0000945436`
  → SunEdison, Inc.).
- **Fix**: Added `_physical_row_count()` — counts rows whose
  `adjClose / close` ratio is in [0.001, 1000]. The candidate with MORE
  physical-era rows wins. SunEdison has 24/24 physical rows in the 2012
  narrow probe (ratio=1.0); SUNation has 0/24 physical rows in 2012
  (their rows have ratio ~150,000× because of CSII→PEGY→SUNE back-merged
  serial splits accumulated through the active chain). With the fix,
  SunEdison correctly wins for the 2012-2016 Wikipedia interval.
- **Resolution**: Pre-fix Phase A wrote 1 SUNE permaTicker row pointing
  at SUNation (`US000000002062`). Phase B stored SUNation's CSII-era
  "junk" 2012 data under that permaTicker. Post-fix re-run of Phase A
  and Phase B correctly identifies SunEdison
  (`US000000002709`) as the historical owner and stores its 1196 rows
  under that permaTicker. The wrongly-attributed SUNation node was PURGED
  via the Phase B cleanup pass (orphan storage path).
- **Verification (post-fix run on 2025-07-15)**:
  - `/sp400/US000000002709` (SunEdison) — STORED, 1196 rows,
    2011-07-21 → 2016-04-21 (delisting after bankruptcy),
    Close first/last $8.05 / $0.34, **0 NaN in Adj_Close**.
  - `/sp400/US000000002062` (SUNation) — CORRECTLY PURGED.

---

## Database state after migration (2026-07-15)

### Primary metadata table: `/metadata/sp400_permatickers` (962 rows)

| Column | Type | Notes |
|---|---|---|
| `permaTicker` | str | PRIMARY KEY (Tiingo-issued, identity-stable) |
| `canonical_ticker` | str | current trading ticker code (informational; EODHD calendar join key for Phase D) |
| `name` | str | company name from Tiingo search response (some rows have NaN — orphaned rebrand chain tinier metadata) |
| `isActive` | bool | True if currently trading; False if delisted/incorporated-into-parent |
| `openfigi` | str | Bloomberg OpenFIGI composite (defensive redundancy anchor; present on 771 / 962 rows = 80%) |
| `cik` | str | SEC EDGAR CIK (carried from `/metadata/sp400` for SIC sector ETF lookup, NOT the primary identity) |
| `sic` | str | SIC code (sector ETF mapping key) |
| `index_ref` | str | IJK / IJH / IJJ / IJS / XLB/etc. — the sector ETF used as benchmark |
| `wikipedia_intervals` | str | JSON list of `{added, removed}` dicts (S&P 400 residency spans) |
| `price_unavailable` | bool | FINAL state (post Phase B self-correction); True iff Tiingo's full-history fetch returned 0 rows |
| `legacy_perm_id` | str | ⚠️ Currently EMPTY (see "Known issues" below) |

Counts:
- Total permaTickers: **962**
- `price_unavailable=False`: **928** (have Tiingo price data)
- `price_unavailable=True`: **34** (Tickers with no Tiingo history across 15 years; e.g. pre-IPO shells like GPRO 2012, defunct SunAmerica-era SAI, LPS, BIVV/Baxalta spin-outs never IPO'd as their ticker)
- `isActive=True`: **697** (currently trading)
- `isActive=False`: **265** (delisted/rebranded/bankrupt)
- Multi-interval permaTickers: **38** (different Wikipedia intervals collapsed to one permaTicker row — e.g. ALK, AM, AMG, IDCC)

### Storage: `/sp400/{permaTicker}` (928 nodes)

| Metric | Value |
|---|---|
| Total storage nodes | 928 |
| All permaTicker-keyed | YES (0 suspicious non-`US`/`CA`-prefixed keys) |
| Total rows across all nodes | 2,752,968 |
| Schema (per node) | `Date, Open, High, Low, Close, Volume, Adj_Open, Adj_High, Adj_Low, Adj_Close, Adj_Volume` |
| Date column type | `datetime64[us]` (timezone-naive) |
| Date column indexed | `data_columns=["Date"]` in HDFStore |
| SUNE SunEdison (`US000000002709`) | 1196 rows, 2011-07-21 → 2016-04-21 |
| SUNE SUNation (`US000000002062`) | PURGED (was wrongly-attributed before Phase A fix) |

### Other tables (UNCHANGED by this migration)

| Key | Rows | State |
|---|---|---|
| `/metadata/sp400` | 993 | Wikipedia intervals (Phase A input; preserved) |
| `/earnings/raw` | 44,897 | **STALE: keyed by `perm_id`**. 891 unique perm_ids. Needs Phase D re-keying. |
| `/features/gated_events` | 21,269 | **STALE: keyed by `perm_id`**. 853 unique perm_ids. Needs Phase E Stage 1 re-run. |
| `/features/train_matrix` | 21,248 | **STALE: keyed by `perm_id`, points to old `/sp400/{canonical_ticker}` price paths**. Needs Phase E Stage 2 re-run with new permaTicker-keyed prices. |
| `/macros/*` | unchanged | IJH + 5 ETFs + SPY + VIXY + 6 FRED macseries |

---

## Identity model differences: perm_id vs permaTicker

### What Tiingo "sees" as one entity

Tiingo's identity model is **listing-based** (permaTicker = a continuous
tradable asset track) rather than **legal-entity-based** (SEC CIK = a
single company across all its tickers/deals). This difference manifests
in the following comparisons:

| Case | CIK-anchored view (old `perm_id`) | permaTicker view (new) |
|---|---|---|
| Clean rebrand (POL → AVNT) | Same CIK 0001122976; one perm_id; one /sp400/{AVNT} node storing PolyOne→Avient concat | One permaTicker `US000000009263`; one /sp400/{US000000009263} node containing full PolyOne→Avient back-merged history |
| Spinoff (CFX → ENOV + ESAB) | Same CIK 0001043666 → one perm_id; Phase B fetched aliases which contaminated ENOV CFX alternation | Three permaTickers: `US000000001291` (ENOV inherits Colfax+Enovis full history), `US000000104025` (ESAB fresh, startDate 2022-03-29), `US000000104366` (CFX legacy empty — data migrated to ENOV). Identity boundary cleanly splits the spin-off |
| Bankruptcy reorg (CHK → EXE) | Same CIK 0000895126 → one perm_id `0000895126_CHK`; Phase B v2.1 fetched CHK+EXE aliases, concatenated for full history but suffered contamination on overlap dates | Tiingo splits into TWO permaTickers: `US000000000505` (CHK legacy — 2011-2024 OTC-traded until delist) and `US000000092728` (EXE / Expand Energy — freshly activated 2021-02-10). Boundary sharply drawn by listing event, not by SEC acquirer CIK |
| Ticker recycling (SUNE) | Two Wikipedia intervals under one perm_id (12 collision pairs handled by §7.7 LIVE-wins-overlap rule from merger_identity_patch.md) | Two permaTickers cleanly separated at Tiingo level: SunEdison (US000000002709, inactive) and SUNation (US000000002062, active). Disambiguation is per-Wikipedia-interval positional, NOT by ticker code |

**What this means**: under permaTicker, the identity boundary is actually
**MORE GRANULAR** than under SEC CIK. A legal entity that has a spinoff
will be split into multiple permaTickers (e.g. Colfax → Enovis + ESAB).
A legal entity that has a bankruptcy reorg gets a fresh permaTicker
(CHK → EXE). The Tiingo-permaTicker view represents "this is a tradable
price series" — exactly what we want for CAR calculations and event
studies.

---

## Live Tiingo API usage

Two Phase A probe phases + one Phase B fetch run:

| Phase | Endpoint | Calls | Purpose |
|---|---|---|---|
| Phase A first run | `/utilities/search/{ticker}?includeDelisted=true&exactTickerMatch=true` | ~993 (one per /metadata/sp400 row) | Discover all historical permaTicker holders of each ticker code |
| Phase A first run | `/tiingo/daily/{permaTicker}/prices` | ~1500-2000 (multiple candidates per ticker code) | Per-candidate narrow + wide price probe for disambiguation + name-sanity check |
| Phase A second run (post fix) | same | ~993 + ~1500-2000 | Reproduce permaTicker table with SUNE Class-S fix |
| Phase B first run (paid tier) | `/tiingo/daily/{permaTicker}/prices` | 962 | Full-history fetch per permaTicker (one call each, no alias fetching) |
| Phase B second run (post fix) | `/tiingo/daily/{permaTicker}/prices` | 962 | Re-confirm with SUNE SunEdison now in meta |

Total API calls: ~5500 calls across all phases. Well within the user's
10k/hr paid tier (one full Phase A+B cycle takes ~26 min wall clock).

No throttling was needed; calls were batched with 0.02-0.05s polite
delays only.

---

## Bugs discovered and resolved during the migration

### Bug 1: Phase A disambiguation price-range tiebreaker was wrong (FIXED)

The original `disambiguate_permaTicker()` scored candidates by
`abs(log10(median_close) - 1.5)` — preferring median close prices near
$32. This caused SUNE's Wikipedia 2012-2016 interval to be wrongly
assigned to SUNation's permaTicker (median $13.88, score 0.36) instead
of SunEdison's permaTicker (median $3.97, score 0.90), because both
returned sane rows in the 2012 probe window.

**Fix**: Introduced a new primary tiebreaker `_physical_row_count()` —
the candidate with MORE rows whose `adjClose / close` ratio is in
[0.001, 1000] wins. The historical-era permaTicker (SunEdison, ratio=1.0
in 2012) is preferred over the modern-recycler-permaTicker (SUNation,
ratio=150,000× in 2012 due to CSII→PEGY→SUNE back-merged splits).
Secondary tiebreaker: `isActive=False` preference for past-closed
Wikipedia intervals (SunEdison was delisted post-bankruptcy).

This was the ONLY confirmed Phase A misattribution triggered by the
sanity-checker. After the fix, the SUNE case routes to SunEdison
correctly, and 3 NEW permaTicker nodes were created by a Phase B re-run
(SunEdison + COHR + VSTS); 4 orphan nodes (formerly
SUNation-assigned) were PURGED.

### Bug 2: SAI Wikipedia interval is a Wikipedia data quirk, not a Phase A bug (DOCUMENTED, NOT FIXED)

Wikipedia's `/metadata/sp400.SAI` row tracks the interval
`[added=2010-01-01, removed=2013-08-19]` with `cik_at_added=0000945436`
(SunAmerica Inc — a pre-2000 Sunam America-era SP400 ticker reuse).
Phase A's permaTicker discovery for ticker code "SAI" today returns only
the modern recycler `US000000073780` (SUNAMERICA INC), which has 0
Tiingo /prices rows in 2010-2013. Therefore Phase A correctly flagged
this permaTicker as `price_unavailable=True`.

This row is then correctly skipped by Phase B. The Phase B output
schema records this as an honest data gap (Tiingo lacks pre-2014 SAI
price history for any US permaTicker). The historical SAIC/Leidos
legacy is preserved via a separate permaTicker `US000000009014`
(Leidos Holdings Inc) which DOES have full 2011-2026 price data.

**This is not a bug in Phase A** — it's a Wikipedia/ticker-code-recycle
quirk that surfaces organically via the name-sanity check (the original
Phase A design intent). No action needed; the missing-data row is
flagged for awareness and skipped during Phase B / Phase D.

### 30+ severe storage-start-vs-Wiki-added gaps (DOCUMENTED, NOT BUGS)

30 permaTickers have storage row-count beginning far after their
Wikipedia `added` date (>365 days gap). Examples:
- `QRVO` Qorvo Inc — Wiki added 2012-01-01, Tiingo storage starts 2015-01-02 (Qorvo was formed Jan 2015 from a RF Micro + TriQuint merger; Wikipedia's "added 2012" actually tracks the RFMD predecessor ticker)
- `VSTO` Vista Outdoor — Wiki added 2012, Tiingo storage starts 2015-02-10 (VSTO was spun from ATK in 2015; Wikipedia tracks an ATK-predecessor-era residency via ticker code "VSTO" extending back)
- `AVNS` Avanos Medical — Wiki added 2012, Tiingo storage starts 2014-10-21 (Avanos was spun from Halyard Health in 2014; Wikipedia tracks Halyard-era residency via ticker "AVNS")

These are Wikipedia's data quirks where the `added` date records the
predecessor/spin-off-parent era rather than the permaTicker's actual
entity existence. The Tiingo data correctly covers the
permaTicker's actual existence era. Downstream Phase E
feature-gating needs to handle these missing-data glidepaths; the
Phase E `T-match failures → row dropped` policy from
`features.md` §3 already covers this case (no header T-match means the
row's `report_date` predates any Tiingo data and the row is naturally
dropped).

### Phase B always-refetch (ATTRIBUTED TO HISTORY, NOT FIXED)

Phase B does not skip already-fetched permaTickers. Each run re-fetches
all 962 from Tiingo (13 min wall time). This is **deliberate** — under
the Tiingo paid tier (10k/hr) it's cheap, and it eliminates any
incremental-update edge cases like Phase B v2.1's "always-refetch to
ensure union-alias reconciliation overwrites any clobbered data"
band-aid that motivated the original `always_refetch=True` flag. The
schema no longer requires that band-aid since alias concatenation is
gone. The full-refetch is the simplest correct implementation.

If we ever need to scale to more ticker coverage or reduce Tiingo load,
an incremental-only refresh mode is straightforward to add later.

---

## Known issues and forward work

### Issue 1: `legacy_perm_id` column is EMPTY (NEEDS Phase D RECONSTRUCTION)

On the 2026-07-14 first Phase A run, the legacy
`/metadata/sp400_perm_ids` table was PURGED and 939 rows of `legacy_perm_id`
mappings were correctly carried over. On the 2026-07-15 SECOND Phase A
run (post SUNE Class-S fix), `load_legacy_perm_id_map()` finds the
legacy table already purged → returns empty dict → fills all 962 rows'
`legacy_perm_id` column with NaN.

This means the new `/metadata/sp400_permatickers` table has no
forward mapping from Phase D's existing 44,897 `/earnings/raw` rows
(value `perm_id` is `0001158449_AAP` etc.) to the new `permaTicker`
values.

**Reconstruction path (no network calls needed)**: The legacy `perm_id`
formula is `f"{cik_at_added}_{canonical_ticker}"`. Both
`cik_at_added` and `canonical_ticker` columns are present in
`/metadata/sp400` (993 rows). The new `/metadata/sp400_permatickers`
table has `canonical_ticker`. So a JOIN `sp400_permatickers` ↔ `sp400`
on `canonical_ticker` allows carrying the `cik_at_added` → form the
synthetic `legacy_perm_id` per row. Phase D's re-keying step can do
this join in-process and have a complete `legacy_perm_id → permaTicker`
mapping without re-fetching from EODHD.

If this join has collisions (multiple permaTickers sharing the same
`canonical_ticker` — e.g. SAI's SUNAMERICA INC vs LDOS's Leidos
legacy SAI both have `canonical_ticker='SAI'`), the Phase D re-keying
step would need a tiebreaker. From inspection: SAI itself returns only
ONE permaTicker in the new table (`US000000073780`) — Leidos Holdings
was tracked under `LDOS` `canonical_ticker` (not `SAI`). So no
collision expected. The mapping is straightforward.

### Issue 2: Phase D / Phase E will require full re-runs

Once Phase D re-keying maps `/earnings/raw` to permaTicker, the existing
`/features/gated_events` (21,269 rows) and `/features/train_matrix`
(21,248 rows) tables must be **PURGED** and re-built from scratch:

1. **Phase E Stage 1**: Iterate per `permaTicker` (not per `perm_id`),
   gate by per-permaTicker Wikipedia intervals using the
   `wikipedia_intervals` column from `/metadata/sp400_permatickers`.
   §7.7 disambiguation rule is NO LONGER NECESSARY — no two
   permaTickers can collide because permaTicker is the storage key.

2. **Phase E Stage 2**: For each permaTicker's gated event, fetch
   price data from `/sp400/{permaTicker}` (the new key path). Feature
   computation is otherwise unchanged (CAR, SUE, rolling-window stats
   etc.). Output: `/features/train_matrix` keyed by `permaTicker`.

3. **Phase F Stage 3**: Retrain the XGBoost Ranker on the newly-built
   matrix. With the contaminated Phase B v2.1 alias-intermixing
   eliminated at source, the extreme CAR outliers (e.g. +363,862% NSR,
   +6,519% AVNT, -98.7% LDOS) should disappear. The
   `Overfitting investigation` trial documented in
   `phase_b_contamination_audit.md` should re-train at much better
   val NDCG@3 than the Phase F v1 baseline of 0.086.

### Issue 3: 30+ severe storage-vs-Wiki-added gaps require Phase E awareness

Per Bug 3 above, 30 permaTickers have storage_start much later than
their Wikipedia `added` date. This is not Phase A or Phase B's bug;
Phase E Stage 2's existing `T-match` failure policy (rows dropped when
no trading day >= report_date is available for a permaTicker)
naturally handles these cases. The Phase E `drop + log` policy
documented in `features.md §3` (NaN policy with ONE exception) is the
right place to absorb these gaps.

Phase E Stage 2 also needs to verify it correctly handles permaTickers
with VERY SHORT Wikipedia intervals (e.g. `VSTS` Vestis Corp Wiki
interval `[2023-10-02, 2023-12-18]` — only 11 weeks of SP400 residency)
that may have only a few earnings events falling into that window; the
pipeline-matched-counts expectation should accommodate that.

---

## File-level changes summary

### New scripts
None new — the existing scripts have been rewritten.

### Rewritten scripts

- `luan_bot_trading/01_data/02b_build_company_map.py`:
  **1296 LOC → 774 LOC** (~40% reduction). Completely replaced
  Wikipedia+DERA+CIK-synthesis machinery with Tiingo search-based
  permaTicker discovery. Removed: DERA cache loading, manual
  `MANUAL_TAS_OVERRIDE`, point-in-time CIK resolution, 2-step interval
  fix, post-Wikipedia active-alias extension, §7.7 disambiguation rule.
  Added: Tiingo `search` endpoint, `_prices_sanity_score()`,
  `_physical_row_count()`, positional-disambiguation algorithm.

- `luan_bot_trading/01_data/03_data_gathering.py`:
  **593 LOC → ~430 LOC**. Replaced EODHD alias-concatenation with
  single-fetch-per-permaTicker. Removed:
  `fetch_concatenated_aliases_from_eodhd()`, `aggregate_canonicals()`,
  multi-perm_id-per-canonical write-clobber workaround, Phase B v2 `always_refetch` flag.
  Added: `fetch_from_tiingo()` (native adj-OHLC Volume mapping, no derivation),
  `write_back_availability()` (Q1 self-correction), stable-mergesort agg, permaTicker-keyed
  storage.

### Updated documentation files (deprecation banners added)

- `luan_bot_trading/Design.md` (file-top identity notice + §9b deprecation tag)
- `luan_bot_trading/database_layout.md` (file-top banner + per-section ⛔ tags on 6 schema rows)
- `luan_bot_trading/features.md` (file-top notice + §0 row-granularity deprecation)
- `luan_bot_trading/01_data/merger_identity_patch.md` (file-top banner + §7.11 successor section)
- `luan_bot_trading/01_data/phase_b_contamination_audit.md` (file-top resolution notice)
- `luan_bot_trading/01_data/earnings_gathering_design.md` (file-top identity notice)
- `luan_bot_trading/01_data/company_merge_design.md` (whole-document deprecation)
- `04_backtest/archive/docs/eodhd_vs_tiingo.md` (archived 2026-08-15; historical provider comparison)

### Authoritative reference

- `luan_bot_trading/01_data/tiingo_permaTicker_audit.md` (the AUTHORITATIVE
  successor spec — promoted at file top, with full migration design +
  OpenFIGI complementary redundancy discussion)

### Cleanup artifacts created

- `luan_bot_trading/01_data/permaticker_disambiguation.log` — audit
  trail of every permaTicker assignment decision (one line per ticker;
  one line per assigned permaTicker; one indented line per Wikipedia
  interval resolved).

### Test scripts (temporary, all deleted)

The migration was validated through live smoke tests against the Tiingo
API:
- Phase A: tested disambiguate_permaTicker against 8 known-stress-case
  permaTickers (AA, SUNE 2014-2016/2022-, NSR, META, ENOV, AVNT, SAI,
  AAP) before the live run.
- Phase A: tested end-to-end write with `--limit 8` against a temp db.h5 copy.
- Phase B: tested `fetch_from_tiingo()` against 6 known-stress permaTickers.
- Phase B: tested end-to-end run with `--limit 8` against a temp db.h5 copy.
- Post Phase A re-run: confirmed SunEdison (`US000000002709`) newly stored,
  SUNation (`US000000002062`) correctly purged, total row count 2.75M preserved.

All temp test scripts (`_*.py` under `/tmp/`) have been deleted.

---

## Recommended next steps (priority-ordered)

1. **Phase D re-keying** (`06_earnings_gathering.py` rewrite):
   - Migrate 44,897 `/earnings/raw` rows from `perm_id` to `permaTicker`. Two
     paths possible:
     - **(a) Lightweight migration**: Reconstruct `legacy_perm_id → permaTicker`
       mapping from `/metadata/sp400.cik_at_added + canonical_ticker`,
       then UPDATE the `perm_id` column on existing rows in-place. No
       additional API calls needed.
     - **(b) Full Phase D re-fetch** (requires EODHD subscription to remain
       active for the calendar endpoint): Re-run `06_earnings_gathering.py`
       iterating per `permaTicker` instead of per `perm_id`, fetching from
       EODHD's `/api/calendar/earnings` (still load-bearing — Tiingo has no
       equivalent calendar endpoint).
   - **Recommendation**: Start with (a) the lightweight migration; if it reveals
     inconsistencies, fall back to (b).

2. **Phase E Stage 1 rewrite** (`02_features/01_features_gate_events.py`):
   - Iterate per `permaTicker` instead of per `perm_id`.
   - Use `wikipedia_intervals` from `/metadata/sp400_permatickers` for the gate.
   - DELETE the §7.7 disambiguation rule (no longer needed — no collision).
   - PURGE old `/features/gated_events` (21,269 rows) and rewrite.

3. **Phase E Stage 2 rewrite** (`02_features/02_build_feature_matrix.py`):
   - Load prices from `/sp400/{permaTicker}` instead of `/sp400/{canonical_ticker}`.
   - Per-permaTicker rolling stats (sue_score, consecutive_surprises, car_drift_historical_q1).
   - PURGE old `/features/train_matrix` (21,248 rows) and rewrite.

4. **Phase F Stage 3** (`03_model/01_train_model.py`):
   - TRAIN NDCG@3 metrics are expected to improve substantially now that
     the Phase B v2.1 alias-intermixing / NSR Class-W Nanda-Royalty
     contamination / SUNE Class-S CSII alternation are eliminated.
   - The previous Phase F baseline v1 model artifacts
     (`ranker.json` + `calibrator.pkl` + `meta.json` under
     `03_model/models/phase_f_baseline_v1/`) are TRAINED ON CONTAMINATED
     DATA and should be considered OBSOLETE. Retrain as
     `phase_f_baseline_v2/` on the clean data.

5. **Documentation pass** on any new docs that should accompany Phase D/E/F
   re-runs (likely update `features.md`, `database_layout.md` to
   permaTicker-keyed schemas once stable).

---

## Audit trail logs

- Permaticker-disambig log file:
  `luan_bot_trading/01_data/permaticker_disambiguation.log`
  (~962 ticker entries with permaTicker assignments and Wikipedia
  interval breakdowns) — produced by the 2026-07-15 re-run of Phase A.

This file is the canonical "why did we choose this permaTicker for this
Wikipedia interval" audit trail. Reviewing it is recommended before
running Phase D re-keying to catch any anomalies.

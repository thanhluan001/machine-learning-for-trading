# Quantitative Architecture Specification: Identity Patch

> **⛔ DEPRECATED (2026-07-14):** The entire `perm_id` identity layer
> specified in this document is **OBSOLETE**. The pipeline is migrating to
> **Tiingo's `permaTicker`** as the primary entity identifier, which is
> identity-stable across rebrands, mergers, delistings, and spin-offs
> (empirically verified across all bug classes from the Phase B audit:
> Class U/V/W/S are all natively resolved at source by permaTicker).
> 
> See [`01_data/tiingo_permaTicker_audit.md`](tiingo_permaTicker_audit.md) for
> the live probe-based evidence and **the new identity design** that replaces:
> - The synthetic `perm_id = f"{cik}_{start_ticker}"` derivation (§2, §3.1 below)
> - The Wikipedia-interval + DERA + CIK-synthesis Phase A `02b_build_company_map.py` (§7.7 disambiguation, §7.3 active-alias extension, §7.4 canonical selection)
> - The Phase B alias-concatenation bug workaround (§7.7 v2/v2.1 fixes)
> - The Phase D `(perm_id, fiscal_period_end)` dedup key (§7.7 implication)
> - The Phase E §7.9 disambiguation-at-gate rule (no longer needed once
>   identity is `permaTicker`-keyed)
> 
> This document is **retained only as historical audit** of why `perm_id`
> was introduced (the survivor-CIK collision bug of §1 is real, and
> `permaTicker` solves it correctly — but the underlying bug analysis
> stays useful context). All forward work MUST reference the audit doc
> above instead of the `perm_id` machinery defined below.

## Target Component: Asset Identification and Merger Processing Architecture
* **Status:** Operational Patch Implementation (July 2026) -- ⛔ see deprecation banner above.
* **Impacted Code Modules:** `02b_build_company_map.py`, `03_data_gathering.py`, `04_feature_building.py`

---

## 1. Problem Statement: The Survivor-CIK Collision Bug
In the previous implementation version, the pipeline utilized the SEC Central Index Key (`cik`) as the unique canonical anchor for an individual company across the 15-year lookback window[cite: 1]. While a CIK represents a single legal corporate entity over time, it creates a point-in-time backtesting failure when two active S&P 400 constituents merge[cite: 1].

When Company A acquires Company B:
1. The SEC retroactively maps historical filings or target entity snapshots to the surviving parent company’s CIK in consolidated data stores[cite: 1].
2. In the old design, the pipeline compressed overlapping index intervals under the same CIK, assuming they belonged to the same company[cite: 1].
3. This resulted in the **accidental deletion of Company B's independent pre-merger history**, wiping out its independent earnings events, pricing footprints, and valid cross-sectional alpha rows[cite: 1].

---

## 2. Solution: The Composite Asset ID (`perm_id`)
To decouple legal corporate entity tracking from tradable historical asset lifecycles, the pipeline introduces a unique **Composite Asset ID** (`perm_id`). 

$$\text{perm\_id} = \text{CIK} + \text{"\_"} + \text{First\_Observed\_Ticker}$$

Every distinct constituent asset entering the S&P 400 universe receives an immutable `perm_id` string token. If two components share the same CIK post-merger, they remain structurally separated in the database because their tracking anchors reflect their unique historical entry ticker origins.

---

## 3. Data Structure Restructuring (`db.h5`)

The HDF5 data schemas are updated to shift the primary lookup key from `cik` to the composite asset tracking layer.

### 3.1 Metadata Table Schema: `/metadata/sp400_companies`
Instead of exactly one row per CIK, this table now contains **one row per unique `perm_id`**[cite: 1].

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `perm_id` | `string` (Key) | **Primary Key Anchor:** `f"{cik}_{first_ticker}"` |
| `cik` | `string` | SEC Central Index Key (used for fundamental mapping)[cite: 1]. |
| `canonical_ticker`| `string` | The ultimate current or final trading ticker alias[cite: 1]. |
| `aliases` | `JSON string` | List of all historical ticker tokens used *by this specific asset track*[cite: 1]. |
| `combined_intervals`| `JSON string` | Non-overlapping index residency spans for *this asset track only*[cite: 1]. |

### 3.2 Feature Matrix Output Schema: `/features/...`
The metadata row tracking definitions within the generated training data matrix must include both tracking levels[cite: 2]:
* `perm_id` (str) — Primary tracking entity anchor.
* `canonical_ticker` (str) — Price series mapping anchor[cite: 2].
* `cik` (str) — Fundamental parsing anchor[cite: 2].

---

## 4. Pipeline Execution Rules

### Step 1: Mapping Isolation (`02b_build_company_map.py`)
When parsing the historical S&P 400 constituency modifications timeline, the mapping engine evaluates intervals sequentially. 
* If a CIK entry appears with an index membership window that overlaps an existing active record for the same CIK, **do not merge the intervals**[cite: 1].
* Fork the record instantly by initializing a new `perm_id` using the current reporting ticker name.
* Allocate the specific residency intervals to that asset's history independently.

### Step 2: Earnings Timeline Separation (`04_feature_building.py`)
When flattening out earnings rows from `/earnings/raw` into the training matrix, events are mapped by matching the `cik` **and** validating that the historical `report_date` falls cleanly within the specific `perm_id`'s `combined_intervals` residency window[cite: 1, 2]. 

* Pre-merger earnings from Company B map to `perm_id_B` and leverage its historical ticker price series.
* Pre-merger earnings from Company A map to `perm_id_A`.
* Post-merger earnings flow exclusively to the surviving parent `perm_id_A`.

---

## 5. Reference Implementation: Interval Forking Layer

The tracking dictionary maps incoming constituents by evaluating timeline overlaps programmatically:

```python
import pandas as pd

def build_perm_id_map(raw_constituents_log):
    """
    Processes historical constituent changes to construct a point-in-time 
    compliant asset tracking map, splitting overlapping corporate mergers.
    """
    perm_id_table = {}
    
    for record in raw_constituents_log:
        cik = record['cik']
        ticker = record['ticker']
        new_interval = {
            'added': pd.to_datetime(record['added']),
            'removed': pd.to_datetime(record['removed']) if record['removed'] != "None" else pd.Timestamp.now()
        }
        
        # Search for identity collisions within the same corporate CIK
        existing_tracks = {k: v for k, v in perm_id_table.items() if v['cik'] == cik}
        
        assigned = False
        for pid, track in existing_tracks.items():
            # Check for temporal overlap against current recorded spans
            has_overlap = False
            for active_int in track['intervals']:
                if not (new_interval['removed'] < active_int['added'] or new_interval['added'] > active_int['removed']):
                    has_overlap = True
                    break
            
            # If no temporal overlap exists, this is a clean re-entry of the same company
            if not has_overlap:
                track['intervals'].append(new_interval)
                track['aliases'].append(ticker)
                assigned = True
                break
                
        if not assigned:
            # Identity collision or new asset detected: instantiate unique perm_id tracking space
            new_pid = f"{cik}_{ticker}"
            perm_id_table[new_pid] = {
                'perm_id': new_pid,
                'cik': cik,
                'first_observed_ticker': ticker,
                'aliases': [ticker],
                'intervals': [new_interval]
            }
            
    return perm_id_table

---

## §7. Phase A — Production Implementation Notes (2026-07-13)

Phase A rewrote `02b_build_company_map.py` per this spec. Key design
refinements and deviations from §4-5 above, audited empirically against the
full SP400 universe (993 Wikipedia ticker rows -> 1037 interval entries):

### 7.1 Point-in-time CIK lookup (replaces the patch's "present-day cik")
The patch's §4 pseudocode takes `record['cik']` as a fixed present-day field.
That was precisely the bug source: present-day SEC `company_tickers_exchange.json`
retroactively maps a target's pre-merger CIK into the surviving acquirer's
CIK. Phase A replaces this with **point-in-time lookup**:
For each Wikipedia (ticker, added_year) interval entry, we resolve the CIK
*at that added_year* via DERA `sub_{YYYY}.txt` Q4 snapshots using a
year-walkback/forward within the range [2010, 2025]. Manual overrides
(MANUAL_TAS_OVERRIDE dict) handle stale DERA `instance` columns (e.g. RBC
2023-2025 -> 1324948 RBC Bearings, since DERA's `instance` column is stuck
on "RBC" from Regal Beloit). Active SEC and cached ticker.txt are fallbacks.

Result: 1033 / 1037 Wiki entries resolve to a CIK; 4 `__nocik_*` perm_ids.

### 7.2 Fork decision rule — replaced "overlap -> fork" with "different CIK -> fork"

The patch's §4 rule ("same CIK + temporal overlap -> fork") fails for the
empirically common rebrand case where both ticker aliases are live in the
SP400 simultaneously during the rebrand day (Wikipedia reports open-ended
intervals for both aliases that overlap on the transition day). Those cases
(GAP+GPS, FHI+FII, AVNS+HYH, EHC+HLS, CHX+APY, CHRD+OAS, FBHS+FBIN) ARE
the SAME legal entity but were incorrectly forked under the patch's rule.

Phase A's refined rule: **same point-in-time CIK ALWAYS merges into one
perm_id regardless of interval overlap**; **different point-in-time CIK
SEPARATES into different perm_ids**. The point-in-time lookup upstream
prevents the survivor-CIK collision bug from ever re-entering the merge
step, making the overlap check redundant.

Result: 0 CIKs shared across multiple perm_ids (survivor-CIK collision bug
eliminated). 31 multi-alias perm_ids (24 same-CIK rebrand merges + 7
post-Wikipedia active-alias extensions — see §7.3).

### 7.3 Post-Wikipedia active-alias extension (acquirer-rebrand recovery)

For perm_ids whose CIK is currently active in SEC `company_tickers_exchange.json`
under a ticker symbol NOT present in the perm_id's Wiki-tracked aliases,
Phase A adds that active alias so Phase B can fetch post-rebrand prices.
This recovers the **acquirer-rebrand** pattern (acquirer keeps its CIK,
target's ticker symbol is retired by Wikipedia post-delisting, survivor
adopts the ticker):

| perm_id | aliases added | start | reason |
|---|---|---|---|
| `0001360604_HTA` | HR | 2022-07-21 | Healthcare Realty + Healthcare Trust of America merger (surviving CIK 1360604) |
| `0001069183_AAXN` | AXON | 2023-05-04 | TASER / Axon Enterprises rebrand |
| `0000820318_IIVI` | COHR | 2026-03-23 | II-VI -> Coherent Corp post-merger |
| `0000933974_BRKS` | AZTA | 2024-11-25 | Brooks Automation -> Azenta rebrand |
| `0001590895_ERI` | CZR | 2021-03-22 | Eldorado Resorts -> Caesars post-merger |
| `0000895126_CHK` | EXE | 2025-03-24 | Chesapeake Energy -> Expand Energy rebrand |
| `0001336920_SAI` | LDOS | 2019-08-09 | SAIC -> Leidos (walkback-contamination case) |
| `0000837465_MODG` | CALY | 2012-01-01 | Topgolf Callaway rebrand |
| `0000750004_LNW` | LAWIL | 2012-01-01 | Light & Wonder / Scientific Games rebrand |
| `0000730464_ATGE` | CVSA | 2012-01-01 | Adtalem / DeVry CIK + Covista cik collision |

Safety guards (all three must hold to ADD an active alias):
  (a) perm_id's combined_intervals' last `removed` is None (perm_id still
      in SP400 per Wikipedia).
  (b) NO OTHER perm_id exists with the same start_ticker AND its
      combined_intervals' last `removed` is None (i.e. the active alias's
      ticker is not still-owned by another live perm_id).
  (c) If a conflicting CLOSED perm_id exists with the active alias as its
      start_ticker, its CIK must DIFFER from ours (acquirer-rebrand case)
      and its removed date >= our added date (the acquirer adopted the
      ticker post-target-delisting). If conflicting perm_id has SAME-CIK
      as ours, defensive skip.

### 7.4 Canonical-ticker selection — strict CIK match (Covista bug)

The patch's rule ("aliases that are active-in-SEC win") accidentally
picked a different company's ticker for our CIK on the ATGE / Covista
case: SEC's cache had CIK 730464 -> "CVSA" (Covista Inc.) mistakenly
pinned to our Adtalem perm_id (CIK 730464's actual historical filings are
DeVry / Adtalem, per DERA). The new canonical-selection rule:

  1. Filter aliases to those whose active-SEC CIK == our perm_id's CIK.
     Pick among them by latest-added-date tiebreak.
  2. If none match (perm_id's CIK has no alias active in SEC), fall back to
     alias with the latest latest-added date across all aliases.

This results in exactly 30 / 31 multi-alias perm_ids having a canonical
whose active-SEC CIK matches our perm_id's CIK. One remaining "mismatch"
(`0001610950_SYNH`) is a Wikipedia edge case (Syneos went private 2023,
INCR ticker was reassigned post-deprecation); left as acceptable Phase A v1
limitation since Phase B's EODHD probe will resolve it.

### 7.5 Output schema (final implementation)

Replacement: `/metadata/sp400_perm_ids` (DELETE `/metadata/sp400_companies`).
10 cols, see `luan_bot_trading/database_layout.md` for the schema table.

### 7.7 Known Phase A v1 limitation: 12 canonical-ticker collision pairs (post-run finding)

Phase B's post-run audit surfaced 12 perm_id pairs whose canonical_ticker
COLLIDES (same ticker symbol picked canonical for two distinct perm_ids).
These are an unavoidable side-effect of the post-Wiki acquirer-rebrand
extension in §7.3 -- the extension adds the surviving ticker to the
acquirer perm_id's aliases, but Phase A's strict-CIK-match canonical
selection (§7.4) can then pick that same ticker canonical for both the
acquirer (with the post-Wiki-extended aliases list) AND the historical
predecessor perm_id (whose own Wikipedia row was originally under that
ticker). 8 of the 12 pairs have OVERLAPPING combined_intervals, so
Phase E cannot uniquely attribute /sp400/{canon} rows in the overlap
zone purely by date.

The empirically observed structure makes this tractable: in 11 of 12
pairs, exactly one perm_id is LIVE (last combined_intervals' `removed` is
`None`) and the other is CLOSED. The rows in the stored /sp400/{canon}
series correspond to whichever company is CURRENTLY traded under that
EODHD symbol -- which is the LIVE perm_id's modern corporate lineage.
The CLOSED perm_id's price history in its Wikipedia-overlap period is
NOT separately available on EODHD (the older phantom ticker-of-the-same-
symbol company has no EODHD series). One pair (CC) has both perm_ids
CLOSED; disambiguate there by later `removed` date (ticker-had-it-last).

Pairs (12 total):

| canon | LIVE perm_id | CLOSED perm_id |
|---|---|---|
| ACI  | 0001646972_ACI | 0001037676_ACI |
| AM   | 0001598968_AM  | 0000005133_AM  |
| AXON | 0001069183_AAXN (rebranded to AXON) | 0001636050_AXON (phantom 2012-2023 XMGS) |
| AZTA | 0000933974_BRKS (rebranded to AZTA) | 0001518749_AZTA |
| CC   | (both CLOSED -- disambiguate by later `removed`= 2025-03-24 → 0001627223_CC) | 0001577095_CC |
| COHR | 0000820318_IIVI (rebranded to COHR) | 0000021510_COHR |
| CZR  | 0001590895_ERI (rebranded to CZR) | 0000858339_CZR |
| EXE  | 0000895126_CHK (rebranded to EXE)  | 0001075736_EXE |
| HR   | 0001360604_HTA (rebranded to HR)   | 0000899749_HR  |
| JEF  | 0000096223_JEF  | 0001084580_JEF  |
| LDOS | 0001336920_SAI (rebranded to LDOS) | 0000353394_LDOS |
| VAL  | 0000314808_ESV (rebranded to VAL) | 0000102741_VAL |

**Phase E disambiguation rule** (LOCKED):
  When two perm_ids share `canonical_ticker` AND their `combined_intervals`
  overlap, rows in the overlap zone belong ONLY to whichever perm_id has the
  LATER effective "end date" (where `null` removed = +infinity). Concretely,
  for the LIVE vs CLOSED cases, ALL stored rows in the overlap zone belong to
  the LIVE perm_id; the CLOSED perm_id's CAR for events with `report_date` in
  the overlap zone is NaN-dropped (logged). For both-CLOSED cases (CC),
  disambiguate by `removed` date -- the one with later `removed` wins.

**Phase D implication** (subjects: `06_earnings_gathering.py`): the dedup
rule by `(canonical_ticker, report_date)` is WRONG for these 12 pairs --
it could drop a LIVE perm_id's earnings row if the CLOSED perm_id also had
an event on the same `report_date` (unlikely but possible). Phase D must
key earnings by `perm_id` (not `canonical_ticker`) at write-time, so each
perm_id has its own earnings rows and the dedup is per-perm_id.

**Phase B v2 fix: per-canonical aggregation (post-run bug fix)**

Phase B v1's per-perm_id write loop has a write-clobber bug for the 12
collision pairs: each perm_id fetched-then-stored its concatenated result
under the SHARED `/sp400/{canonical_ticker}` node, and the LAST-perm_id
processed (alphabetical perm_id order from HDFStore) wins. For 3 of 12
cases (AXON, AZTA, EXE), the last perm_id's aliases list did NOT include
the pre-rebrand ticker, clobbering the union history:
  - `/sp400/EXE` stored 1360 rows from `0001075736_EXE` (EXE-only), losing
    `0000895126_CHK`'s 2645-row CHK+EXE concatenation (CHK pre-rebrand
    history from 2016-2020 was dropped).
  - `/sp400/AXON` similarly stored 3768 rows from `0001636050_AXON`.
  - `/sp400/AZTA` similarly stored 3768 rows from `0001518749_AZTA`.

Phase B v2 fix: `03_data_gathering.py` now aggregates perm_ids by
`canonical_ticker` BEFORE the fetch loop, UNION-ing their aliases.
For each canonical, it fetches ONCE with the union alias list and stores
ONCE. Stored `/sp400/{canonical}` now contains the UNION of all perm_ids'
alias histories, and Phase E's interval-gating + §7.7 disambiguation rule
attribute rows back to each perm_id.

**Phase B v2.1 fix: always-refetch (post-run bug fix #2)**

Phase B v2's first run kept the v1-clobbered nodes for EXE/AXON/AZTA
because of an over-eager freshness check: the v1 fetch had stored data
up to END_DATE (today), so v2's `latest_date >= END_DATE` skip fired and
left the clobbered 1360-row /sp400/EXE node in place. The freshness skip
was designed for incremental updates but defeats the union-alias
reconciliation when v1 clobbered data exists.

Phase B v2.1 fix: `update_canonical_node()` now ALWAYS refetches when the
node already exists (no freshness skip). EODHD subscription is effectively
unlimited so always-refetch is cheap and the only way to guarantee the
stored /sp400/{canon} reflects the union alias history. The `prev_gap=N d`
field in the new log line is informational only (shows how stale the
previous node was).

  Verified empirically: re-running `aggregate_canonicals` + per-canonical
  fetch for the EXE aggregate returned 2645 rows (EXE+CHK, 1285 pre-rebrand
  + 1360 post-rebrand rows), confirming the fix resolves the clobber.

### 7.8 Pipeline-wide impact summary

  - 974 perm_ids built (vs 966 old companies).
  - 958 UNIQUE canonical_tickers among the 970 available perm_ids (12 pairs
    collide as documented in §7.7). /sp400/* has exactly 958 nodes post
    Phase B (one node per canonical_ticker; collisions dedup automatically
    on the storage key).
  - 4 unavailable (RE, WXS, TMST, CDAY).
  - All 14 originally-flagged BAD merges correctly resolved (see summary in
    Phase A's release notes).
  - Phase B (`03_data_gathering.py`) re-fetched all 970 perm_ids' price
    history using the new canonical/alias system. Alias CONCATENATION (not
    fallback-pick-first) is mandatory because EODHD does not retro-relabel
    rebrands -- see 03_data_gathering.py docstring and database_layout.md
    `/sp400/{canonical}` section.
  - Phase C+E (`06_earnings_gathering.py`) will key by `perm_id` (not
    `canonical_ticker`) to avoid the §7.7 collision-dedup problem.
  - Phase E feature builder MUST apply the §7.7 disambiguation rule when
    attributing /sp400/{canon} rows to a perm_id.

### 7.9 Phase E implementation plan (LOCKED 2026-07-14)

**Stage 1 gate (`02_features/01_features_gate_events.py`) rewrite:**

  - Iterate per **perm_id** from `/metadata/sp400_perm_ids` (replaces per-
    canonical iteration).
  - For each perm_id, gate events from `/earnings/raw` keyed by **`perm_id`**
    (NOT `canonical_ticker`) to avoid the §7.7 collision-dedup problem.
  - Window membership: `event.report_date in [added + 90d, removed]` for any
    of the perm_id's `combined_intervals`.
  - Output schema gains a `perm_id` column (alongside `canonical_ticker`
    for the downstream Stage-2 price-series join). `cik` retained for audit.
  - `price_unavailable=True` perm_ids produce ZERO gated events.
  - Nan-drop per §7.7 happens here for the **loser** perm_id at this stage:
    if the perm_id is a §7.7 LOSER (its effective end precedes the other
    perm_id sharing canonical_ticker), all its events in the OVERLAP zone are
    dropped at the gate. This removes them at the EARLIEST stage so Phase 2's
    price-carried computations don't waste effort on rows that would be
    NaN-dropped later anyway.
  - Empirical impact: 105 events dropped (0.23% of all `/earnings/raw` rows).
    Distribution across 7 pairs (COHR 26, LDOS 24, AZTA 16, AXON 15, VAL 14, CZR 9,
    EXE 1). 5 pairs (ACI, AM, CC, JEF, HR) have NO overlap -> no drop.

**Stage 2 feature matrix (`02_features/02_build_feature_matrix.py`)
implementation updates:**

  - `gated_df` now has `perm_id` column; iterate per-perm_id (not per-canonical).
  - Price-series loading STILL uses `canonical_ticker` (e.g. `/sp400/EXE` for
    both `0000895126_CHK` and `0001075736_EXE`). The §7.7 disambiguation at
    Stage 1 ensures we ONLY compute features for events belonging to the
    winner perm_id at points where the live asset was actually traded.
  - Pass A (per-event CAR windows + Block 2 + Block 3): unchanged logic;
    per-perm_id loop. Each perm_id loads its canonical's price node
    (deduplicated load via `stock_prices_cache[canonical]` so the 12
    collision pairs share the loaded DataFrame).
  - Pass B (Block 1 SUE family + `car_drift_historical_q1`):
    `car_60d_pass1.shift(1)` is now PER-PERM-ID, not per-canonical. Because
    the §7.7 NaN-drop removed loser events in overlap, each winner perm_id's
    earnings history is clean (no spillover from loser perm_id).
  - `index_ref` join moves from `/metadata/sp400_companies` (deleted) to
    `/metadata/sp400_perm_ids` (the new schema has `index_ref` column
    per-perm_id; this is the Phase-A-derived index_ref already joined at
    02b time).
  - `canonical_ticker` remains in the output schema as metadata for
    price-series joins; `perm_id` is the row anchor.

**Schema deltas in `/features/train_matrix` (29 -> 30 columns):**

  + `perm_id` (str): the Phase-A perm_id anchor; row identifier.
  - `canonical_ticker` (str): informational; identifies the price node.
  (All other schema unchanged.)

### 7.10 T-match failures / EODHD price-feed gaps (Phase E acceptance, Tiingo fallback deferred)

After Phase E Stage 2's live run, 21 gated events failed T-match (i.e., no
trading day `>= report_date` exists in the canonical's `/sp400/{canon}` price
node). These 21 events are DROPPED from `/features/train_matrix` per the
NaN policy (features.md §4 "T-match failure is the ONE documented drop").

**Root cause: EODHD price-history feed gaps.** For each dropping perm_id
the EODHD `/api/eod/{CANON}.US` endpoint returns fewer rows than the
perm_id's effective membership window. EODHD's earnings-calendar feed and
price-history feed can be inconsistent per symbol.

**Empirically observed (2026-07-14 Phase E run):**

| canonical | # drops | comment |
|---|---|---|
| `SIX`  | 11 | Six Flags Entertainment NYSE:"SIX" EODHD price-history feed has only 251 rows for 2015-01-02 to 2015-12-31 (with volume=0 -- a phantom SIX-variant instrument, NOT real NYSE SIX data). EODHD's symbol-search returns no results for "Six Flags" or "SIX" -- the Six Flags NYSE SIX listing is genuinely absent from EODHD's symbol universe for 2017-2023. EODHD's `/api/calendar/earnings` however HAS 28 SIX.US rows for 2017-2023 (those 11 surviving-gated events are the T-match failures). |
| `BERY` |  1 | BERY delisted 2025-04-29 (perm_id `removed`=2025-05-01). The 2025-04-30 earnings report fell one day AFTER the last stored price row. |
| `IGT`  |  1 | EODHD price-history ends before the 2025-11-11 report_date; post-reorganization rebrand timing problem. |
| `PSTG` |  1 | EODHD price-history ends before the 2026-05-27 report_date; recent delisting/restructuring. |
| Others |  7 | Various single-event drops at series end (similar -- report_date past the last stored price day). |

**Acceptance:** 21 drops = 0.10% of 21,269 gated events. Acceptable for
Phase E. The pipeline correctly follows the NaN-policy drop + log path.

**Tiingo fallback option (DEFERRED):**

If EODHD gaps start eating into the training universe
beyond a chosen tolerance (e.g. >2% of train rows lost to T-match failures),
we can re-introduce Tiingo as a **conditional fallback** specifically for
perm_ids whose canonical's `/sp400/{canon}` node is too short for the
perm_id's effective membership window. Implementation outline (future
Phase F ticket):

  1. Add a `02_features/...` pre-flight probe: for each gated-perm_id, count
     `/sp400/{canon}` rows that fall inside the perm_id's `combined_intervals`.
     If < 80% of expected trading days, mark the perm_id as `tiingo_fallback`.
  2. In `03_data_gathering.py`'s `update_canonical_node()`, add a path that,
     when called with `--repair-tiingo`, re-fetches the canonical from
     Tiingo's `/daily/{ticker}` endpoint using the same `aliases` list.
  3. Schema is identical (11-col OHLCV + derived adj_* via close/adjusted_close
     in Tiingo returns). Merge with existing EODHD rows (dedup-keep-last per Date).
  4. Re-run Stage 1+2 to repair the dropped events.

---

## §7.11 ARCHITECTURE SUCCESSOR: permaTicker migration (2026-07-14)

After Phase B's contamination audit surfaced 3 distinct bug classes
(`phase_b_contamination_audit.md`), a deeper probe-based comparison
(`04_backtest/archive/docs/eodhd_vs_tiingo.md` (archived), `tiingo_permaTicker_audit.md`) led to the
conclusion that the synthetic `perm_id` mechanism defined in this
spec was solving a problem Tiingo's data model natively solves.
Tiingo's `permaTicker` field (queried via
`/tiingo/utilities/search/{query}?includeDelisted=true&exactTickerMatch=true`)
returns identity-stable primary keys that survive:
- ticker renames (META was FB's permaTicker; prices under META perma fetch
  back to 2013 correctly)
- multi-ticker rebrand chains (CSII -> PEGY -> SUNE all share one permaTicker,
  prices retrievable across the entire chain via the permaTicker key)
- bankruptcy reorgs with a fresh ticker code (Chesapeake CHK perma `US000000000505`,
  Expand EXE perma `US000000092728` -- distinct even though same SEC CIK)
- spinoffs producing two separately-traded descendants (Colfax -> ENOV
  inherits full history + ESAB starts fresh, distinct permaTickers)
- symbol recycling where two different companies held the same ticker code
  in different eras (SUNE was SunEdison `US000000002709` AND SUNation
  `US000000002062` -- both identities retrivable independently via perma)

All bug classes (Class U/V/W/S) documented in `phase_b_contamination_audit.md`
are eliminated at source by switching the primary key to `permaTicker`:
- Class U (alias-intertwining): eliminated -- no alias-merge needed; one
  `/prices` fetch per permaTicker retrieves the full clean history
  (Tiingo back-merged rebrand history into the modern canonical)
- Class V (alternating adjusted_close): same as Class U (reclassified as Class U)
- Class W (NSR retroactive rewiring): eliminated -- Neustar `US000000006945`
  cleanly returns 2015 real prices, separate from Nomad's perma
- Class S (SUNE modern era confusion): eliminated -- SunEdison and the
  CSII/Pineapple/SUNation chain are separate permaTickers

See `tiingo_permaTicker_audit.md` for the full evidence table, lookup
mechanism, and migration design. Forward Phase A rewrite (`02b_build_company_map.py`)
will be permaTicker-keyed against the Tiingo search endpoint; all `perm_id`
references below (and the `02b` algorithm in §7.1-§7.10) are obsolete.

---

This option is intentionally DEFERRED for Phase E. It is documented here
so a future Phase F can take it up without re-discovering the issue.
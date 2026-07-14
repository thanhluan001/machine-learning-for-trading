# Quantitative Architecture Specification: Identity Patch

## Target Component: Asset Identification and Merger Processing Architecture
* **Status:** Operational Patch Implementation (July 2026)
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

### 7.6 Pipeline-wide impact summary

  - 974 perm_ids built (vs 966 old companies).
  - 970 effective training universe after EODHD availability probe
    (4 unavailable: RE, WXS, TMST, CDAY).
  - All 14 originally-flagged BAD merges correctly resolved (see summary in
    Phase A's release notes).
  - Phase B (`03_data_gathering.py`) will re-fetch all 970 perm_ids' price
    history using the new canonical/alias system.
  - Phase C+E (`06_earnings_gathering.py`) will re-fetch all 970 perm_ids'
    earnings using perm_id and the new write-time dedup rule.
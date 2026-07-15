#!/usr/bin/env python3
"""
Earnings Gathering (EODHD) - 06 — Phase D rewrite
=================================================

Fetches full 15-year historical earnings (per perm_id) from EODHD's
`/api/calendar/earnings` endpoint and stores raw rows under
`/earnings/raw` in db.h5. One row per (perm_id, fiscal_period_end).

Phase D (2026-07-14): migrates the dedup key from `(canonical_ticker,
report_date)` to `(perm_id, fiscal_period_end)`. Why:
    - Phase A rewrote `/metadata/sp400_perm_ids` with point-in-time CIK
      anchoring + interval-forked perm_ids (decoupling legal entity from
      tradable asset). See `01_data/merger_identity_patch.md`.
    - 12 perm_id pairs share `canonical_ticker` (post-Wiki acquirer-rebrand
      extension side-effect, documented in merger_identity_patch.md §7.7).
      v1's `(canonical_ticker, report_date)` dedup could WRONGLY drop one
      perm_id's event if the other perm_id had a same-day event.
    - Keying by `perm_id` (and the more-specific `fiscal_period_end`) isolates
      each perm_id's earnings without cross-perm_id collisions.

Reads from: `/metadata/sp400_perm_ids` (Phase A output; replaces the
deleted `/metadata/sp400_companies`).

Dedup-at-write-time rule (LOCKED per merger_identity_patch.md §7.7 + Phase A
release):
    - Dedup key:     (perm_id, fiscal_period_end)
    - Tiebreak 1:    prefer the row whose `code` (alias EODHD returned the row
                     under) is the perm_id's canonical alias (i.e.
                     `code == canonical_ticker + ".US"`).
    - Tiebreak 2:    latest `report_date`
    - Tiebreak 3:    lexicographic `code` (deterministic fallback)

Iteration
---------
- Iterate per **perm_id** from `/metadata/sp400_perm_ids`.
- For each perm_id, EODHD fetch by `symbols=` with ALL its aliases appended
  `.US` (e.g. `CHK.US,EXE.US` for the CHK+EXE rebrand perm_id).
- `from`/`to` required even with `symbols=` set (live-verified).
- Skip perm_ids with `price_unavailable=True` (matches 03_data_gathering.py
  so the earnings universe and price universe are aligned).

EODHD subscription: effectively unlimited. With ~970 available perm_ids
the full run is ~970 EODHD Calendar API calls, finishing in minutes. No
checkpoint (idempotent via `store.remove` + put pattern).

Output schema (writes to `/earnings/raw`):
    report_date          : datetime  -- announcement date T (PEAD event time)
    fiscal_period_end    : datetime  -- fiscal quarter end (NOT the event date)
    code                 : str       -- EODHD alias the row was returned under
                                       (e.g. 'AAXN.US', 'AXON.US')
    perm_id              : str       -- candidate-tradable-asset-track anchor
                                       (Phase A: f"{cik}_{start_ticker}")
    canonical_ticker     : str       -- perm_id's canonical alias (informational)
    cik                  : str|None  -- perm_id's CIK (10-digit; None for
                                       __nocik_* perm_ids)
    actual               : float|None
    estimate             : float|None  (NaN estimates -> difference=0.0 per
                                        EODHD convention; kept in /earnings/raw
                                        as the denominator block per sues)
    difference           : float|None  (actual - estimate)
    percent              : float|None  (surprise %, maps to eps_surprise_pct)
    before_after_market  : str|None    ('Bmo' / 'AfterMarket')
    currency             : str|None    (usually 'USD')

NOTE: This replaces the v1 schema (had `canonical_ticker` as PRIMARY; no
`perm_id` column). Phase E feature builder will iterate per perm_id and
join to `/sp400/{canonical_ticker}` for price data via
`/metadata/sp400_perm_ids`.

Usage:
    python 06_earnings_gathering.py
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Explicit .env path so the script works regardless of CWD (matches 02b/03).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

EODHD_API_KEY = os.getenv("EODHD_API_KEY")
if not EODHD_API_KEY:
    raise ValueError(
        "EODHD_API_KEY not found in .env. Add it before running this script."
    )

DB_FILE = Path(__file__).parent / "db.h5"
PERM_IDS_KEY = "/metadata/sp400_perm_ids"
EARNINGS_KEY = "/earnings/raw"

HISTORY_YEARS = 15
START_DATE = (datetime.now() - timedelta(days=HISTORY_YEARS * 365)).strftime("%Y-%m-%d")
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

EARNINGS_URL = "https://eodhd.com/api/calendar/earnings"


# ------------------------------------------------------------------
# Load perm_id universe
# ------------------------------------------------------------------

def load_perm_ids() -> list[dict]:
    """Read /metadata/sp400_perm_ids and return a list of dicts with the
    fields we need: perm_id, canonical_ticker, cik, aliases,
    price_unavailable (informational pass-thru).
    """
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"{DB_FILE} not found. Run 01 -> 02 -> 02b_build_company_map.py (Phase A) first."
        )
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if PERM_IDS_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {PERM_IDS_KEY} missing in {DB_FILE}. "
                f"Run 02b_build_company_map.py (Phase A) first."
            )
    df = pd.read_hdf(DB_FILE, key=PERM_IDS_KEY)
    perm_ids = []
    for _, row in df.iterrows():
        try:
            aliases = row["aliases"]
            if isinstance(aliases, str):
                aliases = json.loads(aliases)
        except Exception:
            aliases = []
        if not aliases:
            aliases = [row["canonical_ticker"]]
        perm_ids.append({
            "perm_id": None if pd.isna(row.get("perm_id")) else str(row["perm_id"]),
            "canonical_ticker": str(row["canonical_ticker"]),
            "cik": None if pd.isna(row.get("cik")) else str(row["cik"]),
            "aliases": [str(a) for a in aliases],
            "price_unavailable": bool(row["price_unavailable"]),
        })
    return perm_ids


# ------------------------------------------------------------------
# EODHD fetcher
# ------------------------------------------------------------------

def fetch_perm_earnings(symbols_us: list[str]) -> list[dict]:
    """GET /api/calendar/earnings for one perm_id's aliases.

    Args:
        symbols_us: list of aliases already suffixed with .US
                    (e.g. ['AAXN.US', 'AXON.US']) -- the perm_id's full alias list
                    (Phase B's aggregate_canonicals is NOT used here; we want
                    each perm_id's OWN aliases so cross-perm_id rows stay sorted
                    by perm_id).

    Returns:
        list of raw EODHD earnings-row dicts (empty if none / error).
    """
    params = {
        "symbols": ",".join(symbols_us),
        "from": START_DATE,
        "to": END_DATE,
        "api_token": EODHD_API_KEY,
        "fmt": "json",
    }
    try:
        r = requests.get(EARNINGS_URL, params=params, timeout=45)
        r.raise_for_status()
    except Exception as e:
        print(f"      EODHD request error for {symbols_us}: {e}")
        return []

    body = r.json()
    if not isinstance(body, dict):
        return []
    rows = body.get("earnings") or []
    if not isinstance(rows, list):
        return []
    return rows


# ------------------------------------------------------------------
# Per-event row normalization
# ------------------------------------------------------------------

def _to_date_str(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _to_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        if pd.isna(f):
            return None
        return f
    except Exception:
        return None


def normalize_rows(raw_rows: list[dict],
                   perm_id: str | None,
                   canonical_ticker: str,
                   cik: str | None,
                   canonical_code: str) -> list[dict]:
    """Convert raw EODHD rows to the stored schema, joined with perm_id."""
    out = []
    for r in raw_rows:
        report_date = _to_date_str(r.get("report_date"))
        if report_date is None:
            continue
        # Safety-filter window (defensive; EODHD usually honors from/to)
        if report_date < START_DATE or report_date > END_DATE:
            continue
        out.append({
            "report_date": report_date,
            "fiscal_period_end": _to_date_str(r.get("date")),
            "code": r.get("code"),
            "perm_id": perm_id,
            "canonical_ticker": canonical_ticker,
            "cik": cik,
            "actual": _to_float(r.get("actual")),
            "estimate": _to_float(r.get("estimate")),
            "difference": _to_float(r.get("difference")),
            "percent": _to_float(r.get("percent")),
            "before_after_market": r.get("before_after_market"),
            "currency": r.get("currency"),
            # Helper column used only for dedup; dropped before storage.
            "_is_canonical_code": (r.get("code") == canonical_code),
        })
    return out


# ------------------------------------------------------------------
# Deduplication (handles rebrand-transition overlap)
# ------------------------------------------------------------------

def deduplicate(rows: list[dict]) -> list[dict]:
    """Drop duplicates keyed by (perm_id, fiscal_period_end), keeping the
    best row per phase A's "dedup-at-write-time rule":
        Tiebreak 1 (primary): prefer the row whose `code` equals the
                  perm_id's canonical alias's EODHD code
                  (`canonical_ticker + '.US'`).
                  Encoded as the helper boole `_is_canonical_code=True`.
        Tiebreak 2: latest `report_date` (later report = later filing).
        Tiebreak 3: lexicographic `code` (deterministic).

    Implementation: stable sort with the priority order
        (perm_id, fiscal_period_end, _is_canonical_code DESC, report_date DESC, code ASC)
    ...then drop_duplicates keeping first. The DESC tiebreaks are realized
    by negating booleans / reversing dates so sort_values can stay ASC.
    """
    if not rows:
        return []
    df = pd.DataFrame(rows)
    # For ASC-sort + drop_duplicates-keep-first with:
    #   - prefer canonical code (Tiebreak 1): use `~_is_canonical_code` so
    #     canonical (True) becomes False (sorts FIRST under ASC).
    #   - prefer latest report_date (Tiebreak 2): use negative date so later
    #     dates sort FIRST under ASC.
    #   - lexicographic `code` ASC (Tiebreak 3): already ASC.
    df["_sort_canonical_pref"] = ~df["_is_canonical_code"].astype(bool)
    df["_sort_report_date_desc"] = pd.to_datetime(
        df["report_date"], errors="coerce"
    ).apply(lambda x: -x.value if pd.notna(x) else 0)
    df = df.sort_values(
        ["perm_id", "fiscal_period_end", "_sort_canonical_pref",
         "_sort_report_date_desc", "code"],
        kind="stable",
        na_position="last",
    )
    df = df.drop_duplicates(
        subset=["perm_id", "fiscal_period_end"], keep="first"
    )
    # Drop helper columns; persist only the documented schema.
    df = df.drop(columns=["_sort_canonical_pref", "_sort_report_date_desc", "_is_canonical_code"])
    return df.to_dict("records")


# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------

def store_rows(rows: list[dict]) -> None:
    """Persist all per-event rows under /earnings/raw.

    Uses the HDFStore('a') + store.remove() pattern (never `mode='w'` on an
    existing DB).
    """
    if not rows:
        # Still remove the existing node so a re-run doesn't leave stale data.
        with pd.HDFStore(DB_FILE, mode="a") as store:
            if EARNINGS_KEY in store:
                store.remove(EARNINGS_KEY)
        return

    df = pd.DataFrame(rows)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"], errors="coerce")

    with pd.HDFStore(DB_FILE, mode="a") as store:
        if EARNINGS_KEY in store:
            store.remove(EARNINGS_KEY)
        store.put(EARNINGS_KEY, df, format="table")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  06 - Earnings Gathering (EODHD, per perm_id)  [Phase D rewrite]")
    print("=" * 70)
    print(f"  History:  {HISTORY_YEARS} years ({START_DATE} to {END_DATE})")
    print(f"  Source:   EODHD /api/calendar/earnings")
    print(f"  Iteration: per perm_id (canonical + aliases)")
    print(f"  Dedup:    (perm_id, fiscal_period_end) -- canonical alias preferred")
    print("=" * 70)

    print(f"\n[1/3] Loading {PERM_IDS_KEY} ...")
    perm_ids = load_perm_ids()
    n_total = len(perm_ids)
    n_skipped = sum(1 for p in perm_ids if p["price_unavailable"])
    print(f"   {n_total} perm_ids total; {n_skipped} skipped (price_unavailable).")

    print(f"\n[2/3] Fetching EODHD earnings per perm_id ...")
    all_rows: list[dict] = []
    perm_ids_with_events = 0
    perm_ids_zero_events: list[str] = []
    perm_ids_skipped: list[str] = []

    fetched = 0
    for i, p in enumerate(perm_ids, 1):
        perm_id = p["perm_id"]
        canonical = p["canonical_ticker"]
        if p["price_unavailable"] or not perm_id:
            perm_ids_skipped.append(perm_id or canonical)
            continue

        symbols_us = [f"{a}.US" for a in p["aliases"]]
        raw = fetch_perm_earnings(symbols_us)
        canonical_code = f"{canonical}.US"
        rows = normalize_rows(raw,
                              perm_id=perm_id,
                              canonical_ticker=canonical,
                              cik=p["cik"],
                              canonical_code=canonical_code)
        fetched += 1

        if rows:
            perm_ids_with_events += 1
            all_rows.extend(rows)
        else:
            perm_ids_zero_events.append(perm_id)

        if i % 50 == 0 or i == n_total:
            print(
                f"   progress: {i}/{n_total}  "
                f"(fetched={fetched}, with_events={perm_ids_with_events}, "
                f"cum_rows={len(all_rows)})"
            )
        # No throttle needed (EODHD subscription is effectively unlimited);
        # keep a tiny sleep for OS scheduler hygiene.
        time.sleep(0.02)

    print(f"\n   Fetched perm_ids:         {fetched}")
    print(f"   With >=1 event:           {perm_ids_with_events}")
    print(f"   With zero events:         {len(perm_ids_zero_events)}")

    print("\n[3/3] Deduplicating + storing /earnings/raw ...")
    deduped = deduplicate(all_rows)
    print(f"   Pre-dedup rows:  {len(all_rows)}")
    print(f"   Post-dedup rows: {len(deduped)}")
    store_rows(deduped)
    print(f"   Wrote {EARNINGS_KEY} ({len(deduped)} rows)")

    # ---- Audit ----
    print("\n" + "=" * 70)
    print("  EARNINGS GATHERING AUDIT (Phase D)")
    print("=" * 70)
    print(f"  Total perm_ids:           {n_total}")
    print(f"  Skipped (price_unavail):  {len(perm_ids_skipped)}")
    print(f"  Fetched from EODHD:      {fetched}")
    print(f"  Perm_ids with events:     {perm_ids_with_events}")
    print(f"  Perm_ids zero events:    {len(perm_ids_zero_events)}")
    print(f"  Total raw events:        {len(deduped)}")

    if perm_ids_zero_events:
        print(f"\n  --- Perm_ids with zero EODHD events (first 30) ---")
        for t in perm_ids_zero_events[:30]:
            print(f"    - {t}")
        if len(perm_ids_zero_events) > 30:
            print(f"    ... and {len(perm_ids_zero_events) - 30} more")
        print("  These contribute 0 rows to the training matrix.")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

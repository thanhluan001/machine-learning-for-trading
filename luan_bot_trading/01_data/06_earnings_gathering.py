#!/usr/bin/env python3
"""
Earnings Gathering (EODHD) - 06
===============================

Fetches full 15-year historical earnings (per company) from EODHD's
`/api/calendar/earnings` endpoint and stores raw rows under
`/earnings/raw` in db.h5. One row per (canonical_ticker, report_date).

Per the design in `luan_bot_trading/earnings_gathering_design.md`:
    - Iterate per COMPANY from /metadata/sp400_companies (not per ticker).
    - For each company, fetch by `symbols=` with ALL its aliases appended
      `.US` (e.g. `AAXN.US,AXON.US`).
    - `from`/`to` ARE required even with `symbols=` set (live-verified).
    - Skip companies with `price_unavailable=True` (matches 03_data_gathering.py).
    - Deduplicate by (canonical_ticker, report_date) to handle rebrand-
      transition overlap where the same report_date appears under both the
      old alias and the new alias.
    - EODHD's `/api/calendar/trends` endpoint is NOT used (forward-looking
      only, cannot backfill historical training estimates).

EODHD subscription: 100k calls/day, 1000 calls/min. With ~930 companies
the full run is ~930 calls, finishing in minutes. No checkpoint needed
(re-runnable; idempotent via `store.remove` + put pattern).

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
COMPANIES_KEY = "/metadata/sp400_companies"
EARNINGS_KEY = "/earnings/raw"

HISTORY_YEARS = 15
START_DATE = (datetime.now() - timedelta(days=HISTORY_YEARS * 365)).strftime("%Y-%m-%d")
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

EARNINGS_URL = "https://eodhd.com/api/calendar/earnings"


# ------------------------------------------------------------------
# Load company universe
# ------------------------------------------------------------------

def load_companies() -> list[dict]:
    """Read /metadata/sp400_companies and return a list of dicts with the
    fields we need: canonical_ticker, cik, aliases, price_unavailable,
    combined_intervals.
    """
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"{DB_FILE} not found. Run 01 -> 02 -> 02b_build_company_map.py first."
        )
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if COMPANIES_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {COMPANIES_KEY} missing in {DB_FILE}. "
                f"Run 02b_build_company_map.py first."
            )
    df = pd.read_hdf(DB_FILE, key=COMPANIES_KEY)
    companies = []
    for _, row in df.iterrows():
        try:
            aliases = row["aliases"]
            if isinstance(aliases, str):
                aliases = json.loads(aliases)
        except Exception:
            aliases = []
        if not aliases:
            aliases = [row["canonical_ticker"]]
        try:
            combined = row["combined_intervals"]
            if isinstance(combined, str):
                combined = json.loads(combined)
        except Exception:
            combined = []
        companies.append({
            "canonical_ticker": str(row["canonical_ticker"]),
            "cik": None if pd.isna(row.get("cik")) else str(row["cik"]),
            "aliases": [str(a) for a in aliases],
            "price_unavailable": bool(row["price_unavailable"]),
            "combined_intervals": combined,
        })
    return companies


# ------------------------------------------------------------------
# EODHD fetcher
# ------------------------------------------------------------------

def fetch_company_earnings(symbols_us: list[str]) -> list[dict]:
    """GET /api/calendar/earnings for one company's aliases.

    Args:
        symbols_us: list of aliases already suffixed with .US
                    (e.g. ['AAXN.US', 'AXON.US'])

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


def normalize_rows(raw_rows: list[dict], canonical_ticker: str, cik: str | None) -> list[dict]:
    """Convert raw EODHD rows to the stored schema, joined with canonical."""
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
            "canonical_ticker": canonical_ticker,
            "cik": cik,
            "actual": _to_float(r.get("actual")),
            "estimate": _to_float(r.get("estimate")),
            "difference": _to_float(r.get("difference")),
            "percent": _to_float(r.get("percent")),
            "before_after_market": r.get("before_after_market"),
            "currency": r.get("currency"),
        })
    return out


# ------------------------------------------------------------------
# Deduplication (handles rebrand-transition overlap)
# ------------------------------------------------------------------

def _alias_active_on_date(code: str | None, combined_intervals: list[dict]) -> bool:
    """Best-effort: did the company (per its combined_intervals) have the alias
    represented by `code` active at the report date? We don't track per-alias
    intervals in this step (that lives in `per_ticker_intervals`), so we use a
    coarse heuristic: the company's combined span is active = True (always,
    since we only queried aliases of the company). Real tiebreak is fallback
    below (keep the first occurrence after stable sort).
    """
    return True  # Cannot resolve per-alias precisely here; tiebreak uses order.


def deduplicate(rows: list[dict]) -> list[dict]:
    """Drop duplicates keyed by (canonical_ticker, report_date), keeping the
    first occurrence after a stable sort by (canonical_ticker, report_date, code asc).

    Rebrand-transition overlap (same report_date under both AAXN and AXON) is
    resolved by keeping whichever row sorts first by `code` (alphabetically),
    which is deterministic but not guaranteed to be the "active alias on that
    date" — for that, the feature builder can re-derive from
    `per_ticker_intervals`. For the raw storage layer, deterministic dedup is
    sufficient.
    """
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["canonical_ticker", "report_date", "code"],
        kind="stable",
        na_position="last",
    )
    df = df.drop_duplicates(
        subset=["canonical_ticker", "report_date"], keep="first"
    )
    return df.to_dict("records")


# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------

def store_rows(rows: list[dict]) -> None:
    """Persist all per-event rows under /earnings/raw.

    Uses the HDFStore('a') + store.remove() pattern (never `mode='w'` on an
    existing DB — that bug class wiped the whole db.h5 in earlier versions).
    """
    if not rows:
        # Still remove the existing node so a re-run doesn't leave stale data
        # if the new run happens to find zero rows (unlikely at the union level
        # but possible).
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
    print("  06 - Earnings Gathering (EODHD, per company)")
    print("=" * 70)
    print(f"  History:  {HISTORY_YEARS} years ({START_DATE} to {END_DATE})")
    print(f"  Source:   EODHD /api/calendar/earnings")
    print(f"  Iteration: per company (canonical + aliases)")
    print("=" * 70)

    print("\n[1/3] Loading /metadata/sp400_companies ...")
    companies = load_companies()
    n_total = len(companies)
    n_skipped = sum(1 for c in companies if c["price_unavailable"])
    print(f"   {n_total} companies total; {n_skipped} skipped (price_unavailable).")

    print(f"\n[2/3] Fetching EODHD earnings per company ...")
    all_rows: list[dict] = []
    companies_with_events = 0
    companies_zero_events: list[str] = []
    companies_skipped: list[str] = []

    fetched = 0
    for i, c in enumerate(companies, 1):
        canonical = c["canonical_ticker"]
        if c["price_unavailable"]:
            companies_skipped.append(canonical)
            continue

        symbols_us = [f"{a}.US" for a in c["aliases"]]
        raw = fetch_company_earnings(symbols_us)
        rows = normalize_rows(raw, canonical_ticker=canonical, cik=c["cik"])
        fetched += 1

        if rows:
            companies_with_events += 1
            all_rows.extend(rows)
        else:
            companies_zero_events.append(canonical)

        if i % 50 == 0 or i == n_total:
            print(
                f"   progress: {i}/{n_total}  "
                f"(fetched={fetched}, with_events={companies_with_events}, "
                f"cum_rows={len(all_rows)})"
            )
        # 1000/min limit -> ~16/sec. Be polite; well below anyway.
        time.sleep(0.05)

    print(f"\n   Fetched companies:        {fetched}")
    print(f"   With >=1 event:           {companies_with_events}")
    print(f"   With zero events:         {len(companies_zero_events)}")

    print("\n[3/3] Deduplicating + storing /earnings/raw ...")
    deduped = deduplicate(all_rows)
    print(f"   Pre-dedup rows:  {len(all_rows)}")
    print(f"   Post-dedup rows:  {len(deduped)}")
    store_rows(deduped)
    print(f"   Wrote {EARNINGS_KEY} ({len(deduped)} rows)")

    # ---- Audit ----
    print("\n" + "=" * 70)
    print("  EARNINGS GATHERING AUDIT")
    print("=" * 70)
    print(f"  Total companies:           {n_total}")
    print(f"  Skipped (price_unavail):   {len(companies_skipped)}")
    print(f"  Fetched from EODHD:        {fetched}")
    print(f"  Companies with events:     {companies_with_events}")
    print(f"  Companies with zero events: {len(companies_zero_events)}")
    print(f"  Total raw events:          {len(deduped)}")

    if companies_zero_events:
        print(f"\n  --- Companies with zero EODHD events (first 30) ---")
        for t in companies_zero_events[:30]:
            print(f"    - {t}")
        if len(companies_zero_events) > 30:
            print(f"    ... and {len(companies_zero_events) - 30} more")
        print("  These contribute 0 rows to the training matrix.")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

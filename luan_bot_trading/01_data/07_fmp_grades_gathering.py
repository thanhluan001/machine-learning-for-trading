#!/usr/bin/env python3
"""
FMP Analyst Grades Gathering — 07
====================================

Fetches full historical analyst upgrade/downgrade grades from FMP's
`/stable/grades` endpoint and stores under `/analyst/grades/{permaTicker}`
in db.h5.

This is the #1 PEAD feature identified in `feature_sourcing_audit.md`:
analyst revision momentum. FMP returns 14 years of daily-granularity
analyst actions from 111 firms (upgrade / downgrade / maintain).

FMP $49/mo plan. Uses `/stable/grades?symbol={ticker}`.
Data source confirmed per `feature_sourcing_audit.md` §4A.3.

Output schema (writes to `/analyst/grades/{permaTicker}`):
    symbol              : str       -- the ticker used for the FMP API call
    permaTicker         : str       -- PRIMARY key (from /metadata/sp400_permatickers)
    date                : datetime   -- date of the analyst action
    grading_company     : str       -- name of the analyst firm (e.g. "Morgan Stanley")
    previous_grade      : str       -- previous rating (e.g. "Hold", "Buy")
    new_grade           : str       -- new rating (e.g. "Buy", "Sell")
    action              : str       -- "upgrade", "downgrade", "maintain", "initiate"

Dedup: FMP may return duplicate grade entries (same date, same firm,
same action). We dedup by (permaTicker, date, grading_company, action),
keeping the first occurrence.

Usage:
    python 07_fmp_grades_gathering.py
    python 07_fmp_grades_gathering.py --limit 10   # smoke test
"""
import argparse
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    raise ValueError("FMP_API_KEY not found in .env.")

DB_FILE = Path(__file__).parent / "db.h5"
PERMATICKERS_KEY = "/metadata/sp400_permatickers"
GRADES_GROUP = "analyst/grades"

FMP_BASE = "https://financialmodelingprep.com/stable"
FMP_TIMEOUT = 30


# ==============================================================================
# PART 1: LOAD PERMATICKER UNIVERSE
# ==============================================================================
def load_permatickers() -> pd.DataFrame:
    """Load /metadata/sp400_permatickers."""
    if not DB_FILE.exists():
        raise FileNotFoundError(f"{DB_FILE} not found. Run 02b_build_company_map.py first.")
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if PERMATICKERS_KEY not in store.keys():
            raise FileNotFoundError(f"Key {PERMATICKERS_KEY} missing. Run 02b first.")
        return store[PERMATICKERS_KEY]


# ==============================================================================
# PART 2: FMP GRADES FETCHER
# ==============================================================================
def fetch_fmp_grades(canonical_ticker: str) -> list[dict]:
    """Fetch /stable/grades for one ticker. Returns the full historical
    analyst upgrade/downgrade list (14+ years for well-covered companies)."""
    url = f"{FMP_BASE}/grades"
    params = {"symbol": canonical_ticker, "apikey": FMP_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=FMP_TIMEOUT)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return data


def normalize_grades(
    raw_rows: list[dict],
    permaTicker: str,
    canonical_ticker: str,
) -> list[dict]:
    """Convert FMP raw grades to the stored schema."""
    out = []
    for r in raw_rows:
        date_str = r.get("date")
        if date_str is None:
            continue
        out.append({
            "symbol": canonical_ticker,
            "permaTicker": permaTicker,
            "date": pd.to_datetime(date_str, errors="coerce"),
            "grading_company": r.get("gradingCompany"),
            "previous_grade": r.get("previousGrade"),
            "new_grade": r.get("newGrade"),
            "action": r.get("action"),
        })
    return out


# ==============================================================================
# PART 3: PER-PERMATICKER STORE
# ==============================================================================
def store_grades(permaTicker: str, rows: list[dict]) -> int:
    """Store grades for one permaTicker under /analyst/grades/{permaTicker}.
    Dedup by (permaTicker, date, grading_company, action), keeping first.
    Uses HDFStore(mode='a') + store.remove() pattern."""
    if not rows:
        # Remove any stale node
        h5_path = f"/{GRADES_GROUP}/{permaTicker}"
        with pd.HDFStore(DB_FILE, mode="a") as store:
            if h5_path in store.keys():
                store.remove(h5_path)
        return 0

    df = pd.DataFrame(rows)
    # Dedup: same date + same firm + same action = duplicate
    n_pre = len(df)
    df = df.drop_duplicates(
        subset=["permaTicker", "date", "grading_company", "action"],
        keep="first",
    )
    n_post = len(df)
    # Sort by date for downstream feature computation
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    h5_path = f"/{GRADES_GROUP}/{permaTicker}"
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if h5_path in store.keys():
            store.remove(h5_path)
        store.put(h5_path, df, format="table", data_columns=["date", "action"])

    return n_post


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="FMP Analyst Grades Gathering")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, limit permaTickers (smoke test).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip permaTickers that already have a grades node (resume after timeout).")
    args = parser.parse_args()

    print("=" * 70)
    print("  07 - FMP Analyst Grades Gathering (analyst revision history)")
    print("=" * 70)
    print(f"  Source:    FMP /stable/grades?symbol={{ticker}}")
    print(f"  Storage:   /analyst/grades/{{permaTicker}}  (per-ticker nodes)")
    print(f"  History:   14 years (2012-present for well-covered companies)")
    print(f"  Fields:    date, grading_company, previous_grade, new_grade, action")
    print(f"  Dedup key: (permaTicker, date, grading_company, action)")
    print("=" * 70)

    print(f"\n[1/3] Loading {PERMATICKERS_KEY} ...")
    pt_df = load_permatickers()

    # Only fetch permaTickers with price_unavailable=False (matching 03/06b)
    fetchable = pt_df[~pt_df["price_unavailable"]].copy()
    n_total = len(fetchable)
    print(f"   {len(pt_df)} total permaTickers; {n_total} fetchable")

    if args.limit > 0:
        fetchable = fetchable.head(args.limit)
        print(f"   --limit={args.limit}: fetching first {len(fetchable)} (smoke)")

    # Check which permaTickers already have grade nodes (for --skip-existing resume)
    existing_keys = set()
    if args.skip_existing:
        with pd.HDFStore(DB_FILE, mode="r") as store:
            existing_keys = {k.split("/")[-1] for k in store.keys()
                             if k.startswith(f"/{GRADES_GROUP}/")}
        print(f"   --skip-existing: {len(existing_keys)} permaTickers already have nodes, skipping.")
        fetchable = fetchable[~fetchable["permaTicker"].isin(existing_keys)]
        print(f"   Remaining to fetch: {len(fetchable)}")

    print(f"\n[2/3] Fetching FMP grades per permaTicker ...")

    n_with_grades = 0
    n_zero_grades = 0
    n_failed = 0
    total_rows = 0
    t0 = time.time()
    progress_every = max(1, len(fetchable) // 20)

    for i, (_, row) in enumerate(fetchable.iterrows()):
        pt = str(row["permaTicker"])
        ct = str(row["canonical_ticker"])

        try:
            raw = fetch_fmp_grades(ct)
            if raw:
                rows = normalize_grades(raw, permaTicker=pt, canonical_ticker=ct)
                stored = store_grades(pt, rows)
                n_with_grades += 1
                total_rows += stored
            else:
                n_zero_grades += 1
                store_grades(pt, [])  # clear stale node if any
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"   [ERROR] {pt} ({ct}): {e}")

        if (i + 1) % progress_every == 0 or (i + 1) == len(fetchable):
            elapsed = time.time() - t0
            print(
                f"   [PROGRESS] {i+1}/{len(fetchable)}  "
                f"with_grades={n_with_grades}  zero={n_zero_grades}  "
                f"failed={n_failed}  total_rows={total_rows}  "
                f"elapsed={elapsed:.1f}s"
            )

        time.sleep(0.02)

    print(f"\n   With grades:      {n_with_grades}")
    print(f"   Zero grades:      {n_zero_grades}")
    print(f"   Failed:            {n_failed}")
    print(f"   Total grade rows:  {total_rows}")

    # ---- Audit ----
    print("\n[3/3] Audit ...")
    with pd.HDFStore(DB_FILE, mode="r") as store:
        grade_keys = [k for k in store.keys() if k.startswith(f"/{GRADES_GROUP}/")]

    print(f"   Stored nodes: {len(grade_keys)}")

    if grade_keys:
        # Sample stats from first few nodes
        all_actions = {}
        all_firms = set()
        date_range_min = None
        date_range_max = None
        for k in grade_keys[:50]:
            df = pd.read_hdf(DB_FILE, key=k)
            if df.empty:
                continue
            for a in df["action"].dropna().unique():
                all_actions[a] = all_actions.get(a, 0) + df[df["action"] == a].shape[0]
            all_firms.update(df["grading_company"].dropna().unique())
            dmin = df["date"].min()
            dmax = df["date"].max()
            if date_range_min is None or dmin < date_range_min:
                date_range_min = dmin
            if date_range_max is None or dmax > date_range_max:
                date_range_max = dmax

        print(f"   Action distribution (first 50 nodes):")
        for a, n in sorted(all_actions.items(), key=lambda x: -x[1]):
            print(f"     {a:12s}: {n}")

        print(f"   Unique grading firms (first 50): {len(all_firms)}")
        if date_range_min and date_range_max:
            print(f"   Date range (first 50): {date_range_min.date()} -> {date_range_max.date()}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

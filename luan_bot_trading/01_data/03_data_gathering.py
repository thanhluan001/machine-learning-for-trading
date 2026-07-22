#!/usr/bin/env python3
"""
Data Gathering - S&P 400 Mid-Cap Universe (permaTicker-Keyed, Tiingo)
=====================================================================

Phase B rewrite (2026-07-14): migrates the data fetcher from the deprecated
perm_id + EODHD alias-concatenation approach to the new
/metadata/sp400_permatickers + Tiingo per-permaTicker single-fetch approach.
See `01_data/tiingo_permaTicker_audit.md` for the identity model.

Fetches 15 years of daily OHLC history for every permaTicker that has ever
been a constituent of the S&P 400 (current + removed), per
/metadata/sp400_permatickers in db.h5.

Rules
-----
- Iterate per **permaTicker** (the new PRIMARY KEY). For each permaTicker:
  - Always refetch full history. EODHD rate limits on the previous pipeline
    aren't a concern on Tiingo paid tier (10k/hr). The Phase A name-sanity
    probe was a narrow 30-day window; the full Phase B fetch covers the
    entire 15-year history. If the full fetch returns 0 rows, mark
    `price_unavailable=True` and SKIP placeholder node creation. This
    self-corrects the Phase A narrow-probe flag.
  - One Tiingo `/tiingo/daily/{permaTicker}/prices` fetch per permaTicker.
    NO alias concatenation -- Tiingo back-merges the full rebrand-covered
    history under the permaTicker server-side, eliminating Phase B v2/v2.1
    "alias-union + non-stable-sort contamination" bug class (Class U/V).

- Storage path: `/sp400/{permaTicker}` (NEW). Legacy `/sp400/{canonical_ticker}`
  nodes from Phase B v1/v2/v2.1 are purged in a cleanup pass.

- Phase A's `price_unavailable` flag is treated as a HINT, not a gate.
  We fetch ALL 962 permaTickers per Q1. The post-fetch row-count check
  decides the official `price_unavailable` state (Q2), which is written
  back to /metadata/sp400_permatickers for Phase E downstream awareness.

Tiingo schema (NO local derivation needed)
------------------------------------------
Tiingo `/tiingo/daily/{permaTicker}/prices` returns rows shaped:
    {date, open, high, low, close, volume,
     adjOpen, adjHigh, adjLow, adjClose, adjVolume,
     divCash, splitFactor}

All adjusted columns are pre-computed by Tiingo (split + dividend
back-adjusted). Direct mapping to our db.h5 column schema:

    Tiingo            ->  db.h5 (OUTPUT_COLUMNS)
    date              ->  Date
    open              ->  Open
    high              ->  High
    low               ->  Low
    close             ->  Close
    volume            ->  Volume
    adjOpen           ->  Adj_Open
    adjHigh           ->  Adj_High
    adjLow            ->  Adj_Low
    adjClose          ->  Adj_Close
    adjVolume         ->  Adj_Volume

Usage:
    python 03_data_gathering.py
"""
import argparse
import json
import os
import sys
import io
import time
from datetime import datetime, timedelta
from pathlib import Path

# Windows console cp1252 support (matches 02b).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import requests
from dotenv import load_dotenv

# ==============================================================================
# CONFIGURATION
# ==============================================================================
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
if not TIINGO_API_KEY:
    raise ValueError(
        "TIINGO_API_KEY not found in .env. 03 fetches price history from "
        "Tiingo's /tiingo/daily/{permaTicker}/prices endpoint (paid tier)."
    )

DB_FILE = Path(__file__).parent / "db.h5"
PERMATICKERS_KEY = "/metadata/sp400_permatickers"

HISTORY_YEARS = 15
START_DATE = (datetime.now() - timedelta(days=HISTORY_YEARS * 365)).strftime("%Y-%m-%d")
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

H5_GROUP = "sp400"

# Output column order -- matches the prior Tiingo-era storage so downstream
# feature builder / discovery notebooks can read /sp400/{permaTicker}
# unchanged with respect to column names. Only the key path changes
# (permaTicker vs. canonical_ticker).
OUTPUT_COLUMNS = [
    "Date",
    "Open", "High", "Low", "Close", "Volume",
    "Adj_Open", "Adj_High", "Adj_Low", "Adj_Close", "Adj_Volume",
]

# Tiingo's /prices endpoint max rows per call: documented around 5000 rows
# (~20 years of business days). Our 15-year fetch is well within this.
TIINGO_TIMEOUT = 60


# ==============================================================================
# PART 1: PERMATICKER UNIVERSE FROM /metadata/sp400_permatickers
# ==============================================================================
def get_all_permatickers() -> list[dict]:
    """Return the full list of permaTicker dicts from
    /metadata/sp400_permatickers (Phase A output).

    Each dict carries the fields the data fetcher needs:
        permaTicker        : str  -- PRIMARY KEY + /sp400/{permaTicker} path key
        canonical_ticker   : str  -- informational (EODHD calendar join key for Phase D)
        name               : str  -- informational; included for logging
        isActive           : bool -- informational
        sic                : str  -- informational
        index_ref          : str  -- informational
        wikipedia_intervals: list[{added, removed}] -- optional filter.
                                     03 fetches full 15-year history by default
                                     (`--no-window`/`--wiki-window` flags below);
                                     users typically want the full history to
                                     support permaTicker-keyed feature building
                                     even pre-Wikipedia-residency era. We
                                     ignore wikipedia_intervals for the storage
                                     window but store the permaTicker with the
                                     full date range Tiingo returns.
        price_unavailable  : bool -- Phase A narrow-probe hint. Treated as a
                                     HINT, not a gate. If True, we still attempt
                                     the fetch (Q1); the row-count check on the
                                     returned data decides the final state.
    """
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"{DB_FILE} not found. Run 01_metadata -> 02 -> 02b_build_company_map.py first."
        )
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if PERMATICKERS_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {PERMATICKERS_KEY} missing in {DB_FILE}. "
                "Run 02b_build_company_map.py (Phase A) first -- it produces "
                "/metadata/sp400_permatickers and is the Phase B input."
            )
        df = pd.read_hdf(DB_FILE, key=PERMATICKERS_KEY)

    out = []
    for _, row in df.iterrows():
        try:
            ivs = (
                json.loads(row["wikipedia_intervals"])
                if isinstance(row["wikipedia_intervals"], str)
                else row["wikipedia_intervals"]
            )
        except Exception:
            ivs = []
        out.append(
            {
                "permaTicker": str(row["permaTicker"]),
                "canonical_ticker": None if pd.isna(row.get("canonical_ticker")) else str(row["canonical_ticker"]),
                "name": None if pd.isna(row.get("name")) else str(row["name"]),
                "isActive": bool(row["isActive"]),
                "sic": None if pd.isna(row.get("sic")) else str(row["sic"]),
                "index_ref": None if pd.isna(row.get("index_ref")) else str(row["index_ref"]),
                "wikipedia_intervals": ivs,
                "price_unavailable": bool(row["price_unavailable"]),
            }
        )
    return out


# ==============================================================================
# PART 2: DATA FETCHER (Tiingo permaTicker-keyed)
# ==============================================================================
_TIINGO_HEADERS = {"Content-Type": "application/json"}


def fetch_from_tiingo(permaTicker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch Tiingo /prices rows for one permaTicker in [start, end].

    Returns an empty DataFrame on any error (network, non-200, empty body,
    missing columns). Empty DataFrame triggers the caller's "skip
    placeholder" path.

    Tiingo's `/prices` endpoint natively returns adjusted OHLC+Volume -- no
    local derivation needed. We map directly to OUTPUT_COLUMNS.

    Conservation of data: Tiingo returns a single `date` field per row with
    ISO 8601 format and a 'T00:00:00.000Z' suffix. We strip to date-only,
    timezone-naive -- matches our current /sp400 schema and the existing
    feature builder expectation.
    """
    url = f"https://api.tiingo.com/tiingo/daily/{requests.utils.quote(permaTicker)}/prices"
    params = {
        "token": TIINGO_API_KEY,
        "startDate": start,
        "endDate": end,
    }
    try:
        response = requests.get(url, params=params, headers=_TIINGO_HEADERS, timeout=TIINGO_TIMEOUT)
    except Exception:
        return pd.DataFrame()

    if response.status_code != 200:
        return pd.DataFrame()

    try:
        data = response.json()
    except Exception:
        return pd.DataFrame()

    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Sanity: required Tiingo columns must be present.
    required = ["date", "open", "high", "low", "close", "volume",
                "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    out = pd.DataFrame()
    # Tiingo's `date` is an ISO 8601 string like "2021-12-01T00:00:00.000Z".
    # Truncate to date-only via pd.to_datetime, then strip the tz.
    out["Date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    out["Open"] = df["open"].astype(float)
    out["High"] = df["high"].astype(float)
    out["Low"] = df["low"].astype(float)
    out["Close"] = df["close"].astype(float)
    out["Volume"] = df["volume"].astype(float)
    out["Adj_Open"] = df["adjOpen"].astype(float)
    out["Adj_High"] = df["adjHigh"].astype(float)
    out["Adj_Low"] = df["adjLow"].astype(float)
    out["Adj_Close"] = df["adjClose"].astype(float)
    out["Adj_Volume"] = df["adjVolume"].astype(float)

    out = out[OUTPUT_COLUMNS]

    # Drop rows with NaN in the adjusted columns (shouldn't happen but
    # defensive -- matches the prior Pipeline behavior).
    out = out.dropna(subset=["Adj_Close", "Adj_Volume"]).reset_index(drop=True)
    # Stable sort by Date, dedup-keep-last (Tiingo normally returns one row
    # per date -- the dedup is purely defensive)
    out = out.sort_values("Date", kind="mergesort").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    return out


# ==============================================================================
# PART 3: STORAGE HELPERS
# ==============================================================================
def store_data(permaTicker: str, data: pd.DataFrame, group: str = H5_GROUP):
    """Overwrite a single /sp400/{permaTicker} node, leaving the rest of
    db.h5 untouched. Uses HDFStore mode='a' + store.remove() (never
    mode='w' on the whole DB file -- documented write-safety pattern).
    """
    h5_path = f"/{group}/{permaTicker}"
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if h5_path in store:
            store.remove(h5_path)
        store.put(h5_path, data, format="table", data_columns=["Date"])


def node_exists(permaTicker: str, group: str = H5_GROUP) -> bool:
    if not DB_FILE.exists():
        return False
    with pd.HDFStore(DB_FILE, mode="r") as store:
        return f"/{group}/{permaTicker}" in store.keys()


def get_latest_date(permaTicker: str, group: str = H5_GROUP) -> pd.Timestamp | None:
    if not node_exists(permaTicker, group=group):
        return None
    with pd.HDFStore(DB_FILE, mode="r") as store:
        df = store[f"/{group}/{permaTicker}"]
        if df.empty:
            return None
        latest = df["Date"].max()
        if hasattr(latest, "tz") and latest.tz is not None:
            latest = latest.tz_localize(None)
        return latest


def remove_node(permaTicker: str, group: str = H5_GROUP):
    """Remove a single /sp400/{permaTicker} node (used by stale-node cleanup)."""
    h5_path = f"/{group}/{permaTicker}"
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if h5_path in store:
            store.remove(h5_path)


# ==============================================================================
# PART 4: PER-PERMATICKER FETCH+STORE
# ==============================================================================
def update_permaTicker_node(pt_row: dict, group: str = H5_GROUP) -> tuple[str, int, bool]:
    """Fetch + store Tiingo's full history for one permaTicker row.

    Returns (status_str, n_rows, price_unavailable_final).
        status_str: "stored" | "skipped_empty" | "skipped_unavailable"
        n_rows:     number of rows stored (0 on skip)
        final_price_unavailable: True if Tiingo actually returned 0 rows
                                  (Q1-Q2 design: post-fetch self-correction
                                  of Phase A's narrow probe flag).
    """
    pt = pt_row["permaTicker"]
    name = pt_row.get("name") or "<no-name>"
    a_hint = " [Phase-A-flagged unavailable]" if pt_row["price_unavailable"] else ""
    print(f"  {pt} ({name}){a_hint}: fetching {START_DATE}..{END_DATE} ...", flush=True)

    data = fetch_from_tiingo(pt, START_DATE, END_DATE)
    n_rows = len(data)

    if n_rows == 0:
        # Tiingo genuinely has no rows for this permaTicker in the full
        # 15-year window (Phase A's narrow probe flag was correct, OR the
        # permaTicker is genuinely not in Tiingo's covered data). Skip
        # placeholder node creation, mark price_unavailable=True.
        if node_exists(pt, group=group):
            remove_node(pt, group=group)
            print(f"     -> 0 rows. Purged stale /sp400/{pt} (if any).")
        else:
            print(f"     -> 0 rows. No node created (price_unavailable=True).")
        return "skipped_empty", 0, True

    # Got rows. Store under /sp400/{permaTicker}.
    store_data(pt, data, group=group)
    # Final flag: False (Tiingo has data for this permaTicker in the full
    # 15-year window -- the Q1 self-correction).
    print(f"     -> stored {n_rows} rows under /sp400/{pt}")
    return "stored", n_rows, False


# ==============================================================================
# PART 5: STALE-NODE CLEANUP (PURGE LEGACY /sp400/{canonical_ticker})
# ==============================================================================
def cleanup_stale_nodes(live_permaTickers: set[str], group: str = H5_GROUP) -> list[str]:
    """Phase-B-permaTicker-migration cleanup pass: remove /sp400/{KEY} nodes
    whose KEY is no longer in `live_permaTickers`. These are leftovers from
    the pre-Phase-B-migration (Phase B v1/v2/v2.1) where the key was
    canonical_ticker (e.g. AAP, ENOV, etc.) instead of permaTicker.

    Idempotent and assertive.
    """
    if not DB_FILE.exists():
        return []

    with pd.HDFStore(DB_FILE, mode="r") as store:
        stored_keys = [k for k in store.keys() if k.startswith(f"/{group}/")]

    removed = []
    for k in stored_keys:
        # Strip the /sp400/ prefix to get the key.
        key = k.replace(f"/{group}/", "", 1)
        # Defensive: don't touch nested keys (none in current schema,
        # but if anyone added sub-groups we wouldn't want to nuke them).
        if "/" in key:
            continue
        if key in live_permaTickers:
            continue
        remove_node(key, group=group)
        removed.append(key)
    return removed


# ==============================================================================
# PART 6: WRITE-BACK final price_unavailable state to /metadata/sp400_permatickers
# ==============================================================================
def write_back_availability(
    final_unavailable: dict[str, bool],
) -> None:
    """Update /metadata/sp400_permatickers.price_unavailability to the final
    state computed during Phase B's full-history fetch.

    `final_unavailable` is dict[permaTicker -> bool] for all permaTickers
    actually fetched in this run (permaTickers that raised an exception
    before reaching the row-count check are NOT included here -- they retain
    their prior flag state).

    The write-back is the Q1 design -- Phase A's flag was a narrow-probe
    hint; Phase B's actual row counts are authoritative.
    """
    if not final_unavailable:
        return
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if PERMATICKERS_KEY not in store:
            print(f"[WARN] Cannot write-back price_unavailable: "
                  f"{PERMATICKERS_KEY} not in db.h5")
            return
        df = store[PERMATICKERS_KEY]
        # Apply updates per permaTicker (PRIMARY KEY).
        updates = 0
        for pt, flag in final_unavailable.items():
            mask = df["permaTicker"] == pt
            if mask.any():
                df.loc[mask, "price_unavailable"] = bool(flag)
                updates += 1
        if updates:
            store.remove(PERMATICKERS_KEY)
            store.put(PERMATICKERS_KEY, df, format="table")
            print(f"\n[WRITEBACK] Updated price_unavailable on {updates} "
                  f"/metadata/sp400_permatickers rows.")
        else:
            print(f"\n[WRITEBACK] No rows needed updating.")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase B: Tiingo permaTicker fetch")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="If >0, limit the number of permaTickers fetched (dry-run/smoke).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  DATA GATHERING - S&P 400 (Phase B: permaTicker / Tiingo)")
    print("=" * 70)
    print(f"  History:   {HISTORY_YEARS} years ({START_DATE} .. {END_DATE})")
    print(f"  Source:    Tiingo /tiingo/daily/{{permaTicker}}/prices")
    print(f"  Storage:   /sp400/{{permaTicker}}  (NEW path)")
    print(f"  Throttle:  none (paid tier 10k/hr)")
    print("=" * 70)

    pt_rows = get_all_permatickers()
    n_total = len(pt_rows)
    print(f"\n[INFO] Universe: {n_total} permaTickers (from {PERMATICKERS_KEY})")
    if args.limit > 0:
        pt_rows = pt_rows[:args.limit]
        print(f"[INFO] --limit={args.limit}: fetching only first {len(pt_rows)} (smoke mode)")
    if n_total == 0:
        print("[INFO] Nothing to do.")
        return

    n_unavail_hint = sum(1 for p in pt_rows if p.get("price_unavailable"))
    print(f"[INFO] Phase-A-flagged price_unavailable: {n_unavail_hint} "
          f"(treated as HINT -- will attempt full fetch per Q1)")
    print(f"[INFO] Effective fetch universe: {len(pt_rows)} permaTickers "
          f"(ALL, ignoring the hint per Q1)")

    progress_every = max(1, len(pt_rows) // 20)
    t0 = time.time()
    done = skipped_empty = 0
    failed = 0
    final_unavailable: dict[str, bool] = {}

    for i, pt_row in enumerate(pt_rows):
        try:
            status, n_rows, final_flag = update_permaTicker_node(pt_row)
            if status == "stored":
                done += 1
            elif status == "skipped_empty":
                skipped_empty += 1
            final_unavailable[pt_row["permaTicker"]] = final_flag
        except Exception as e:
            failed += 1
            print(f" [ERROR] /sp400/{pt_row['permaTicker']}: {e}")
        if (i + 1) % progress_every == 0 or (i + 1) == len(pt_rows):
            elapsed = time.time() - t0
            print(
                f"[PROGRESS] {i + 1}/{len(pt_rows)} permaTickers  |  "
                f"stored={done}, empty={skipped_empty}, failed={failed}  |  "
                f"elapsed={elapsed:.1f}s, eta={(len(pt_rows)-i-1)/max((i+1)/max(elapsed,0.001),0.001):.1f}s"
            )

    # Write-back final price_unavailable state (Q1 self-correction).
    write_back_availability(final_unavailable)

    # Cleanup pass: purge /sp400/{KEY} nodes whose KEY is NOT a current
    # permaTicker (legacy canonical_ticker-keyed nodes from the pre-
    # migration storage schema).
    print("\n[CLEANUP] Purging legacy /sp400/{canonical_ticker} nodes "
          "(not in live permaTicker set)...")
    removed = cleanup_stale_nodes({p["permaTicker"] for p in pt_rows}, group=H5_GROUP)
    if removed:
        print(f"[CLEANUP] Purged {len(removed)} legacy /sp400/* nodes:")
        for k in removed[:50]:
            print(f"    /sp400/{k}")
        if len(removed) > 50:
            print(f"    ... and {len(removed)-50} more")
    else:
        print("[CLEANUP] No legacy nodes to purge.")

    print()
    elapsed = time.time() - t0
    print("=" * 70)
    print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Database: {DB_FILE}")
    print(f"  Total permaTickers:      {len(pt_rows)}")
    print(f"  Stored (had rows):       {done}")
    print(f"  Skipped (0 rows):        {skipped_empty}")
    print(f"  Failed (exception):      {failed}")
    print(f"  Final price_unavailable:  "
          f"{sum(1 for v in final_unavailable.values() if v)}")
    print(f"  Legacy nodes purged:      {len(removed)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

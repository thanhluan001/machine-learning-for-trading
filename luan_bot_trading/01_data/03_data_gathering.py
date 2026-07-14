#!/usr/bin/env python3
"""
Data Gathering - S&P 400 Mid-Cap Universe (perm_id-Level, EODHD)
================================================================

Phase B rewrite (2026-07-14): migrates the data fetcher from the deprecated
/metadata/sp400_companies (per-CIK collapsing, dropped in Phase A) to the new
/metadata/sp400_perm_ids (per-perm_id, point-in-time-CIK anchored, interval-
forked). See 01_data/merger_identity_patch.md and database_layout.md for the
Phase A design and the survivor-CIK collision bug that motivated it.

Fetches full EODHD adjusted OHLCV history for every perm_id that has ever
been a constituent of the S&P 400 (current + removed), per
/metadata/sp400_perm_ids in db.h5.

Rules
-----
- Iterate per **perm_id** (not per ticker, not per CIK). For each perm_id:
  - If ``price_unavailable=True`` -> SKIP entirely (no placeholder node).
  - Otherwise, fetch each alias from EODHD (as {ticker}.US) and **concatenate**
    the responses on Date, dedup-keep-last. This is REQUIRED because EODHD
    does NOT retro-relabel: when a company rebrands, EODHD keeps the old
    ticker series as a dead series (ending at the rebrand day) and starts a
    new series under the new ticker -- they are NOT concatenated by EODHD.
    Picking the first-non-empty alias alone (the pre-Phase-B approach) loses
    the pre- OR post-rebrand segment. Concatenating all aliases preserves
    the full perm_id history across rebrands.
    The concatenated series is stored under /sp400/{canonical_ticker};
    aliases are NOT stored individually. Empirically validated:
      * CHK + EXE  (Chesapeake -> Expand Energy): 2201 + 1360 = 2645 rows
        2016-2026, single-alias fetch would have lost 841 rows either way.
      * SYNH + INCR (Syneos): 2238 + 1738 = 2934 rows, 2014-2026.
      * ESV + VAL  (Valaris): 2157 + 1304 = 3461 rows, 2011-2026.
      * FTR + FYBR (Frontier): 2346 + 1185 = 3531 rows, 2011-2026.
      * LNW + SGMS + LAWIL (Light&Wonder): 3846 + 2893 + 1 = 3846 rows.

- Always fetch full 15-year history per alias; no partial-range logic. EODHD
  is effectively unlimited -- no throttle, no batching, no offset checkpoint.
- Stale /sp400/{TICKER} nodes whose TICKER is no longer canonical for any
  perm_id are PURGED at the end of the run (leftover from the pre-Phase-A
  /metadata/sp400_companies schema). E.g. pre-Phase-A canonicals ``DV``,
  ``POL``, ``SGMS``, ``CHK``, ``ZI`` are now non-canonical aliases of
  perm_ids whose canonical is the post-rebrand ticker (CVSA, AVNT, LAWIL,
  EXE, GTM respectively).

EODHD schema adaptation
-----------------------
EODHD ``/api/eod/{TICKER}.US`` returns rows shaped:
    {date, open, high, low, close, adjusted_close, volume}

It does NOT expose adj_open / adj_high / adj_low / adj_volume directly. We
derive all four locally via the close/adjusted_close ratio (encodes the
cumulative split + dividend reinvestment factor). This is exactly how Tiingo
computes `adjVolume` (same convention). Validated empirically in
``validate_eodhd_adjclose.py`` (7/7 probe tickers PASS). Purely local -- no
extra split/dividend lookup credits consumed.

Local derivation (per row):
    adj_factor  = close / adjusted_close     # cumulative split+div factor
    adj_open    = open    / adj_factor
    adj_high    = high    / adj_factor
    adj_low     = low    / adj_factor
    adj_close   = adjusted_close
    adj_volume  = volume  * adj_factor

Storage columns (matches prior Tiingo-era output for downstream feature
builder compatibility):
    Date, Open, High, Low, Close, Volume,
    Adj_Open, Adj_High, Adj_Low, Adj_Close, Adj_Volume

Usage:
    python 03_data_gathering.py
"""
import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Explicit .env path so the script works regardless of CWD (matches 02b).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
EODHD_API_KEY = os.getenv("EODHD_API_KEY")
if not EODHD_API_KEY:
    raise ValueError(
        "EODHD_API_KEY not found in .env. 03 fetches price history from EODHD "
        "(replacing Tiingo) for schema/stability alignment with the earnings pipeline."
    )

DB_FILE = Path(__file__).parent / "db.h5"
# Phase B: read perm_id view produced by 02b_build_company_map.py (Phase A).
PERM_IDS_KEY = "/metadata/sp400_perm_ids"

HISTORY_YEARS = 15
START_DATE = (datetime.now() - timedelta(days=HISTORY_YEARS * 365)).strftime("%Y-%m-%d")
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

H5_GROUP = "sp400"

# Output column order matches the prior Tiingo-era storage so downstream
# feature builder / discovery notebooks can read /sp400/{TICKER} unchanged.
# EODHD returns raw OHLC + adjusted_close + raw volume only; we derive the
# 4 adjusted columns locally (validated by validate_eodhd_adjclose.py).
OUTPUT_COLUMNS = [
    "Date",
    "Open", "High", "Low", "Close", "Volume",
    "Adj_Open", "Adj_High", "Adj_Low", "Adj_Close", "Adj_Volume",
]


# ==============================================================================
# PART 1: PERM_ID UNIVERSE FROM /metadata/sp400_perm_ids
# ==============================================================================
def get_all_perm_ids() -> list[dict]:
    """Return the full list of perm_id dicts from /metadata/sp400_perm_ids.

    Phase B replacement for the pre-Phase-A get_all_companies(). Each dict
    carries the fields the data fetcher needs:
        perm_id           : str         -- informational; not used as the
                                           storage key (we still key on
                                           canonical_ticker for backward
                                           compatibility with downstream
                                           feature builder path lookups).
        canonical_ticker  : str         -- storage key + first alias to try
                                           on EODHD.
        cik               : str | None  -- informational; written to /earnings
                                           by 06, not used by 03.
        aliases           : list[str]   -- EODHD fetch fallback order
                                           (Phase A insertion order -- earliest
                                           Wikipedia interval first).
        name              : str | None  -- informational
        sic               : str | None  -- informational
        index_ref         : str | None  -- informational
        combined_intervals: list[dict]  -- informational; Phase E feature
                                           builder will use this for row
                                           attribution.
        per_ticker_intervals: dict      -- informational; not used by 03.
        price_unavailable : bool        -- True -> skip fetch + skip node.
    """
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"{DB_FILE} not found. Run 01_metadata -> 02 -> 02b_build_company_map.py first."
        )
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if PERM_IDS_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {PERM_IDS_KEY} missing in {DB_FILE}. "
                "Run 02b_build_company_map.py (Phase A) first -- it produces "
                "/metadata/sp400_perm_ids and is the Phase B input."
            )
        df = pd.read_hdf(DB_FILE, key=PERM_IDS_KEY)

    perm_ids = []
    for _, row in df.iterrows():
        # Parse JSON-encoded aliases (Phase A stores as JSON str).
        try:
            aliases = (
                json.loads(row["aliases"]) if isinstance(row["aliases"], str) else row["aliases"]
            )
        except Exception:
            aliases = [row["canonical_ticker"]]
        if not aliases:
            aliases = [row["canonical_ticker"]]
        aliases = [str(a) for a in aliases]

        # Parse JSON-encoded combined_intervals (informational for 03 but
        # carried so we don't lose it if downstream stages want to read it
        # via the same dict shape).
        try:
            combined = (
                json.loads(row["combined_intervals"])
                if isinstance(row["combined_intervals"], str)
                else row["combined_intervals"]
            )
        except Exception:
            combined = []

        # per_ticker_intervals is a JSON object (dict); leave as-is string for
        # here -- 03 doesn't use it but other phases will read it from the
        #/metadata/sp400_perm_ids table directly, not via this list.
        perm_ids.append(
            {
                "perm_id": None if pd.isna(row.get("perm_id")) else str(row["perm_id"]),
                "canonical_ticker": str(row["canonical_ticker"]),
                "cik": None if pd.isna(row.get("cik")) else str(row["cik"]),
                "aliases": aliases,
                "name": None if pd.isna(row.get("name")) else str(row["name"]),
                "sic": None if pd.isna(row.get("sic")) else str(row["sic"]),
                "index_ref": None if pd.isna(row.get("index_ref")) else str(row["index_ref"]),
                "combined_intervals": combined,
                "price_unavailable": bool(row["price_unavailable"]),
            }
        )
    return perm_ids


# ==============================================================================
# PART 2: DATA FETCHER (EODHD)
# ==============================================================================
def fetch_from_eodhd(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch EOD daily rows from EODHD and derive adjusted OHLC+Volume locally.

    Returns an empty DataFrame on any error (network, non-200, empty body,
    missing columns). The caller tries the next alias on the empty case.
    """
    url = f"https://eodhd.com/api/eod/{ticker}.US"
    params = {
        "from": start,
        "to": end,
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "period": "d",
    }
    try:
        response = requests.get(url, params=params, timeout=60)
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
    for col in ("open", "high", "low", "close", "adjusted_close", "volume"):
        if col not in df.columns:
            return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])

    df_clean = pd.DataFrame()
    df_clean["Date"] = df["date"]
    df_clean["Open"] = df["open"].astype(float)
    df_clean["High"] = df["high"].astype(float)
    df_clean["Low"] = df["low"].astype(float)
    df_clean["Close"] = df["close"].astype(float)
    df_clean["Volume"] = df["volume"].astype(float)

    adj_close = df["adjusted_close"].astype(float)
    # Guard against zero/negative adjusted_close (shouldn't happen, but a NaN
    # in the ratio would silently corrupt the entire adj-derived block).
    adj_close_safe = adj_close.where(adj_close > 0)
    df_clean["adj_factor"] = (df_clean["Close"] / adj_close_safe).astype(float)

    df_clean["Adj_Open"] = df_clean["Open"] / df_clean["adj_factor"]
    df_clean["Adj_High"] = df_clean["High"] / df_clean["adj_factor"]
    df_clean["Adj_Low"] = df_clean["Low"] / df_clean["adj_factor"]
    df_clean["Adj_Close"] = adj_close
    df_clean["Adj_Volume"] = df_clean["Volume"] * df_clean["adj_factor"]

    df_clean = df_clean[OUTPUT_COLUMNS]

    if df_clean["Date"].dt.tz is not None:
        df_clean["Date"] = df_clean["Date"].dt.tz_localize(None)

    df_clean = df_clean.dropna(subset=["Adj_Close", "Adj_Volume"]).reset_index(drop=True)
    return df_clean


def fetch_concatenated_aliases_from_eodhd(aliases: list[str]) -> tuple[list[str], pd.DataFrame]:
    """Fetch each alias from EODHD and CONCATENATE the responses on Date,
    dedup-keep-last (latest-write-wins on overlap days).

    Required because EODHD does NOT retro-relabel rebrands. When a company
    rebrands (e.g. CHK -> EXE), EODHD keeps the old ticker ``CHK.US`` as a
    dead series ending at the rebrand day and starts a fresh ``EXE.US``
    series -- they are NOT concatenated server-side. Fetching only the
    first non-empty alias would lose the pre- OR post-rebrand segment.
    Concatenating all aliases preserves the full perm_id history.

    Returns (list-of-aliases-that-returned-data, concatenated_df).
    The concatenated df is sorted by Date and dedup-keep-last on Date.
    """
    pieces = []
    successful_aliases = []
    for alias in aliases:
        data = fetch_from_eodhd(alias, START_DATE, END_DATE)
        if not data.empty:
            pieces.append(data)
            successful_aliases.append(alias)
        # 404 / empty / network error -> try next alias (no throttle needed).
    if not pieces:
        return [], pd.DataFrame()
    big = pd.concat(pieces, ignore_index=True)
    big = big.sort_values("Date")
    # Dedup: on overlap days (rare -- rebrand day is usually a single row
    # duplicated across two alias series), keep the LAST. Last-write-wins is
    # safe because rebrand-day closes are byte-identical across the two
    # series for the same underlying security.
    big = big.drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    return successful_aliases, big


# ==============================================================================
# PART 3: STORAGE HELPERS
# ==============================================================================
def store_data(canonical_ticker: str, data: pd.DataFrame, group: str = H5_GROUP):
    """Overwrite a single /sp400/{canonical_ticker} node, leaving the rest
    of db.h5 untouched. Uses HDFStore mode='a' + store.remove() (never
    mode='w' on the whole DB file -- documented write-safety pattern).
    """
    h5_path = f"/{group}/{canonical_ticker}"
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if h5_path in store:
            store.remove(h5_path)
        store.put(h5_path, data, format="table", data_columns=["Date"])


def canonical_exists(canonical_ticker: str, group: str = H5_GROUP) -> bool:
    if not DB_FILE.exists():
        return False
    with pd.HDFStore(DB_FILE, mode="r") as store:
        return f"/{group}/{canonical_ticker}" in store.keys()


def get_latest_date(canonical_ticker: str, group: str = H5_GROUP) -> pd.Timestamp:
    with pd.HDFStore(DB_FILE, mode="r") as store:
        df = store[f"/{group}/{canonical_ticker}"]
        latest = df["Date"].max()
        if hasattr(latest, "tz") and latest.tz is not None:
            latest = latest.tz_localize(None)
        return latest


def remove_canonical(canonical_ticker: str, group: str = H5_GROUP):
    """Remove a stale /sp400/{canonical_ticker} node (used by Phase B's
    cleanup pass for tickers that are no longer canonical for any perm_id)."""
    h5_path = f"/{group}/{canonical_ticker}"
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if h5_path in store:
            store.remove(h5_path)


# ==============================================================================
# PART 4: UPDATE LOGIC (per perm_id)
# ==============================================================================
def _ordered_aliases(perm_id: dict) -> list[str]:
    """Return the EODHD fetch-order for the perm_id's aliases.

    We now fetch ALL aliases for each perm_id (concatenated on Date) -- not
    first-non-empty fallback. So the order doesn't affect which aliases are
    fetched, but it does affect the log/output ordering. We keep canonical
    first for output readability.
    """
    canonical = perm_id["canonical_ticker"]
    aliases = list(perm_id["aliases"] or [canonical])
    # Put canonical first; preserve remainder order.
    ordered = [canonical] + [a for a in aliases if a != canonical]
    # Dedup while preserving order.
    seen = set()
    out = []
    for a in ordered:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def update_perm_id(perm_id: dict, group: str = H5_GROUP):
    """Fetch + store full EODHD history for one perm_id (keyed by
    canonical_ticker) using alias fallback. See module docstring for rules.
    """
    canonical = perm_id["canonical_ticker"]
    ordered_aliases = _ordered_aliases(perm_id)

    if perm_id.get("price_unavailable"):
        print(f"  {canonical} (perm_id={perm_id.get('perm_id')}): SKIP "
              f"(price_unavailable=True). aliases={perm_id['aliases']}")
        # Also ensure no stale node lingers for a perm_id that became
        # unavailable post-rebuild (defensive cleanup).
        if canonical_exists(canonical, group=group):
            remove_canonical(canonical, group=group)
            print(f"      Purged stale /sp400/{canonical} node.")
        return

    # Fetch-once-and-store (no incremental / freshness branch needed now that
    # the subscription is unlimited and the canonical selection is the only
    # thing that changes between runs). We refetch the full 15y history every
    # run; cheap enough on EODHD and avoids stale-data bugs.
    if not DB_FILE.exists():
        # First-ever run: create DB in mode='w' (safe because DB_FILE absent).
        print(f"  {canonical}: Initial fetch (aliases: {ordered_aliases})...")
        used, data = fetch_concatenated_aliases_from_eodhd(ordered_aliases)
        if not data.empty:
            with pd.HDFStore(DB_FILE, mode="w") as store:
                store.put(f"/{group}/{canonical}", data, format="table", data_columns=["Date"])
            print(f"      Stored {len(data)} rows via aliases={used}")
        else:
            print(f"      No data from any alias.")
        return

    if not canonical_exists(canonical, group=group):
        print(f"  {canonical} (perm_id={perm_id.get('perm_id')}): New perm_id. "
              f"Fetching (aliases: {ordered_aliases})...")
        used, data = fetch_concatenated_aliases_from_eodhd(ordered_aliases)
        if not data.empty:
            store_data(canonical, data, group=group)
            print(f"      Stored {len(data)} rows via aliases={used}")
        else:
            print(f"      No data from any alias.")
        return

    # Existing node -- check freshness. If up-to-date, skip the refetch.
    latest_date = get_latest_date(canonical, group=group)
    if latest_date.strftime("%Y-%m-%d") >= END_DATE:
        print(f"  {canonical}: Up to date (thru {latest_date.strftime('%Y-%m-%d')}).")
        return

    gap = (datetime.strptime(END_DATE, "%Y-%m-%d") - latest_date).days
    print(f"  {canonical}: Refetch full history (gap={gap}d, aliases: {ordered_aliases})...")
    used, data = fetch_concatenated_aliases_from_eodhd(ordered_aliases)
    if not data.empty:
        store_data(canonical, data, group=group)
        print(f"      Stored {len(data)} rows via aliases={used}")
    else:
        # All aliases returned empty even though we had data previously.
        # Most likely cause: temporary EODHD failure or the ticker was
        # delisted mid-window. Do NOT purge the existing node -- preserve
        # historical data; logged for surfacing.
        print(f"      [WARN] All aliases returned empty; keeping existing "
              f"/sp400/{canonical} node.")


def cleanup_stale_nodes(live_canonicals: set[str], group: str = H5_GROUP):
    """Phase B cleanup pass: remove /sp400/{TICKER} nodes whose TICKER is no
    longer canonical for any perm_id. These are leftovers from the pre-Phase-A
    /metadata/sp400_companies schema where a different ticker was chosen
    canonical (e.g. DV subsumed under ATGE->CVSA, POL under AVNT, SGMS under
    LNW->LAWIL, CHK under EXE).

    Idempotent and assertive: leaves only the canonicals that match
    /metadata/sp400_perm_ids. Skips nodes whose key starts with the group
    prefix but isn't a ticker (none currently, but defensive).
    """
    if not DB_FILE.exists():
        return
    removed = []
    with pd.HDFStore(DB_FILE, mode="r") as store:
        stored_keys = [k for k in store.keys() if k.startswith(f"/{group}/")]
    for k in stored_keys:
        tick = k.replace(f"/{group}/", "", 1)
        if tick in live_canonicals:
            continue
        # Defensive: don't touch nested keys (none in current schema).
        if "/" in tick:
            continue
        remove_canonical(tick, group=group)
        removed.append(tick)
    if removed:
        print(f"\n[CLEANUP] Purged {len(removed)} stale /sp400/* nodes no longer "
              f"canonical for any perm_id:")
        for t in removed:
            print(f"    /sp400/{t}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("  DATA GATHERING - S&P 400 Historical Universe (Phase B: perm_id)")
    print("=" * 60)
    print(f"  History:   {HISTORY_YEARS} years ({START_DATE} .. {END_DATE})")
    print(f"  Source:    EODHD /api/eod (full history, canonical-first alias fallback)")
    print(f"  Throttle:  none (unlimited subscription)")
    print("=" * 60)

    perm_ids = get_all_perm_ids()
    n_total = len(perm_ids)
    print(f"\n[INFO] Universe: {n_total} perm_ids (from {PERM_IDS_KEY})")
    if n_total == 0:
        print("[INFO] Nothing to do.")
        return
    n_unavail = sum(1 for p in perm_ids if p.get("price_unavailable"))
    print(f"[INFO] price_unavailable=True: {n_unavail} (skip + purge stale node)")
    print(f"[INFO] Effective fetch universe: {n_total - n_unavail}")

    progress_every = max(1, n_total // 20)  # log ~20 progress steps
    t0 = time.time()
    done = skipped = failed = 0

    for i, pid in enumerate(perm_ids):
        canonical = pid["canonical_ticker"]
        try:
            update_perm_id(pid)
            if pid.get("price_unavailable"):
                skipped += 1
            else:
                done += 1
        except Exception as e:
            failed += 1
            print(f" [ERROR] Failed to update {canonical} (perm_id={pid.get('perm_id')}): {e}")

        if (i + 1) % progress_every == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            print(
                f"[PROGRESS] {i + 1}/{n_total} perm_ids  |  "
                f"stored={done}, skipped={skipped}, failed={failed}  |  "
                f"elapsed={elapsed:.1f}s"
            )

    # Cleanup pass: purge /sp400/{TICKER} nodes whose TICKER is no longer
    # canonical for any perm_id (Phase B schema-migration cleanup).
    live_canonicals = {p["canonical_ticker"] for p in perm_ids}
    cleanup_stale_nodes(live_canonicals, group=H5_GROUP)

    print()
    elapsed = time.time() - t0
    print("=" * 60)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Database: {DB_FILE}")
    print(f"  Total perm_ids:  {n_total}")
    print(f"  Stored/Refreshed: {done}")
    print(f"  Skipped (unavail): {skipped}")
    print(f"  Failed:           {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()

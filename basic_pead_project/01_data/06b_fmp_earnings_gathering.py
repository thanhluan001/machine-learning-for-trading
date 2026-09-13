#!/usr/bin/env python3
"""
FMP Earnings Gathering — 06b
=============================

Fetches full historical earnings from FMP's `/stable/earnings` endpoint
and stores under `/earnings/fmp` in db.h5. Replaces EODHD's
`/earnings/raw` (Phase D) with richer data (revenue estimates, BMO/AMC
in clean "bmo"/"amc" format, fiscalPeriod labels, 41 years of history).

FMP $49/mo plan. Uses `/stable/earnings?symbol={ticker}&includeReportTimes=true`.
Data source confirmed per `feature_sourcing_audit.md` §4A.

Key differences from EODHD (06):
  - FMP uses regular tickers (AAPL, AOS), NOT permaTickers (US000...).
    We iterate permaTickers from /metadata/sp400_permatickers and join
    via canonical_ticker for the API call.
  - FMP returns 41 years of history (1985-present) vs EODHD's 15 years.
  - FMP includes revenueActual + revenueEstimated (EODHD doesn't).
  - FMP's `time` field is "bmo"/"amc" (clean) vs EODHD's CamelCase.
  - FMP's fiscalPeriod ("Q3") and fiscalYear (2026) are separate fields.
  - FMP does NOT return `difference` or `percent` -- we derive them.
  - FMP does NOT return `currency` -- we assume USD for S&P 400.

Output schema (writes to `/earnings/fmp`):
    permaTicker         : str        -- PRIMARY key (from /metadata/sp400_permatickers)
    canonical_ticker    : str        -- the ticker used for the FMP API call
    cik                  : str|None   -- from /metadata/sp400_permatickers
    report_date          : datetime   -- announcement date (FMP: `date`)
    period_ending        : datetime   -- fiscal quarter end (FMP: `periodEnding`)
    fiscal_period        : str        -- "Q1".."Q4" (FMP: `fiscalPeriod`)
    fiscal_year          : int        -- e.g. 2026 (FMP: `fiscalYear`)
    eps_actual            : float|None -- reported EPS (FMP: `epsActual`)
    eps_estimated         : float|None -- consensus estimate (FMP: `epsEstimated`)
    eps_difference        : float|None -- derived: eps_actual - eps_estimated
    eps_surprise_pct      : float|None -- derived: (actual-est)/est * 100
    revenue_actual        : float|None -- reported revenue (FMP: `revenueActual`)
    revenue_estimated     : float|None -- consensus revenue (FMP: `revenueEstimated`)
    revenue_difference    : float|None -- derived: rev_actual - rev_estimated
    revenue_surprise_pct  : float|None -- derived: (rev_actual-rev_est)/rev_est * 100
    before_after_market   : str        -- "bmo" or "amc" (FMP: `time`)
    confirmed             : bool       -- whether earnings date is confirmed (FMP: `confirmed`)
    last_updated          : datetime   -- FMP data freshness (FMP: `lastUpdated`)

Dedup key: (permaTicker, report_date) -- one row per earnings event.
Tiebreak: keep the row with the latest `lastUpdated` (most recent data).

Usage:
    python 06b_fmp_earnings_gathering.py
    python 06b_fmp_earnings_gathering.py --limit 10   # smoke test
"""
import argparse
import io
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Windows console cp1252 support
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    raise ValueError(
        "FMP_API_KEY not found in .env. Add it before running this script."
    )

DB_FILE = Path(__file__).parent / "db.h5"
PERMATICKERS_KEY = "/metadata/sp400_permatickers"
EARNINGS_KEY = "/earnings/fmp"

FMP_BASE = "https://financialmodelingprep.com/stable"
FMP_TIMEOUT = 30


# ==============================================================================
# PART 1: LOAD PERMATICKER UNIVERSE
# ==============================================================================
def load_permatickers() -> pd.DataFrame:
    """Load /metadata/sp400_permatickers. Returns permaTicker, canonical_ticker,
    cik, price_unavailable for all 962 rows."""
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"{DB_FILE} not found. Run 02b_build_company_map.py (Phase A) first."
        )
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if PERMATICKERS_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {PERMATICKERS_KEY} missing in {DB_FILE}. "
                "Run 02b_build_company_map.py (Phase A) first."
            )
        return store[PERMATICKERS_KEY]


# ==============================================================================
# PART 2: FMP EARNINGS FETCHER
# ==============================================================================
def fetch_fmp_earnings(canonical_ticker: str) -> list[dict]:
    """Fetch /stable/earnings for one ticker with includeReportTimes=true.

    Returns the full historical earnings list (up to ~164 entries per ticker,
    going back to 1985 for well-established companies).
    """
    url = f"{FMP_BASE}/earnings"
    params = {
        "symbol": canonical_ticker,
        "apikey": FMP_API_KEY,
        "includeReportTimes": "true",
    }
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

    if not isinstance(data, list) or not data:
        return []

    return data


def _to_float(v):
    """Safely convert to float, returning None for NaN/null."""
    if v is None:
        return None
    try:
        f = float(v)
        if pd.isna(f):
            return None
        return f
    except Exception:
        return None


def _derive_difference(actual, estimated):
    """actual - estimated. Returns None if either is None."""
    if actual is None or estimated is None:
        return None
    return actual - estimated


def _derive_surprise_pct(actual, estimated):
    """(actual - estimated) / |estimated| * 100. Returns None if estimated is 0 or None."""
    if actual is None or estimated is None or estimated == 0:
        return None
    return (actual - estimated) / abs(estimated) * 100


def normalize_fmp_rows(
    raw_rows: list[dict],
    permaTicker: str,
    canonical_ticker: str,
    cik: str | None,
) -> list[dict]:
    """Convert FMP raw rows to the stored schema, joined with permaTicker."""
    out = []
    for r in raw_rows:
        date_str = r.get("date")
        if date_str is None:
            continue

        eps_actual = _to_float(r.get("epsActual"))
        eps_estimated = _to_float(r.get("epsEstimated"))
        rev_actual = _to_float(r.get("revenueActual"))
        rev_estimated = _to_float(r.get("revenueEstimated"))

        # FMP's `time` field: "bmo" or "amc" (clean, no CamelCase parsing)
        bam = r.get("time")
        if bam and isinstance(bam, str):
            bam = bam.strip().lower()

        # FMP's fiscalYear as int (may come as string)
        fy = r.get("fiscalYear")
        try:
            fy = int(fy) if fy is not None else None
        except (ValueError, TypeError):
            fy = None

        out.append({
            "permaTicker": permaTicker,
            "canonical_ticker": canonical_ticker,
            "cik": cik,
            "report_date": pd.to_datetime(date_str, errors="coerce"),
            "period_ending": pd.to_datetime(r.get("periodEnding"), errors="coerce"),
            "fiscal_period": r.get("fiscalPeriod"),
            "fiscal_year": fy,
            "eps_actual": eps_actual,
            "eps_estimated": eps_estimated,
            "eps_difference": _derive_difference(eps_actual, eps_estimated),
            "eps_surprise_pct": _derive_surprise_pct(eps_actual, eps_estimated),
            "revenue_actual": rev_actual,
            "revenue_estimated": rev_estimated,
            "revenue_difference": _derive_difference(rev_actual, rev_estimated),
            "revenue_surprise_pct": _derive_surprise_pct(rev_actual, rev_estimated),
            "before_after_market": bam,
            "confirmed": bool(r.get("confirmed", False)),
            "last_updated": pd.to_datetime(r.get("lastUpdated"), errors="coerce"),
        })
    return out


# ==============================================================================
# PART 3: DEDUPLICATION
# ==============================================================================
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates keyed by (permaTicker, report_date), keeping the row
    with the latest lastUpdated (most recent data).

    FMP occasionally returns duplicate entries for the same earnings event
    (e.g. when estimates get revised). We keep the freshest row.
    """
    if df.empty:
        return df

    n_pre = len(df)
    df = df.sort_values(
        ["permaTicker", "report_date", "last_updated"],
        kind="mergesort",
        na_position="last",
    )
    df = df.drop_duplicates(subset=["permaTicker", "report_date"], keep="last")
    n_post = len(df)
    if n_pre != n_post:
        print(f"     [dedup] {n_pre} -> {n_post} rows ({n_pre - n_post} dups removed)")
    return df.reset_index(drop=True)


# ==============================================================================
# PART 4: STORAGE
# ==============================================================================
def store_earnings(df: pd.DataFrame) -> None:
    """Persist /earnings/fmp. Uses HDFStore(mode='a') + store.remove()
    (never mode='w' on existing DB)."""
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if EARNINGS_KEY in store.keys():
            store.remove(EARNINGS_KEY)
        store.put(
            EARNINGS_KEY,
            df,
            format="table",
            data_columns=["permaTicker", "report_date", "canonical_ticker"],
        )


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="FMP Earnings Gathering (replaces EODHD)")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="If >0, limit the number of permaTickers fetched (smoke test).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  06b - FMP Earnings Gathering (replaces EODHD)")
    print("=" * 70)
    print(f"  Source:    FMP /stable/earnings?includeReportTimes=true")
    print(f"  Storage:   /earnings/fmp  (NEW path; /earnings/raw kept as backup)")
    print(f"  History:   41 years (FMP returns back to 1985)")
    print(f"  Dedup key: (permaTicker, report_date) -- keep latest lastUpdated")
    print(f"  Derive:    eps_difference, eps_surprise_pct, revenue_difference,")
    print(f"             revenue_surprise_pct (FMP does not return these)")
    print("=" * 70)

    print(f"\n[1/3] Loading {PERMATICKERS_KEY} ...")
    pt_df = load_permatickers()

    # Only fetch permaTickers with price_unavailable=False (matching 03_data_gathering.py)
    fetchable = pt_df[~pt_df["price_unavailable"]].copy()
    n_total = len(fetchable)
    print(f"   {len(pt_df)} total permaTickers; {n_total} fetchable (price_unavailable=False)")

    if args.limit > 0:
        fetchable = fetchable.head(args.limit)
        print(f"   --limit={args.limit}: fetching only first {len(fetchable)} (smoke mode)")

    print(f"\n[2/3] Fetching FMP earnings per permaTicker ...")

    all_rows: list[dict] = []
    n_with_events = 0
    n_zero_events = 0
    n_failed = 0
    t0 = time.time()
    progress_every = max(1, len(fetchable) // 20)

    for i, (_, row) in enumerate(fetchable.iterrows()):
        pt = str(row["permaTicker"])
        ct = str(row["canonical_ticker"])
        cik = str(row["cik"]) if pd.notna(row.get("cik")) else None
        name = str(row.get("name", ""))[:25]

        try:
            raw = fetch_fmp_earnings(ct)
            if raw:
                rows = normalize_fmp_rows(raw, permaTicker=pt, canonical_ticker=ct, cik=cik)
                all_rows.extend(rows)
                n_with_events += 1
            else:
                n_zero_events += 1
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"   [ERROR] {pt} ({ct}): {e}")

        if (i + 1) % progress_every == 0 or (i + 1) == len(fetchable):
            elapsed = time.time() - t0
            print(
                f"   [PROGRESS] {i+1}/{len(fetchable)}  "
                f"with_events={n_with_events}  zero={n_zero_events}  "
                f"failed={n_failed}  cum_rows={len(all_rows)}  "
                f"elapsed={elapsed:.1f}s"
            )

        # FMP rate limit: paid plan is generous, minimal sleep for OS hygiene
        time.sleep(0.02)

    print(f"\n   With events:     {n_with_events}")
    print(f"   Zero events:     {n_zero_events}")
    print(f"   Failed:           {n_failed}")
    print(f"   Total raw rows:   {len(all_rows)}")

    print(f"\n[3/3] Deduplicating + storing {EARNINGS_KEY} ...")
    if all_rows:
        df = pd.DataFrame(all_rows)
        df = deduplicate(df)
        store_earnings(df)
        print(f"   Wrote {EARNINGS_KEY}: {len(df)} rows x {len(df.columns)} cols")
    else:
        print("   [WARN] No earnings rows fetched. Nothing to store.")

    # ---- Audit ----
    print("\n" + "=" * 70)
    print("  FMP EARNINGS GATHERING AUDIT")
    print("=" * 70)
    print(f"  permaTickers fetchable:   {n_total}")
    print(f"  With events:               {n_with_events}")
    print(f"  Zero events:               {n_zero_events}")
    print(f"  Failed (exception):        {n_failed}")
    print(f"  Total stored rows:         {len(all_rows)}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = deduplicate(df)
        print(f"  After dedup:               {len(df)}")
        print(f"  Date range:                {df['report_date'].min()} -> {df['report_date'].max()}")
        print(f"  Unique permaTickers:       {df['permaTicker'].nunique()}")
        # Check BMO/AMC coverage
        bam_valid = df["before_after_market"].notna().sum()
        print(f"  BMO/AMC coverage:          {bam_valid}/{len(df)} ({bam_valid/len(df)*100:.1f}%)")
        # Revenue coverage
        rev_cov = df["revenue_actual"].notna().sum()
        print(f"  Revenue actual coverage:   {rev_cov}/{len(df)} ({rev_cov/len(df)*100:.1f}%)")
        # EPS actual coverage
        eps_cov = df["eps_actual"].notna().sum()
        print(f"  EPS actual coverage:       {eps_cov}/{len(df)} ({eps_cov/len(df)*100:.1f}%)")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

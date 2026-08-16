#!/usr/bin/env python3
"""69_insider_cluster_drift.py — Event study: SEC Form 4 insider buying drift on S&P 400.

RESEARCH QUESTION (Slow-week edge candidate)
--------------------------------------------
Do open-market insider purchases (SEC Form 4, Transaction Code P) — especially
cluster buying by multiple officers/directors — generate tradeable abnormal
drift on S&P 400 constituents? And does this edge provide sufficient supply in
off-earnings shoulder months (September, December, March, June)?

EXECUTION ASSUMPTIONS (look-ahead free)
---------------------------------------
- Filing Date: date the Form 4 was accepted by SEC EDGAR.
- Execution Point: Close[T+1] (the first trading close AFTER filing date).
- Returns: Benchmark-relative abnormal return (CAR) vs /macros/IJH.
- Horizons: 5d, 10d, 20d, 40d, 60d forward trading days.

SUB-POPULATIONS TESTED
----------------------
1. All open-market purchases (P-Purchase)
2. Materiality filters: $ Value >= $25k, $50k, $100k, $250k
3. Role breakdown: C-Suite (CEO/CFO/COO/Pres) vs Director vs 10% Owner
4. Cluster Buying: >= 2 distinct insiders buying within a 14-calendar-day window
5. Seasonal breakdown: Distribution and performance by calendar month (Sep/Dec/Mar/Jun)
6. Temporal stability: Year-by-year performance (2015–2026)

DATA / CACHE
------------
- Raw insider data fetched from FMP /stable/insider-trading/search and cached
  in 01_data/db_insider.h5.
- Stock prices from /sp400/{permaTicker} in db.h5.
- Benchmark from /macros/IJH in db.h5.
- Production db.h5 is read-only.
"""
from __future__ import annotations
import io
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent          # archive/edge_search_2026/
ROOT = HERE.parents[2]                             # luan_bot_trading/ (2 up from archive/edge_search_2026)
load_dotenv(ROOT / ".env")
FMP_API_KEY = os.getenv("FMP_API_KEY")

DB_PROD = ROOT / "01_data" / "db.h5"
DB_INSIDER = ROOT / "01_data" / "db_insider.h5"
FMP_BASE = "https://financialmodelingprep.com/stable"
HORIZONS = [5, 10, 20, 40, 60]
START_YEAR = 2015


# ==============================================================================
# DATA FETCHING & CACHING
# ==============================================================================

def fetch_insider_for_ticker(ticker: str) -> pd.DataFrame | None:
    """Fetch all historical insider filings for a ticker (limit 1000 per call)."""
    all_data = []
    page = 0
    while page < 5:  # up to 5,000 filings per ticker
        url = f"{FMP_BASE}/insider-trading/search"
        params = {"symbol": ticker, "apikey": FMP_API_KEY, "limit": 1000, "page": page}
        for _ in range(2):
            try:
                r = requests.get(url, params=params, timeout=25)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list) and data:
                        all_data.extend(data)
                        if len(data) < 1000:
                            page = 999  # last page
                        break
                    else:
                        page = 999
                        break
                elif r.status_code == 429:
                    time.sleep(1)
            except Exception:
                time.sleep(1)
        page += 1
        time.sleep(0.08)

    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    need = ["symbol", "filingDate", "transactionDate", "transactionType",
            "reportingName", "typeOfOwner", "securitiesTransacted", "price"]
    avail = [c for c in need if c in df.columns]
    df = df[avail].copy()
    if "filingDate" in df.columns:
        df["filingDate"] = pd.to_datetime(df["filingDate"]).dt.tz_localize(None).dt.normalize()
    if "transactionDate" in df.columns:
        df["transactionDate"] = pd.to_datetime(df["transactionDate"]).dt.tz_localize(None).dt.normalize()
    if "securitiesTransacted" in df.columns:
        df["securitiesTransacted"] = pd.to_numeric(df["securitiesTransacted"], errors="coerce").fillna(0)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    return df


def ensure_insider_data(meta_df: pd.DataFrame, refresh: bool = False) -> None:
    """Download and cache insider transactions for all S&P 400 constituents."""
    mode = "a" if DB_INSIDER.exists() else "w"
    if refresh and DB_INSIDER.exists():
        DB_INSIDER.unlink()
        mode = "w"

    with pd.HDFStore(DB_INSIDER, mode=mode) as store:
        keys = set(store.keys())
        needed = []
        for _, row in meta_df.iterrows():
            pt = row["permaTicker"]
            sym = row["canonical_ticker"]
            if f"/insider/{pt}" not in keys:
                needed.append((pt, sym))

        print(f"[1] Gathering insider data: {len(needed)} tickers needed (already cached: {len(keys)}) ...")
        n_fetched = 0
        for i, (pt, sym) in enumerate(needed, 1):
            df = fetch_insider_for_ticker(sym)
            if df is not None and not df.empty:
                store.put(f"/insider/{pt}", df, format="table")
                n_fetched += 1
            if i % 100 == 0:
                print(f"    [{i}/{len(needed)}] fetched {n_fetched} insider series ...")
        print(f"    Done gathering. Total cached nodes: {len(store.keys())}")


# ==============================================================================
# ROLE & CLUSTER CLASSIFICATION
# ==============================================================================

def classify_role(owner_str: str) -> str:
    """Classify the insider role into standard hierarchy."""
    if not isinstance(owner_str, str):
        return "other"
    s = owner_str.lower()
    if any(k in s for k in ["ceo", "chief executive", "cfo", "chief financial", "coo", "chief operating", "president"]):
        return "c_suite"
    elif any(k in s for k in ["officer", "vp", "vice president", "general counsel", "controller", "treasurer"]):
        return "officer"
    elif "director" in s:
        return "director"
    elif "10%" in s or "ten percent" in s or "owner" in s:
        return "10pct_owner"
    return "other"


def build_events(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Load cached insider transactions, filter open-market buys, compute clusters & forward CAR."""
    print("[2] Processing open-market purchases and computing forward abnormal returns ...")

    with pd.HDFStore(DB_PROD, mode="r") as ps:
        ijh = ps["/macros/IJH"].copy()
        ijh["Date"] = pd.to_datetime(ijh["Date"]).dt.tz_localize(None).dt.normalize()
        ijh = ijh.sort_values("Date").reset_index(drop=True)
        i_dates = ijh["Date"].to_numpy()
        i_close = ijh["Close"].to_numpy(float)
        all_prod_keys = set(ps.keys())

        px_cache = {}
        for _, row in meta_df.iterrows():
            pt = row["permaTicker"]
            pk = f"/sp400/{pt}"
            if pk in all_prod_keys:
                p = ps[pk].copy()
                p["Date"] = pd.to_datetime(p["Date"]).dt.tz_localize(None).dt.normalize()
                p = p.sort_values("Date").reset_index(drop=True)
                px_cache[pt] = (p["Date"].to_numpy(), p["Adj_Close"].to_numpy(float))
            else:
                px_cache[pt] = None

    all_buys = []
    with pd.HDFStore(DB_INSIDER, mode="r") as store:
        for _, row in meta_df.iterrows():
            pt = row["permaTicker"]
            sym = row["canonical_ticker"]
            ik = f"/insider/{pt}"
            if ik not in store.keys():
                continue
            df = store[ik]
            if df.empty:
                continue

            # Strict Open Market Purchase filter
            tt = df["transactionType"].astype(str)
            p_mask = tt.str.startswith("P") | (tt == "P-Purchase")
            buys = df[p_mask].copy()
            if buys.empty:
                continue

            buys["permaTicker"] = pt
            buys["canonical_ticker"] = sym
            buys["dollar_value"] = buys["price"] * buys["securitiesTransacted"]
            buys["role"] = buys["typeOfOwner"].apply(classify_role)
            all_buys.append(buys)

    if not all_buys:
        return pd.DataFrame()

    raw_buys = pd.concat(all_buys, ignore_index=True)
    raw_buys = raw_buys.dropna(subset=["filingDate"]).sort_values(["permaTicker", "filingDate"]).reset_index(drop=True)
    print(f"    Total raw open-market purchase transactions: {len(raw_buys):,}")

    # Deduplicate multiple transactions filed on the same day by the same person
    # and compute daily aggregates per insider
    daily_insider = raw_buys.groupby(["permaTicker", "canonical_ticker", "filingDate", "reportingName", "role"], as_index=False).agg(
        total_dollars=("dollar_value", "sum"),
        total_shares=("securitiesTransacted", "sum"),
        avg_price=("price", "mean")
    )

    # Compute Company-level Cluster Signals (2+ distinct insiders buying within 14 days)
    events = []
    for pt, grp in daily_insider.groupby("permaTicker"):
        pl = px_cache.get(pt)
        if pl is None:
            continue
        pdates, pclose = pl
        grp = grp.sort_values("filingDate").reset_index(drop=True)

        for i in range(len(grp)):
            r = grp.iloc[i]
            fdate = r["filingDate"]
            if fdate.year < START_YEAR:
                continue

            # Find trading day index on/after filing date
            t = int(np.searchsorted(pdates, np.datetime64(fdate), side="left"))
            if t + 1 + max(HORIZONS) >= len(pclose) or t < 0:
                continue

            # Benchmark alignment
            bt = int(np.searchsorted(i_dates, pdates[t], side="left"))
            if bt + 1 + max(HORIZONS) >= len(i_close) or bt < 0:
                continue

            # Cluster check: look back 14 calendar days for other distinct insiders
            window_start = fdate - pd.Timedelta(days=14)
            prior_window = grp[(grp["filingDate"] >= window_start) & (grp["filingDate"] <= fdate)]
            distinct_insiders = prior_window["reportingName"].nunique()
            cluster_dollars = prior_window["total_dollars"].sum()
            is_cluster = (distinct_insiders >= 2) and (cluster_dollars >= 50_000)

            # Execution from Close[T+1] forward
            fwd_car = {}
            for h in HORIZONS:
                stk_ret = float(pclose[t + 1 + h] / pclose[t + 1] - 1.0)
                ijh_ret = float(i_close[bt + 1 + h] / i_close[bt + 1] - 1.0)
                fwd_car[f"car_{h}d"] = stk_ret - ijh_ret

            events.append({
                "permaTicker": pt,
                "ticker": r["canonical_ticker"],
                "filingDate": fdate,
                "year": fdate.year,
                "month": fdate.month,
                "reportingName": r["reportingName"],
                "role": r["role"],
                "dollars": r["total_dollars"],
                "is_cluster": is_cluster,
                "cluster_insiders": distinct_insiders,
                "cluster_dollars": cluster_dollars,
                **fwd_car
            })

    res_df = pd.DataFrame(events)
    print(f"    Computable insider purchase events (2015+): {len(res_df):,}")
    return res_df


# ==============================================================================
# REPORTING
# ==============================================================================

def print_table(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        print(f"  {name:38s} n=0")
        return
    n = len(df)
    c5 = df["car_5d"].to_numpy(float)
    c10 = df["car_10d"].to_numpy(float)
    c20 = df["car_20d"].to_numpy(float)
    c60 = df["car_60d"].to_numpy(float)

    print(f"  {name:38s} n={n:5,} | "
          f"5d: {np.mean(c5)*100:+5.2f}% ({np.mean(c5>0)*100:4.1f}%w) | "
          f"10d: {np.mean(c10)*100:+5.2f}% ({np.mean(c10>0)*100:4.1f}%w) | "
          f"20d: {np.mean(c20)*100:+5.2f}% ({np.mean(c20>0)*100:4.1f}%w) | "
          f"60d: {np.mean(c60)*100:+5.2f}% ({np.mean(c60>0)*100:4.1f}%w)")


def main():
    print("=" * 86)
    print("SEC FORM 4 INSIDER BUYING DRIFT EVENT STUDY — S&P 400 (2015–2026)")
    print("=" * 86)

    meta_df = pd.read_hdf(DB_PROD, "/metadata/sp400_permatickers")
    refresh = "--refresh" in sys.argv
    ensure_insider_data(meta_df, refresh=refresh)

    df = build_events(meta_df)
    if df.empty:
        print("No events found.")
        return

    print("\n" + "=" * 86)
    print("1. OVERALL & MATERIALITY CUTS (Abnormal CAR vs IJH from Close[T+1])")
    print("=" * 86)
    print_table(df, "All P-Purchases (no filter)")
    print_table(df[df.dollars >= 25_000], "Value >= $25k")
    print_table(df[df.dollars >= 50_000], "Value >= $50k")
    print_table(df[df.dollars >= 100_000], "Value >= $100k")
    print_table(df[df.dollars >= 250_000], "Value >= $250k")

    print("\n" + "=" * 86)
    print("2. ROLE HIERARCHY (Value >= $50k)")
    print("=" * 86)
    m50 = df[df.dollars >= 50_000]
    print_table(m50[m50.role == "c_suite"], "C-Suite (CEO/CFO/COO/Pres)")
    print_table(m50[m50.role == "director"], "Board Directors (non-employee)")
    print_table(m50[m50.role == "officer"], "Other Officers (VP/Counsel)")
    print_table(m50[m50.role == "10pct_owner"], "10% Beneficial Owners")

    print("\n" + "=" * 86)
    print("3. CLUSTER BUYING vs SINGLE INSIDER (Value >= $50k)")
    print("=" * 86)
    print_table(m50[~m50.is_cluster], "Single Insider Only")
    print_table(m50[m50.is_cluster], "Cluster (>=2 insiders in 14d)")
    print_table(m50[m50.is_cluster & (m50.role.isin(["c_suite", "director"]))], "Cluster + (C-Suite or Director)")

    print("\n" + "=" * 86)
    print("4. CALENDAR MONTH DISTRIBUTION (Cluster Buying >= $50k — Slow-week supply)")
    print("=" * 86)
    clusters = m50[m50.is_cluster]
    mo_counts = clusters["month"].value_counts().sort_index()
    mnames = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    for m in range(1, 13):
        cnt = mo_counts.get(m, 0)
        bar = "#" * int(cnt / 3)
        sub = clusters[clusters.month == m]
        c20_mean = f"{sub.car_20d.mean()*100:+5.2f}%" if len(sub) else " N/A "
        season_tag = " [PEAD EARNINGS]" if m in [2, 5, 8, 11] else (" [SLOW MONTH]" if m in [3, 6, 9, 12] else "")
        print(f"  {mnames[m-1]:3s}  n={cnt:4d} | 20d CAR={c20_mean} | {bar:<35s}{season_tag}")

    print("\n" + "=" * 86)
    print("5. TEMPORAL STABILITY: Cluster Buying by Year")
    print("=" * 86)
    for y in sorted(clusters.year.unique()):
        print_table(clusters[clusters.year == y], f"Year {y}")


if __name__ == "__main__":
    main()

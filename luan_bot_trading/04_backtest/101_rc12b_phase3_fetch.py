"""RC-12b Phase 3, step 1 — permaTicker mapping + Tiingo prices for the
331 point-in-time SP600 tickers not already in hand (removed/graduated
before today, absent from db_sp600.h5 current-members tables).

Pre-registered in findings/rc12b_sp600_transfer_findings.md (2026-09-05).
Checkpointable: mapping results and prices append to db_sp600.h5 as they
land; re-running skips finished work.

Data-quality gates (pre-registered):
  - >=90% of 331 must map to a Tiingo permaTicker (failures by name)
  - >=90% of mapped must have >=1 in-window (2018-09-01+) price row
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
DB_SP400 = ROOT / "01_data" / "db.h5"
W0 = pd.Timestamp("2018-09-01")

TIINGO_KEY = os.environ.get("TIINGO_API_KEY", "")
if not TIINGO_KEY:  # .env fallback
    env = ROOT / ".env"
    if env.exists():
        for ln in env.read_text().splitlines():
            if ln.strip().startswith("TIINGO_API_KEY"):
                TIINGO_KEY = ln.split("=", 1)[1].strip().strip('"').strip("'")

NEED_CSV = Path(os.environ.get("TEMP", "/tmp")) / "_need600.csv"


def parse_intervals(raw: str) -> list[dict]:
    try:
        return json.loads(str(raw).replace("'", '"').replace("None", "null"))
    except Exception:
        return []


def build_need_list() -> list[str]:
    """Tickers with an interval overlapping the window, minus current members
    minus SP400-tables overlap (those 161 reuse db.h5)."""
    with pd.HDFStore(DB_SP600, "r") as s:
        meta = s["/metadata/sp600"]
        have = set(s["/metadata/sp600_ptmap"].ticker)
    with pd.HDFStore(DB_SP400, "r") as s:
        sp400 = set(s["/metadata/sp400_permatickers"].canonical_ticker.astype(str))
    W1 = pd.Timestamp("2026-06-30")
    need = []
    for r in meta.itertuples(index=False):
        tk = str(r.ticker)
        if tk in have or tk in sp400:
            continue
        for iv in parse_intervals(r.intervals):
            a = pd.Timestamp(iv.get("added"))
            rm = iv.get("removed")
            rm = pd.Timestamp(rm) if rm not in (None, "null", "") else pd.NaT
            if a <= W1 and (pd.isna(rm) or rm >= W0):
                need.append(tk)
                break
    return sorted(set(need))


def tiingo_meta(ticker: str) -> dict | None:
    """Identity lookup via /tiingo/utilities/search (the endpoint the
    permaTicker audit standardized; returns permaTicker + isActive +
    date ranges for delisted names too)."""
    url = f"https://api.tiingo.com/tiingo/utilities/search/{requests.utils.quote(ticker)}"
    for attempt in range(3):
        try:
            r = requests.get(url, params={"token": TIINGO_KEY,
                                         "includeDelisted": "true",
                                         "exactTickerMatch": "true"},
                             timeout=20)
            if r.status_code >= 500 and attempt < 2:
                time.sleep(2.0 * (attempt + 1)); continue
            if r.status_code != 200:
                return None
            data = r.json()
            if not isinstance(data, list) or not data:
                return None
            # exact-ticker match: prefer a row whose ticker == query, else first
            for row in data:
                if str(row.get("ticker", "")).upper() == ticker.upper():
                    return row
            return data[0]
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2.0 * (attempt + 1))
    return None


def tiingo_prices(ticker: str) -> pd.DataFrame | None:
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    r = requests.get(url, params={"token": TIINGO_KEY, "startDate": "2015-01-01",
                                  "format": "json", "resampleFrequency": "1day"},
                     timeout=60)
    if r.status_code != 200 or not r.json():
        return None
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df


def main() -> None:
    need = build_need_list()
    print(f"phase-3 fetch list: {len(need)} tickers")

    with pd.HDFStore(DB_SP600, "a") as s:
        if "/metadata/sp600_ptmap3" in s.keys():
            done_map = dict(zip(s["/metadata/sp600_ptmap3"].ticker,
                                s["/metadata/sp600_ptmap3"].permaTicker))
        else:
            done_map = {}
        done_px = {k.split("/")[-1] for k in s.keys() if k.startswith("/sp600/")}

    todo = [t for t in need if t not in done_map]
    print(f"mapping todo: {len(todo)} (done: {len(done_map)})")

    rows, failed_map = [], []
    t0 = time.time()
    for i, tk in enumerate(todo):
        if time.time() - t0 > 780:  # checkpoint before bash timeout
            print(f"[checkpoint] stopping at {i}/{len(todo)} for timeout")
            break
        m = tiingo_meta(tk)
        if m is None or not m.get("permaTicker"):
            failed_map.append(tk)
            rows.append({"ticker": tk, "source": "tiingo-failed", "permaTicker": ""})
        else:
            rows.append({"ticker": tk, "source": "tiingo", "permaTicker": m["permaTicker"]})
        if (i + 1) % 40 == 0:
            print(f"  mapped {i+1}/{len(todo)} (fails so far: {len(failed_map)})")

    if rows:
        new = pd.DataFrame(rows)
        with pd.HDFStore(DB_SP600, "a") as s:
            if "/metadata/sp600_ptmap3" in s.keys():
                old = s["/metadata/sp600_ptmap3"]
                new = pd.concat([old[~old.ticker.isin(new.ticker)], new], ignore_index=True)
                s.remove("/metadata/sp600_ptmap3")
            s.put("/metadata/sp600_ptmap3", new, format="table",
                  data_columns=["ticker"])

    # prices for everything mapped and not yet fetched
    with pd.HDFStore(DB_SP600, "a") as s:
        pm = s["/metadata/sp600_ptmap3"]
        have_px = {k.split("/")[-1] for k in s.keys() if k.startswith("/sp600/")}
    todo_px = [(r.ticker, r.permaTicker) for r in pm.itertuples(index=False)
               if r.permaTicker and r.permaTicker not in have_px]
    print(f"\nprice fetch todo: {len(todo_px)}")
    ok_px, no_px = 0, []
    t0 = time.time()
    for i, (tk, pt) in enumerate(todo_px):
        if time.time() - t0 > 780:
            print(f"[checkpoint] stopping prices at {i}/{len(todo_px)}")
            break
        df = tiingo_prices(tk)
        with pd.HDFStore(DB_SP600, "a") as s:
            if df is not None and (df["date"] >= W0).any():
                df = df.rename(columns={"date": "Date", "adjClose": "Adj_Close",
                                        "adjVolume": "Adj_Volume"})
                keep = [c for c in ["Date", "Close", "Adj_Close", "Volume", "Adj_Volume"] if c in df.columns]
                s.put(f"/sp600/{pt}", df[keep].sort_values("Date"), format="table")
                ok_px += 1
            else:
                no_px.append(tk)
        if (i + 1) % 25 == 0:
            print(f"  prices {i+1}/{len(todo_px)} ok={ok_px} empty={len(no_px)}")

    print(f"\nSUMMARY: mapped_ok={len(pm)-len([f for f in failed_map])} "
          f"map_fail={len(failed_map)} -> {failed_map[:20]}")
    print(f"prices stored: {ok_px}, no-data: {len(no_px)} -> {no_px[:20]}")
    print("re-run this script to continue from checkpoint if stopped early")


if __name__ == "__main__":
    main()

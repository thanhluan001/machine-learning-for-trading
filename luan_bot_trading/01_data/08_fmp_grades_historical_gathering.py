#!/usr/bin/env python3
"""Fetch FMP monthly aggregate analyst ratings.

Source: /stable/grades-historical?symbol={canonical_ticker}
Storage: /analyst/grades_historical/{permaTicker}

The endpoint returns monthly snapshots, not individual analyst actions. Each
snapshot is point-in-time and can be safely joined backward to an earnings
feature cutoff.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    raise ValueError("FMP_API_KEY not found in .env")

DB_FILE = Path(__file__).parent / "db.h5"
META_KEY = "/metadata/sp400_permatickers"
GROUP = "/analyst/grades_historical"
BASE = "https://financialmodelingprep.com/stable"
TIMEOUT = 30


def fetch(symbol: str, session: requests.Session) -> list[dict]:
    r = session.get(
        f"{BASE}/grades-historical",
        params={"symbol": symbol, "apikey": FMP_API_KEY},
        timeout=TIMEOUT,
    )
    if r.status_code != 200:
        return []
    data = r.json()
    return data if isinstance(data, list) else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    with pd.HDFStore(DB_FILE, mode="r") as store:
        meta = store[META_KEY]
        existing = {
            k.rsplit("/", 1)[-1]
            for k in store.keys()
            if k.startswith(GROUP + "/")
        }

    rows = meta[["permaTicker", "canonical_ticker", "price_unavailable"]].copy()
    rows = rows[~rows["price_unavailable"].astype(bool)]
    if args.skip_existing:
        rows = rows[~rows["permaTicker"].isin(existing)]
    if args.limit:
        rows = rows.head(args.limit)

    session = requests.Session()
    ok = zero = failed = total = 0
    print(f"Fetching {len(rows)} permaTickers -> {GROUP}/{{permaTicker}}")
    for i, row in enumerate(rows.itertuples(index=False), 1):
        pt = str(row.permaTicker)
        symbol = str(row.canonical_ticker)
        try:
            raw = fetch(symbol, session)
            out = []
            for x in raw:
                dt = pd.to_datetime(x.get("date"), errors="coerce")
                if pd.isna(dt):
                    continue
                out.append({
                    "permaTicker": pt,
                    "canonical_ticker": symbol,
                    "date": dt,
                    "analystRatingsStrongBuy": x.get("analystRatingsStrongBuy"),
                    "analystRatingsBuy": x.get("analystRatingsBuy"),
                    "analystRatingsHold": x.get("analystRatingsHold"),
                    "analystRatingsSell": x.get("analystRatingsSell"),
                    "analystRatingsStrongSell": x.get("analystRatingsStrongSell"),
                })
            key = f"{GROUP}/{pt}"
            with pd.HDFStore(DB_FILE, mode="a") as store:
                if key in store.keys():
                    store.remove(key)
                if out:
                    store.put(key, pd.DataFrame(out).sort_values("date"),
                              format="table", data_columns=["date"])
            if out:
                ok += 1
                total += len(out)
            else:
                zero += 1
        except Exception as exc:
            failed += 1
            if failed <= 5:
                print(f"[ERROR] {symbol}/{pt}: {exc}")
        if i % 50 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] with_data={ok} zero={zero} failed={failed} rows={total}")
        time.sleep(0.02)

    print(f"Done: with_data={ok}, zero={zero}, failed={failed}, rows={total}")


if __name__ == "__main__":
    main()

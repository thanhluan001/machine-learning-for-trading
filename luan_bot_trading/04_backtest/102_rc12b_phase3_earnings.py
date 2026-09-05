"""RC-12b Phase 3, step 2 — FMP earnings + grades for the 320 mapped
point-in-time SP600 tickers (removed/graduated names).

Same conventions as Phase 1 (98_rc12b_phase1_matrix): FMP /stable/earnings
with includeReportTimes, /stable/grades; stored under
/sp600/earnings_full/{ticker} and /sp600/grades/{ticker} so the matrix
builder uses one lookup path for all tickers. Checkpointed.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
FMP_BASE = "https://financialmodelingprep.com/stable"

FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
if not FMP_API_KEY:
    env = ROOT / ".env"
    if env.exists():
        for ln in env.read_text().splitlines():
            if ln.strip().startswith("FMP_API_KEY"):
                FMP_API_KEY = ln.split("=", 1)[1].strip().strip('"').strip("'")


def _fmp_get(url, params):
    for _ in range(2):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                d = r.json()
                return d if isinstance(d, list) else []
            time.sleep(1.0)
        except Exception:
            time.sleep(1.0)
    return []


def fetch_earnings_full(sym):
    d = _fmp_get(f"{FMP_BASE}/earnings", {"symbol": sym, "apikey": FMP_API_KEY,
                                          "includeReportTimes": "true"})
    if not d or "date" not in pd.DataFrame(d).columns:
        return None
    df = pd.DataFrame(d)
    out = pd.DataFrame({
        "report_date": pd.to_datetime(df["date"]).dt.normalize(),
        "actual": pd.to_numeric(df.get("epsActual"), errors="coerce"),
        "estimate": pd.to_numeric(df.get("epsEstimated"), errors="coerce"),
        "time": df.get("time"),
    })
    return out.sort_values("report_date").reset_index(drop=True)


def fetch_grades(sym):
    d = _fmp_get(f"{FMP_BASE}/grades", {"symbol": sym, "apikey": FMP_API_KEY})
    if not d:
        return None
    rows = []
    for r in d:
        if r.get("date") is None:
            continue
        rows.append({
            "date": pd.to_datetime(r["date"], errors="coerce"),
            "grading_company": r.get("gradingCompany"),
            "previous_grade": r.get("previousGrade"),
            "new_grade": r.get("newGrade"),
            "action": r.get("action"),
        })
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    pm = pd.read_hdf(DB_SP600, "/metadata/sp600_ptmap3")
    todo = [r.ticker for r in pm.itertuples() if r.permaTicker]
    print(f"phase-3 earnings/grades fetch list: {len(todo)}")

    with pd.HDFStore(DB_SP600, "a") as store:
        keys = set(store.keys())
        todo = [t for t in todo if f"/sp600/earnings_full/{t}" not in keys]
    print(f"todo after checkpoint: {len(todo)}")

    ok_e, ok_g, miss = 0, 0, []
    t0 = time.time()
    for i, sym in enumerate(todo, 1):
        if time.time() - t0 > 780:
            print(f"[checkpoint] stopping at {i}/{len(todo)} — re-run to continue")
            break
        e = fetch_earnings_full(sym)
        g = fetch_grades(sym)
        with pd.HDFStore(DB_SP600, "a") as store:
            if e is not None and not e.empty:
                store.put(f"/sp600/earnings_full/{sym}", e, format="table")
                ok_e += 1
            else:
                miss.append(sym)
            if g is not None and not g.empty:
                store.put(f"/sp600/grades/{sym}", g, format="table")
                ok_g += 1
        if i % 40 == 0:
            print(f"  [{i}/{len(todo)}] earnings_ok={ok_e} grades_ok={ok_g} "
                  f"miss={len(miss)} ({time.time()-t0:.0f}s)")
        time.sleep(0.05)

    print(f"\nSUMMARY: earnings_ok={ok_e} grades_ok={ok_g} no-earnings={len(miss)}")
    if miss:
        print("no-earnings tickers:", miss[:25])


if __name__ == "__main__":
    main()

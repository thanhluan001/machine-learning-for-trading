"""97_rc12b_phase0_universe.py — RC-12b Phase 0, step 1: SP600 universe build.

Pre-registration: 04_backtest/rc12b_pre_registration.md (2026-08-31).

Reuses the SP400 Wikipedia machinery (constituents + changes tables ->
interval arrays) pointed at the S&P 600 page, then applies the
defensive-closure rule learned 2026-08-30 (stale nameless changes rows
with open intervals and absent from the constituents table get closed
at refresh date — never deleted).

Writes /metadata/sp600 (intervals table) to a NEW store db_sp600.h5 —
production db.h5 untouched.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "01_data"
DB_SP600 = DATA / "db_sp600.h5"
SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse_intervals(raw) -> list[dict]:
    if raw is None:
        return []
    s = raw
    if not isinstance(s, str):
        try:
            s = json.dumps(s, default=str)
        except Exception:
            return []
    if s.strip() in {"", "nan", "None", "[]", "{}"}:
        return []
    try:
        data = json.loads(s)
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            out.append({"added": item.get("added"),
                        "removed": item.get("removed")})
        return out
    except Exception:
        return []


def is_open(ivs: list[dict]) -> bool:
    if not ivs:
        return True
    last = ivs[-1]
    r = last.get("removed")
    return r is None or (isinstance(r, str) and r.strip() in ("", "NaT"))


def main() -> None:
    print("=" * 100)
    print("RC-12b PHASE 0 / STEP 1 — SP600 point-in-time universe build")
    print("=" * 100)

    meta = load("meta_gather", DATA / "01_metadata_gathering.py")
    meta.WIKI_URL = SP600_URL

    fresh = meta.build_unified_metadata()
    if fresh is None or fresh.empty:
        raise RuntimeError("SP600 wiki parse returned nothing")
    fresh["ticker"] = fresh["ticker"].astype(str)
    print(f"\nfresh SP600 tickers: {len(fresh)}")

    # constituents set = ground truth for "current"
    cons = meta.fetch_constituents()
    cons_set = {str(t) for t in cons.index}
    print(f"constituents table: {len(cons_set)} tickers")

    # defensive closure (rule of 2026-08-30): nameless + open + absent
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    closed = []
    out = []
    for r in fresh.itertuples(index=False):
        tk = str(r.ticker)
        name = getattr(r, "name", "")
        name_empty = (name is None) or (isinstance(name, float)) or (not str(name).strip())
        ivs = parse_intervals(r.intervals)
        if is_open(ivs) and tk not in cons_set and name_empty:
            if ivs:
                ivs[-1]["removed"] = today
            closed.append(tk)
            r = r._replace(intervals=json.dumps(
                [{"added": iv.get("added"), "removed": iv.get("removed")} for iv in ivs], default=str))
        out.append(r)
    fresh = pd.DataFrame(out)
    if closed:
        print(f"defensively closed {len(closed)} stale nameless rows: "
              f"{', '.join(sorted(closed)[:15])}{'...' if len(closed) > 15 else ''}")

    fresh["open"] = fresh["intervals"].apply(lambda s: is_open(parse_intervals(s)))
    print(f"open-interval (current) tickers: {int(fresh['open'].sum())}")

    # save to the new store
    with pd.HDFStore(DB_SP600, "a") as s:
        if "/metadata/sp600" in s.keys():
            s.remove("/metadata/sp600")
        s.put("/metadata/sp600", fresh.drop(columns=["open"]), format="table")
    print(f"wrote /metadata/sp600 -> {DB_SP600}")

    # overlap with SP400 (canonical ticker space)
    with pd.HDFStore(DATA / "db.h5", "r") as s:
        pt400 = s["/metadata/sp400_permatickers"]
    sp400_t = set(pt400.canonical_ticker.astype(str))
    sp600_t = set(fresh.ticker)
    ov = sp400_t & sp600_t
    print(f"\noverlap with SP400 canonical tickers: {len(ov)} "
          f"({', '.join(sorted(ov)[:12])}{'...' if len(ov) > 12 else ''})")

    # decade coverage of intervals (backtest depth sanity)
    years = []
    for s_ in fresh["intervals"]:
        for iv in parse_intervals(s_):
            if iv.get("added"):
                try:
                    years.append(pd.Timestamp(iv["added"]).year)
                except Exception:
                    pass
    ys = pd.Series(years).value_counts().sort_index()
    print("\ninterval-add events per year (coverage depth):")
    for y, c in ys.items():
        if int(y) >= 2005:
            print(f"  {y}: {c}")


if __name__ == "__main__":
    main()

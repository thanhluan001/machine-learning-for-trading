"""RC-18 amendment prep — SPEC ARITHMETIC ONLY (read-only diagnostic).

Answers three questions for the proposed peer read-across family (F):
  1. COVERAGE: how often does a candidate event have a same-sub-industry
     peer reporting in the trailing window (1-2 weeks)?
  2. FEASIBILITY: of those, how many peers have (a) day-0 reaction known by
     my cutoff, (b) 3-session partial, (c) 11-session COMPLETED drift?
  3. GWRE case mechanics (motivating anecdote: beat EPS+rev, dropped).

NO feature-vs-outcome relationship is computed here (that is P1, and P1 is
gated behind the P0 firewall). This is only coverage + clock arithmetic.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.chdir(ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def log(*a):
    print(*a, flush=True)


DB = str(HERE.parent / "01_data" / "db.h5")
DB6 = str(HERE.parent / "01_data" / "db_sp600.h5")

m = pd.read_hdf(DB, "/features/train_matrix_v6c")
meta = pd.read_hdf(DB, "/metadata/sp400")[["ticker", "gics_sector", "gics_sub_industry"]]
m = m.merge(meta, left_on="canonical_ticker", right_on="ticker", how="left")
m["report_date"] = pd.to_datetime(m["report_date"])
m["T"] = pd.to_datetime(m["T"])

ijh = pd.read_hdf(DB, "/macros/IJH")
cal = pd.to_datetime(pd.Series(ijh.index if ijh.index.dtype.kind == "M" else ijh["Date"])).sort_values().to_numpy()
m["pos"] = np.searchsorted(cal, m["T"].to_numpy(), side="left")
m["cut"] = m["pos"] - np.where(m["is_bmo"].astype(bool), 2, 1)   # entry cutoff position
log(f"events: {len(m):,} | tickers: {m.permaTicker.nunique()} | "
    f"sub-industry coverage: {m.gics_sub_industry.notna().mean():.1%} | "
    f"date range: {m.report_date.min().date()} .. {m.report_date.max().date()}")

# ---------------------------------------------------------------- coverage
def coverage(group_col, K, label):
    ev = m[["permaTicker", "pos", "cut", group_col, "report_date"]].dropna(subset=[group_col])
    rows = []
    for g, sub in ev.groupby(group_col, sort=False):
        if len(sub) < 2:
            continue
        sub = sub.sort_values("pos")
        p = sub["pos"].to_numpy(); c = sub["cut"].to_numpy()
        for i in range(len(sub)):
            lo = p[i] - K
            # peers strictly before me, within K sessions, same group
            j = (p < p[i]) & (p >= lo)
            if not j.any():
                rows.append((sub.index[i], 0, False, False, False))
                continue
            n = int(j.sum())
            pj = p[j]
            rows.append((sub.index[i], n,
                         bool((pj + 1 <= c[i]).any()),   # day-0 reaction known at my cutoff
                         bool((pj + 3 <= c[i]).any()),   # 3-session partial
                         bool((pj + 11 <= c[i]).any())))  # 11-session completed drift
    r = pd.DataFrame(rows, columns=["idx", "n_peers", "gap_known", "partial3", "completed11"]).set_index("idx")
    r = r.reindex(ev.index).fillna({"n_peers": 0, "gap_known": False, "partial3": False, "completed11": False})
    log(f"\n[{label}] window={K} sessions")
    log(f"  >=1 peer in window        : {float((r.n_peers > 0).mean()):.1%}")
    log(f"  peer day-0 reaction known : {float(r.gap_known.mean()):.1%}")
    log(f"  peer 3-session partial    : {float(r.partial3.mean()):.1%}")
    log(f"  peer 11-session COMPLETED : {float(r.completed11.mean()):.1%}")
    log(f"  median peers when any     : {float(r.loc[r.n_peers > 0, 'n_peers'].median()):.0f}")
    return r

r_sub = coverage("gics_sub_industry", 10, "sub-industry")
coverage("gics_sub_industry", 5, "sub-industry")
coverage("gics_sub_industry", 15, "sub-industry")
coverage("sector", 10, "sector-ETF (coarse)")

# seasonality of coverage (sub-industry, K=10)
m2 = m.loc[r_sub.index].copy()
m2["n_peers"] = r_sub["n_peers"].values
m2["completed11"] = r_sub["completed11"].values
m2["month"] = m2.report_date.dt.month
log("\n[seasonality] share of events with >=1 peer (K=10) by month:")
log(m2.groupby("month").apply(lambda d: f"{float((d.n_peers>0).mean()):.0%} (n={len(d)})",
                              include_groups=False).to_string())
log("\n[by year] >=1 peer | completed11 available:")
log(m2.groupby(m2.report_date.dt.year).apply(
    lambda d: f"{(d.n_peers>0).mean():.0%} | {d.completed11.mean():.0%} (n={len(d)})",
    include_groups=False).to_string())

# ------------------------------------------------------------- GWRE case
log("\n" + "=" * 78)
log("GWRE CASE MECHANICS")
log("=" * 78)
g = m[m.canonical_ticker == "GWRE"]
if g.empty:
    log("GWRE not found in the SP400 matrix; checking SP600 side ...")
    m6 = pd.read_hdf(DB6, "/features/train_matrix_sp600_pt_v7c")
    m6["report_date"] = pd.to_datetime(m6.report_date)
    g6 = m6[m6.ticker == "GWRE"]
    log(g6[["ticker", "report_date", "sue_score" if "sue_score" in g6 else "car_10d",
            "car_10d", "pregap_return"]].tail(6).to_string() if len(g6) else "not in SP600 either")
else:
    cols = [c for c in ["canonical_ticker", "report_date", "is_bmo", "sue_score",
                        "car_10d", "pass_g1", "pass_g2", "pass_g3", "pead_pass",
                        "pregap_return", "gics_sub_industry"] if c in g.columns]
    log(g[cols].tail(6).to_string())
    pt = g.iloc[-1]["permaTicker"]
    try:
        px = pd.read_hdf(DB, f"/sp400/{pt}")
        px["Date"] = pd.to_datetime(px["Date"])
        px = px.sort_values("Date").set_index("Date")
        rd = pd.to_datetime(g.iloc[-1]["report_date"])
        near = px.loc[(px.index >= rd - pd.Timedelta(days=4)) & (px.index <= rd + pd.Timedelta(days=8))]
        c = near["Adj_Close"]
        log(f"\nprice path around {rd.date()} (adj close):")
        log((c / c.iloc[0] - 1).mul(100).round(2).to_string())
    except Exception as e:
        log(f"price read failed: {e}")

    sub = g.iloc[-1]["gics_sub_industry"]
    rd = pd.to_datetime(g.iloc[-1]["report_date"])
    peers = m[(m.gics_sub_industry == sub) & (m.report_date != rd)
              & (m.report_date >= rd - pd.Timedelta(days=21))
              & (m.report_date <= rd + pd.Timedelta(days=21))]
    log(f"\nsame-sub-industry ({sub}) events within +/-21d of GWRE's print:")
    if peers.empty:
        log("  none in the matrix window")
    else:
        log(peers[["canonical_ticker", "report_date", "sue_score", "car_10d",
                   "pead_pass", "pregap_return"]].sort_values("report_date").to_string())

log("\nDONE (spec arithmetic only; no outcome relationship measured)")

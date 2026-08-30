#!/usr/bin/env python3
"""75_megatrend_phase2b_pivot.py — RC-4 Phase 2b: gradual momentum pivot vs floor.

HYPOTHESIS (user, 2026-08-16)
-----------------------------
Phase 2 failed with BINARY cluster selection (top-1 = all-in bet). The
proposed alternative: start equal-weight across all above-trend assets, then
GRADUALLY tilt toward demonstrated winners (trailing 12m momentum), selling
losers slowly. Megatrend persistence (Phase 1: multi-year runs) means the
pivot doesn't need to be early — only eventually right. Future refinement
(Phase 2c if this passes): tilt by CAPEX FLOWS (capital-cycle confirmation)
instead of / in addition to price momentum.

MECHANIC UNDER TEST
-------------------
Monthly. Eligibility = Phase-2 floor (close > MA10m). Weight per eligible
asset ∝ clip(1 + k * z(12m momentum), 0.5, 2.0), renormalized. k=0 is the
floor. Small k = slow pivot. The clip bounds single-asset tilt; monthly
cadence bounds turnover.

KILL CRITERIA (fixed before running):
  K1 full universe:  pivot(k*) beats floor on total AND maxDD.
  K2 stress (drop 7 hand-picked mega-winners): pivot total >= 0.8 * floor
     total AND pivot maxDD not worse than floor by more than 5pp.
  PASS = K1 and K2. The stress test is the one that killed top-cluster —
  concentration into trailing winners in 2019-21 = pandemic cluster = the
  known failure mode of momentum tilts (procyclicality at trend death).

Universe: the Phase-2 24 trend assets (ex SPY/QQQ refs).
"""
from __future__ import annotations
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_MT = ROOT / "01_data" / "db_megatrend.h5"

PHASE1_ASSETS = {'ARKK','COIN','EEM','GDX','IBB','ICLN','KWEB','LIT','MSTR','NVDA',
                 'PTON','PYPL','QQQ','SHOP','SMH','SPAK','SPY','TAN','TSM','URA',
                 'VNQ','XBI','XLE','XLU','XYZ','ZM'}
STRESS_DROP = ['MSTR', 'SHOP', 'NVDA', 'SMH', 'TSM', 'XYZ', 'PYPL']


def load_monthly():
    with pd.HDFStore(DB_MT, mode="r") as s:
        px = {k.split("/")[-1]: s[k].set_index("date")["adjClose"] for k in s.keys()}
    M = pd.DataFrame({sym: ser.resample("ME").last() for sym, ser in px.items()})
    return M


def pivot_backtest(M: pd.DataFrame, k: float, drop=None):
    rets = np.log(M / M.shift(1))
    ma10 = M.rolling(10).mean()
    assets = [c for c in M.columns if c not in PHASE1_ASSETS or c in M.columns]
    assets = [c for c in M.columns if c not in ("SPY", "QQQ")]
    if drop:
        assets = [c for c in assets if c not in drop]
    months = M.index.tolist()
    rows = []
    for i in range(13, len(months) - 1):   # need 12m momentum + 10m MA
        nxt = months[i + 1]
        elig = [c for c in assets
                if pd.notna(ma10[c].iloc[i]) and M[c].iloc[i] > ma10[c].iloc[i]
                and rets[c].iloc[i - 12:i].notna().all()]
        if not elig:
            rows.append((nxt, 0.0, 0))
            continue
        mom = np.array([rets[c].iloc[i - 12:i].sum() for c in elig])
        sd = mom.std()
        z = (mom - mom.mean()) / sd if sd > 1e-9 else np.zeros_like(mom)
        mult = np.clip(1 + k * z, 0.5, 2.0)
        w = mult / mult.sum()
        r = float(np.dot(w, [rets.loc[nxt, c] for c in elig]))
        rows.append((nxt, r, len(elig)))
    df = pd.DataFrame([(d, r) for d, r, _ in rows], columns=["date", "ret"]).set_index("date")
    df = df[df.index >= "2018-08-01"]
    nav = np.exp(df.ret.cumsum())
    return dict(total=float(df.ret.sum()), ann=float(df.ret.mean() * 12),
                dd=float((nav / nav.cummax() - 1).min()),
                time_in=float((df.ret != 0).mean()),
                by_year=df.groupby(df.index.year).ret.sum().to_dict(),
                n_months=len(df))


def show(label, r):
    print(f"  {label:>28} {np.expm1(r['total'])*100:>+7.0f}% {r['ann']*100:>+5.1f}% "
          f"{r['dd']*100:>6.1f}% {r['time_in']*100:>4.0f}%")


def main():
    print("=" * 92)
    print("RC-4 PHASE 2b — GRADUAL MOMENTUM PIVOT vs FLOOR (with survivorship stress)")
    print("=" * 92)
    M = load_monthly()

    print(f"\n[1] FULL universe (24 trend assets, 2018-08 -> 2026-08):")
    print(f"  {'variant':>28} {'total':>8} {'ann%':>6} {'maxDD':>7} {'in%':>5}")
    base = pivot_backtest(M, 0.0)
    show("FLOOR (k=0, equal wt)", base)
    results = {}
    for k in (0.25, 0.5, 1.0):
        r = pivot_backtest(M, k)
        results[k] = r
        show(f"PIVOT k={k}", r)

    print(f"\n[2] STRESS universe (drop 7 mega-winners: {STRESS_DROP}):")
    print(f"  {'variant':>28} {'total':>8} {'ann%':>6} {'maxDD':>7} {'in%':>5}")
    base_s = pivot_backtest(M, 0.0, drop=STRESS_DROP)
    show("FLOOR (k=0, equal wt)", base_s)
    for k in (0.25, 0.5, 1.0):
        r = pivot_backtest(M, k, drop=STRESS_DROP)
        results[("stress", k)] = r
        show(f"PIVOT k={k}", r)

    print(f"\n[3] Year-by-year, full universe (k=0.5 vs floor):")
    years = sorted(set(base["by_year"]) | set(results[0.5]["by_year"]))
    print(f"  {'year':>6} {'floor':>7} {'k=0.25':>7} {'k=0.5':>7} {'k=1.0':>7}")
    for y in years:
        row = f"  {y:>6}"
        for r in (base, results[0.25], results[0.5], results[1.0]):
            v = r["by_year"].get(y, np.nan)
            row += f"{v*100:>+6.0f}%"
        print(row)

    # kill criteria
    print(f"\n[4] KILL CRITERIA:")
    best_k = max((0.25, 0.5, 1.0), key=lambda k: results[k]["total"])
    rb = results[best_k]
    k1 = (rb["total"] > base["total"]) and (rb["dd"] > base["dd"])
    rb_s = results[("stress", best_k)]
    k2 = (rb_s["total"] >= 0.8 * base_s["total"]) and (rb_s["dd"] >= base_s["dd"] - 0.05)
    print(f"  K1 full: best k={best_k} beats floor on total+DD? {k1} "
          f"({np.expm1(rb['total'])*100:+.0f}%/{rb['dd']*100:.1f}% vs "
          f"{np.expm1(base['total'])*100:+.0f}%/{base['dd']*100:.1f}%)")
    print(f"  K2 stress: no collapse? {k2} "
          f"({np.expm1(rb_s['total'])*100:+.0f}%/{rb_s['dd']*100:.1f}% vs "
          f"{np.expm1(base_s['total'])*100:+.0f}%/{base_s['dd']*100:.1f}%)")
    print(f"\n  VERDICT: PASS = {k1 and k2}")


if __name__ == "__main__":
    main()

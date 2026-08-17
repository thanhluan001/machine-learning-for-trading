#!/usr/bin/env python3
"""73_megatrend_phase3_broad_regime.py — RC-4 Phase 3: broad-universe regime
table for the FLOOR strategy (per Phase-2 redirect).

QUESTION (the open one from Phase 2)
------------------------------------
On a BROAD deployment universe (GICS sectors + broad/international/bond/gold +
major theme ETFs — the universe a real carve-out would trade), does the Phase-2
floor (equal-weight basket of assets above their 10-month mean, monthly
rebalance, member exits below MA10m) beat SPY buy-and-hold with lower
drawdown — INCLUDING through the regime table (2008, 2011, 2015, 2018Q4,
2020 crash, 2022, 2023 chop)?

Survivorship note: this universe is chosen by CATEGORY (all 11 GICS sectors,
all major regions/asset classes, the standard theme-ETF menu), not by which
trends won. GXTG (2019 launch) and PAVE/XLC/XLRE late starts are structural,
not selection.

STRATEGIES
----------
  FLOOR-27   : equal-weight ALL 27 broad assets above their own MA10m,
               monthly rebalance (the Phase-2 winner structure, real universe)
  FLOOR-EW   : 60/40 reference — 60% FLOOR-27, 40% AGG (deployment-realistic
               sizing for a cautious core carve-out)
  SPY B&H    : reference
  60/40      : SPY/AGG static reference

The 2008 question is the honest one: pre-2008 the universe had ~17 of 27
assets (theme ETFs didn't exist), which is itself the historical reality —
the machinery must work with what existed THEN. Assets enter the backtest
from their first valid MA10m month (rolling, expanding universe).

KILL CRITERIA (fixed before running):
  P1: FLOOR-27 maxDD < SPY maxDD (materially: at least 10pp better)
  P2: FLOOR-27 total return > SPY total return
  P3: regime table — no regime year worse than SPY by more than 15pp
  P4: FLOOR-27 beats 60/40 static on return
  PASS = P1 and P2 and P3 and P4.
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
SPY = "SPY"; AGG = "AGG"


def load_monthly():
    with pd.HDFStore(DB_MT, mode="r") as s:
        px = {k.split("/")[-1]: s[k].set_index("date")["adjClose"] for k in s.keys()}
    M = pd.DataFrame({sym: ser.resample("ME").last() for sym, ser in px.items()})
    broad = [c for c in M.columns if c not in PHASE1_ASSETS]
    cols, seen = [], set()
    for c in [SPY, AGG] + sorted(broad):
        if c not in seen:
            cols.append(c); seen.add(c)   # AGG is in broad too — dedup
    M = M[cols]
    return M


def floor_backtest(M, risk_assets=None, agg_weight=0.0, start="2006-06-30"):
    """Equal-weight members above their own 10m mean; monthly; expanding universe."""
    rets = np.log(M / M.shift(1))
    ma10 = M.rolling(10).mean()
    months = M.index.tolist()
    assets = [c for c in M.columns if c not in (SPY,)]  # AGG included if risk_assets None
    if risk_assets is not None:
        assets = risk_assets
    rows = []
    for i in range(10, len(months) - 1):
        nxt = months[i + 1]
        sel = [c for c in assets
               if pd.notna(ma10[c].iloc[i]) and pd.notna(M[c].iloc[i])
               and M[c].iloc[i] > ma10[c].iloc[i]]
        r_risk = float(np.mean([rets.loc[nxt, c] for c in sel])) if sel else 0.0
        r = agg_weight * rets.loc[nxt, AGG] + (1 - agg_weight) * r_risk
        rows.append((nxt, r, sel))
    df = pd.DataFrame([(d, r) for d, r, _ in rows], columns=["date", "ret"]).set_index("date")
    df = df[df.index >= start]
    nav = np.exp(df.ret.cumsum())
    return dict(nav=nav, total=float(df.ret.sum()), ann=float(df.ret.mean() * 12),
                dd=float((nav / nav.cummax() - 1).min()),
                time_in=float((df.ret != 0).mean()),
                by_year=df.groupby(df.index.year).ret.sum().to_dict(),
                n_months=len(df),
                holdings={(d, tuple(s)) for d, _, s in rows if s and d >= pd.Timestamp(start)},
                avg_positions=float(np.mean([len(s) for d, _, s in rows if d >= pd.Timestamp(start)])))


def bh(M, col, start="2006-06-30"):
    ser = M[col]
    r = pd.Series(np.log(ser.astype(float) / ser.astype(float).shift(1)), index=ser.index).dropna()
    r = r[r.index >= start]
    nav = np.exp(r.cumsum())
    return dict(nav=nav, total=float(r.sum()), ann=float(r.mean() * 12),
                dd=float((nav / nav.cummax() - 1).min()), time_in=1.0,
                by_year=r.groupby(r.index.year).sum().to_dict(),
                n_months=len(r), holdings=set(), avg_positions=1.0)


def main():
    print("=" * 92)
    print("RC-4 PHASE 3 — BROAD-UNIVERSE REGIME TABLE for the FLOOR (2006-2026)")
    print("=" * 92)
    M = load_monthly()
    n_broad = M.shape[1] - 2
    print(f"[1] Universe: {n_broad} broad assets + SPY/AGG reference "
          f"(expanding universe, entries from first valid MA10m)")

    spy = bh(M, SPY)
    agg = bh(M, AGG)
    floor = floor_backtest(M)                       # all broad assets incl AGG
    floor6040 = floor_backtest(M, agg_weight=0.40)  # 60% floor / 40% AGG
    static6040 = dict(total=0.6 * spy["total"] + 0.4 * agg["total"],
                      dd=np.nan, by_year={y: 0.6 * spy["by_year"].get(y, 0) + 0.4 * agg["by_year"].get(y, 0)
                                          for y in set(spy["by_year"]) | set(agg["by_year"])},
                      ann=0.6 * spy["ann"] + 0.4 * agg["ann"], time_in=1.0, n_months=spy["n_months"],
                      nav=None, holdings=set(), avg_positions=1.0)

    print(f"\n[2] Results (2006-06 -> 2026-08, monthly, no costs):")
    print(f"  {'strategy':>26} {'total%':>9} {'ann%':>6} {'maxDD%':>7} {'in%':>5} {'avg#':>5}")
    for label, r in [("FLOOR-27 (broad)", floor), ("FLOOR 60/40 AGG", floor6040),
                     ("SPY B&H", spy), ("static 60/40", static6040)]:
        print(f"  {label:>26} {np.expm1(r['total'])*100:>+8.0f}% {r['ann']*100:>+5.1f}% "
              f"{r['dd']*100:>6.1f}% {r['time_in']*100:>4.0f}% {r['avg_positions']:>5.1f}")

    print(f"\n[3] REGIME TABLE — log-return by year (%):")
    years = sorted(set(spy["by_year"]) | set(floor["by_year"]))
    print(f"  {'year':>6} {'FLOOR':>8} {'F6040':>7} {'SPY':>7} {'60/40':>7} {'F-SPY':>7}")
    for y in years:
        f = floor["by_year"].get(y, np.nan); s = spy["by_year"].get(y, np.nan)
        f6 = floor6040["by_year"].get(y, np.nan); s6 = static6040["by_year"].get(y, np.nan)
        print(f"  {y:>6} {f*100:>+7.0f}% {f6*100:>+6.0f}% {s*100:>+6.0f}% {s6*100:>+6.0f}% "
              f"{(f-s)*100:>+6.0f}%")

    print(f"\n[4] KILL CRITERIA:")
    p1 = floor["dd"] > spy["dd"] + 0.10
    p2 = floor["total"] > spy["total"]
    worst = min((floor["by_year"].get(y, 0) - spy["by_year"].get(y, 0) for y in years),
                default=0)
    p3 = worst > -0.15
    p4 = floor["total"] > static6040["total"]
    print(f"  P1 DD (FLOOR {floor['dd']*100:.1f}% vs SPY {spy['dd']*100:.1f}%, need >=10pp better): {p1}")
    print(f"  P2 return ({np.expm1(floor['total'])*100:+.0f}% vs {np.expm1(spy['total'])*100:+.0f}%): {p2}")
    print(f"  P3 worst regime-year vs SPY ({worst*100:+.0f}pp, need > -15pp): {p3}")
    print(f"  P4 beats static 60/40 ({np.expm1(floor['total'])*100:+.0f}% vs {np.expm1(static6040['total'])*100:+.0f}%): {p4}")
    print(f"\n  VERDICT: PASS = {p1 and p2 and p3 and p4}")

    # 2008 anatomy + current state for the warning role
    print(f"\n[5] 2008-09 anatomy (monthly log-returns):")
    sub = floor["nav"].loc["2008-01-01":"2009-06-30"]
    spysub = spy["nav"].loc["2008-01-01":"2009-06-30"]
    f_dd = float((sub / sub.cummax() - 1).min()); s_dd = float((spysub / spysub.cummax() - 1).min())
    print(f"  2008 window: FLOOR maxDD {f_dd*100:.1f}% vs SPY {s_dd*100:.1f}%")

    # current signal state
    last = M.index[-1]
    ma10 = M.rolling(10).mean()
    active = [c for c in M.columns if c not in (SPY, AGG)
              and pd.notna(ma10[c].iloc[-1]) and M[c].iloc[-1] > ma10[c].iloc[-1]]
    print(f"\n[6] CURRENT STATE ({last.date()}, warning-role output):")
    print(f"  assets above MA10m: {len(active)}/{n_broad}")
    print(f"  {sorted(active)}")


if __name__ == "__main__":
    main()

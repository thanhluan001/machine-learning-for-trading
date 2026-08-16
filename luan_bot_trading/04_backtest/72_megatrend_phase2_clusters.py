#!/usr/bin/env python3
"""72_megatrend_phase2_clusters.py — RC-4 Phase 2: cluster construction + composite scoring.

KILL TEST C (from Design.md §18 RC-4 + Phase 1 findings)
--------------------------------------------------------
The cluster machinery (L1-L3 of the design skeleton) must BEAT the Phase-1
trivial floor (MA200 monthly state machine run across all assets) on the
same universe. If it doesn't, the doctrine says: use the trivial rule.

MACHINERY UNDER TEST (monthly cadence throughout, matching Phase 1 finding
that month-end decisions are operationally viable):
  1. BETA-RESIDUALIZATION: monthly log-returns vs SPY, trailing 36m rolling
     regression per asset (kills the "everything correlates in QE regimes"
     problem found in exploration: raw rho>=0.6 merged ALL assets in 2020-21).
  2. AVG-LINKAGE CLUSTERING on residual correlation, threshold 0.35.
  3. CLUSTER SCORING: mean member 12m momentum (log).
  4. BREADTH GATE: >= 60% of cluster members above their own 10m mean.
  5. SELECTION: hold top-scoring cluster (equal-weight) if it passes breadth;
     else cash. Evaluated at month-end close, positions effective next month.
  6. EXIT LAYER: member dropped when below its MA200 at month-end (Phase-1
     floor provides the exit within the cluster).

COMPARISONS (all monthly, 2018-07 -> 2026-08 where the 36m window allows):
  A. CLUSTER strategy (machinery above)
  B. FLOOR-ALL: MA200/monthly on ALL 26 assets, equal-weight basket
  C. FLOOR-SPY: MA200/monthly on SPY alone
  D. SPY buy-and-hold (reference)

PASS = A beats B on: return per unit time-in-market, AND max drawdown, AND
2022 (crash year) behavior. A pure return win with equal risk-adjusted
profile is sufficient given the overlay role.

Universe: the 26 Phase-1 assets (dead_2021 + cycles + live_2026 + refs).
This is deliberately trend-asset-heavy; the core-book deployment would run
the same machinery on a broad sector/industry ETF list (Phase 3).
"""
from __future__ import annotations
import itertools
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

MA_LEN = 200        # exit floor (Phase-1 validated)
MOM = 12            # cluster momentum window (months)
BREADTH_MIN = 0.60  # fraction of members above 10m mean
RHO = 0.35          # residual-correlation clustering threshold
CORR_WIN = 36       # months


def load_monthly():
    with pd.HDFStore(DB_MT, mode="r") as s:
        px = {k.split("/")[-1]: s[k].set_index("date")["adjClose"] for k in s.keys()}
    M = pd.DataFrame({sym: ser.resample("ME").last() for sym, ser in px.items()})
    return M[M.index >= "2015-01-01"]


def avg_link_clusters(corr, ns, thresh):
    pairs = sorted(((corr.loc[a, b], a, b) for a, b in itertools.combinations(ns, 2)),
                   reverse=True)
    assigned, clusters = {}, []
    for rho, a, b in pairs:
        if rho < thresh:
            break
        ca, cb = assigned.get(a), assigned.get(b)
        if ca is None and cb is None:
            clusters.append([a, b]); assigned[a] = assigned[b] = len(clusters) - 1
        elif cb is None:
            clusters[ca].append(b); assigned[b] = ca
        elif ca is None:
            clusters[cb].append(a); assigned[a] = cb
        elif ca != cb:
            A, B = clusters[ca], clusters[cb]
            if np.mean([corr.loc[x, y] for x in A for y in B]) >= thresh:
                clusters[ca] = A + B
                for x in B:
                    assigned[x] = ca
    return [sorted(x) for x in clusters if len(x) >= 2]


def backtest(M: pd.DataFrame, mode: str):
    rets = np.log(M / M.shift(1))
    names = [c for c in M.columns if c not in ("SPY", "QQQ")]
    months = M.index.tolist()
    # MA200 on daily approximated by 10-month mean on monthly closes (classic
    # trend-following equivalence; consistent across strategies compared here)
    ma10 = M.rolling(10).mean()

    monthly_pos, holdings = [], []
    for i in range(max(CORR_WIN, MOM) + 1, len(months) - 1):
        end = months[i]
        nxt = months[i + 1]
        r_next = rets.loc[nxt]

        if mode in ("floor_all", "floor_spy"):
            pool = names if mode == "floor_all" else ["SPY"]
            sel = [c for c in pool if pd.notna(ma10[c].iloc[i]) and M[c].iloc[i] > ma10[c].iloc[i]]
        else:  # cluster
            R36 = rets.iloc[i - CORR_WIN:i]
            valid = [c for c in names if R36[c].notna().all()]
            sel = []
            if len(valid) >= 4:
                y = R36["SPY"]
                Rr = pd.DataFrame(index=R36.index)
                for c in valid:
                    beta = np.polyfit(y, R36[c], 1)[0]
                    Rr[c] = R36[c] - beta * y
                clusters = avg_link_clusters(Rr.corr(), valid, RHO)
                if clusters:
                    scored = []
                    for cl in clusters:
                        mom = np.mean([rets[c].iloc[i - MOM:i].sum() for c in cl])
                        breadth = np.mean([M[c].iloc[i] > M[c].iloc[i - 10:i].mean()
                                           for c in cl])
                        scored.append((mom, breadth, cl))
                    scored.sort(reverse=True)
                    top = scored[0]
                    if top[1] >= BREADTH_MIN:
                        sel = [c for c in top[2]
                               if pd.notna(ma10[c].iloc[i]) and M[c].iloc[i] > ma10[c].iloc[i]]
        r = float(np.mean([r_next[c] for c in sel])) if sel else 0.0
        monthly_pos.append((nxt, r, sel))

    df = pd.DataFrame([(d, r) for d, r, _ in monthly_pos], columns=["date", "ret"]).set_index("date")
    nav = np.exp(df.ret.cumsum())
    dd = float((nav / nav.cummax() - 1).min())
    ann = float(df.ret.mean() * 12)
    yrs = df.groupby(df.index.year).ret.sum()
    return dict(nav=nav, total=float(df.ret.sum()), ann=ann, dd=dd,
                time_in=float((df.ret != 0).mean()),
                by_year=yrs.to_dict(),
                n_months=len(df),
                holdings={d: s for d, _, s in monthly_pos if s})


def main():
    print("=" * 92)
    print("RC-4 PHASE 2 — CLUSTER MACHINERY vs PHASE-1 FLOOR (Kill Test C), monthly cadence")
    print("=" * 92)
    M = load_monthly()
    print(f"[1] Monthly matrix: {M.shape[0]} months x {M.shape[1]} assets")

    res = {}
    for mode, label in [("cluster", "A. CLUSTER machinery"),
                        ("floor_all", "B. FLOOR: MA10m on all 26 assets"),
                        ("floor_spy", "C. FLOOR: MA10m on SPY"),
                        ("spy_bh", "D. SPY buy-and-hold")]:
        if mode == "spy_bh":
            r = np.log(M["SPY"] / M["SPY"].shift(1)).dropna()
            r = r[r.index >= "2018-08-01"]
            nav = np.exp(r.cumsum())
            res[mode] = dict(nav=nav, total=float(r.sum()), ann=float(r.mean() * 12),
                             dd=float((nav / nav.cummax() - 1).min()),
                             time_in=1.0, by_year=r.groupby(r.index.year).sum().to_dict(),
                             n_months=len(r), holdings={})
        else:
            res[mode] = backtest(M, mode)

    print(f"\n[2] Results (2018-08 -> 2026-08, monthly, no costs):")
    print(f"  {'strategy':>32} {'total%':>8} {'ann%':>6} {'maxDD%':>7} {'in-mkt%':>8}")
    for mode, label in [("cluster", "A. CLUSTER machinery"),
                        ("floor_all", "B. FLOOR: MA10m all assets"),
                        ("floor_spy", "C. FLOOR: MA10m SPY"),
                        ("spy_bh", "D. SPY buy-and-hold")]:
        r = res[mode]
        print(f"  {label:>32} {np.expm1(r['total'])*100:>+7.0f}% {r['ann']*100:>+5.1f}% "
              f"{r['dd']*100:>6.1f}% {r['time_in']*100:>7.0f}%")

    print(f"\n[3] Year-by-year (log-return sums, %):")
    hdr = f"  {'year':>6}" + "".join(f"{m:>9}" for m in ["A cluster", "B floor", "C spyMA", "D spyBH"])
    print(hdr)
    years = sorted(set().union(*[set(res[m]['by_year']) for m in res]))
    for y in years:
        row = f"  {y:>6}"
        for m in ["cluster", "floor_all", "floor_spy", "spy_bh"]:
            v = res[m]["by_year"].get(y, np.nan)
            row += f"{v*100:>+8.0f}%" if not np.isnan(v) else f"{'—':>9}"
        print(row)

    # 2022 focus (the warning-indicator year)
    print(f"\n[4] 2022 crash-year behavior (the overlay's job):")
    for m, label in [("cluster", "A"), ("floor_all", "B"), ("floor_spy", "C"), ("spy_bh", "D")]:
        v22 = res[m]["by_year"].get(2022, np.nan)
        print(f"  {label}: {v22*100:+.0f}%")

    # cluster holdings timeline
    h = res["cluster"]["holdings"]
    print(f"\n[5] Cluster strategy in-position months: {len(h)}/{res['cluster']['n_months']}")
    prev = None
    for d, sel in sorted(h.items()):
        if sel != prev:
            print(f"  {d.date()}  {sel}")
            prev = sel

    # verdict
    a, b = res["cluster"], res["floor_all"]
    beats_ret = a["total"] > b["total"]
    beats_dd = a["dd"] > b["dd"]
    print(f"\n[6] KILL TEST C: cluster beats floor-all? return={beats_ret} "
          f"({np.expm1(a['total'])*100:+.0f}% vs {np.expm1(b['total'])*100:+.0f}%), "
          f"DD={beats_dd} ({a['dd']*100:.1f}% vs {b['dd']*100:.1f}%)")


if __name__ == "__main__":
    main()

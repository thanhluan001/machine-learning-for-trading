#!/usr/bin/env python3
"""68_analyst_upgrade_drift.py — Event study: post-analyst-upgrade drift (sp400).

RESEARCH QUESTION (slow-week edge candidate)
--------------------------------------------
Do analyst UPGRADES on S&P 400 names produce capturable drift? The hypothesis:
in earnings dead zones (Sep/Dec/Mar/Jun) upgrade events could fill idle slots.

EXECUTION ASSUMPTIONS (deliberately honest)
-------------------------------------------
- Grade events are published intraday -> we can trade no earlier than the NEXT
  close. Drift measured from Close[T+1] (first close after the event) forward.
- The announcement-day gap (Open[T+1]/Close[T]) is reported but marked
  UNTRADEABLE.
- Returns are RELATIVE to IJH (same benchmark convention as the pipeline).
- Same-day/near earnings contamination: upgrades issued within +-3 trading
  days of the ticker's earnings report are split out — that drift would be
  PEAD, not an independent edge.

DATA
----
- /analyst/grades/{pt} (db.h5, FMP /stable/grades) — 807 sp400 nodes.
- /sp400/{pt} prices for forward returns.
- gated events (earnings dates) from train matrix report dates.

No model training here — event study only. Diagnostic; writes nothing.
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

ROOT = Path(__file__).resolve().parents[3]  # luan_bot_trading/ (3 up from archive/edge_search_2026)
DB = ROOT / "01_data" / "db.h5"
MATRIX = "/features/train_matrix_v4_timing_correct"
NEAR_EARN_DAYS = 3          # +- trading days for contamination split
FWD_HORIZONS = [5, 10]      # trading days from Close[T+1]

GRADE_ORDINAL = {
    "strong sell": 1, "sell": 1, "underweight": 1, "underperform": 1,
    "reduce": 1, "reduce in price": 1,
    "negative": 2, "below average": 2, "below market": 2,
    "market underperform": 2, "market underperformer": 2,
    "hold": 3, "neutral": 3, "sector weight": 3, "sector perform": 3,
    "market perform": 3, "in-line": 3, "peer perform": 3,
    "equal-weight": 3, "equal weight": 3, "market weight": 3,
    "fair value": 3, "average": 3, "maintain": 3,
    "accumulate": 4, "overweight": 4, "outperform": 4, "market outperform": 4,
    "add": 4, "add position": 4, "positive": 4, "above average": 4,
    "above market": 4, "mild buy": 4, "long-term buy": 4,
    "strong buy": 5, "buy": 5, "trading buy": 5, "market outperformer": 5,
}


def ord_of(g):
    if g is None or not isinstance(g, str):
        return None
    g = g.strip().lower()
    if g in GRADE_ORDINAL:
        return GRADE_ORDINAL[g]
    for k, v in GRADE_ORDINAL.items():
        if k in g:
            return v
    return None


def main():
    print("=" * 78)
    print("ANALYST UPGRADE DRIFT EVENT STUDY (S&P 400, FMP /stable/grades)")
    print("=" * 78)

    # earnings dates per permaTicker (for contamination split)
    mat = pd.read_hdf(DB, MATRIX, columns=["permaTicker", "report_date"])
    earn_by_pt = {pt: pd.to_datetime(g.report_date).to_numpy()
                  for pt, g in mat.groupby("permaTicker")}

    with pd.HDFStore(DB, mode="r") as s:
        ijh = s["/macros/IJH"].copy()
        ijh["Date"] = pd.to_datetime(ijh["Date"]).dt.tz_localize(None).dt.normalize()
        ijh = ijh.sort_values("Date").reset_index(drop=True)
        i_idx = ijh["Date"].to_numpy()
        i_close = ijh["Close"].to_numpy(float)

        gkeys = [k for k in s.keys() if k.startswith("/analyst/grades/")]
        all_keys = set(s.keys())  # cache once: s.keys() rescans every node
        px_cache = {}

        def px_of(pt):
            if pt not in px_cache:
                k = f"/sp400/{pt}"
                if k in all_keys:
                    p = s[k].copy()
                    p["Date"] = pd.to_datetime(p["Date"]).dt.tz_localize(None).dt.normalize()
                    p = p.sort_values("Date").reset_index(drop=True)
                    px_cache[pt] = (p["Date"].to_numpy(), p["Adj_Close"].to_numpy(float))
                else:
                    px_cache[pt] = None
            return px_cache[pt]

        rows = []
        n_done = 0
        for k in gkeys:
            n_done += 1
            if n_done % 100 == 0:
                print(f"    [{n_done}/{len(gkeys)}] nodes, events so far {len(rows):,}", flush=True)
            pt = k.split("/")[-1]
            g = s[k]
            if g.empty:
                continue
            pl = px_of(pt)
            if pl is None:
                continue
            pdates, pclose = pl
            earn = earn_by_pt.get(pt)
            for r in g.itertuples(index=False):
                if r.action not in ("upgrade", "downgrade"):
                    continue
                d = pd.Timestamp(r.date)
                # event day = first trading day on/after grade date
                t = int(np.searchsorted(pdates, np.datetime64(d), side="left"))
                if t + 1 + max(FWD_HORIZONS) >= len(pclose):
                    continue
                if t < 1:
                    continue
                # benchmark alignment: IJH index for trading day t
                bt = int(np.searchsorted(i_idx, pdates[t], side="left"))
                if bt + 1 + max(FWD_HORIZONS) >= len(i_close):
                    continue
                near_earn = False
                if earn is not None and len(earn):
                    dd = np.abs((earn - np.datetime64(d)).astype("timedelta64[D]").astype(int))
                    near_earn = bool((dd <= NEAR_EARN_DAYS * 2 - 1).any())  # ~3 trading days
                gap = float(pclose[t] / pclose[t - 1] - 1.0)  # day-of move (approx; publish intraday)
                fwd = {}
                for h in FWD_HORIZONS:
                    e = t + 1 + h
                    fwd[f"rel_{h}d"] = float(
                        (pclose[e] / pclose[t + 1])
                        - (i_close[bt + 1 + h] / i_close[bt + 1]))
                rows.append({
                    "pt": pt, "date": d, "action": r.action,
                    "new_ord": ord_of(r.new_grade), "prev_ord": ord_of(r.previous_grade),
                    "ord_delta": (ord_of(r.new_grade) - ord_of(r.previous_grade))
                    if ord_of(r.new_grade) and ord_of(r.previous_grade) else None,
                    "near_earn": near_earn, "gap_day0": gap, **fwd,
                })

    df = pd.DataFrame(rows)
    df["year"] = df.date.dt.year
    print(f"\nevents analyzed: {len(df):,} "
          f"(upgrades {int((df.action=='upgrade').sum()):,} / "
          f"downgrades {int((df.action=='downgrade').sum()):,})")
    print(f"near-earnings (+-{NEAR_EARN_DAYS} td) share of upgrades: "
          f"{df[df.action=='upgrade'].near_earn.mean()*100:.1f}%")

    def summarize(d, name):
        if d.empty:
            print(f"  {name:34s} n=0")
            return
        r5 = d.rel_5d.to_numpy(float); r10 = d.rel_10d.to_numpy(float)
        print(f"  {name:34s} n={len(d):5,}  rel5d={np.mean(r5)*100:+6.2f}% "
              f"win5={np.mean(r5>0)*100:5.1f}%  rel10d={np.mean(r10)*100:+6.2f}% "
              f"win10={np.mean(r10>0)*100:5.1f}%")

    print("\n--- UPGRADES vs DOWNGRADES (all) ---")
    summarize(df[df.action == "upgrade"], "all upgrades")
    summarize(df[df.action == "downgrade"], "all downgrades (reference)")

    up = df[(df.action == "upgrade") & df.near_earn]
    print("\n--- UPGRADES split by earnings proximity ---")
    summarize(up, "upgrade NEAR earnings (+-3td)")
    summarize(df[(df.action == "upgrade") & ~df.near_earn], "upgrade FAR from earnings")

    far = df[(df.action == "upgrade") & ~df.near_earn].copy()
    print("\n--- FAR-from-earnings upgrades by new rating ordinal (1=sell..5=strong buy) ---")
    for o in sorted(far.new_ord.dropna().unique()):
        summarize(far[far.new_ord == o], f"new_ord={int(o)}")

    print("\n--- FAR upgrades by ordinal delta ---")
    for dl in sorted(far.ord_delta.dropna().unique()):
        if dl > 0:
            summarize(far[far.ord_delta == dl], f"delta=+{int(dl)}")

    print("\n--- FAR upgrades to ord 4-5, by year (stability) ---")
    best = far[far.new_ord >= 4]
    for y in sorted(best.year.unique()):
        summarize(best[best.year == y], f"{y}")

    print("\n--- Announcement-DAY move (untradeable; where the edge went?) ---")
    for nm, dsub in [("FAR upgrades to ord 4-5", far[far.new_ord >= 4]),
                     ("all upgrades", df[df.action == 'upgrade']),
                     ("downgrades", df[df.action == 'downgrade'])]:
        gp = dsub.gap_day0.to_numpy(float)
        print(f"  {nm:26s} n={len(dsub):6,}  day0 move mean={np.mean(gp)*100:+.2f}%  median={np.median(gp)*100:+.2f}%")

    print("\n--- Monthly volume of FAR upgrades to ord 4-5 (slow-week supply) ---")
    mo = best.date.dt.month.value_counts().sort_index()
    mnames = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    for m, c in mo.items():
        bar = "#" * int(c / 5)
        print(f"  {mnames[m-1]}  {c:5,}  {bar}")
    sep = best[best.date.dt.month == 9]
    per_sep_yr = len(sep) / max(best.year.nunique(), 1)
    print(f"  -> Sep avg/yr: {per_sep_yr:.0f} events, median rel5d "
          f"{np.median(sep.rel_5d.to_numpy(float))*100:+.2f}%" if len(sep) else "  -> no Sep events")


if __name__ == "__main__":
    main()

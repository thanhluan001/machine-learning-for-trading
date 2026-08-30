"""93_rc11_phase0_audit.py — RC-11 Phase 0: attention-stratum audit & supply.

Pre-registration: 04_backtest/rc11_pre_registration.md (2026-08-31).
User hypothesis: neglect pricing — the LOW-attention corner of SP400
ex-div events (never tested by script 70) may carry positive pre-ex
run-up CAR.

Frozen window: entry Close[T-6] -> exit Close[T-1] (5 trading sessions,
never holding through the ex-date drop). CAR vs IJH (log).

KILL GATE 0 (pre-registered): if NO attention stratum (ADV20 decile)
has mean CAR >= +0.05% with n >= 500 -> close RC-11 at Phase 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "01_data" / "db.h5"
DIV_DB = HERE.parent / "01_data" / "db_div.h5"
OUT = Path(__import__("os").environ.get("TEMP", "/tmp")) / "_rc11_phase0.json"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> None:
    print("=" * 100)
    print("RC-11 PHASE 0 — attention-stratum audit (neglect-pricing hypothesis)")
    print("=" * 100)

    with pd.HDFStore(DB, "r") as s:
        ijh = s["/macros/IJH"]
        keys = set(s.keys())
        sp_frames = {}
        for k in keys:
            if k.startswith("/sp400/"):
                df = s[k]
                df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
                sp_frames[k.split("/")[-1]] = df.set_index("Date")[
                    ["Adj_Close", "Volume", "Adj_Volume"]]

    ijh["Date"] = pd.to_datetime(ijh["Date"]).dt.normalize()
    ijh_s = ijh.set_index("Date")["Close"].astype(float)

    rows = []
    with pd.HDFStore(DIV_DB, "r") as s:
        div_keys = s.keys()
        print(f"div frames: {len(div_keys)}")
        for k in div_keys:
            pt = k.split("/")[-1]
            sp = sp_frames.get(pt)
            if sp is None or len(sp) < 60:
                continue
            d = s[k]
            d["date"] = pd.to_datetime(d["date"]).dt.normalize()
            ex = d[d["divCash"] > 0].set_index("date")["divCash"]
            if ex.empty:
                continue
            # ticker calendar from db.h5 prices
            cal = sp.index
            for T, div_amt in ex.items():
                pos = cal.searchsorted(T)
                if pos >= len(cal) or cal[pos] != T:
                    continue
                if pos < 27 or pos + 1 >= len(cal):
                    continue
                i6, i1 = pos - 6, pos - 1
                p6, p1 = sp["Adj_Close"].iloc[i6], sp["Adj_Close"].iloc[i1]
                b6, b1 = ijh_s.reindex([cal[i6], cal[i1]]).values
                if any(v is None or np.isnan(v) for v in (p6, p1, b6, b1)) or p6 <= 0:
                    continue
                # ADV20 strictly pre-entry: sessions [i6-20, i6)
                lo = max(0, i6 - 20)
                dv = (sp["Adj_Close"].iloc[lo:i6] * sp["Adj_Volume"].iloc[lo:i6])
                adv20 = float(dv.mean()) if len(dv) else np.nan
                # TTM yield: divCash sum over trailing 365d / entry price
                tr = ex[(ex.index > T - pd.Timedelta(days=365)) & (ex.index <= T)]
                yld = float(tr.sum() / p6) if p6 > 0 else np.nan
                rows.append({
                    "pt": pt, "T": T,
                    "car": float(np.log(p1 / p6) - np.log(b1 / b6)),
                    "adv20": adv20, "yield_ttm": yld,
                    "month": T.month, "year": T.year,
                })

    d = pd.DataFrame(rows)
    print(f"\nevents with full data: {len(d):,} (script 70 had 20,015)")
    print(f"reconciliation: frozen-window mean CAR {d.car.mean():+.3%} "
          f"(script 70 T-5: -0.05%)")

    # ---- attention strata: ADV20 deciles -----------------------------------
    d = d.dropna(subset=["adv20"]).copy()
    d["adv_decile"] = pd.qcut(d.adv20, 10, labels=False, duplicates="drop") + 1
    print("\n" + "-" * 100)
    print("CAR by ADV20 decile (1 = most neglected / smallest dollar volume):")
    g = d.groupby("adv_decile").agg(n=("car", "size"), mean_car=("car", "mean"),
                                    win=("car", lambda v: (v > 0).mean()))
    best = None
    for dec, r in g.iterrows():
        marker = ""
        if r.mean_car >= 0.0005 and r.n >= 500:
            marker = "  <-- PASSES stratum bar (+0.05%, n>=500)"
            best = (dec, r)
        print(f"  D{int(dec):<2d} n={int(r.n):>6,}  mean {r.mean_car:+.3%}  "
              f"win {r.win:.0%}{marker}")

    # low-attention terciles x yield quartiles (interaction peek)
    d["low_adv"] = d.adv_decile <= 3
    print("\nLow-ADV tercile (D1-D3) x yield quartile:")
    dq = d[d.low_adv].copy()
    dq["yq"] = pd.qcut(dq.yield_ttm, 4, labels=False, duplicates="drop") + 1
    g2 = dq.groupby("yq").agg(n=("car", "size"), mean_car=("car", "mean"))
    for yq, r in g2.iterrows():
        print(f"  yield Q{int(yq)} n={int(r.n):>6,}  mean {r.mean_car:+.3%}")

    # ---- supply ------------------------------------------------------------
    print("\nSupply (events/month, low-ADV tercile D1-D3):")
    sup = d[d.low_adv].groupby("month").size()
    for m in range(1, 13):
        print(f"  month {m:2d}: {sup.get(m, 0):>5,}", end="")
        if m % 4 == 0:
            print()
    sep_all = d[~d.low_adv].groupby("month").size().get(9, 0)
    print(f"\n  September: low-ADV {sup.get(9, 0):,}/yr avg, high-ADV {sep_all:,} (informational)")

    verdict = "PASS — stratum found" if best else "FAIL — CLOSE RC-11 AT PHASE 0"
    print("\n" + "=" * 100)
    print(f"PHASE 0 VERDICT: {verdict}")
    print("=" * 100)
    d.to_json(OUT, orient="records", date_format="iso")
    print(f"saved per-event frame -> {OUT}")


if __name__ == "__main__":
    main()

"""94_rc12a_phase0_labels.py — RC-12a Phase 0: short-side labels + base rates.

Pre-registration: 04_backtest/rc12a_pre_registration.md (2026-08-31).

Mirror of the long strategy at the SAME entry point (close before the
print): short_ret_1d = -(entry -> d1 log return) — the gap capture.
5-day variant includes the mirrored -10% stop with ADVERSE fills:
a stop on a short (price +10%) fills at max(stop_price, next_open) —
overnight gaps through the stop are the uninsurable tail.

Borrow: flat 1% per 5-day hold, subtracted from 5d EV (1d EV shown
gross and net-of-0.2% prorated).

KILL GATE 0: unconditional mirrored-short EV negative AND no feature
stratum (sue_lag_1 quartiles, streak terciles, rel_ret_20d quartiles)
with short-EV >= +0.5%/event (n>=200) -> close RC-12a at Phase 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "01_data" / "db.h5"
OUT = Path(__import__("os").environ.get("TEMP", "/tmp")) / "_rc12a_phase0.json"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BORROW_5D = 0.010  # 1% per 5-day hold (pre-registered)
STOP = 0.10


def main() -> None:
    print("=" * 100)
    print("RC-12a PHASE 0 — mirrored short labels, base rates, stratum EV")
    print("=" * 100)

    mx = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
    mx["report_date"] = pd.to_datetime(mx.report_date)
    with pd.HDFStore(DB, "r") as _s:
        ev = _s["/earnings/fmp"][["permaTicker", "report_date", "before_after_market"]]
    ev["report_date"] = pd.to_datetime(ev["report_date"])
    ev = ev.drop_duplicates(["permaTicker", "report_date"])
    mx = mx.merge(ev, on=["permaTicker", "report_date"], how="left")
    print(f"bam merged: {mx.before_after_market.notna().mean():.0%} of events")
    # entry/exit via pregap dates (same as the simulator)
    e = pd.to_datetime(mx.pregap_entry_date)
    x = pd.to_datetime(mx.pregap_exit_date)
    mx["entry_date"] = e

    with pd.HDFStore(DB, "r") as store:
        keys = set(store.keys())

        def prices(pt):
            k = f"/sp400/{pt}"
            if k not in keys:
                return None
            df = store[k]
            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
            return df.set_index("Date")[["Adj_Close", "Adj_Open"]]

        rows = []
        _err = [0]
        for pt, g in mx.groupby("permaTicker"):
            pl = prices(pt)
            if pl is None:
                continue
            ac = pl["Adj_Close"]
            for _, r in g.iterrows():
                try:
                    i0 = ac.index.get_indexer([pd.Timestamp(r.pregap_entry_date)], method="nearest")[0]
                    d1 = ac.index.get_indexer([pd.Timestamp(r.report_date)], method="nearest")[0]
                    # first post-print close: AMC -> report day + 1; BMO -> report day
                    if r.before_after_market == "amc":
                        d1 += 1
                    if i0 < 1 or d1 >= len(ac) - 1 or d1 + 5 >= len(ac):
                        continue
                    if (ac.index[d1] - ac.index[i0]) > pd.Timedelta(days=7):
                        continue
                    p0 = ac.iloc[i0]
                    if not np.isfinite(p0) or p0 <= 0:
                        continue
                    sr1 = -np.log(ac.iloc[d1] / p0)  # short ret entry->d1

                    # 5-day with adverse-fill stop: cover if price RISES 10%
                    stop_px = p0 * (1 + STOP)
                    worst = 0.0
                    stopped = False
                    stop_ret = None
                    for j in range(i0 + 1, min(d1 + 6, len(ac))):
                        cj = ac.iloc[j]
                        if not np.isfinite(cj):
                            continue
                        # adverse fill: overnight gap through stop -> fill at open
                        oj = pl["Adj_Open"].iloc[j] if np.isfinite(pl["Adj_Open"].iloc[j]) else cj
                        if oj >= stop_px:
                            stop_ret = -np.log(max(stop_px, oj) / p0)
                            stopped = True
                            break
                        if cj >= stop_px:
                            stop_ret = -np.log(stop_px / p0)
                            stopped = True
                            break
                    if stopped:
                        sr5 = stop_ret
                    else:
                        x5i = min(d1 + 5, len(ac) - 1)
                        sr5 = -np.log(ac.iloc[x5i] / p0)
                    rows.append({"pt": pt, "report_date": r.report_date,
                                 "short_1d": sr1, "short_5d": sr5,
                                 "stopped": stopped,
                                 "sue1": r.get("sue_lag_1", np.nan),
                                 "streak": r.get("consecutive_surprises_pre", np.nan),
                                 "rr20": r.get("rel_ret_20d", np.nan)})
                except Exception as ex:
                    if _err[0] < 3:
                        print(f"  [WARN] {pt} {str(r.report_date)[:10]}: {type(ex).__name__}: {ex}")
                        _err[0] += 1
                    continue

    d = pd.DataFrame(rows)
    n = len(d)
    print(f"\nevents with mirrored labels: {n:,}")

    print("\n--- base rates (pooled) ---")
    print(f"  short_1d  mean {d.short_1d.mean():+.3%}  median {d.short_1d.median():+.3%}  "
          f"win {((d.short_1d > 0)).mean():.0%}")
    print(f"  P(stock fell >1%) { (d.short_1d > 0.01).mean():.1%} | >2% { (d.short_1d > 0.02).mean():.1%} "
          f"| >3% { (d.short_1d > 0.03).mean():.1%}")
    print(f"  short_5d  mean {d.short_5d.mean():+.3%}  median {d.short_5d.median():+.3%}  "
          f"win {(d.short_5d > 0).mean():.0%}")
    print(f"  stop hit rate (adverse +10%): {d.stopped.mean():.1%}")
    print(f"  short_5d net of 1% borrow   : {d.short_5d.mean() - BORROW_5D:+.3%}")

    # adverse tail table (the SQNS mirror check)
    worst = d.nsmallest(5, "short_5d")
    print("\n  worst 5 adverse outcomes for shorts (the SQNS-mirror check):")
    for _, r in worst.iterrows():
        print(f"    {r.pt} {str(r.report_date)[:10]}  short_5d {r.short_5d:+.1%}"
              f"{' STOPPED' if r.stopped else ''}")

    # ---- stratum EV (kill gate 0 second half) ----------------------------
    print("\n--- short-EV by stratum (5d, net of borrow) ---")
    d["ev5"] = d.short_5d - BORROW_5D
    any_pass = False
    for col, name, q in [("sue1", "sue_lag_1 quartile", 4),
                         ("streak", "beat-streak tercile", 3),
                         ("rr20", "rel_ret_20d quartile", 4)]:
        sub = d.dropna(subset=[col]).copy()
        if len(sub) < 500:
            continue
        sub["b"] = pd.qcut(sub[col], q, labels=False, duplicates="drop")
        g = sub.groupby("b").agg(n=("ev5", "size"), mean=("ev5", "mean"),
                                 win=("ev5", lambda v: (v > 0).mean()))
        print(f"  {name} ({col}):")
        for b, r in g.iterrows():
            flag = ""
            if r["mean"] >= 0.005 and r["n"] >= 200:
                flag = "  <-- passes stratum bar (+0.5%, n>=200)"
                any_pass = True
            print(f"    Q{int(b)+1}: n={int(r['n']):>6,}  EV5 {r['mean']:+.3%}  "
                  f"win {r['win']:.0%}{flag}")

    print("\n" + "=" * 100)
    base_ok = (d.short_1d.mean() - 0.002) >= 0 or (d.short_5d.mean() - BORROW_5D) >= 0
    verdict = "PASS (floor exists)" if (base_ok or any_pass) else \
        "FAIL — no floor: unconditional EV negative AND no stratum >= +0.5%"
    print(f"PHASE 0 VERDICT: {verdict}")
    print("=" * 100)
    d.to_json(OUT, orient="records", date_format="iso")


if __name__ == "__main__":
    main()

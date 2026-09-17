"""Short-side economics diagnostic (exploratory, non-gating).

User idea: misses are heavily punished (-3.4% to -5.4%) and the model can
identify them -> short the miss side; also keep the high-confidence long.

CRITICAL: pregap_return embeds a LONG-side 10% stop, so -pregap_return
would OVERSTATE short returns (upside gaps truncated). This script
recomputes from prices:
  long_raw      exit/entry - 1                       (no stop)
  short_raw     -(exit/entry - 1)                    (no stop)
  short_stop    short with a +10% upside stop (cover at the crossing price)
  long_stop     long with the -10% stop (sanity: should ~match pregap_return)
Returns are 5-session holds from the same entry/exit dates as the matrix.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import os

os.chdir(ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = HERE / "archive" / "experiments" / "short_side"
OUT.mkdir(parents=True, exist_ok=True)
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = load("bt_short", HERE / "51_hp_theta_sweep_23feat.py")
DB = bt.DB

# prices
with pd.HDFStore(DB, "r") as st:
    px = {}
    for k in st.keys():
        if k.startswith("/sp400/"):
            df = st[k]
            col = "Adj_Close" if "Adj_Close" in df.columns else "Close"
            d = pd.to_datetime(df["Date"]).to_numpy().astype("datetime64[D]")
            o = np.argsort(d)
            px[k.split("/")[-1]] = (d[o], df[col].to_numpy(dtype=float)[o])
with pd.HDFStore(DB_SP600, "r") as st:
    for k in st.keys():
        parts = k.split("/")
        if len(parts) == 3 and parts[1] == "sp600":
            df = st[k]
            if "Date" in df.columns and "Adj_Close" in df.columns:
                d = pd.to_datetime(df["Date"]).to_numpy().astype("datetime64[D]")
                o = np.argsort(d)
                px[parts[2]] = (d[o], df["Adj_Close"].to_numpy(dtype=float)[o])

P = pd.read_hdf(HERE / "archive" / "experiments" / "beat_prediction" / "oos_predictions.h5", "p")
v7 = pd.read_hdf(DB, "/features/train_matrix_v7c")
v7["report_date"] = pd.to_datetime(v7["report_date"])
v7["entry_date"] = pd.to_datetime(v7["entry_date"])
v7["exit_date"] = pd.to_datetime(v7["exit_date"])
P = P.merge(v7[["permaTicker", "report_date", "entry_date", "exit_date"]],
            on=["permaTicker", "report_date"], how="left")

long_raw, short_raw, short_stop, long_stop = [], [], [], []
for pt, ed, xd in zip(P.permaTicker, P.entry_date, P.exit_date):
    lr = sr = ss = ls = np.nan
    if pt in px and pd.notna(ed) and pd.notna(xd):
        d, c = px[pt]
        e = np.searchsorted(d, ed.to_datetime64().astype("datetime64[D]"))
        x = np.searchsorted(d, xd.to_datetime64().astype("datetime64[D]"))
        if e < len(d) and x < len(d) and e < x and c[e] > 0:
            ep = c[e]
            lr = c[x] / ep - 1.0
            sr = -lr
            path = c[e:x + 1]
            path = path[np.isfinite(path)]
            sp_up, sp_dn = ep * 1.10, ep * 0.90
            hit_up = path[1:][path[1:] >= sp_up]
            ss = -(hit_up[0] / ep - 1.0) if len(hit_up) else sr
            hit_dn = path[1:][path[1:] <= sp_dn]
            ls = (hit_dn[0] / ep - 1.0) if len(hit_dn) else lr
    long_raw.append(lr); short_raw.append(sr); short_stop.append(ss); long_stop.append(ls)
P["long_raw"] = long_raw; P["short_raw"] = short_raw
P["short_stop"] = short_stop; P["long_stop"] = long_stop
P = P[P.long_raw.notna()].copy()
print(f"events with recomputed returns: {len(P):,}")
print(f"sanity: corr(long_stop, pregap_return) = {P.long_stop.corr(P.pregap_return):.4f}  "
      f"mean long_stop={P.long_stop.mean()*100:+.3f}% vs pregap={P.pregap_return.mean()*100:+.3f}%")
print(f"short_raw mean={P.short_raw.mean()*100:+.3f}%  short_stop mean={P.short_stop.mean()*100:+.3f}% "
      f"(stop costs {abs(P.short_stop.mean()-P.short_raw.mean())*100:.3f}pp)")

print("\n=== SHORT-SIDE by p_beat (predicted-miss confidence) ===")
print(f"{'thr<':>5} {'n':>5} {'miss%':>6} {'short_raw%':>11} {'SE':>5} {'t':>6} {'short_stop%':>12} "
      f"{'P(>+10%)':>9} {'max loss%':>10}")
for thr in (0.5, 0.4, 0.3, 0.2, 0.1):
    s = P[P.p_beat < thr]
    if len(s) < 30:
        continue
    se = s.short_raw.std(ddof=1) / np.sqrt(len(s))
    print(f"{thr:>5.1f} {len(s):>5} {(1-s.beat.mean())*100:>6.1f} {s.short_raw.mean()*100:>11.3f} "
          f"{se*100:>5.2f} {s.short_raw.mean()/se:>6.2f} {s.short_stop.mean()*100:>12.3f} "
          f"{(s.long_raw>0.10).mean()*100:>8.1f}% {s.long_raw.max()*100:>9.1f}%")

print("\n=== LONG high-confidence beat (same recomputed returns) ===")
print(f"{'thr>=':>6} {'n':>5} {'beat%':>6} {'long_raw%':>10} {'SE':>5} {'t':>6} {'long_stop%':>11}")
for thr in (0.6, 0.7, 0.8, 0.9):
    s = P[P.p_beat >= thr]
    se = s.long_raw.std(ddof=1) / np.sqrt(len(s))
    print(f"{thr:>6.1f} {len(s):>5} {s.beat.mean()*100:>6.1f} {s.long_raw.mean()*100:>10.3f} "
          f"{se*100:>5.2f} {s.long_raw.mean()/se:>6.2f} {s.long_stop.mean()*100:>11.3f}")

print("\n=== COMBINED long-short (equal weight, gross) ===")
print(f"{'long>=':>7} {'short<':>7} {'nL':>5} {'nS':>5} {'L%':>7} {'S%':>7} {'avg%':>7} {'t':>6}")
for lt, st_ in ((0.8, 0.3), (0.8, 0.2), (0.7, 0.3), (0.9, 0.2)):
    L = P[P.p_beat >= lt]; S = P[P.p_beat < st_]
    if len(L) < 20 or len(S) < 20:
        continue
    comb = np.concatenate([L.long_raw.to_numpy(), S.short_raw.to_numpy()])
    se = comb.std(ddof=1) / np.sqrt(len(comb))
    print(f"{lt:>7.1f} {st_:>7.1f} {len(L):>5} {len(S):>5} {L.long_raw.mean()*100:>7.3f} "
          f"{S.short_raw.mean()*100:>7.3f} {comb.mean()*100:>7.3f} {comb.mean()/se:>6.2f}")

print("\n=== TAIL RISK of shorting predicted misses ===")
S = P[P.p_beat < 0.3]
q = np.percentile(S.long_raw, [50, 90, 95, 99, 99.9])
print(f"  long_raw quantiles 50/90/95/99/99.9 = {q.round(3)*100}%  (short loses these)")
print(f"  P(long_raw > +10%) = {(S.long_raw>0.10).mean()*100:.1f}%   P(> +20%) = {(S.long_raw>0.20).mean()*100:.1f}%")
print(f"  worst single name: {S.long_raw.max()*100:+.1f}%")
print(f"  short_raw mean {S.short_raw.mean()*100:+.3f}%  vs short_stop {S.short_stop.mean()*100:+.3f}% "
      f"-> stop costs {(S.short_stop.mean()-S.short_raw.mean())*100:+.3f}pp")

with open(OUT / "report.json", "w") as f:
    json.dump({"n": int(len(P)),
               "short_by_thr": {str(t): {"n": int((P.p_beat < t).sum()),
                                         "short_raw_pct": round(float(P[P.p_beat < t].short_raw.mean() * 100), 3),
                                         "short_stop_pct": round(float(P[P.p_beat < t].short_stop.mean() * 100), 3)}
                                for t in (0.5, 0.4, 0.3, 0.2, 0.1)},
               "long_by_thr": {str(t): {"n": int((P.p_beat >= t).sum()),
                                        "long_raw_pct": round(float(P[P.p_beat >= t].long_raw.mean() * 100), 3)}
                               for t in (0.6, 0.7, 0.8, 0.9)}}, f, indent=2)
P[["permaTicker", "report_date", "is_sp400", "p_beat", "beat", "long_raw", "short_raw",
   "short_stop", "fold"]].to_hdf(OUT / "returns.h5", key="r", format="table")
print(f"\nwrote {OUT}/report.json")

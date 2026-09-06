"""RC-13 — V7 threshold sensitivity + bootstrap sweep (60/61-equivalent).

Full portfolio construct (4-slot, stop 10%, force-refresh, ADV, XLF-excl,
per-event 10/30bp costs) at each threshold. Reports DEV folds 1-3 and
holdout 2026H1: trades, win, cost-adj mean, bootstrap CI95, NAV, plus
per-universe splits and border-band economics.

Robustness criteria (mirroring the V6 freeze process):
- selected threshold must have DEV mean CI excluding 0 AND holdout > 0
- no cliff: neighboring thresholds remain positive on both windows
- border bands show monotone-or-flat economics (no inverted gradient)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v108 = load("v108", HERE / "108_rc13_v7_full_bar.py")
v104 = v108.v104
fr = v108.fr
bt = v108.bt

DB = bt.DB
MATRIX_KEY = "/features/train_matrix_combined"
FEATURES = v108.FEATURES
GATES = v108.GATES
XLF = v108.XLF
GRID = [0.28, 0.30, 0.32, 0.33, 0.35, 0.38, 0.40, 0.43]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def build_cands(oos, w0, w1, th):
    m = ((oos.score >= th) & (~oos.sector.isin(XLF)) & oos.pregap_return.notna()
         & oos.adv_pass
         & (pd.to_datetime(oos.report_date) >= pd.Timestamp(w0))
         & (pd.to_datetime(oos.report_date) <= pd.Timestamp(w1)))
    c = oos[m].copy()
    c["entry_date"] = pd.to_datetime(c.entry_date)
    rows = []
    for r in c.itertuples(index=False):
        pl = v104.get_prices(r.permaTicker)
        if pl is None:
            continue
        dates, closes = pl
        e = int(np.searchsorted(dates, np.datetime64(r.entry_date), side="left"))
        x = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(r.exit_date)), side="left"))
        if e >= len(closes) or x >= len(closes) or x <= e:
            continue
        ret = fr.ret_with_stop(closes, e, x, stop=v104.STOP)
        if ret is None:
            continue
        rows.append({"permaTicker": r.permaTicker, "entry_date": r.entry_date,
                     "exit_date": pd.Timestamp(dates[x]), "score": r.score,
                     "is_sp400": int(r.is_sp400), "cost": float(r.cost),
                     "entry_idx": e, "ret5": ret, "ret_cost": ret - float(r.cost)})
    out = pd.DataFrame(rows)
    return out.sort_values(["entry_date", "score"], ascending=[True, False]).reset_index(drop=True)


def sim_pool(oos, windows, th):
    trades = []
    for w0, w1 in windows:
        trades.extend(v108.simulate(fr.weekly_slate(build_cands(oos, w0, w1, th))))
    return pd.DataFrame(trades)


def stat_line(df):
    if df is None or df.empty:
        return "n=0"
    r = df.ret_cost.astype(float)
    lo, hi = v108.boot_ci(r.to_numpy())
    nav = float(np.prod(1 + r.sort_index(kind="stable") / v104.N_SLOTS))
    return (f"n={len(df):3d} win={(r>0).mean():.0%} mean {r.mean():+.2%} "
            f"CI[{lo:+.1%},{hi:+.1%}] NAV {nav:.2f}x")


def main() -> None:
    print("=" * 100)
    print("RC-13 V7 THRESHOLD SENSITIVITY + BOOTSTRAP (full sim construct)")
    print("=" * 100)
    d = pd.read_hdf(DB, MATRIX_KEY)
    s6 = pd.read_hdf(v104.DB_SP600, "/features/train_matrix_sp600_pt",
                     columns=["permaTicker", "report_date", "adv20"])
    s6["report_date"] = pd.to_datetime(s6.report_date)
    d = d.merge(s6, on=["permaTicker", "report_date"], how="left")
    d["adv_pass"] = ~(d.adv20.notna() & (d.adv20 < 1e7))

    frames = []
    for fi, tr, sv, ts in v108.folds_for(d):
        train = pd.concat([tr, sv], ignore_index=True)
        for g in GATES:
            y = train[g].astype(int).to_numpy()
            yt = ts[g].astype(int).to_numpy()
            clf = v108.fit(train[FEATURES], y, ts[FEATURES], yt)
            ts[f"p_{g}"] = clf.predict_proba(ts[FEATURES])[:, 1]
        ts["fold"] = fi
        ts["score"] = ts[[f"p_{g}" for g in GATES]].min(axis=1)
        frames.append(ts)
        print(f"  fold {fi} scored")
    oos = pd.concat(frames, ignore_index=True)
    v104._bulk_load()

    dev_windows = [(w0, w1) for _, w0, w1 in v104.FOLDS[:3]]
    res = {}
    print(f"\n{'th':>5} | {'DEV folds 1-3':58s} | HOLDOUT 2026H1")
    print("-" * 130)
    pooled_dev = {}
    for th in GRID:
        dev = sim_pool(oos, dev_windows, th)
        ho = sim_pool(oos, [("2026-01-01", "2026-06-30")], th)
        pooled_dev[th] = dev
        print(f"{th:5.2f} | {stat_line(dev):58s} | {stat_line(ho)}")
        r = dev.ret_cost.astype(float) if len(dev) else pd.Series(dtype=float)
        rh = ho.ret_cost.astype(float) if len(ho) else pd.Series(dtype=float)
        res[f"{th}"] = {
            "dev": {"n": int(len(r)), "mean": float(r.mean()) if len(r) else None,
                    "win": float((r > 0).mean()) if len(r) else None},
            "holdout": {"n": int(len(rh)), "mean": float(rh.mean()) if len(rh) else None,
                        "win": float((rh > 0).mean()) if len(rh) else None},
        }

    # per-universe at selected threshold
    th = 0.33
    dev, ho = pooled_dev[th], sim_pool(oos, [("2026-01-01", "2026-06-30")], th)
    print(f"\nper-universe @ {th}:")
    for lab, df in [("DEV", dev), ("HOLDOUT", ho)]:
        for u, ul in [(1, "SP400"), (0, "SP600")]:
            sub = df[df.is_sp400 == u]
            r = sub.ret_cost.astype(float)
            print(f"  {lab:8s} {ul}: {stat_line(sub) if len(sub) else 'n=0'}")

    # border-band economics on executed trades (all thresholds pooled)
    alltr = pd.concat([df for df in pooled_dev.values() if len(df)], ignore_index=True)
    alltr = alltr.drop_duplicates(subset=["permaTicker", "entry_date"])
    print(f"\nborder bands (unique executed DEV trades, n={len(alltr)}):")
    for lo, hi in [(0.28, 0.30), (0.30, 0.33), (0.33, 0.35), (0.35, 0.40), (0.40, 1.0)]:
        b = alltr[(alltr.score >= lo) & (alltr.score < hi)].ret_cost.astype(float)
        if len(b):
            print(f"  [{lo:.2f},{hi:.2f}): n={len(b):3d} mean {b.mean():+.2%} "
                  f"win {(b>0).mean():.0%}")

    out = HERE / "archive" / "experiments" / "gate_decomposition_v6" / "rc13_v7_threshold_sweep.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nsaved", out)


if __name__ == "__main__":
    main()

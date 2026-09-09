"""RC-13 Amendment A — no-flag-bar validation (pre-registered
rc13_amendment_a_no_flag_bar.md, 2026-09-09).

ELIGIBILITY: score_noflag >= theta_elig (is_sp400 forced 0 in scoring).
RANKING:     score_real (flag as trained).
theta_elig swept on DEV (NAV-max, >=50 trades), then holdout read.

Gates A1: DEV mean > 0 AND >= +4.39% (baseline V7); holdout mean > 0
AND >= +3.5% (baseline +4.26% within noise).
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
BASE_DEV = 0.0439
BASE_HO = 0.035  # pre-registered noise-adjusted holdout floor

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def build_cands(oos, w0, w1, th):
    m = ((oos.score_noflag >= th)          # <-- AMENDMENT: eligibility bar
         & (~oos.sector.isin(XLF)) & oos.pregap_return.notna() & oos.adv_pass
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
                     "exit_date": pd.Timestamp(dates[x]),
                     "score": r.score,             # <-- RANKING: real score
                     "score_noflag": r.score_noflag, "is_sp400": int(r.is_sp400),
                     "cost": float(r.cost), "entry_idx": e, "ret5": ret,
                     "ret_cost": ret - float(r.cost)})
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
    return (f"n={len(df):3d} win={(r > 0).mean():.0%} mean {r.mean():+.3%} "
            f"CI[{lo:+.1%},{hi:+.1%}] NAV {nav:.2f}x")


def main() -> None:
    print("=" * 100)
    print("RC-13 AMENDMENT A — no-flag-bar (eligibility on score_noflag, ranking on score_real)")
    print("=" * 100)
    d = pd.read_hdf(DB, MATRIX_KEY)
    s6 = pd.read_hdf(v104.DB_SP600, "/features/train_matrix_sp600_pt",
                     columns=["permaTicker", "report_date", "adv20"])
    s6["report_date"] = pd.to_datetime(s6.report_date)
    d = d.merge(s6, on=["permaTicker", "report_date"], how="left")
    d["adv_pass"] = ~(d.adv20.notna() & (d.adv20 < 1e7))

    import xgboost as xgb
    frames = []
    for fi, tr, sv, ts in v108.folds_for(d):
        train = pd.concat([tr, sv], ignore_index=True)
        for g in GATES:
            y = train[g].astype(int).to_numpy()
            yt = ts[g].astype(int).to_numpy()
            clf = v108.fit(train[FEATURES], y, ts[FEATURES], yt)
            ts[f"p_{g}"] = clf.predict_proba(ts[FEATURES])[:, 1]
            Xcf = ts[FEATURES].copy()
            Xcf["is_sp400"] = 0.0
            ts[f"p0_{g}"] = clf.predict_proba(Xcf)[:, 1]
        ts["fold"] = fi
        ts["score"] = ts[[f"p_{g}" for g in GATES]].min(axis=1)
        ts["score_noflag"] = ts[[f"p0_{g}" for g in GATES]].min(axis=1)
        frames.append(ts)
        print(f"  fold {fi} scored")
    oos = pd.concat(frames, ignore_index=True)
    v104._bulk_load()

    dev_windows = [(w0, w1) for _, w0, w1 in v104.FOLDS[:3]]
    print(f"\n{'th':>5} | {'DEV (no-flag eligibility)':58s} | HOLDOUT")
    best = None
    for th in [0.28, 0.30, 0.31, 0.32, 0.33, 0.35]:
        dev = sim_pool(oos, dev_windows, th)
        ho = sim_pool(oos, [("2026-01-01", "2026-06-30")], th)
        print(f"{th:5.2f} | {stat_line(dev):58s} | {stat_line(ho)}")
        nav = float(np.prod(1 + dev.ret_cost.astype(float).sort_index(kind="stable") / v104.N_SLOTS))
        if len(dev) >= 50 and (best is None or nav > best[1]):
            best = (th, nav)
    th_sel = best[0] if best else 0.31
    print(f"\nDEV-selected theta_elig: {th_sel:.2f}")

    dev = sim_pool(oos, dev_windows, th_sel)
    ho = sim_pool(oos, [("2026-01-01", "2026-06-30")], th_sel)
    dm = dev.ret_cost.astype(float).mean() if len(dev) else float("nan")
    hm = ho.ret_cost.astype(float).mean() if len(ho) else float("nan")
    print(f"\nA1 CHECK: DEV {dm:+.3%} (need > 0 and >= +4.39%) | "
          f"HOLDOUT {hm:+.3%} (need > 0 and >= +3.5%)")
    a1 = (dm > 0 and dm >= BASE_DEV - 0.0005 and hm > 0 and hm >= BASE_HO - 0.0005)
    print(f"A1 VERDICT: {'PASS' if a1 else 'FAIL'}")

    # residual flag-decisive-style trades under the amendment
    resid = ho[(ho.is_sp400 == 1) & (ho.score_noflag >= th_sel)]
    band = resid[(resid.score_noflag < th_sel + 0.02)]
    if len(band):
        r = band.ret_cost.astype(float)
        print(f"\ntie-breaker: marginal no-flag band [{th_sel:.2f},{th_sel+0.02}) in holdout: "
              f"n={len(band)} mean {r.mean():+.3%}")

    out = HERE / "archive" / "experiments" / "gate_decomposition_v6" / "rc13_amendment_a.json"
    out.write_text(json.dumps({
        "theta_elig": th_sel,
        "dev": {"n": int(len(dev)), "mean": float(dm)},
        "holdout": {"n": int(len(ho)), "mean": float(hm)},
        "A1": bool(a1)}, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()

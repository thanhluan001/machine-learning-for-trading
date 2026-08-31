"""99_rc12b_phase2_transfer.py — RC-12b Phase 2: frozen-V6 transfer to SP600.

Pre-registration: 04_backtest/rc12b_pre_registration.md (2026-08-31).

Frozen phase_g_v6_gate_decomposition gates (SP400-trained, never retrained)
score the SP600 matrix. Policy identical to live: min-gate >= 0.33, XLF
excluded, weekly top-4 slate, force-refresh mh=4, 10% stop. Binding Phase-0
safeguards: ADV20 >= $10M entry filter; spread-adjusted returns
(-0.30% round trip, pre-registered haircut = 1.1x realized SP400 slippage
band) reported alongside raw.

Windows:
  full   2019-2026 (features NaN-heavy pre-2022 — informational)
  clean  2022-01+  (full feature population)
  folds  the four V6 fold windows (DEV 1-3 + 2026-H1 holdout) for
         like-for-like comparison with script-88 SP400 baselines.

GATE (pre-registered): transfer trade-edge positive (spread-adjusted) on
DEV folds AND holdout -> candidate shadow book. Fail + base-rate < 1.5x
-> close (base rate measured 1.10x in Phase 1, so Phase 3 is NOT
auto-triggered).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
DB_PROD = ROOT / "01_data" / "db.h5"
MODEL_DIR = ROOT / "03_model" / "models" / "phase_g_v6_gate_decomposition"
THRESH = 0.33
STOP = 0.10
N_SLOTS = 4
SPREAD_ADJ = 0.0030  # 30bp round trip, pre-registered
XLF = {"XLF"}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


fr = _load("fr", HERE / "63_force_refresh_backtest.py")
bt = fr.bt
DEPLOY_FEATURES = bt.DEPLOY_FEATURES
FOLDS = [(1, "2024-07-01", "2024-12-31"), (2, "2025-01-01", "2025-06-30"),
         (3, "2025-07-01", "2025-12-31"), (4, "2026-01-01", "2026-06-30")]


_PCACHE: dict = {}


def get_prices(pt):
    if pt in _PCACHE:
        return _PCACHE[pt]
    out = None
    with pd.HDFStore(DB_SP600, "r") as s6:
        k = f"/sp600/{pt}"
        if k in s6.keys():
            p = s6[k]
            p["Date"] = pd.to_datetime(p["Date"]).dt.normalize()
            out = (p["Date"].values, p["Adj_Close"].to_numpy(float))
    if out is None:
        with pd.HDFStore(DB_PROD, "r") as sp:
            k = f"/sp400/{pt}"
            if k in sp.keys():
                p = sp[k]
                p["Date"] = pd.to_datetime(p["Date"]).dt.normalize()
                out = (p["Date"].values, p["Adj_Close"].to_numpy(float))
    _PCACHE[pt] = out
    return out


def score_v6(df):
    X = xgb.DMatrix(df[DEPLOY_FEATURES])
    for g in ("pass_g1", "pass_g2", "pass_g3"):
        m = xgb.Booster()
        m.load_model(str(MODEL_DIR / g / "classifier.json"))
        df["p_" + g] = m.predict(X)
    df["score"] = df[["p_pass_g1", "p_pass_g2", "p_pass_g3"]].min(axis=1)
    return df


def build_cands(d, w0, w1):
    m = ((d.score >= THRESH) & (~d.sector.isin(XLF)) & d.pregap_return.notna()
         & (d.adv20 >= 1e7)
         & (pd.to_datetime(d.report_date) >= pd.Timestamp(w0))
         & (pd.to_datetime(d.report_date) <= pd.Timestamp(w1)))
    c = d[m].copy()
    c["entry_date"] = pd.to_datetime(c.entry_date)
    rows = []
    for _, r in c.iterrows():
        pl = get_prices(r.permaTicker)
        if pl is None:
            continue
        dates, closes = pl
        e = int(np.searchsorted(dates, np.datetime64(r.entry_date), side="left"))
        x = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(r.exit_date)), side="left"))
        if e >= len(closes) or x >= len(closes) or x <= e:
            continue
        ret = fr.ret_with_stop(closes, e, x, stop=STOP)
        if ret is None:
            continue
        rows.append({"permaTicker": r.permaTicker, "entry_date": r.entry_date,
                     "exit_date": pd.Timestamp(dates[x]), "score": r.score,
                     "entry_idx": e, "ret5": ret})
    out = pd.DataFrame(rows)
    return out.sort_values(["entry_date", "score"], ascending=[True, False]).reset_index(drop=True)


def simulate(slate):
    slots, trades = [], []
    for ev in slate.itertuples(index=False):
        ed = ev.entry_date
        evd = ev._asdict()
        kept = []
        for s in slots:
            if s["exit_date"] <= ed:
                trades.append({**s, "return": s["ret5"], "exit_reason": "natural"})
            else:
                kept.append(s)
        slots = kept
        if len(slots) < N_SLOTS:
            slots.append(evd)
        else:
            scored = []
            for s in slots:
                if fr.iso_week(s["entry_date"]) >= fr.iso_week(ed):
                    continue
                pl = _PCACHE.get(s["permaTicker"])
                if pl is None:
                    continue
                vdates, vcloses = pl
                sidx = int(np.searchsorted(vdates, np.datetime64(ed), side="right")) - 1
                if sidx - s["entry_idx"] < 4:
                    continue
                scored.append((s, sidx, vcloses))
            if scored:
                victim, sidx, vcloses = min(scored, key=lambda z: z[0]["entry_date"])
                part = fr.ret_with_stop(vcloses, victim["entry_idx"], sidx, stop=STOP)
                trades.append({**victim, "return": part if part is not None else victim["ret5"],
                               "exit_reason": "force_refresh"})
                slots.remove(victim)
                slots.append(evd)
    for s in slots:
        trades.append({**s, "return": s["ret5"], "exit_reason": "end"})
    return trades


def report(label, trades):
    df = pd.DataFrame(trades)
    if df.empty:
        print(f"\n[{label}] NO TRADES")
        return
    r = df["return"].astype(float)
    nav = float(np.prod(1 + r.sort_index(kind="stable") / N_SLOTS))
    print(f"\n[{label}] trades {len(df)} | win {(r > 0).mean():.0%} | "
          f"raw mean {r.mean():+.3%} | spread-adj mean {r.mean() - SPREAD_ADJ:+.3%} | NAV {nav:.3f}x")
    yrs = pd.to_datetime(df.entry_date).dt.year
    for y in sorted(yrs.unique()):
        ry = r[yrs == y]
        print(f"    {y}: n={len(ry):>3}  mean {ry.mean():+.2%}  win {(ry > 0).mean():.0%}")
    return df


def main() -> None:
    print("=" * 100)
    print("RC-12b PHASE 2 — frozen V6 (SP400-trained) transferred to SP600")
    print("=" * 100)
    d = pd.read_hdf(DB_SP600, "/features/train_matrix_sp600")
    print(f"events: {len(d):,}")
    d = score_v6(d)
    print(f"scored: min-gate >= {THRESH}: {(d.score >= THRESH).sum()} "
          f"({(d.score >= THRESH).mean():.1%}) | XLF-excluded + ADV>=10M next")

    report("FULL 2019-2026 (features NaN-heavy pre-2022)", simulate(fr.weekly_slate(build_cands(d, "2018-09-01", "2026-06-30"))))
    report("CLEAN 2022+ (full feature population)", simulate(fr.weekly_slate(build_cands(d, "2022-01-01", "2026-06-30"))))

    print("\n--- fold windows (V6 folds; SP400 script-88 baselines quoted) ---")
    dev_all = []
    for i, w0, w1 in FOLDS:
        t = simulate(fr.weekly_slate(build_cands(d, w0, w1)))
        report(f"fold {i} {w0[:7]}..{w1[:7]}", t)
        if i < 4:
            dev_all.extend(t)
    report("DEV folds 1-3 (SP400 baseline: mean +3.08%, NAV 2.41x)", dev_all)
    t4 = simulate(fr.weekly_slate(build_cands(d, "2026-01-01", "2026-06-30")))
    report("HOLDOUT 2026-H1 (SP400 baseline: mean +4.15%, NAV 1.84x)", t4)

    out = HERE / "archive" / "experiments" / "gate_decomposition_v6" / "rc12b_phase2_transfer.json"
    res = {}
    for lbl, tr in [("full", simulate(fr.weekly_slate(build_cands(d, "2018-09-01", "2026-06-30")))),
                    ("clean2022", simulate(fr.weekly_slate(build_cands(d, "2022-01-01", "2026-06-30")))),
                    ("holdout2026H1", simulate(fr.weekly_slate(build_cands(d, "2026-01-01", "2026-06-30"))))]:
        df = pd.DataFrame(tr)
        res[lbl] = {"n": len(df), "mean": float(df["return"].mean()) if len(df) else None,
                    "win": float((df["return"] > 0).mean()) if len(df) else None}
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nsaved", out)


if __name__ == "__main__":
    main()

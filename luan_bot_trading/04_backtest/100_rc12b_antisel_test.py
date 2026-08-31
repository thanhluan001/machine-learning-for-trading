"""100_rc12b_antisel_test.py — anti-selection diagnostic (pre-RC decision).

User question (2026-08-31): the V6 score is anti-informative on SP600
(rank-IC -0.054 on DEV). Before opening a new RC, characterize the
anti-selection as a TRADE RULE with fold + holdout statistics.

Rules tested (declared before running):
  R1  events with score < 0.33 (all gate-rejected names)
  R2  bottom-quintile by score (strongest anti-selection)
Policy constraints identical to Phase 2: ADV>=10M, XLF excluded,
v4 timing contract, 10% stop, 30bp spread haircut. Both event-level
(no capacity) and the 4-slot weekly slate (lowest scores first).

CRITICAL CAVEAT carried in every table: the SP600 matrix is
current-members-backward (survivorship). Low-fundamental-score names
that survived look better than the pt-in-time truth. This diagnostic
CANNOT separate anti-selection edge from survivorship inflation —
only the RC-13 Phase-0 rebuild can.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
DB_PROD = ROOT / "01_data" / "db.h5"
MODEL_DIR = ROOT / "03_model" / "models" / "phase_g_v6_gate_decomposition"
STOP = 0.10
N_SLOTS = 4
SPREAD_ADJ = 0.0030
FOLDS = [("fold1 DEV", "2024-07-01", "2024-12-31"),
         ("fold2 DEV", "2025-01-01", "2025-06-30"),
         ("fold3 DEV", "2025-07-01", "2025-12-31"),
         ("holdout", "2026-01-01", "2026-06-30")]

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

_PCACHE = {}


def _bulk_load():
    """One pass per store — per-ticker HDFStore opens are the proven killer."""
    with pd.HDFStore(DB_SP600, "r") as s6:
        for k in s6.keys():
            if k.startswith("/sp600/") and k.count("/") == 2:
                p = s6[k]
                _PCACHE[k.split("/")[-1]] = (pd.to_datetime(p["Date"]).dt.normalize().values,
                                             p["Adj_Close"].to_numpy(float))
    with pd.HDFStore(DB_PROD, "r") as sp:
        for k in sp.keys():
            if k.startswith("/sp400/") and k.count("/") == 2:
                pt = k.split("/")[-1]
                if pt not in _PCACHE:
                    p = sp[k]
                    _PCACHE[pt] = (pd.to_datetime(p["Date"]).dt.normalize().values,
                                   p["Adj_Close"].to_numpy(float))


def get_prices(pt):
    return _PCACHE.get(pt)


def bootstrap_ci(x, n=10000, seed=11):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    boots = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return np.percentile(boots, [2.5, 97.5])


def main() -> None:
    print("=" * 100)
    print("ANTI-SELECTION DIAGNOSTIC — SP600, V6 score inverted as signal")
    print("=" * 100)
    d = pd.read_hdf(DB_SP600, "/features/train_matrix_sp600")
    d["report_date"] = pd.to_datetime(d.report_date)
    print("bulk-loading prices (one pass per store)...")
    _bulk_load()
    print(f"  price cache: {len(_PCACHE)} tickers")
    X = xgb.DMatrix(d[DEPLOY_FEATURES])
    preds = []
    for g in ("pass_g1", "pass_g2", "pass_g3"):
        m = xgb.Booster()
        m.load_model(str(MODEL_DIR / g / "classifier.json"))
        preds.append(m.predict(X))
    d["score"] = np.min(preds, axis=0)
    base = d[(d.adv20 >= 1e7) & (~d.sector.isin({"XLF"})) & d.pregap_return.notna()].copy()

    print("\n--- rank-IC(score, return) per window (stability of the inversion) ---")
    dev_mask = pd.Series(False, index=base.index)
    for lbl, w0, w1 in FOLDS:
        m = (base.report_date >= pd.Timestamp(w0)) & (base.report_date <= pd.Timestamp(w1))
        ic, p = stats.spearmanr(base.loc[m, "score"], base.loc[m, "pregap_return"])
        print(f"  {lbl:12s} {w0[:7]}..{w1[:7]}  n={int(m.sum()):>5,}  IC {ic:+.3f} (p={p:.3f})"
              + ("  *" if p < 0.05 else ""))
        if "DEV" in lbl:
            dev_mask |= m
    ic, p = stats.spearmanr(base.loc[dev_mask, "score"], base.loc[dev_mask, "pregap_return"])
    print(f"  DEV pooled                     n={int(dev_mask.sum()):>5,}  IC {ic:+.3f} (p={p:.4f})  *")

    # quintile table on DEV pooled + holdout
    for lbl, mask in [("DEV pooled", dev_mask),
                      ("holdout", (base.report_date >= pd.Timestamp("2026-01-01")) & (base.report_date <= pd.Timestamp("2026-06-30")))]:
        sub = base[mask].copy()
        sub["q"] = pd.qcut(sub.score, 5, labels=False)
        print(f"\n--- {lbl}: mean return by score quintile (Q1 = lowest score = most rejected) ---")
        g = sub.groupby("q").agg(n=("pregap_return", "size"), mean=("pregap_return", "mean"),
                                 win=("pregap_return", lambda v: (v > 0).mean()))
        for q, r in g.iterrows():
            print(f"  Q{int(q)+1}: n={int(r['n']):>5,}  mean {r['mean']:+.3%}  win {r['win']:.0%}")

    # ---- R1 / R2 event-level + slate sim per window ------------------------
    def ret5(r):
        pl = get_prices(r.permaTicker)
        if pl is None:
            return None
        dates, closes = pl
        e = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(r.entry_date)), side="left"))
        x = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(r.exit_date)), side="left"))
        if e >= len(closes) or x >= len(closes) or x <= e:
            return None
        return fr.ret_with_stop(closes, e, x, stop=STOP)

    cache_path = Path(__import__("os").environ.get("TEMP", "/tmp")) / "_antisel_rets.json"
    if cache_path.exists():
        rr = json.loads(cache_path.read_text())
    else:
        rr = {}
    for _, r in base.iterrows():
        k = f"{r.permaTicker}|{str(r.report_date)[:10]}"
        if k not in rr:
            rr[k] = ret5(r)
        if len(rr) % 5000 == 0:
            cache_path.write_text(json.dumps({a: b for a, b in rr.items()}, default=float))
    cache_path.write_text(json.dumps(rr, default=float))
    base["ret5"] = [rr.get(f"{r.permaTicker}|{str(r.report_date)[:10]}") for _, r in base.iterrows()]
    base = base[base.ret5.notna()].copy()
    base["ret5"] = base.ret5.astype(float)

    print("\n--- R1 (score < 0.33) event-level per window ---")
    dev_r1 = []
    for lbl, w0, w1 in FOLDS:
        m = (base.report_date >= pd.Timestamp(w0)) & (base.report_date <= pd.Timestamp(w1))
        sub = base[m & (base.score < 0.33)]
        r = sub.ret5
        lo, hi = bootstrap_ci(r.values)
        print(f"  {lbl:12s} n={len(r):>5,}  raw {r.mean():+.3%}  net30bp {r.mean()-SPREAD_ADJ:+.3%}  "
              f"win {(r>0).mean():.0%}  CI95[{lo-SPREAD_ADJ:+.3%},{hi-SPREAD_ADJ:+.3%}]")
        if "DEV" in lbl:
            dev_r1.append(sub)
    dev = pd.concat(dev_r1)
    lo, hi = bootstrap_ci(dev.ret5.values)
    print(f"  DEV pooled   n={len(dev):>5,}  raw {dev.ret5.mean():+.3%}  net30bp {dev.ret5.mean()-SPREAD_ADJ:+.3%}  "
          f"win {(dev.ret5>0).mean():.0%}  CI95[{lo-SPREAD_ADJ:+.3%},{hi-SPREAD_ADJ:+.3%}]")

    print("\n--- R2 (bottom quintile) event-level per window ---")
    for lbl, w0, w1 in FOLDS:
        m = (base.report_date >= pd.Timestamp(w0)) & (base.report_date <= pd.Timestamp(w1))
        sub = base[m].copy()
        if len(sub) < 50:
            continue
        q20 = sub.score.quantile(0.20)
        r = sub[sub.score <= q20].ret5
        print(f"  {lbl:12s} n={len(r):>5,}  raw {r.mean():+.3%}  net30bp {r.mean()-SPREAD_ADJ:+.3%}  win {(r>0).mean():.0%}")

    # ---- slate sim on R1 (weekly top-4 LOWEST scores) ----------------------
    print("\n--- R1 as a 4-slot weekly slate (lowest-score-first), stop+force-refresh ---")
    def slate_sim(sub):
        slots, trades = [], []
        c = sub.sort_values(["entry_date", "score"], ascending=[True, True]).reset_index(drop=True)
        c["entry_date"] = pd.to_datetime(c.entry_date)
        keep = []
        for wk, gdf in c.groupby(c.entry_date.map(fr.iso_week), sort=True):
            keep.append(gdf.sort_values("score", ascending=True).head(N_SLOTS))
        slate = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
        for ev in slate.itertuples(index=False):
            ed = ev.entry_date
            evd = {"permaTicker": ev.permaTicker, "entry_date": ev.entry_date,
                   "exit_date": pd.Timestamp(ev.exit_date), "score": ev.score, "ret5": ev.ret5}
            kept = []
            for s in slots:
                if s["exit_date"] <= ed:
                    trades.append({**s, "return": s["ret5"]})
                else:
                    kept.append(s)
            slots = kept
            if len(slots) < N_SLOTS:
                slots.append(evd)
            else:
                stale = [s for s in slots if fr.iso_week(s["entry_date"]) < fr.iso_week(ed)]
                if stale:
                    victim = min(stale, key=lambda z: z["entry_date"])
                    trades.append({**victim, "return": victim["ret5"]})
                    slots.remove(victim)
                    slots.append(evd)
        for s in slots:
            trades.append({**s, "return": s["ret5"]})
        return trades

    for lbl, w0, w1 in FOLDS:
        m = (base.report_date >= pd.Timestamp(w0)) & (base.report_date <= pd.Timestamp(w1))
        t = slate_sim(base[m & (base.score < 0.33)])
        r = pd.Series([x["return"] for x in t], dtype=float)
        if len(r) == 0:
            continue
        nav = float(np.prod(1 + r / N_SLOTS))
        lo, hi = bootstrap_ci(r.values)
        print(f"  {lbl:12s} trades {len(r):>3}  raw {r.mean():+.3%}  net30bp {r.mean()-SPREAD_ADJ:+.3%}  "
              f"win {(r>0).mean():.0%}  NAV {nav:.3f}x  CI95[{lo-SPREAD_ADJ:+.3%},{hi-SPREAD_ADJ:+.3%}]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bootstrap validation for the nested v6 gate-decomposition experiment.

Reads the saved nested v6 executed trades, reconstructs the fixed v4 executed
trades from the persisted v4 matrix, and computes trade/week/fold uncertainty.
No production artifacts or HDF5 nodes are written.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parents[2])
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
DB = None
V4_HP = {"gamma": 3, "min_child_weight": 100, "max_depth": 2, "n_estimators": 300}
N_BOOT = 10_000
SEED = 20260807

def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

bt = load("bt_validate_v6", HERE / "51_hp_theta_sweep_23feat.py")
DB = bt.DB
MATRIX = "/features/train_matrix_v4_timing_correct"
OUT = HERE / "archive" / "experiments" / "gate_decomposition_v6"


def json_default(v):
    if isinstance(v, (pd.Timestamp, np.datetime64)): return pd.Timestamp(v).isoformat()
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return float(v)
    raise TypeError(type(v).__name__)


def reconstruct_v4(df):
    """Reproduce v4's four outer test predictions and weekly slot selection."""
    import xgboost as xgb
    rd = pd.to_datetime(df.report_date); parts = []
    for fi, (te, sve, tse) in enumerate(bt.DEFAULT_FOLDS, 1):
        tr = df[rd <= pd.Timestamp(te)]
        sv = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))]
        ts = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        train = pd.concat([tr, sv], ignore_index=True)
        clf = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric=["logloss", "auc"],
            learning_rate=.05, reg_lambda=1., subsample=.7,
            colsample_bytree=.7, random_state=42, n_jobs=-1, **V4_HP)
        clf.fit(train[bt.DEPLOY_FEATURES], train.pead_pass.astype(int),
                eval_set=[(ts[bt.DEPLOY_FEATURES], ts.pead_pass.astype(int))], verbose=False)
        ts["p"] = clf.predict_proba(ts[bt.DEPLOY_FEATURES])[:, 1]
        ts["fold"] = fi
        mask = (ts.p >= .20) & ts.pregap_return.notna() & (~ts.sector.isin(bt.EXCLUDE_SECTORS))
        raw = ts[mask].copy()
        if raw.empty: continue
        raw["entry_date"] = pd.to_datetime(raw.pregap_entry_date)
        raw["exit_date"] = pd.to_datetime(raw.pregap_exit_date)
        selected = bt.select_weekly(raw, bt.N_SLOTS)
        if not selected.empty: parts.append(selected)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def trade_summary(x):
    r = np.asarray(x, dtype=float); r = r[np.isfinite(r)]
    if not len(r): return {"n": 0}
    wins = r > 0
    return {"n": int(len(r)), "win_rate_pct": float(wins.mean()*100),
            "avg_trade_pct": float(r.mean()*100), "median_trade_pct": float(np.median(r)*100),
            "avg_win_pct": float(r[wins].mean()*100) if wins.any() else 0.,
            "avg_loss_pct": float(r[~wins].mean()*100) if (~wins).any() else 0.}


def weekly_returns(ex):
    if ex.empty: return np.array([])
    z = ex.copy(); z.entry_date = pd.to_datetime(z.entry_date)
    iso = z.entry_date.dt.isocalendar()
    z["week"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    return z.groupby("week", sort=True).pregap_return.sum().to_numpy(dtype=float) / bt.N_SLOTS


def bootstrap_ci(rng, values, stat, n=N_BOOT):
    values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    if not len(values): return {"estimate": None, "ci95": [None, None]}
    idx = rng.integers(0, len(values), size=(n, len(values)))
    samples = values[idx]
    vals = stat(samples)
    estimate = float(np.asarray(stat(values[None, :])).reshape(-1)[0])
    return {"estimate": estimate, "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def bootstrap_weekly_nav(rng, weekly, n=N_BOOT):
    weekly = np.asarray(weekly, dtype=float); weekly = weekly[np.isfinite(weekly)]
    if not len(weekly): return {"estimate_nav_pct": None, "ci95_nav_pct": [None, None]}
    idx = rng.integers(0, len(weekly), size=(n, len(weekly)))
    navs = np.prod(1. + weekly[idx], axis=1) - 1.
    return {"estimate_nav_pct": float((np.prod(1.+weekly)-1)*100),
            "ci95_nav_pct": [float(np.percentile(navs, 2.5)*100), float(np.percentile(navs, 97.5)*100)],
            "weeks": int(len(weekly))}


def bootstrap_difference(rng, a, b, n=N_BOOT):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    ia = rng.integers(0, len(a), size=(n, len(a)))
    ib = rng.integers(0, len(b), size=(n, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return {"estimate_delta_pct_points": float((a.mean()-b.mean())*100),
            "ci95_delta_pct_points": [float(np.percentile(d,2.5)*100), float(np.percentile(d,97.5)*100)],
            "prob_v6_mean_gt_v4": float((d > 0).mean())}


def main():
    print("="*100); print("V6 VALIDATION: bootstrap and exact v4 reconstruction"); print("="*100)
    saved = json.load(open(OUT / "nested_results.json", encoding="utf-8"))
    v6 = pd.DataFrame(saved["executed_trades"])
    with pd.HDFStore(DB, "r") as store: df = store[MATRIX]
    print(f"v6 saved trades={len(v6)}; reconstructing v4 ...")
    v4 = reconstruct_v4(df)
    print(f"v4 reconstructed trades={len(v4)}")
    if len(v6) != 168 or len(v4) != 99: raise RuntimeError("Unexpected trade counts")

    rng = np.random.default_rng(SEED)
    v6_r = v6.pregap_return.to_numpy(float); v4_r = v4.pregap_return.to_numpy(float)
    v6_w = weekly_returns(v6); v4_w = weekly_returns(v4)
    result = {
        "model_version": "phase_g_v6_gate_decomposition",
        "baseline": "phase_g_v4_timing_correct",
        "matrix": MATRIX, "n_boot": N_BOOT, "seed": SEED,
        "point_estimates": {"v6": trade_summary(v6_r), "v4": trade_summary(v4_r)},
        "weekly_nav_bootstrap": {"v6": bootstrap_weekly_nav(rng, v6_w), "v4": bootstrap_weekly_nav(rng, v4_w)},
        "trade_bootstrap": {
            "v6_avg_trade": bootstrap_ci(rng, v6_r, lambda x: x.mean(axis=1)),
            "v4_avg_trade": bootstrap_ci(rng, v4_r, lambda x: x.mean(axis=1)),
            "v6_win_rate": bootstrap_ci(rng, (v6_r > 0).astype(float), lambda x: x.mean(axis=1)),
            "v4_win_rate": bootstrap_ci(rng, (v4_r > 0).astype(float), lambda x: x.mean(axis=1)),
            "v6_minus_v4_avg_trade": bootstrap_difference(rng, v6_r, v4_r),
        },
        "folds": [],
        "v6_trades": v6.to_dict(orient="records"),
        "v4_trades": v4.to_dict(orient="records"),
    }
    # Per-fold realized statistics; no fold resampling is substituted for this.
    for fi in range(1, 5):
        a = v6[v6.fold.astype(int) == fi]; b = v4[v4.fold.astype(int) == fi]
        result["folds"].append({"fold": fi, "v6": trade_summary(a.pregap_return), "v4": trade_summary(b.pregap_return),
                                "v6_weekly_nav": bootstrap_weekly_nav(rng, weekly_returns(a)),
                                "v4_weekly_nav": bootstrap_weekly_nav(rng, weekly_returns(b))})
    with open(OUT / "validation.json", "w", encoding="utf-8") as f: json.dump(result, f, indent=2, default=json_default)
    print(json.dumps({k: result[k] for k in ["point_estimates", "weekly_nav_bootstrap", "trade_bootstrap", "folds"]}, indent=2, default=json_default))
    print(f"Saved {OUT / 'validation.json'}")

if __name__ == "__main__": main()

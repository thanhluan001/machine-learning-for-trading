"""EXPLORATORY probe (not an RC program; no registration): threshold
discovery + anti-signal diagnostic on the RC-16/17 corrected arms.

Read-only. Any actionable finding requires a new registration + shadow.
Context: v6c avg trade -1.914% CI [-1.744,-0.067] (marginal negative);
v6n -1.581% CI [-1.684,+0.122]. User hypothesis: threshold 0.33 too low
for leak-free models (calibrated on leaky probability scales); possibly
anti-signal strong enough to flip.

Outputs:
  1. score-decile vs outcome (is there ANY monotone signal, either sign?)
  2. threshold sweep 0.30..0.65: executed, wr, avg trade, week-block CI
  3. the same thresholds FLIPPED (short the picks, paper-level)
Arms: v6c (23 feats) and v6n (22 feats), identical machinery to RC-16/17.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Absolute paths FIRST, then chdir (relative __file__ + chdir = wrong root)
HERE = Path(__file__).resolve().parent
os.chdir(HERE.parents[2])
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

N_BOOT = 10_000
SEED = 20260807
GATES = ["pass_g1", "pass_g2", "pass_g3"]
V6_HP = {
    "pass_g1": {"gamma": 8, "min_child_weight": 20, "max_depth": 3},
    "pass_g2": {"gamma": 12, "min_child_weight": 50, "max_depth": 3},
    "pass_g3": {"gamma": 1, "min_child_weight": 50, "max_depth": 3},
}
N_TREES = 300


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = load("bt_thr_probe", HERE / "51_hp_theta_sweep_23feat.py")
DB = bt.DB
FEATURES23 = list(bt.DEPLOY_FEATURES)
FEATURES22 = [f for f in FEATURES23 if f != "car_drift_historical_q1"]


def fit_gate(X, y, Xe, ye, hp):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=N_TREES, learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1,
    ).fit(X, y, eval_set=[(Xe, ye)], verbose=False)


def oos_scores(df, feats, hp_map):
    out = []
    rd = df["label_end"] if "label_end" in df.columns else pd.to_datetime(df["report_date"])
    for fi, (te, sve, tse) in enumerate(bt.DEFAULT_FOLDS, 1):
        tr = df[rd <= pd.Timestamp(te)]
        sv = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))]
        ts = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        train = pd.concat([tr, sv], ignore_index=True)
        for i, g in enumerate(GATES, 1):
            clf = fit_gate(train[feats], train[g].astype(int).to_numpy(),
                           ts[feats], ts[g].astype(int).to_numpy(), hp_map[g])
            ts[f"p{i}"] = clf.predict_proba(ts[feats])[:, 1]
        ts["score"] = ts[["p1", "p2", "p3"]].min(axis=1)
        ts["fold"] = fi
        out.append(ts)
    return pd.concat(out, ignore_index=True)


def week_series(ex):
    z = ex.copy()
    z["entry_date"] = pd.to_datetime(z["entry_date"])
    iso = z["entry_date"].dt.isocalendar()
    z["wk"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    return z.groupby("wk", sort=True)["pregap_return"].sum() / bt.N_SLOTS


def boot_ci_mean(w, rng):
    v = w.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    m = v[idx].mean(axis=1)
    return [round(float(np.percentile(m, 2.5)) * 100, 3),
            round(float(np.percentile(m, 97.5)) * 100, 3)]


def execute(pred, t, flip=False):
    mask = ((pred["score"] >= t) & (~pred["sector"].isin(bt.EXCLUDE_SECTORS))
            & pred["pregap_return"].notna())
    raw = pred[mask].copy()
    raw["p"] = raw["score"]
    raw["entry_date"] = pd.to_datetime(raw["pregap_entry_date"])
    raw["exit_date"] = pd.to_datetime(raw["pregap_exit_date"])
    if flip:
        raw["pregap_return"] = -raw["pregap_return"]
    pieces = [bt.select_weekly(raw[raw["fold"] == fi], bt.N_SLOTS) for fi in range(1, 5)
              if not raw[raw["fold"] == fi].empty]
    ex = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if ex.empty:
        return {"exec": 0}
    r = ex["pregap_return"].astype(float)
    return {"exec": int(len(r)), "wr": round(float((r > 0).mean() * 100), 1),
            "avg": round(float(r.mean() * 100), 3)}


def main():
    rng = np.random.default_rng(SEED)
    v6c = pd.read_hdf(DB, "/features/train_matrix_v6c")
    old = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
    for d in (old, v6c):
        d["report_date"] = pd.to_datetime(d["report_date"])
    ko = set(zip(old.permaTicker, old.report_date))
    kc = set(zip(v6c.permaTicker, v6c.report_date))
    common = ko & kc
    v6c_c = v6c[v6c.set_index(["permaTicker", "report_date"]).index.isin(common)].copy()

    arms = {"v6c(23f)": (v6c_c, FEATURES23), "v6n(22f)": (v6c_c, FEATURES22)}
    results = {}
    for name, (df, feats) in arms.items():
        print(f"\n{'=' * 96}\n{name}: score deciles vs outcome (ALL OOS rows)\n{'=' * 96}")
        pred = oos_scores(df, feats, V6_HP)
        results[name] = {}
        pred["dec"] = pd.qcut(pred["score"], 10, labels=False, duplicates="drop")
        tab = pred.groupby("dec").agg(
            n=("score", "size"),
            score_lo=("score", "min"), score_hi=("score", "max"),
            mean_car_10d=("car_10d", lambda s: round(float(pd.to_numeric(s).mean()) * 100, 3)),
            pead_rate=("pead_pass", "mean"))
        print(tab.round(4).to_string())

        # rank correlation score vs car_10d (Spearman via pandas — no scipy dep)
        s_ok = pred[["score", "car_10d"]].dropna()
        rho = s_ok["score"].corr(s_ok["car_10d"], method="spearman")
        print(f"spearman(score, car_10d) = {rho:+.4f}  (n={len(s_ok):,})")

        print(f"\n{name}: threshold sweep (long | flip)")
        print(f"{'t':>5} | {'exec':>4} {'wr%':>5} {'avg%':>7} {'CI':>18} | {'exec':>4} {'wr%':>5} {'avg%':>7} {'CI':>18}")
        for t in [0.30, 0.33, 0.37, 0.41, 0.45, 0.49, 0.53, 0.57, 0.61, 0.65]:
            row_l, row_s = {}, {}
            # long
            mask = ((pred["score"] >= t) & (~pred["sector"].isin(bt.EXCLUDE_SECTORS))
                    & pred["pregap_return"].notna())
            raw = pred[mask].copy()
            raw["p"] = raw["score"]
            raw["entry_date"] = pd.to_datetime(raw["pregap_entry_date"])
            raw["exit_date"] = pd.to_datetime(raw["pregap_exit_date"])
            pieces = [bt.select_weekly(raw[raw["fold"] == fi], bt.N_SLOTS) for fi in range(1, 5)
                      if not raw[raw["fold"] == fi].empty]
            ex = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
            if ex.empty:
                row_l = {"exec": 0, "wr": None, "avg": None, "ci": None}
            else:
                r = ex["pregap_return"].astype(float)
                row_l = {"exec": len(r), "wr": round(float((r > 0).mean() * 100), 1),
                         "avg": round(float(r.mean() * 100), 3),
                         "ci": boot_ci_mean(week_series(ex), rng)}
            # flipped (short the same picks, paper-level: negate returns)
            raw2 = raw.copy()
            raw2["pregap_return"] = -raw2["pregap_return"]
            pieces = [bt.select_weekly(raw2[raw2["fold"] == fi], bt.N_SLOTS) for fi in range(1, 5)
                      if not raw2[raw2["fold"] == fi].empty]
            ex2 = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
            if ex2.empty:
                row_s = {"exec": 0, "wr": None, "avg": None, "ci": None}
            else:
                r2 = ex2["pregap_return"].astype(float)
                row_s = {"exec": len(r2), "wr": round(float((r2 > 0).mean() * 100), 1),
                         "avg": round(float(r2.mean() * 100), 3),
                         "ci": boot_ci_mean(week_series(ex2), rng)}
            print(f"{t:5.2f} | {row_l['exec']:4d} {str(row_l['wr']):>5} {str(row_l['avg']):>7} "
                  f"{str(row_l['ci']):>18} | {row_s['exec']:4d} {str(row_s['wr']):>5} {str(row_s['avg']):>7} "
                  f"{str(row_s['ci']):>18}")
            results[name][t] = {"long": row_l, "flip": row_s}

        # low-score tail diagnostic (anti-signal check)
        print(f"\n{name}: LOW-score tail (score <= 0.05): n={(pred['score'] <= 0.05).sum():,}, "
              f"mean car {pd.to_numeric(pred.loc[pred['score'] <= 0.05, 'car_10d']).mean() * 100:+.3f}%, "
              f"pead {pred.loc[pred['score'] <= 0.05, 'pead_pass'].mean():.1%}")

    with open(HERE / "archive" / "experiments" / "rc18_threshold_probe.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwrote archive/experiments/rc18_threshold_probe.json (EXPLORATORY)")


if __name__ == "__main__":
    main()

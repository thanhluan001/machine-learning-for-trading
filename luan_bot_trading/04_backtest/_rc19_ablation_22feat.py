"""RC-19 ablation diagnostic (declared, non-gating): label-only effect.

Fills the missing cell: 22 retired features + SINGLE pass_g1 classifier
(thr 0.33, frozen V6 g1 HPs) — no F1_sb_h3, no A2. Decomposes RC-19's
+2.30pp swing into label-swap vs feature-addition. Reports stats +
paired weekly diffs vs v6n and vs rc19 (both loaded from rc19/executed.h5,
no retraining). Diagnostic only: no gates, no sweeps.
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

N_BOOT, SEED, THRESH = 10_000, 20260807, 0.33
HP1 = {"gamma": 8, "min_child_weight": 20, "max_depth": 3}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = load("bt_abl", HERE / "51_hp_theta_sweep_23feat.py")
DB = bt.DB
FEATURES22 = [f for f in bt.DEPLOY_FEATURES if f != "car_drift_historical_q1"]

m = pd.read_hdf(DB, "/features/train_matrix_v6c")
old = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
for d in (old, m):
    d["report_date"] = pd.to_datetime(d["report_date"])
ko = set(zip(old.permaTicker, old.report_date))
kc = set(zip(m.permaTicker, m.report_date))
m = m[m.set_index(["permaTicker", "report_date"]).index.isin(ko & kc)].copy()
print(f"matrix {len(m):,}", flush=True)


def fit_gate(X, y, Xe, ye, hp):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=300, learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1,
    ).fit(X, y, eval_set=[(Xe, ye)], verbose=False)


def run_arm(df, feats):
    rd = df["label_end"]
    out = []
    for fi, (te, sve, tse) in enumerate(bt.DEFAULT_FOLDS, 1):
        tr = df[rd <= pd.Timestamp(te)]
        sv = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))]
        ts = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        train = pd.concat([tr, sv], ignore_index=True)
        clf = fit_gate(train[feats], train["pass_g1"].astype(int).to_numpy(),
                       ts[feats], ts["pass_g1"].astype(int).to_numpy(), HP1)
        ts["score"] = clf.predict_proba(ts[feats])[:, 1]
        ts["fold"] = fi
        out.append(ts)
    pred = pd.concat(out, ignore_index=True)
    mask = ((pred["score"] >= THRESH) & (~pred["sector"].isin(bt.EXCLUDE_SECTORS))
            & pred["pregap_return"].notna())
    raw = pred[mask].copy()
    raw["p"] = raw["score"]
    raw["entry_date"] = pd.to_datetime(raw["pregap_entry_date"])
    raw["exit_date"] = pd.to_datetime(raw["pregap_exit_date"])
    pieces = [bt.select_weekly(raw[raw["fold"] == fi], bt.N_SLOTS) for fi in range(1, 5)
              if not raw[raw["fold"] == fi].empty]
    ex = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return raw, ex


def weekly(ex):
    z = ex.copy()
    ec = "entry_date" if "entry_date" in z.columns else "pregap_entry_date"
    z["entry_date"] = pd.to_datetime(z[ec])
    iso = z["entry_date"].dt.isocalendar()
    z["wk"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    return z.groupby("wk", sort=True)["pregap_return"].sum() / bt.N_SLOTS


def stats(ex):
    r = ex["pregap_return"].astype(float)
    w = weekly(ex)
    return {"executed": int(len(r)), "win_rate_pct": round(float((r > 0).mean() * 100), 1),
            "avg_trade_pct": round(float(r.mean() * 100), 3), "weeks": int(len(w)),
            "nav_pct": round(float(((1 + w).prod() - 1) * 100), 2),
            "raw_precision": round(float(ex["pead_pass"].mean() * 100), 1),
            "g1_rate": round(float(ex["pass_g1"].mean() * 100), 1)}


def paired(wa, wb):
    rng = np.random.default_rng(SEED)
    j = pd.concat({"a": wa, "b": wb}, axis=1, join="inner").dropna()
    d = (j["a"] - j["b"]).to_numpy(dtype=float)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    mm = d[idx].mean(axis=1)
    return {"paired_weeks": int(len(d)), "mean_diff_pct": round(float(d.mean() * 100), 4),
            "ci95_pct": [round(float(np.percentile(mm, 2.5)) * 100, 4),
                         round(float(np.percentile(mm, 97.5)) * 100, 4)],
            "prob_pos": round(float((d > 0).mean()), 4)}


_, ex22 = run_arm(m, FEATURES22)
print("rc19_22 (22 feats, pure drift):", json.dumps(stats(ex22)), flush=True)

st = pd.HDFStore(HERE / "archive" / "experiments" / "rc19" / "executed.h5", "r")
w = {"v6n": weekly(st["/v6n"]), "rc19_24": weekly(st["/rc19"]), "rc19_22": weekly(ex22)}
st.close()

print("paired rc19_22 - v6n:   ", json.dumps(paired(w["rc19_22"], w["v6n"])), flush=True)
print("paired rc19_22 - rc19_24:", json.dumps(paired(w["rc19_22"], w["rc19_24"])), flush=True)
print("paired rc19_24 - v6n:   ", json.dumps(paired(w["rc19_24"], w["v6n"])), flush=True)

out = HERE / "archive" / "experiments" / "rc19_ablation"
out.mkdir(parents=True, exist_ok=True)
with open(out / "report.json", "w") as f:
    json.dump({"rc19_22": stats(ex22),
               "paired_22_vs_v6n": paired(w["rc19_22"], w["v6n"]),
               "paired_22_vs_24": paired(w["rc19_22"], w["rc19_24"]),
               "paired_24_vs_v6n": paired(w["rc19_24"], w["v6n"]),
               "note": "diagnostic only, non-gating"}, f, indent=2, default=str)
print("wrote", out / "report.json")

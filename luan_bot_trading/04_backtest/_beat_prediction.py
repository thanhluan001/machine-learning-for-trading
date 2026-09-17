"""Beat-prediction diagnostic (exploratory, non-gating).

Target: BEAT the earnings (intermediate goal before drift).
  SP400: sue_score > 0            (v6c, standardized surprise)
  SP600: actual > estimate        (earnings_full; weaker, simple surprise)

Reports:
  1. base rates — overall, by universe, by year, by sector
  2. SUE distribution (SP400)
  3. economics — 5-session harvest and car_10d for beaters vs misses
  4. predictability — XGBoost (22 deploy features), label-matured folds,
     OOS AUC + decile lift table + economic bridge (harvest by predicted
     beat-probability decile)
  5. ceiling — what a perfect beat predictor would earn
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
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

OUT = HERE / "archive" / "experiments" / "beat_prediction"
OUT.mkdir(parents=True, exist_ok=True)
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"


def log(*a):
    print(*a, flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = load("bt_beat", HERE / "51_hp_theta_sweep_23feat.py")
DB = bt.DB
FEATURES22 = [f for f in bt.DEPLOY_FEATURES if f != "car_drift_historical_q1"]
t0 = time.time()

# ---------------- build beat label ----------------
v7 = pd.read_hdf(DB, "/features/train_matrix_v7c")
v7["report_date"] = pd.to_datetime(v7["report_date"])
v7["label_end"] = pd.to_datetime(v7["label_end"])
v6 = pd.read_hdf(DB, "/features/train_matrix_v6c")
v6["report_date"] = pd.to_datetime(v6["report_date"])
j = v6.set_index(["permaTicker", "report_date"])["sue_score"]
v7["sue_score"] = j.reindex(pd.MultiIndex.from_arrays([v7["permaTicker"], v7["report_date"]])).to_numpy()
v7["sue_score"] = pd.to_numeric(v7["sue_score"], errors="coerce")

with pd.HDFStore(DB_SP600, "r") as st:
    s6 = st["/features/train_matrix_sp600_pt_v7c"]
    s6["report_date"] = pd.to_datetime(s6["report_date"])
    tk = s6.set_index(["permaTicker", "report_date"])["ticker"]
    earn = {}
    for k in st.keys():
        if k.startswith("/sp600/earnings_full/"):
            earn[k.split("/")[-1]] = st[k]
sel6 = v7["is_sp400"] == 0
tkr = tk.reindex(pd.MultiIndex.from_arrays([v7.loc[sel6, "permaTicker"], v7.loc[sel6, "report_date"]])).to_numpy()
beat6, surp6 = [], []
for rd, t_ in zip(v7.loc[sel6, "report_date"], tkr):
    b, s = np.nan, np.nan
    if isinstance(t_, str) and t_ in earn:
        e = earn[t_]
        e = e[pd.to_datetime(e["report_date"]) == rd]
        if len(e):
            r0 = e.iloc[-1]
            a_, est_ = float(r0["actual"]), float(r0["estimate"])
            if np.isfinite(a_) and np.isfinite(est_):
                b = 1.0 if a_ > est_ else 0.0
                s = a_ - est_
    beat6.append(b)
    surp6.append(s)
v7.loc[sel6, "beat_eps"] = beat6
v7.loc[sel6, "surprise_eps"] = surp6
v7["beat"] = np.where(v7["is_sp400"] == 1, (v7["sue_score"] > 0).astype(float),
                      v7["beat_eps"])
v7.loc[v7["is_sp400"] == 1, "beat"] = np.where(v7.loc[v7["is_sp400"] == 1, "sue_score"].isna(),
                                               np.nan, v7.loc[v7["is_sp400"] == 1, "beat"])
d = v7[~v7["sector"].isin(bt.EXCLUDE_SECTORS)].copy()
d["pregap_return"] = pd.to_numeric(d["pregap_return"], errors="coerce")
d["car_10d"] = pd.to_numeric(d["car_10d"], errors="coerce")
log(f"events {len(d):,} | beat defined {d.beat.notna().mean():.1%} ({time.time()-t0:.0f}s)")

# ---------------- 1-2. base rates ----------------
b = d[d["beat"].notna()]
log("\n=== BEAT BASE RATES ===")
log(f"overall            : {b.beat.mean()*100:.1f}%  (n={len(b):,})")
for tag, m in (("SP400 (sue>0)", b.is_sp400 == 1), ("SP600 (eps>est)", b.is_sp400 == 0)):
    log(f"{tag:19}: {b.loc[m,'beat'].mean()*100:.1f}%  (n={int(m.sum()):,})")
log("\nby year:")
yr = b.assign(y=b["report_date"].dt.year).groupby("y").agg(n=("beat", "size"), beat=("beat", "mean"))
log("  " + "  ".join(f"{int(i)}:{r.beat*100:.0f}%({int(r.n)})" for i, r in yr.iterrows()))
log("\nby sector (SP400 side):")
sec = b[b.is_sp400 == 1].groupby("sector").agg(n=("beat", "size"), beat=("beat", "mean")).sort_values("beat")
log("  " + "  ".join(f"{i}:{r.beat*100:.0f}%({int(r.n)})" for i, r in sec.iterrows()))
sue = pd.to_numeric(b.loc[b.is_sp400 == 1, "sue_score"], errors="coerce").dropna()
log(f"\nSUE (SP400): n={len(sue):,} mean={sue.mean():.3f} sd={sue.std():.3f} "
    f"| quantiles 10/25/50/75/90: {np.percentile(sue,[10,25,50,75,90]).round(2)}")

# ---------------- 3. economics of the beat ----------------
log("\n=== ECONOMICS: beat vs miss (5-session harvest / car_10d) ===")
for tag, sub in (("SP400", b[b.is_sp400 == 1]), ("SP600", b[b.is_sp400 == 0]), ("ALL", b)):
    for gv, gl in ((1.0, "beat"), (0.0, "miss")):
        s = sub[sub.beat == gv]
        r = s.pregap_return.dropna()
        c = s.car_10d.dropna()
        log(f"  {tag:6} {gl:5}: n={len(s):>5}  harvest={r.mean()*100:+7.3f}% (SE {r.std(ddof=1)/np.sqrt(len(r))*100:.2f})  "
            f"car_10d={c.mean()*100:+7.3f}%")

# ---------------- 4. predictability ----------------
FEATS = [f for f in FEATURES22 if f in d.columns]
import xgboost as xgb
from sklearn.metrics import roc_auc_score

tr_mask = np.zeros(len(d), dtype=bool)
preds = []
for fi, (te, sve, tse) in enumerate(bt.DEFAULT_FOLDS, 1):
    tr = d[(d["label_end"] <= pd.Timestamp(sve)) & d["beat"].notna()]
    ts = d[(d["label_end"] > pd.Timestamp(sve)) & (d["label_end"] <= pd.Timestamp(tse)) & d["beat"].notna()].copy()
    clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", n_estimators=300,
                            learning_rate=0.05, max_depth=3, min_child_weight=20, gamma=8,
                            reg_lambda=1.0, subsample=0.7, colsample_bytree=0.7,
                            random_state=42, n_jobs=-1)
    clf.fit(tr[FEATS], tr["beat"].astype(int), eval_set=[(ts[FEATS], ts["beat"].astype(int))], verbose=False)
    ts["p_beat"] = clf.predict_proba(ts[FEATS])[:, 1]
    ts["fold"] = fi
    auc = roc_auc_score(ts["beat"].astype(int), ts["p_beat"])
    log(f"fold {fi}: train {len(tr):,} test {len(ts):,} AUC={auc:.4f} base={ts.beat.mean()*100:.1f}% ({time.time()-t0:.0f}s)")
    preds.append(ts)
P = pd.concat(preds, ignore_index=True)
log(f"\nOOS overall AUC(beat) = {roc_auc_score(P['beat'].astype(int), P['p_beat']):.4f}  (n={len(P):,})")
for tag, m in (("SP400", P.is_sp400 == 1), ("SP600", P.is_sp400 == 0)):
    if m.sum() > 100:
        log(f"  {tag}: AUC={roc_auc_score(P.loc[m,'beat'].astype(int), P.loc[m,'p_beat']):.4f} (n={int(m.sum()):,})")

P["dec"] = pd.qcut(P["p_beat"].rank(method="first"), 10, labels=False) + 1
tab = P.groupby("dec").agg(n=("beat", "size"), p_beat=("p_beat", "mean"), beat_rate=("beat", "mean"),
                           harvest=("pregap_return", "mean"), car10=("car_10d", "mean"))
tab["p_beat"] = tab["p_beat"].round(3)
tab["beat_rate"] = (tab["beat_rate"] * 100).round(1)
tab["harvest"] = (tab["harvest"] * 100).round(3)
tab["car10"] = (tab["car10"] * 100).round(3)
log("\n=== DECILE LIFT (OOS) ===")
log(tab.to_string())

top = P[P.dec >= 9]
log(f"\ntop-2-decile (p_beat) n={len(top)}: beat_rate={top.beat.mean()*100:.1f}%  "
    f"harvest={top.pregap_return.mean()*100:+.3f}%  car_10d={top.car_10d.mean()*100:+.3f}%")
beat_ceiling = P[P.beat == 1]
log(f"CEILING (perfect beat prediction) n={len(beat_ceiling)}: harvest={beat_ceiling.pregap_return.mean()*100:+.3f}%  "
    f"car_10d={beat_ceiling.car_10d.mean()*100:+.3f}%")
log(f"unconditional OOS: harvest={P.pregap_return.mean()*100:+.3f}%  car_10d={P.car_10d.mean()*100:+.3f}%")

gain = pd.Series(clf.feature_importances_, index=FEATS).sort_values(ascending=False)
log("\ntop-8 gain: " + ", ".join(f"{k}={v:.3f}" for k, v in gain.head(8).items()))

with open(OUT / "report.json", "w") as f:
    json.dump({"base_rate": float(b.beat.mean()), "n": int(len(d)),
               "oos_auc": float(roc_auc_score(P['beat'].astype(int), P['p_beat'])),
               "deciles": tab.reset_index().to_dict("records"),
               "ceiling": {"harvest_pct": round(float(beat_ceiling.pregap_return.mean() * 100), 3),
                           "car10_pct": round(float(beat_ceiling.car_10d.mean() * 100), 3)},
               "gain": gain.round(4).to_dict(), "seconds": round(time.time() - t0)}, f, indent=2, default=str)
P[["permaTicker", "report_date", "is_sp400", "p_beat", "beat", "pregap_return", "car_10d",
   "fold"]].to_hdf(OUT / "oos_predictions.h5", key="p", format="table")
log(f"\nwrote {OUT}/report.json ({round(time.time()-t0)}s)")

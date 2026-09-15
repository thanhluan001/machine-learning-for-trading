"""EXPLORATORY probe (no RC registration): threshold discovery + anti-signal
diagnostic on the RC-16/17 corrected arms. Any actionable finding requires a
new registration + shadow. Outputs to archive/experiments/rc18_threshold_probe/.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.chdir(ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def pearson_np(x, y):
    """Manual Pearson — pandas Series.corr/cov NATIVELY CRASHES this env
    (pandas 3.0.3 / numpy 2.4.6): rc=127, no traceback. Do not use .corr()."""
    x = np.asarray(x, dtype="float64"); y = np.asarray(y, dtype="float64")
    m = np.isfinite(x) & np.isfinite(y)
    xm = x[m] - x[m].mean(); ym = y[m] - y[m].mean()
    d = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float((xm * ym).sum() / d) if d > 0 else float("nan")


def log(*a):
    print(*a, flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = load("bt_thr_probe2", HERE / "51_hp_theta_sweep_23feat.py")
DB = bt.DB
N_BOOT, SEED = 10_000, 20260807
GATES = ["pass_g1", "pass_g2", "pass_g3"]
V6_HP = {
    "pass_g1": {"gamma": 8, "min_child_weight": 20, "max_depth": 3},
    "pass_g2": {"gamma": 12, "min_child_weight": 50, "max_depth": 3},
    "pass_g3": {"gamma": 1, "min_child_weight": 50, "max_depth": 3},
}
FEATURES23 = list(bt.DEPLOY_FEATURES)
FEATURES22 = [f for f in FEATURES23 if f != "car_drift_historical_q1"]
OUT = HERE / "archive" / "experiments" / "rc18_threshold_probe"
OUT.mkdir(parents=True, exist_ok=True)


def fit_gate(X, y, Xe, ye, hp):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=300, learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1,
    ).fit(X, y, eval_set=[(Xe, ye)], verbose=False)


def oos_scores(df, feats):
    out = []
    rd = df["label_end"]
    for fi, (te, sve, tse) in enumerate(bt.DEFAULT_FOLDS, 1):
        tr = df[rd <= pd.Timestamp(te)]
        sv = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))]
        ts = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        train = pd.concat([tr, sv], ignore_index=True)
        for i, g in enumerate(GATES, 1):
            clf = fit_gate(train[feats], train[g].astype(int).to_numpy(),
                           ts[feats], ts[g].astype(int).to_numpy(), V6_HP[g])
            ts[f"p{i}"] = clf.predict_proba(ts[feats])[:, 1]
        ts["score"] = ts[["p1", "p2", "p3"]].min(axis=1)
        ts["fold"] = fi
        out.append(ts)
        log(f"  arm fold {fi} done ({len(ts)} rows)")
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


def run_thresholds(pred, rng, name, results):
    log(f"\n{name}: threshold sweep (long | flip-short)")
    log(f"{'t':>5} | {'exec':>4} {'wr%':>5} {'avg%':>7} {'CI95':>19} | {'exec':>4} {'wr%':>5} {'avg%':>7} {'CI95':>19}")
    for t in [0.30, 0.33, 0.37, 0.41, 0.45, 0.49, 0.53, 0.57, 0.61, 0.65]:
        row = {}
        for side in ("long", "flip"):
            mask = ((pred["score"] >= t) & (~pred["sector"].isin(bt.EXCLUDE_SECTORS))
                    & pred["pregap_return"].notna())
            raw = pred[mask].copy()
            raw["p"] = raw["score"]
            raw["entry_date"] = pd.to_datetime(raw["pregap_entry_date"])
            raw["exit_date"] = pd.to_datetime(raw["pregap_exit_date"])
            if side == "flip":
                raw["pregap_return"] = -raw["pregap_return"]
            pieces = [bt.select_weekly(raw[raw["fold"] == fi], bt.N_SLOTS) for fi in range(1, 5)
                      if not raw[raw["fold"] == fi].empty]
            ex = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
            if ex.empty:
                row[side] = {"exec": 0, "wr": None, "avg": None, "ci": None}
            else:
                r = ex["pregap_return"].astype(float)
                row[side] = {"exec": int(len(r)),
                             "wr": round(float((r > 0).mean() * 100), 1),
                             "avg": round(float(r.mean() * 100), 3),
                             "ci": boot_ci_mean(week_series(ex), rng)}
        L, S = row["long"], row["flip"]
        log(f"{t:5.2f} | {L['exec']:4d} {str(L['wr']):>5} {str(L['avg']):>7} {str(L['ci']):>19} "
            f"| {S['exec']:4d} {str(S['wr']):>5} {str(S['avg']):>7} {str(S['ci']):>19}")
        results.setdefault(name, {})[t] = row


def main():
    rng = np.random.default_rng(SEED)
    log("RC-18 EXPLORATORY threshold probe — v6c/v6n, deciles + sweep + flip")
    v6c = pd.read_hdf(DB, "/features/train_matrix_v6c")
    old = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
    for d in (old, v6c):
        d["report_date"] = pd.to_datetime(d["report_date"])
    ko = set(zip(old.permaTicker, old.report_date))
    kc = set(zip(v6c.permaTicker, v6c.report_date))
    common = ko & kc
    v6c_c = v6c[v6c.set_index(["permaTicker", "report_date"]).index.isin(common)].copy()
    log(f"common keys {len(common):,}")

    results = {}
    for name, feats in (("v6c_23f", FEATURES23), ("v6n_22f", FEATURES22)):
        log(f"\n===== {name}: OOS scoring =====")
        pred = oos_scores(v6c_c, feats)
        pred.to_pickle(OUT / f"pred_{name}.pkl")

        log(f"\n{name}: score DECILES vs outcome (all OOS rows)")
        d_ok = pred[["score", "car_10d", "pead_pass"]].dropna(subset=["car_10d"])
        d_ok = d_ok.assign(dec=pd.qcut(d_ok["score"], 10, labels=False, duplicates="drop"))
        tab = d_ok.groupby("dec").agg(
            n=("score", "size"), score_lo=("score", "min"), score_hi=("score", "max"),
            mean_car_pct=("car_10d", lambda s: round(float(pd.to_numeric(s).mean()) * 100, 3)),
            pead_rate=("pead_pass", "mean"))
        log(tab.round(4).to_string())
        sc = pd.to_numeric(d_ok["score"], errors="coerce").astype("float64")
        cc = pd.to_numeric(d_ok["car_10d"], errors="coerce").astype("float64")
        # spearman = Pearson on average ranks (manual — see pearson_np note)
        rho = pearson_np(sc.rank().to_numpy(), cc.rank().to_numpy())
        log(f"spearman(score, car_10d) = {rho:+.4f} (n={len(d_ok):,})")
        results[name] = {"spearman": round(float(rho), 5),
                         "deciles": json.loads(tab.reset_index().to_json(orient="records"))}

        run_thresholds(pred, rng, name, results)
        low = pred[pred["score"] <= 0.05]
        log(f"\n{name}: LOW-score tail (<=0.05): n={len(low):,}, "
            f"mean car {pd.to_numeric(low['car_10d']).mean() * 100:+.3f}%, "
            f"pead {low['pead_pass'].mean():.1%}")

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nwrote {OUT / 'results.json'} (EXPLORATORY)")


if __name__ == "__main__":
    main()

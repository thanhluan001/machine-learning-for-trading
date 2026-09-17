"""RC-23 — momentum-conditioned high-confidence beat, EXCESS returns.

Registered 3c4f5b4. Rule: beat model (frozen), p_beat >= 0.80 AND
rel_ret_20d >= 0.0369. Return: EXCESS (stock - own benchmark, identical
windows, recomputed from prices). Gate G1: mean excess > 0 AND week-block
bootstrap CI excludes 0.
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

OUT = HERE / "archive" / "experiments" / "rc23"
OUT.mkdir(parents=True, exist_ok=True)
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
N_BOOT, SEED = 10_000, 20260807
P_BEAT_THR, RR20_THR = 0.80, 0.0369


def log(*a):
    print(*a, flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = load("bt_rc23", HERE / "51_hp_theta_sweep_23feat.py")
DB = bt.DB
FEATS22 = [f for f in bt.DEPLOY_FEATURES if f != "car_drift_historical_q1"]
t0 = time.time()

# ---------------- benchmarks + prices ----------------
ijh = pd.read_hdf(DB, "/macros/IJH")
bd = pd.to_datetime(ijh["Date"]).to_numpy().astype("datetime64[D]")
bc = ijh["Close"].to_numpy(float)
o = np.argsort(bd)
bd, bc = bd[o], bc[o]
with pd.HDFStore(DB_SP600, "r") as st:
    ijr = st["/sp600/benchmark_IJR"]
b6 = pd.to_datetime(ijr["Date"]).to_numpy().astype("datetime64[D]")
c6 = ijr["Adj_Close"].to_numpy(float)
o6 = np.argsort(b6)
b6, c6 = b6[o6], c6[o6]

with pd.HDFStore(DB, "r") as st:
    px = {}
    for k in st.keys():
        if k.startswith("/sp400/"):
            df = st[k]
            col = "Adj_Close" if "Adj_Close" in df.columns else "Close"
            d = pd.to_datetime(df["Date"]).to_numpy().astype("datetime64[D]")
            oo = np.argsort(d)
            px[k.split("/")[-1]] = (d[oo], df[col].to_numpy(float)[oo])
with pd.HDFStore(DB_SP600, "r") as st:
    for k in st.keys():
        parts = k.split("/")
        if len(parts) == 3 and parts[1] == "sp600":
            df = st[k]
            if "Date" in df.columns and "Adj_Close" in df.columns:
                d = pd.to_datetime(df["Date"]).to_numpy().astype("datetime64[D]")
                oo = np.argsort(d)
                px[parts[2]] = (d[oo], df["Adj_Close"].to_numpy(float)[oo])
log(f"prices/benches loaded ({time.time()-t0:.0f}s)")


def excess_ret(pt, ed, xd, sp4):
    if pt not in px or pd.isna(ed) or pd.isna(xd):
        return np.nan
    d, c = px[pt]
    i = np.searchsorted(d, ed)
    j = np.searchsorted(d, xd)
    if i >= len(d) or j >= len(d) or i >= j or c[i] <= 0:
        return np.nan
    sr = c[j] / c[i] - 1.0
    bdd, bcc = (bd, bc) if sp4 == 1 else (b6, c6)
    bi = np.searchsorted(bdd, ed)
    bj = np.searchsorted(bdd, xd)
    if bi >= len(bdd) or bj >= len(bdd) or bcc[bi] <= 0:
        return np.nan
    return sr - (bcc[bj] / bcc[bi] - 1.0)


# ---------------- beat label ----------------
v7 = pd.read_hdf(DB, "/features/train_matrix_v7c")
v7["report_date"] = pd.to_datetime(v7["report_date"])
v7["label_end"] = pd.to_datetime(v7["label_end"])
v7["entry_date"] = pd.to_datetime(v7["entry_date"])
v7["exit_date"] = pd.to_datetime(v7["exit_date"])
v6 = pd.read_hdf(DB, "/features/train_matrix_v6c")
v6["report_date"] = pd.to_datetime(v6["report_date"])
j = v6.set_index(["permaTicker", "report_date"])["sue_score"]
v7["sue"] = pd.to_numeric(j.reindex(pd.MultiIndex.from_arrays([v7.permaTicker, v7.report_date])).to_numpy(),
                          errors="coerce")
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
beat6 = []
for rd, t_ in zip(v7.loc[sel6, "report_date"], tkr):
    b = np.nan
    if isinstance(t_, str) and t_ in earn:
        e = earn[t_]
        e = e[pd.to_datetime(e["report_date"]) == rd]
        if len(e):
            r0 = e.iloc[-1]
            a_, est_ = float(r0["actual"]), float(r0["estimate"])
            if np.isfinite(a_) and np.isfinite(est_):
                b = 1.0 if a_ > est_ else 0.0
    beat6.append(b)
v7.loc[sel6, "beat_eps"] = beat6
v7["beat"] = np.where(v7["is_sp400"] == 1,
                      np.where(v7["sue"] > 0, 1.0, np.where(v7["sue"].isna(), np.nan, 0.0)),
                      v7["beat_eps"])
d = v7[~v7["sector"].isin(bt.EXCLUDE_SECTORS)].copy()
log(f"events {len(d):,} | beat defined {d.beat.notna().mean():.1%} ({time.time()-t0:.0f}s)")

# ---------------- beat model per fold ----------------
import xgboost as xgb

preds = []
for fi, (te, sve, tse) in enumerate(bt.DEFAULT_FOLDS, 1):
    tr = d[(d["label_end"] <= pd.Timestamp(sve)) & d["beat"].notna()]
    ts = d[(d["label_end"] > pd.Timestamp(sve)) & (d["label_end"] <= pd.Timestamp(tse))
           & d["beat"].notna()].copy()
    clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
                            n_estimators=300, learning_rate=0.05, max_depth=3,
                            min_child_weight=20, gamma=8, reg_lambda=1.0,
                            subsample=0.7, colsample_bytree=0.7,
                            random_state=42, n_jobs=-1)
    clf.fit(tr[FEATS22], tr["beat"].astype(int))
    ts["p_beat"] = clf.predict_proba(ts[FEATS22])[:, 1]
    ts["fold"] = fi
    preds.append(ts)
    log(f"fold {fi}: train {len(tr):,} test {len(ts):,} ({time.time()-t0:.0f}s)")
P = pd.concat(preds, ignore_index=True)

# ---------------- apply the frozen rule ----------------
R = P[(P["p_beat"] >= P_BEAT_THR) & (P["rel_ret_20d"] >= RR20_THR)].copy()
R["ex"] = [excess_ret(pt, ed.to_datetime64().astype("datetime64[D]"),
                      xd.to_datetime64().astype("datetime64[D]"), sp4)
           for pt, ed, xd, sp4 in zip(R.permaTicker, R.entry_date, R.exit_date, R.is_sp400)]
R = R[R["ex"].notna()].copy()
B = P[(P["p_beat"] >= P_BEAT_THR)].copy()
B["ex"] = [excess_ret(pt, ed.to_datetime64().astype("datetime64[D]"),
                      xd.to_datetime64().astype("datetime64[D]"), sp4)
           for pt, ed, xd, sp4 in zip(B.permaTicker, B.entry_date, B.exit_date, B.is_sp400)]
B = B[B["ex"].notna()].copy()
log(f"rule picks: {len(R)} (unconditioned p>=0.8: {len(B)})")

iso = R["report_date"].dt.isocalendar()
R["wk"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)


def boot_mean(df):
    rng = np.random.default_rng(SEED)
    g = df.groupby("wk")["ex"].agg(["sum", "count"])
    wks = g.index
    idx = rng.integers(0, len(wks), size=(N_BOOT, len(wks)))
    return g["sum"].to_numpy()[idx].sum(1) / np.maximum(g["count"].to_numpy()[idx].sum(1), 1)


mean_ex = float(R["ex"].mean())
boot = boot_mean(R)
se = float(boot.std(ddof=1))
ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
G1 = bool(mean_ex > 0 and ci[0] > 0)
log(f"\nG1: mean excess = {mean_ex*100:+.3f}%  SE {se*100:.2f}  t={mean_ex/se:.2f}")
log(f"    CI95 [{ci[0]*100:+.3f}, {ci[1]*100:+.3f}]  -> {'PASS' if G1 else 'FAIL'}")

res = {"rule": {"n": int(len(R)), "mean_excess_pct": round(mean_ex * 100, 3),
                "se_pct": round(se * 100, 3), "t": round(mean_ex / se, 2),
                "ci95_pct": [round(ci[0] * 100, 3), round(ci[1] * 100, 3)],
                "G1_pass": G1},
       "baseline_p080": {"n": int(len(B)), "mean_excess_pct": round(float(B['ex'].mean()) * 100, 3)},
       "diagnostics": {"beat_rate": round(float(R['beat'].mean()), 4),
                       "sp400_n": int((R.is_sp400 == 1).sum()),
                       "sp400_ex_pct": round(float(R.loc[R.is_sp400 == 1, 'ex'].mean() * 100), 3) if (R.is_sp400 == 1).any() else None,
                       "sp600_n": int((R.is_sp400 == 0).sum()),
                       "sp600_ex_pct": round(float(R.loc[R.is_sp400 == 0, 'ex'].mean() * 100), 3) if (R.is_sp400 == 0).any() else None,
                       "raw_mean_pct": round(float(R['pregap_return'].astype(float).mean() * 100), 3)},
       "config": {"p_beat_thr": P_BEAT_THR, "rr20_thr": RR20_THR,
                  "n_boot": N_BOOT, "seed": SEED},
       "seconds": round(time.time() - t0)}
with open(OUT / "gauntlet.json", "w") as f:
    json.dump(res, f, indent=2, default=str)
keep = [c for c in ("permaTicker", "report_date", "is_sp400", "p_beat", "rel_ret_20d",
                    "beat", "ex", "fold") if c in R.columns]
R[keep].to_hdf(OUT / "picks.h5", key="p", format="table")
log(f"\n{json.dumps(res, indent=1, default=str)}")
log(f"wrote {OUT / 'gauntlet.json'} ({round(time.time()-t0)}s)")

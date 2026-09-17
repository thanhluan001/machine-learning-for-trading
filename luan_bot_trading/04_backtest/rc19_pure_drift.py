"""RC-19 — pure-drift label construction (registered spec, frozen).

Arms:
  v6n       22 feats, gate trio, min-gate 0.33        (paired baseline)
  rc19      24 feats, SINGLE pass_g1 classifier, thr 0.33   (PRIMARY)
  rc19_cost 24 feats, label car_10d > +0.001          (secondary, non-gating)
  rc19_pos  24 feats, label car_10d > 0               (secondary, non-gating)

Identical machinery to RC-18 P3 (label-matured folds, pregap simulator,
week-block bootstrap 10k seed 20260807). Decoupling diagnostics for the
primary arm: spearman(score, car_10d) + decile table.
"""
from __future__ import annotations

import bisect
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

OUT = HERE / "archive" / "experiments" / "rc19"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT, SEED, THRESH = 10_000, 20260807, 0.33
GATES3 = ["pass_g1", "pass_g2", "pass_g3"]
V6_HP = {
    "pass_g1": {"gamma": 8, "min_child_weight": 20, "max_depth": 3},
    "pass_g2": {"gamma": 12, "min_child_weight": 50, "max_depth": 3},
    "pass_g3": {"gamma": 1, "min_child_weight": 50, "max_depth": 3},
}
TOPK = 10


def log(*a):
    print(*a, flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = load("bt_rc19", HERE / "51_hp_theta_sweep_23feat.py")
p0 = load("rc18_p0_mod_19", HERE / "rc18_p0_firewall.py")
DB = bt.DB
FEATURES22 = [f for f in bt.DEPLOY_FEATURES if f != "car_drift_historical_q1"]
NEW = ["F1_sb_h3", "A2"]

t0 = time.time()
# ---------------- features on ALL events (identical to RC-18 P3) ----------------
ev = p0.load_events()
cal, bclose = p0.load_calendar()
ev["pos"] = np.searchsorted(cal, ev["T"].to_numpy().astype("datetime64[D]"), side="left")
back = np.where(ev["is_bmo"].astype(bool), 2, 1)
ev["cut_date"] = cal[np.clip(ev["pos"] - back, 0, len(cal) - 1)]
prices = p0.load_prices()
ev = p0.compute_drifts(ev, prices, cal, bclose, horizons=(1, 3, 11))
ev3 = ev
log(f"events {len(ev):,} ({time.time()-t0:.0f}s)")

pos = ev["pos"].to_numpy()
pt_arr = ev["permaTicker"].to_numpy()
sue = ev["sue_score"].to_numpy()
cut = ev["cut_date"].to_numpy()
r1 = ev["r1"].to_numpy()
r3 = ev3["r3"].to_numpy()
mat3 = ev3["mat3"].to_numpy()
r11 = ev["r11"].to_numpy()
mat11 = ev["mat11"].to_numpy()
n = len(ev)

A2 = np.full(n, np.nan)
tick_groups = {}
for pt, sub in ev.groupby("permaTicker", sort=False):
    s = sub.sort_values("pos")
    tick_groups[pt] = (s["pos"].to_numpy(), s.index.to_numpy())
for i in range(n):
    p_, ix_ = tick_groups[pt_arr[i]]
    k_ = bisect.bisect_left(p_, pos[i])
    vals = []
    for j in ix_[:k_][::-1]:
        mj = mat11[j]
        if pd.isna(mj) or mj > cut[i] or not np.isfinite(r11[j]) or not np.isfinite(r1[j]):
            continue
        vals.append(r11[j] * np.sign(r1[j]))
        if len(vals) == 8:
            break
    if vals:
        A2[i] = float(np.mean(vals))
log(f"A2 defined {np.isfinite(A2).mean():.1%} ({time.time()-t0:.0f}s)")

order = np.argsort(pos, kind="stable")
pos_sorted = pos[order]


def peers_vec3(i_idx):
    p = pos[i_idx]
    lo = bisect.bisect_left(pos_sorted, p - p0.K_WINDOW)
    hi = bisect.bisect_left(pos_sorted, p)
    w = order[lo:hi]
    if len(w) == 0:
        return w
    m = (pt_arr[w] != pt_arr[i_idx]) & (sue[w] > 0)
    m &= ~pd.isna(mat3[w]) & (mat3[w] <= cut[i_idx]) & np.isfinite(r3[w])
    return w[m]


cp_dates = []
last_m, prev_d = None, None
for d in cal:
    mth = d.astype("M8[M]")
    if last_m is not None and mth != last_m:
        cp_dates.append(prev_d)
    prev_d = d
    last_m = mth
cp_dates.append(cal[-1])
cp_dates = np.array([d for d in cp_dates if d >= np.datetime64("2016-01-01", "D")])
beat_ok = ev[(sue > 0) & ev["r1"].notna() & ev["mat1"].notna()]
Tb = pd.to_datetime(beat_ok["T"]).dt.isocalendar()
wkb = (Tb.year.astype(str) + "-W" + Tb.week.astype(str).str.zfill(2)).to_numpy()
ptb = beat_ok["permaTicker"].to_numpy()
r1b = beat_ok["r1"].to_numpy()
matb = beat_ok["mat1"].to_numpy()
svd_cps = []
for cp in cp_dates:
    sel = matb <= cp
    if sel.sum() < 200:
        continue
    M = pd.pivot_table(pd.DataFrame({"pt": ptb[sel], "wk": wkb[sel], "r": r1b[sel]}),
                       index="pt", columns="wk", values="r", aggfunc="mean")
    M = M.sub(M.mean(axis=0), axis=1).fillna(0.0)
    if M.shape[1] < p0.EMB_DIM:
        continue
    U, S, _ = np.linalg.svd(M.to_numpy(), full_matrices=False)
    E = U[:, :p0.EMB_DIM] * S[:p0.EMB_DIM]
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    svd_cps.append((cp, pd.DataFrame(E / np.where(nrm > 0, nrm, 1.0), index=M.index)))
cp_list = [c for c, _ in svd_cps]
log(f"SVD checkpoints {len(svd_cps)} ({time.time()-t0:.0f}s)")

F1 = np.full(n, np.nan)
for i in range(n):
    w_idx = peers_vec3(i)
    if len(w_idx) == 0:
        continue
    k = bisect.bisect_right(cp_list, cut[i]) - 1
    if k < 0:
        continue
    emb = svd_cps[k][1]
    pt_i = pt_arr[i]
    if pt_i not in emb.index:
        continue
    u = emb.loc[pt_i].to_numpy()
    sims = (emb @ u).clip(lower=0.0)
    ws = np.array([float(sims.get(o, 0.0)) for o in pt_arr[w_idx]])
    if len(ws) > TOPK:
        keep = np.argsort(-ws)[:TOPK]
        w_idx, ws = w_idx[keep], ws[keep]
    rs = r3[w_idx]
    m = ws > 0
    if not m.any():
        continue
    ws, rs = ws[m], rs[m]
    F1[i] = float((ws * rs).sum() / ws.sum())
log(f"F1_sb_h3 defined {np.isfinite(F1).mean():.1%} ({time.time()-t0:.0f}s)")

# ---------------- matrix ----------------
m = pd.read_hdf(DB, "/features/train_matrix_v6c")
old = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
for d in (old, m):
    d["report_date"] = pd.to_datetime(d["report_date"])
ko = set(zip(old.permaTicker, old.report_date))
kc = set(zip(m.permaTicker, m.report_date))
common = ko & kc
m = m[m.set_index(["permaTicker", "report_date"]).index.isin(common)].copy()
car_map = ev.set_index(["permaTicker", ev["report_date"] if "report_date" in ev else pd.to_datetime(ev["T"])])["car_10d"]
m["car_10d_join"] = car_map.reindex(pd.MultiIndex.from_arrays([m.permaTicker, m.report_date])).to_numpy()
feat_df = pd.DataFrame({"F1_sb_h3": F1, "A2": A2}, index=ev.index)
m = m.join(feat_df, how="left")
log(f"matrix {len(m):,} | car_10d nan {pd.isna(m.car_10d_join).mean():.1%} "
    f"| F1 nan {m.F1_sb_h3.isna().mean():.1%} | A2 nan {m.A2.isna().mean():.1%}")

# ---------------- arm machinery ----------------
def fit_gate(X, y, Xe, ye, hp):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=300, learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1,
    ).fit(X, y, eval_set=[(Xe, ye)], verbose=False)


def run_arm(df, feats, gates_list, hp_map, score="min"):
    rd = df["label_end"]
    out = []
    for fi, (te, sve, tse) in enumerate(bt.DEFAULT_FOLDS, 1):
        tr = df[rd <= pd.Timestamp(te)]
        sv = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))]
        ts = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        train = pd.concat([tr, sv], ignore_index=True)
        for i, g in enumerate(gates_list, 1):
            clf = fit_gate(train[feats], train[g].astype(int).to_numpy(),
                           ts[feats], ts[g].astype(int).to_numpy(), hp_map[g])
            ts[f"p{i}"] = clf.predict_proba(ts[feats])[:, 1]
        if score == "min":
            ts["score"] = ts[[f"p{i}" for i in range(1, len(gates_list) + 1)]].min(axis=1)
        else:
            ts["score"] = ts["p1"]
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
    return pred, raw, ex


def weekly(ex):
    if ex is None or len(ex) == 0:
        return pd.Series(dtype=float)
    z = ex.copy()
    z["entry_date"] = pd.to_datetime(z["entry_date"])
    iso = z["entry_date"].dt.isocalendar()
    z["wk"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    return z.groupby("wk", sort=True)["pregap_return"].sum() / bt.N_SLOTS


def stats(ex):
    if ex is None or len(ex) == 0:
        return {"executed": 0}
    r = ex["pregap_return"].astype(float)
    w = weekly(ex)
    return {"executed": int(len(r)), "win_rate_pct": round(float((r > 0).mean() * 100), 1),
            "avg_trade_pct": round(float(r.mean() * 100), 3),
            "weeks": int(len(w)),
            "nav_pct": round(float(((1 + w).prod() - 1) * 100), 2),
            "raw_precision": round(float(ex["pead_pass"].mean() * 100), 1),
            "g1_rate": round(float(ex["pass_g1"].mean() * 100), 1)}


def boot_mean(w, rng):
    v = w.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"mean_pct": None, "ci95_pct": [None, None]}
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    mm = v[idx].mean(axis=1)
    return {"mean_pct": round(float(v.mean() * 100), 4),
            "ci95_pct": [round(float(np.percentile(mm, 2.5)) * 100, 4),
                         round(float(np.percentile(mm, 97.5)) * 100, 4)]}


def paired_diff(wa, wb, rng):
    j = pd.concat({"a": wa, "b": wb}, axis=1, join="inner").dropna()
    d = (j["a"] - j["b"]).to_numpy(dtype=float)
    if not len(d):
        return {"paired_weeks": 0, "mean_diff_pct": None, "ci95_pct": [None, None]}
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    mm = d[idx].mean(axis=1)
    return {"paired_weeks": int(len(d)), "mean_diff_pct": round(float(d.mean() * 100), 4),
            "ci95_pct": [round(float(np.percentile(mm, 2.5)) * 100, 4),
                         round(float(np.percentile(mm, 97.5)) * 100, 4)],
            "prob_pos": round(float((d > 0).mean()), 4)}


rng = np.random.default_rng(SEED)
runs = {}
_, _, ex = run_arm(m, FEATURES22, GATES3, V6_HP, score="min")
runs["v6n"] = {"stats": stats(ex), "boot": boot_mean(weekly(ex), rng), "ex": ex}
log(f"v6n: {json.dumps(runs['v6n']['stats'], default=str)}")

_, _, ex = run_arm(m, FEATURES22 + NEW, ["pass_g1"], V6_HP, score="single")
runs["rc19"] = {"stats": stats(ex), "boot": boot_mean(weekly(ex), rng), "ex": ex,
                "pred": None}
log(f"rc19: {json.dumps(runs['rc19']['stats'], default=str)}")
log(f"  CI: {json.dumps(runs['rc19']['boot'])}")

m2 = m.copy()
m2["pass_cost"] = (m2["car_10d_join"] > 0.001).astype(float)
m2["pass_pos"] = (m2["car_10d_join"] > 0.0).astype(float)
for nm, lab in (("rc19_cost", "pass_cost"), ("rc19_pos", "pass_pos")):
    _, _, exx = run_arm(m2, FEATURES22 + NEW, [lab], {lab: V6_HP["pass_g1"]}, score="single")
    runs[nm] = {"stats": stats(exx), "ex": exx}
    log(f"{nm}: {json.dumps(runs[nm]['stats'], default=str)}")

g2p = paired_diff(weekly(runs["rc19"]["ex"]), weekly(runs["v6n"]["ex"]), rng)
s19, s6n = runs["rc19"]["stats"], runs["v6n"]["stats"]
ci = runs["rc19"]["boot"]["ci95_pct"]
G1 = bool(s19.get("avg_trade_pct", 0) > 0 and ci and (ci[0] > 0 or ci[1] < 0))
G2 = bool(s19.get("avg_trade_pct", 0) > s6n.get("avg_trade_pct", -9e9)
          and g2p["ci95_pct"] and (g2p["ci95_pct"][0] > 0 or g2p["ci95_pct"][1] < 0))

# decoupling diagnostics (primary arm, full OOS rows)
_, raw, _ = run_arm(m, FEATURES22 + NEW, ["pass_g1"], V6_HP, score="single")
dec = raw[["score", "car_10d_join"]].dropna()
deciles = (dec.assign(d=pd.qcut(dec.score.rank(method="first"), 10, labels=False))
           .groupby("d").agg(n=("score", "size"), mean_car_pct=("car_10d_join", lambda x: round(float(x.mean() * 100), 2)),
                             mean_score=("score", "mean")).round(3).to_dict("records"))
sp = float(pd.Series(dec.score).corr(pd.Series(dec.car_10d_join), method="spearman"))
log(f"decoupling: spearman={sp:.4f} | deciles={json.dumps(deciles)}")

verdict = {"G1_primary_positive": {"pass": G1, "boot": runs["rc19"]["boot"]},
           "G2_beats_v6n": {"pass": G2, "paired": g2p},
           "decoupling_spearman": round(sp, 4), "deciles": deciles}
log("VERDICT: " + json.dumps(verdict, indent=1, default=str))

with open(OUT / "gauntlet.json", "w") as f:
    json.dump({"runs": {k: v["stats"] for k, v in runs.items()},
               "boot": {k: v.get("boot") for k, v in runs.items() if v.get("boot")},
               "verdict": verdict, "common_keys": len(common),
               "config": {"n_boot": N_BOOT, "seed": SEED, "thresh": THRESH},
               "seconds": round(time.time() - t0)}, f, indent=2, default=str)
with pd.HDFStore(OUT / "executed.h5", "w") as st:
    for k, v in runs.items():
        if len(v["ex"]):
            keep = [c for c in ("permaTicker", "fold", "pregap_entry_date", "pregap_exit_date",
                                "pregap_return", "score", "pead_pass", "pass_g1") if c in v["ex"].columns]
            st.put(f"/{k}", v["ex"][keep], format="table")
log(f"wrote {OUT / 'gauntlet.json'} ({round(time.time() - t0)}s)")

"""RC-18 P3 — combined model gauntlet (pre-registered gates, untouched).

Arms (identical machinery to RC-16 R3 / RC-17):
  v6n    : 22 retired features (baseline)
  v6c18  : 22 + F1_sb_h3 (SVD read-across) + A2 (follow-through) = 24

Both on /features/train_matrix_v6c restricted to the 16,587 common keys,
frozen V6 gate HPs, min-gate 0.33, label-matured folds, matrix
pregap_return simulator, 10k week-block bootstrap seed 20260807.

GATES (from the amendment; not amendable):
  G1  v6c18 mean trade > 0 AND week-block CI excludes 0
  G2  v6c18 beats v6n: mean trade greater AND paired weekly diff CI excl 0

Features are PIT: F1_sb_h3 via monthly SVD checkpoints (matured cells
only), A2 via matured prior episodes. Both passed truncation-equivalence
(120/120 and 240/240).
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

OUT = HERE / "archive" / "experiments" / "rc18_p3"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT, SEED = 10_000, 20260807
THRESH = 0.33
GATES = ["pass_g1", "pass_g2", "pass_g3"]
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


bt = load("bt_rc18p3", HERE / "51_hp_theta_sweep_23feat.py")
p0 = load("rc18_p0_mod_p3", HERE / "rc18_p0_firewall.py")
DB = bt.DB
FEATURES22 = [f for f in bt.DEPLOY_FEATURES if f != "car_drift_historical_q1"]

t0 = time.time()
# ---------------- features on ALL events (PIT) ----------------
ev = p0.load_events()
cal, bclose = p0.load_calendar()
ev["pos"] = np.searchsorted(cal, ev["T"].to_numpy().astype("datetime64[D]"), side="left")
back = np.where(ev["is_bmo"].astype(bool), 2, 1)
ev["cut_date"] = cal[np.clip(ev["pos"] - back, 0, len(cal) - 1)]
prices = p0.load_prices()
ev = p0.compute_drifts(ev, prices, cal, bclose, horizons=(1, 11))
log(f"events {len(ev):,} ({time.time()-t0:.0f}s)")

# A2: follow-through personality (all events, own ticker, matured)
pos = ev["pos"].to_numpy()
pt_arr = ev["permaTicker"].to_numpy()
sue = ev["sue_score"].to_numpy()
cut = ev["cut_date"].to_numpy()
r1 = ev["r1"].to_numpy()
r11 = ev["r11"].to_numpy()
mat1 = ev["mat1"].to_numpy()
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

# F1_sb_h3: SVD read-across (all events; peers beat, 3-day partial, PIT checkpoints)
order = np.argsort(pos, kind="stable")
pos_sorted = pos[order]

def peers_vec(i_idx, h):
    p = pos[i_idx]
    lo = bisect.bisect_left(pos_sorted, p - p0.K_WINDOW)
    hi = bisect.bisect_left(pos_sorted, p)
    w = order[lo:hi]
    if len(w) == 0:
        return w
    m = (pt_arr[w] != pt_arr[i_idx]) & (sue[w] > 0)
    mj = (mat1 if h == 1 else ev["mat3"].to_numpy())[w] if h != 3 else None
    # h=3 uses mat3
    if h == 3:
        mj = ev["mat3"].to_numpy()[w]
        rj = ev["r3"].to_numpy()[w]
    else:
        mj = mat1[w]
        rj = r1[w]
    m &= ~pd.isna(mj) & (mj <= cut[i_idx]) & np.isfinite(rj)
    return w[m]

ev3 = p0.compute_drifts(ev.copy(), prices, cal, bclose, horizons=(3,))
r3 = ev3["r3"].to_numpy()
mat3 = ev3["mat3"].to_numpy()

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

# SVD checkpoints (monthly, matured cells only)
cp_dates = []
last_m, prev_d = None, None
for d in cal:
    m = d.astype("M8[M]")
    if last_m is not None and m != last_m:
        cp_dates.append(prev_d)
    prev_d = d
    last_m = m
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

# ---------------- assemble matrix ----------------
m = pd.read_hdf(DB, "/features/train_matrix_v6c")
old = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
for d in (old, m):
    d["report_date"] = pd.to_datetime(d["report_date"])
ko = set(zip(old.permaTicker, old.report_date))
kc = set(zip(m.permaTicker, m.report_date))
common = ko & kc
m = m[m.set_index(["permaTicker", "report_date"]).index.isin(common)].copy()
feat_df = pd.DataFrame({"F1_sb_h3": F1, "A2": A2}, index=ev.index)
m = m.join(feat_df, how="left")
log(f"matrix rows {len(m):,} | F1 nan {m.F1_sb_h3.isna().mean():.1%} | A2 nan {m.A2.isna().mean():.1%}")

# ---------------- gauntlet machinery (117-style) ----------------
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
    rd = df["label_end"] if "label_end" in df.columns else pd.to_datetime(df["report_date"])
    out = []
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
            "raw_precision": round(float(ex["pead_pass"].mean() * 100), 1)}


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
                         round(float(np.percentile(mm, 97.5)) * 100), 4],
            "prob_pos": round(float((d > 0).mean()), 4)}


rng = np.random.default_rng(SEED)
runs = {}
for name, feats in (("v6n", FEATURES22),
                    ("v6c18", FEATURES22 + ["F1_sb_h3", "A2"])):
    _, _, ex = run_arm(m, feats)
    w = weekly(ex)
    runs[name] = {"stats": stats(ex), "boot": boot_mean(w, rng), "w": w, "ex": ex}
    log(f"{name}: {json.dumps(runs[name]['stats'], default=str)}")
    log(f"  week-block CI: {json.dumps(runs[name]['boot'])}")

g2p = paired_diff(runs["v6c18"]["w"], runs["v6n"]["w"], rng)
s18, s6n = runs["v6c18"]["stats"], runs["v6n"]["stats"]
ci18 = runs["v6c18"]["boot"]["ci95_pct"]
G1 = bool(s18.get("avg_trade_pct", 0) > 0 and ci18 and (ci18[0] > 0 or ci18[1] < 0))
G2 = bool(s18.get("avg_trade_pct", 0) > s6n.get("avg_trade_pct", 0)
          and g2p["ci95_pct"] and (g2p["ci95_pct"][0] > 0 or g2p["ci95_pct"][1] < 0))
verdict = {"G1_combined_positive": {"pass": G1, "boot": runs["v6c18"]["boot"]},
           "G2_beats_v6n": {"pass": G2, "paired": g2p}}
log("VERDICT: " + json.dumps(verdict, indent=1, default=str))

with open(OUT / "gauntlet.json", "w") as f:
    json.dump({"runs": {k: {kk: vv for kk, vv in v.items() if kk not in ("w", "ex")}
                        for k, v in runs.items()},
               "verdict": verdict, "common_keys": len(common),
               "features_added": ["F1_sb_h3", "A2"],
               "config": {"n_boot": N_BOOT, "seed": SEED, "thresh": THRESH},
               "seconds": round(time.time() - t0)}, f, indent=2, default=str)
with pd.HDFStore(OUT / "executed.h5", "w") as st:
    for k, v in runs.items():
        if len(v["ex"]):
            keep = [c for c in ("permaTicker", "fold", "entry_date", "pregap_exit_date",
                                "pregap_return", "score", "pead_pass") if c in v["ex"].columns]
            st.put(f"/{k}", v["ex"][keep], format="table")
log(f"wrote {OUT / 'gauntlet.json'} ({round(time.time()-t0)}s)")

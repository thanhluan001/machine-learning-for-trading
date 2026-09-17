"""Diagnostic (non-gating): why did mixed training underperform transfer on SP600?

Compares, per fold, on identical SP600 test rows:
  TRANSFER model (SP400-only training, RC-21 protocol)
  MIXED model   (combined-universe training, RC-20 protocol)
Metrics per model x universe: AUC vs pass_g1, spearman(score, car_10d),
pass-rate>=0.33 + pool avg car_10d, SP600-only weekly 4-slot selection
(same simulator both models -> removes slot-competition effect), and
pick overlap. Discriminates: boundary compromise (AUC), noise-chasing
(spearman/pool avg), threshold semantics (pass rates), slot competition
(executed under identical selection).
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

DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
THRESH = 0.33
HP1 = {"gamma": 8, "min_child_weight": 20, "max_depth": 3}
TOPK = 10


def log(*a):
    print(*a, flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = load("bt_diag", HERE / "51_hp_theta_sweep_23feat.py")
p0 = load("p0_diag", HERE / "rc18_p0_firewall.py")
DB = bt.DB
FEATURES22 = [f for f in bt.DEPLOY_FEATURES if f != "car_drift_historical_q1"]
NEW = ["F1_sb_h3", "A2"]
FEATS = FEATURES22 + NEW

t0 = time.time()
cal, ijh_close = p0.load_calendar()
prices = p0.load_prices()
with pd.HDFStore(DB_SP600, "r") as st:
    ijr = st["/sp600/benchmark_IJR"]
    p6 = {}
    for k in st.keys():
        parts = k.split("/")
        if len(parts) == 3 and parts[1] == "sp600":
            df = st[k]
            if "Date" in df.columns and "Adj_Close" in df.columns:
                d = pd.to_datetime(df["Date"]).to_numpy().astype("datetime64[D]")
                o = np.argsort(d)
                p6[parts[2]] = (d[o], df["Adj_Close"].to_numpy(dtype=float)[o])
prices_all = dict(prices)
prices_all.update(p6)
b6 = pd.to_datetime(ijr["Date"]).to_numpy().astype("datetime64[D]")
o6 = np.argsort(b6)
cal6, close6 = b6[o6], ijr["Adj_Close"].to_numpy(dtype=float)[o6]

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


def build_embedding(bf):
    Tb = pd.to_datetime(bf["T"]).dt.isocalendar()
    wkb = (Tb.year.astype(str) + "-W" + Tb.week.astype(str).str.zfill(2)).to_numpy()
    ptb, r1b, matb = bf["permaTicker"].to_numpy(), bf["r1"].to_numpy(), bf["mat1"].to_numpy()
    cps = []
    for cp in cp_dates:
        s_ = matb <= cp
        if s_.sum() < 200:
            continue
        M = pd.pivot_table(pd.DataFrame({"pt": ptb[s_], "wk": wkb[s_], "r": r1b[s_]}),
                           index="pt", columns="wk", values="r", aggfunc="mean")
        M = M.sub(M.mean(axis=0), axis=1).fillna(0.0)
        if M.shape[1] < p0.EMB_DIM:
            continue
        U, S, _ = np.linalg.svd(M.to_numpy(), full_matrices=False)
        E = U[:, :p0.EMB_DIM] * S[:p0.EMB_DIM]
        nrm = np.linalg.norm(E, axis=1, keepdims=True)
        cps.append((cp, pd.DataFrame(E / np.where(nrm > 0, nrm, 1.0), index=M.index)))
    return cps


def compute_A2(evD):
    pta, posa, cuta = evD["permaTicker"].to_numpy(), evD["pos"].to_numpy(), evD["cut_date"].to_numpy()
    r1a, r11a, m11a = evD["r1"].to_numpy(), evD["r11"].to_numpy(), evD["mat11"].to_numpy()
    A = np.full(len(evD), np.nan)
    tg = {}
    for pt, sub in evD.groupby("permaTicker", sort=False):
        s = sub.sort_values("pos")
        tg[pt] = (s["pos"].to_numpy(), s.index.to_numpy())
    for i in range(len(evD)):
        if pd.isna(cuta[i]):
            continue
        p_, ix_ = tg[pta[i]]
        k_ = bisect.bisect_left(p_, posa[i])
        vals = []
        for jx in ix_[:k_][::-1]:
            mj = m11a[jx]
            if pd.isna(mj) or mj > cuta[i] or not np.isfinite(r11a[jx]) or not np.isfinite(r1a[jx]):
                continue
            vals.append(r11a[jx] * np.sign(r1a[jx]))
            if len(vals) == 8:
                break
        if vals:
            A[i] = float(np.mean(vals))
    return A


def compute_F1(evD, cps):
    pta, posa, cuta = evD["permaTicker"].to_numpy(), evD["pos"].to_numpy(), evD["cut_date"].to_numpy()
    beat, r3, mat3 = evD["beatflag"].to_numpy(), evD["r3"].to_numpy(), evD["mat3"].to_numpy()
    cp_list = [c for c, _ in cps]
    order = np.argsort(posa, kind="stable")
    pos_sorted = posa[order]
    F = np.full(len(evD), np.nan)
    for i in range(len(evD)):
        if pd.isna(cuta[i]):
            continue
        p = posa[i]
        lo = bisect.bisect_left(pos_sorted, p - p0.K_WINDOW)
        hi = bisect.bisect_left(pos_sorted, p)
        w = order[lo:hi]
        if len(w) == 0:
            continue
        mm = (pta[w] != pta[i]) & beat[w]
        mm &= ~pd.isna(mat3[w]) & (mat3[w] <= cuta[i]) & np.isfinite(r3[w])
        w_idx = w[mm]
        if len(w_idx) == 0:
            continue
        k = bisect.bisect_right(cp_list, cuta[i]) - 1
        if k < 0:
            continue
        emb = cps[k][1]
        if pta[i] not in emb.index:
            continue
        u = emb.loc[pta[i]].to_numpy()
        sims = (emb @ u).clip(lower=0.0)
        ws = np.array([float(sims.get(o, 0.0)) for o in pta[w_idx]])
        if len(ws) > TOPK:
            keep = np.argsort(-ws)[:TOPK]
            w_idx, ws = w_idx[keep], ws[keep]
        rs = r3[w_idx]
        mk = ws > 0
        if not mk.any():
            continue
        ws, rs = ws[mk], rs[mk]
        F[i] = float((ws * rs).sum() / ws.sum())
    return F


def prep_clocks(evD):
    pos = np.searchsorted(cal, evD["T"].to_numpy().astype("datetime64[D]"), side="left")
    evD["pos"] = pos
    back = np.where(pd.isna(evD["is_bmo"]), -1, np.where(evD["is_bmo"] == 1, 2, 1))
    cut = cal[np.clip(pos - back, 0, len(cal) - 1)]
    cut[pd.isna(evD["is_bmo"])] = np.datetime64("NaT", "D")
    evD["cut_date"] = cut
    return evD


# ---- SP400 (train) side: RC-19 pipeline, common keys ----
v6c = pd.read_hdf(DB, "/features/train_matrix_v6c")
v6c["report_date"] = pd.to_datetime(v6c["report_date"])
old = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
old["report_date"] = pd.to_datetime(old["report_date"])
ko = set(zip(old.permaTicker, old.report_date))
v6c = v6c[v6c.set_index(["permaTicker", "report_date"]).index.isin(ko & set(zip(v6c.permaTicker, v6c.report_date)))].copy()
ev4 = v6c[["permaTicker", "report_date", "T", "is_bmo", "sue_score"]].copy()
ev4["beatflag"] = ev4["sue_score"] > 0
ev4 = p0.compute_drifts(ev4.reset_index(drop=True).copy(), prices, cal, ijh_close, horizons=(1, 3, 11))
ev4 = prep_clocks(ev4)
A2_4 = compute_A2(ev4)
cps4 = build_embedding(ev4[(ev4["beatflag"].to_numpy()) & ev4["r1"].notna() & ev4["mat1"].notna()])
F1_4 = compute_F1(ev4, cps4)
train_df = v6c.reset_index(drop=True).join(pd.DataFrame({"F1_sb_h3": F1_4, "A2": A2_4}))
train_df["label_end"] = pd.to_datetime(train_df["label_end"])
log(f"SP400 side {len(train_df):,} ({time.time()-t0:.0f}s)")

# ---- SP600 (app) side + combined: RC-20 pipeline ----
m = pd.read_hdf(DB, "/features/train_matrix_v7c")
m["report_date"] = pd.to_datetime(m["report_date"])
m["label_end"] = pd.to_datetime(m["label_end"])
ev6 = m[m["is_sp400"] == 0][["permaTicker", "report_date", "is_sp400"]].copy()
ev6["T"] = ev6["report_date"]
ev6["is_bmo"] = np.nan
with pd.HDFStore(DB_SP600, "r") as st:
    s6 = st["/features/train_matrix_sp600_pt_v7c"]
    s6["report_date"] = pd.to_datetime(s6["report_date"])
    tk = s6.set_index(["permaTicker", "report_date"])["ticker"]
    earn = {}
    for k in st.keys():
        if k.startswith("/sp600/earnings_full/"):
            earn[k.split("/")[-1]] = st[k]
tkr = tk.reindex(pd.MultiIndex.from_arrays([ev6["permaTicker"], ev6["report_date"]])).to_numpy()
rt, rb = [], []
for rd, t_ in zip(ev6["report_date"], tkr):
    got = False
    if isinstance(t_, str) and t_ in earn:
        e = earn[t_]
        e = e[pd.to_datetime(e["report_date"]) == rd]
        if len(e):
            r0 = e.iloc[-1]
            rt.append(1.0 if str(r0["time"]).lower() == "bmo" else (0.0 if str(r0["time"]).lower() == "amc" else np.nan))
            rb.append(bool(np.isfinite(float(r0["actual"])) and np.isfinite(float(r0["estimate"]))
                           and float(r0["actual"]) > float(r0["estimate"])))
            got = True
    if not got:
        rt.append(np.nan)
        rb.append(False)
ev6["is_bmo"] = rt
ev6["beatflag"] = rb
idx6 = ev6.index
ev6c = p0.compute_drifts(ev6.reset_index(drop=True).copy(), prices_all, cal6, close6, horizons=(1, 3, 11))
ev6c.index = idx6
ev6c = prep_clocks(ev6c)
ev6c["is_sp400"] = 0
ev4_c = ev4.copy()
ev4_c["is_sp400"] = 1
evC = pd.concat([ev4_c, ev6c], ignore_index=True)
cpsC = build_embedding(evC[(evC["beatflag"].to_numpy()) & evC["r1"].notna() & evC["mat1"].notna()])
evCp = evC.reset_index(drop=True)
n4c = int((evCp["is_sp400"] == 1).sum())
A2_all = compute_A2(evCp)
F1_all = compute_F1(evCp, cpsC)

app_df = m[m["is_sp400"] == 0].copy().reset_index(drop=True)
app_df["F1_sb_h3"] = F1_all[n4c:]
app_df["A2"] = A2_all[n4c:]
comb_df = pd.concat([train_df[["permaTicker", "report_date"]].assign(is_sp400=1),
                     app_df[["permaTicker", "report_date"]].assign(is_sp400=0)]).sort_values(["report_date"])
# mixed training frame: evC-matched SP400 rows (combined-pipeline F1/A2) + SP600 rows
mix400 = train_df.copy()
mix400["is_sp400"] = 1
mix400["F1_sb_h3"] = F1_all[:n4c]
mix400["A2"] = A2_all[:n4c]
mixed_df = pd.concat([mix400, app_df], ignore_index=True)
log(f"frames ready ({time.time()-t0:.0f}s)")


def fit_gate(X, y, Xe, ye):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=300, learning_rate=0.05, max_depth=HP1["max_depth"],
        min_child_weight=HP1["min_child_weight"], gamma=HP1["gamma"],
        reg_lambda=1.0, subsample=0.7, colsample_bytree=0.7,
        random_state=42, n_jobs=-1,
    ).fit(X, y, eval_set=[(Xe, ye)], verbose=False)


from sklearn.metrics import roc_auc_score

rows = []
for fi, (te, sve, tse) in enumerate(bt.DEFAULT_FOLDS, 1):
    lo = pd.Timestamp(sve)
    hi = pd.Timestamp(tse)
    tr_t = train_df[train_df["label_end"] <= lo]
    tr_m = mixed_df[(mixed_df["label_end"] <= lo)]
    ts6 = app_df[(app_df["label_end"] > lo) & (app_df["label_end"] <= hi)].copy()
    ts4 = train_df[(train_df["label_end"] > lo) & (train_df["label_end"] <= hi)].copy()
    ts4m = mixed_df[(mixed_df["is_sp400"] == 1) & (mixed_df["label_end"] > lo) & (mixed_df["label_end"] <= hi)].copy()
    clf_t = fit_gate(tr_t[FEATS], tr_t["pass_g1"].astype(int).to_numpy(), ts6[FEATS], ts6["pass_g1"].astype(int).to_numpy())
    clf_m = fit_gate(tr_m[FEATS], tr_m["pass_g1"].astype(int).to_numpy(), ts6[FEATS], ts6["pass_g1"].astype(int).to_numpy())
    for nm, clf in (("transfer", clf_t), ("mixed", clf_m)):
        s6 = clf.predict_proba(ts6[FEATS])[:, 1]
        s4 = clf.predict_proba(ts4m[FEATS])[:, 1]
        rows.append({"fold": fi, "model": nm,
                     "auc_sp600": round(float(roc_auc_score(ts6["pass_g1"], s6)), 4),
                     "auc_sp400": round(float(roc_auc_score(ts4m["pass_g1"], s4)), 4),
                     "sp600_pass_rate": round(float((s6 >= THRESH).mean()), 3),
                     "sp600_pool_avg_car": round(float(ts6.loc[s6 >= THRESH, "car_10d"].astype(float).mean() * 100), 3)
                     if (s6 >= THRESH).any() else None,
                     "sp400_pass_rate": round(float((s4 >= THRESH).mean()), 3),
                     "sp400_pool_avg_car": round(float(ts4m.loc[s4 >= THRESH, "car_10d"].astype(float).mean() * 100), 3)
                     if (s4 >= THRESH).any() else None})
    ts6["score_t"] = clf_t.predict_proba(ts6[FEATS])[:, 1]
    ts6["score_m"] = clf_m.predict_proba(ts6[FEATS])[:, 1]
    ts6["fold"] = fi
    if fi == 1:
        sp6 = ts6
    else:
        sp6 = pd.concat([sp6, ts6])
    log(f"fold {fi} done ({time.time()-t0:.0f}s)")

agg = pd.DataFrame(rows).groupby("model").mean(numeric_only=True).round(4)
log("PER-FOLD MEANS:\n" + agg.to_string())

# spearman on pooled SP600 OOS rows
sp = {}
for nm, c in (("transfer", "score_t"), ("mixed", "score_m")):
    d = sp6[[c, "car_10d"]].dropna()
    sp[nm] = round(float(pd.Series(d[c]).corr(pd.Series(d["car_10d"]), method="spearman")), 4)
log(f"spearman(score, car_10d) on SP600 OOS rows: {sp}")

# identical SP600-only weekly selection for both models
def weekly_select(col):
    raw = sp6[(sp6[col] >= THRESH) & (~sp6["sector"].isin(bt.EXCLUDE_SECTORS)) & sp6["pregap_return"].notna()].copy()
    raw["p"] = raw[col]
    raw["entry_date"] = pd.to_datetime(raw["entry_date"])
    raw["exit_date"] = pd.to_datetime(raw["exit_date"])
    pieces = [bt.select_weekly(raw[raw["fold"] == f], bt.N_SLOTS) for f in range(1, 5)
              if not raw[raw["fold"] == f].empty]
    ex = pd.concat(pieces, ignore_index=True)
    return raw, ex


raw_t, ex_t = weekly_select("score_t")
raw_m, ex_m = weekly_select("score_m")
log(f"SP600-only weekly selection: transfer {len(ex_t)} trades avg {ex_t['pregap_return'].astype(float).mean()*100:+.3f}% "
    f"| mixed {len(ex_m)} trades avg {ex_m['pregap_return'].astype(float).mean()*100:+.3f}%")
ov_raw = len(set(zip(raw_t.permaTicker, raw_t.report_date)) & set(zip(raw_m.permaTicker, raw_m.report_date)))
log(f"raw >=0.33 pool overlap: {ov_raw} of transfer {len(raw_t)} / mixed {len(raw_m)}")

out = {"per_fold": rows, "means": agg.to_dict(), "spearman_sp600": sp,
       "sp600_only_selection": {"transfer": {"n": int(len(ex_t)), "avg_pct": round(float(ex_t['pregap_return'].astype(float).mean()*100), 3)},
                                 "mixed": {"n": int(len(ex_m)), "avg_pct": round(float(ex_m['pregap_return'].astype(float).mean()*100), 3)}},
       "pool_overlap": {"n": int(ov_raw), "transfer_pool": int(len(raw_t)), "mixed_pool": int(len(raw_m))}}
o = HERE / "archive" / "experiments" / "rc21_why"
o.mkdir(parents=True, exist_ok=True)
with open(o / "report.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
log(f"wrote {o / 'report.json'}")

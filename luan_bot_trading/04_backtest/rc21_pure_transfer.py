"""RC-21 — pure transfer: SP400 rc19 calibration applied UNCHANGED to SP600.

Registered 9f14a85. Per fold: train on SP400 rows (label_end <= sve, v6c
common keys, RC-19 feature values), apply to SP600 rows in (sve, tse]
(RC-20 declared pipeline features). Gate T1 unamendable. 10-event
truncation spot-check on SP600 rows; mismatch aborts.
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

OUT = HERE / "archive" / "experiments" / "rc21"
OUT.mkdir(parents=True, exist_ok=True)
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
N_BOOT, SEED, THRESH = 10_000, 20260807, 0.33
HP1 = {"gamma": 8, "min_child_weight": 20, "max_depth": 3}
TOPK = 10


def log(*a):
    print(*a, flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = load("bt_rc21", HERE / "51_hp_theta_sweep_23feat.py")
p0 = load("rc18_p0_mod_21", HERE / "rc18_p0_firewall.py")
DB = bt.DB
FEATURES22 = [f for f in bt.DEPLOY_FEATURES if f != "car_drift_historical_q1"]
NEW = ["F1_sb_h3", "A2"]

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
log(f"prices {len(prices_all):,} ({time.time()-t0:.0f}s)")


def build_embedding(beat_frame):
    Tb = pd.to_datetime(beat_frame["T"]).dt.isocalendar()
    wkb = (Tb.year.astype(str) + "-W" + Tb.week.astype(str).str.zfill(2)).to_numpy()
    ptb = beat_frame["permaTicker"].to_numpy()
    r1b = beat_frame["r1"].to_numpy()
    matb = beat_frame["mat1"].to_numpy()
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


def compute_A2(evD):
    pos_, cut_, pt_ = evD["pos"], evD["cut_date"], evD["permaTicker"]
    r1_, r11_, mat11_ = evD["r1"], evD["r11"], evD["mat11"]
    A = np.full(len(evD), np.nan)
    tg = {}
    for pt, sub in evD.groupby("permaTicker", sort=False):
        s = sub.sort_values("pos")
        tg[pt] = (s["pos"].to_numpy(), s.index.to_numpy())
    pta = pt_.to_numpy()
    posa = pos_.to_numpy()
    cuta = cut_.to_numpy()
    r1a, r11a, m11a = r1_.to_numpy(), r11_.to_numpy(), mat11_.to_numpy()
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
    posa = evD["pos"].to_numpy()
    cuta = evD["cut_date"].to_numpy()
    pta = evD["permaTicker"].to_numpy()
    beat = evD["beatflag"].to_numpy()
    r3 = evD["r3"].to_numpy()
    mat3 = evD["mat3"].to_numpy()
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


# ---------------- TRAIN SIDE: RC-19 SP400 pipeline ----------------
v6c = pd.read_hdf(DB, "/features/train_matrix_v6c")
v6c["report_date"] = pd.to_datetime(v6c["report_date"])
old = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
old["report_date"] = pd.to_datetime(old["report_date"])
ko = set(zip(old.permaTicker, old.report_date))
kc = set(zip(v6c.permaTicker, v6c.report_date))
v6c = v6c[v6c.set_index(["permaTicker", "report_date"]).index.isin(ko & kc)].copy()

ev4 = v6c[["permaTicker", "report_date", "T", "is_bmo", "sue_score"]].copy()
ev4["beatflag"] = ev4["sue_score"] > 0
ev4 = p0.compute_drifts(ev4.reset_index(drop=True).copy(), prices, cal, ijh_close, horizons=(1, 3, 11))
ev4 = prep_clocks(ev4)
A2_4 = compute_A2(ev4)
cps4 = build_embedding(ev4[(ev4["beatflag"].to_numpy()) & ev4["r1"].notna() & ev4["mat1"].notna()])
F1_4 = compute_F1(ev4, cps4)
train_df = v6c.reset_index(drop=True).join(pd.DataFrame({"F1_sb_h3": F1_4, "A2": A2_4}))
train_df["label_end"] = pd.to_datetime(train_df["label_end"])
log(f"TRAIN side: {len(train_df):,} | F1 nan {train_df.F1_sb_h3.isna().mean():.1%} ({time.time()-t0:.0f}s)")

# ---------------- APPLICATION SIDE: RC-20 pipeline, SP600 rows ----------------
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
rows_time, rows_beat = [], []
for rd, t_ in zip(ev6["report_date"], tkr):
    got = False
    if isinstance(t_, str) and t_ in earn:
        e = earn[t_]
        e = e[pd.to_datetime(e["report_date"]) == rd]
        if len(e):
            r0 = e.iloc[-1]
            rows_time.append(1.0 if str(r0["time"]).lower() == "bmo"
                             else (0.0 if str(r0["time"]).lower() == "amc" else np.nan))
            rows_beat.append(bool(np.isfinite(float(r0["actual"])) and np.isfinite(float(r0["estimate"]))
                                  and float(r0["actual"]) > float(r0["estimate"])))
            got = True
    if not got:
        rows_time.append(np.nan)
        rows_beat.append(False)
ev6["is_bmo"] = rows_time
ev6["beatflag"] = rows_beat

idx6 = ev6.index
ev6c = p0.compute_drifts(ev6.reset_index(drop=True).copy(), prices_all, cal6, close6, horizons=(1, 3, 11))
ev6c.index = idx6
ev6c = prep_clocks(ev6c)

# combined-universe embedding (RC-20 declared pipeline): SP600 app rows + SP400 train rows
ev6c["is_sp400"] = 0
ev4_c = ev4.copy()
ev4_c["is_sp400"] = 1
evC = pd.concat([ev4_c, ev6c], ignore_index=True)
comb_beats = evC[(evC["beatflag"].to_numpy()) & evC["r1"].notna() & evC["mat1"].notna()]
cpsC = build_embedding(comb_beats)
evCp = evC.reset_index(drop=True)
n4c = int((evCp["is_sp400"] == 1).sum())
A2_all = compute_A2(evCp)
F1_all = compute_F1(evCp, cpsC)
A2_6 = A2_all[n4c:]
F1_6 = F1_all[n4c:]
log(f"APP side: {len(ev6c):,} | F1 nan {pd.isna(F1_6).mean():.1%} | A2 nan {pd.isna(A2_6).mean():.1%} ({time.time()-t0:.0f}s)")

app_df = m[m["is_sp400"] == 0].copy()
app_df = app_df.join(pd.DataFrame({"F1_sb_h3": pd.Series(F1_6, index=idx6),
                                   "A2": pd.Series(A2_6, index=idx6)}))

# ---------------- truncation spot-check (10 SP600 rows, combined pool) ----------------
rng_tc = np.random.default_rng(SEED)
cand = np.where(np.isfinite(F1_6))[0]
sample = rng_tc.choice(cand, size=min(10, len(cand)), replace=False)
fails = 0
evCg = evCp
posg = evCg["pos"].to_numpy()
cutg = evCg["cut_date"].to_numpy()
ptg = evCg["permaTicker"].to_numpy()
beatg = evCg["beatflag"].to_numpy()
orderg = np.argsort(posg, kind="stable")
pos_sg = posg[orderg]
sp400g = evCg["is_sp400"].to_numpy() == 1
for i in sample:
    cut_i = cutg[n4c + int(i)]  # i is SP600-frame position
    ptr = {k: (d[d <= cut_i], c[d <= cut_i]) for k, (d, c) in prices_all.items()}
    pc = posg[n4c + int(i)]
    lo = bisect.bisect_left(pos_sg, pc - p0.K_WINDOW)
    hi = bisect.bisect_left(pos_sg, pc)
    w_all = orderg[lo:hi]
    mini = evCg.iloc[w_all]
    orig_idx = mini.index.to_numpy()
    m4 = p0.compute_drifts(mini[sp400g[w_all]].reset_index(drop=True).copy(), ptr, cal, ijh_close, horizons=(3,))
    m4.index = orig_idx[sp400g[w_all]]
    m6 = p0.compute_drifts(mini[~sp400g[w_all]].reset_index(drop=True).copy(), ptr, cal6, close6, horizons=(3,))
    m6.index = orig_idx[~sp400g[w_all]]
    mm_ = pd.concat([m4, m6]).reindex(orig_idx)
    mv3, rv3 = mm_["mat3"].to_numpy(), mm_["r3"].to_numpy()
    msk = (ptg[w_all] != ptg[n4c + int(i)]) & beatg[w_all]
    msk &= ~pd.isna(mv3) & (mv3 <= cut_i) & np.isfinite(rv3)
    w_idx = w_all[msk]
    val = np.nan
    if len(w_idx):
        k = bisect.bisect_right([c for c, _ in cpsC], cut_i) - 1
        if k >= 0:
            emb = cpsC[k][1]
            if ptg[n4c + int(i)] in emb.index:
                u = emb.loc[ptg[n4c + int(i)]].to_numpy()
                sims = (emb @ u).clip(lower=0.0)
                ws = np.array([float(sims.get(o, 0.0)) for o in ptg[w_idx]])
                if len(ws) > TOPK:
                    keep = np.argsort(-ws)[:TOPK]
                    w_idx, ws = w_idx[keep], ws[keep]
                rs = pd.Series(rv3, index=orig_idx).reindex(w_idx).to_numpy(dtype=float)
                mk = ws > 0
                if mk.any():
                    val = float((ws[mk] * rs[mk]).sum() / ws[mk].sum())
    if not np.isclose(val, F1_6[i], equal_nan=True):
        fails += 1
        log(f"  TRUNC MISMATCH {i}: full={F1_6[i]:.6f} trunc={val:.6f}")
log(f"truncation spot-check: {len(sample)} SP600 events, {fails} mismatch")
if fails:
    with open(OUT / "gauntlet.json", "w") as f:
        json.dump({"status": "ABORTED_TRUNCATION_FAIL", "fails": fails}, f)
    sys.exit(2)

# ---------------- fold transfer ----------------
def fit_gate(X, y, Xe, ye):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=300, learning_rate=0.05,
        max_depth=HP1["max_depth"], min_child_weight=HP1["min_child_weight"],
        gamma=HP1["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1,
    ).fit(X, y, eval_set=[(Xe, ye)], verbose=False)


FEATS = FEATURES22 + NEW
preds = []
sanity = []
for fi, (te, sve, tse) in enumerate(bt.DEFAULT_FOLDS, 1):
    tr = train_df[train_df["label_end"] <= pd.Timestamp(sve)]
    ts6 = app_df[(app_df["label_end"] > pd.Timestamp(sve)) & (app_df["label_end"] <= pd.Timestamp(tse))].copy()
    ts4 = train_df[(train_df["label_end"] > pd.Timestamp(sve)) & (train_df["label_end"] <= pd.Timestamp(tse))].copy()
    clf = fit_gate(tr[FEATS], tr["pass_g1"].astype(int).to_numpy(), ts6[FEATS], ts6["pass_g1"].astype(int).to_numpy())
    ts6["score"] = clf.predict_proba(ts6[FEATS])[:, 1]
    ts6["fold"] = fi
    preds.append(ts6)
    ts4["score"] = clf.predict_proba(ts4[FEATS])[:, 1]
    ts4["fold"] = fi
    sanity.append(ts4)
    log(f"fold {fi}: train {len(tr):,} | app {len(ts6):,} | picks(>=.33) {(ts6['score']>=THRESH).sum()} ({time.time()-t0:.0f}s)")

pred = pd.concat(preds, ignore_index=True)
mask = ((pred["score"] >= THRESH) & (~pred["sector"].isin(bt.EXCLUDE_SECTORS))
        & pred["pregap_return"].notna())
raw = pred[mask].copy()
raw["p"] = raw["score"]
raw["entry_date"] = pd.to_datetime(raw["entry_date"])
raw["exit_date"] = pd.to_datetime(raw["exit_date"])
pieces = [bt.select_weekly(raw[raw["fold"] == fi], bt.N_SLOTS) for fi in range(1, 5)
          if not raw[raw["fold"] == fi].empty]
ex = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def weekly(exd):
    z = exd.copy()
    z["entry_date"] = pd.to_datetime(z["entry_date"])
    iso = z["entry_date"].dt.isocalendar()
    z["wk"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    return z.groupby("wk", sort=True)["pregap_return"].sum() / bt.N_SLOTS


def stats(exd):
    if exd is None or len(exd) == 0:
        return {"executed": 0}
    r = exd["pregap_return"].astype(float)
    w = weekly(exd)
    return {"executed": int(len(r)), "win_rate_pct": round(float((r > 0).mean() * 100), 1),
            "avg_trade_pct": round(float(r.mean() * 100), 3), "weeks": int(len(w)),
            "nav_pct": round(float(((1 + w).prod() - 1) * 100), 2),
            "raw_precision": round(float(exd["pead_pass"].mean() * 100), 1),
            "g1_rate": round(float(exd["pass_g1"].mean() * 100), 1)}


rng = np.random.default_rng(SEED)
w = weekly(ex)
v = w.to_numpy(dtype=float)
v = v[np.isfinite(v)]
idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
mm = v[idx].mean(axis=1)
boot = {"mean_pct": round(float(v.mean() * 100), 4),
        "ci95_pct": [round(float(np.percentile(mm, 2.5)) * 100, 4),
                     round(float(np.percentile(mm, 97.5)) * 100, 4)]}
s_t = stats(ex)
T1 = bool(s_t.get("avg_trade_pct", 0) > 0 and boot["ci95_pct"][0] > 0)

san = pd.concat(sanity, ignore_index=True)
san_m = san[(san["score"] >= THRESH) & (~san["sector"].isin(bt.EXCLUDE_SECTORS)) & san["pregap_return"].notna()]
san_stats = {"picked": int(len(san_m)), "avg_trade_pct": round(float(san_m["pregap_return"].astype(float).mean() * 100), 3)}

log(f"TRANSFER: {json.dumps(s_t)}")
log(f"  CI: {json.dumps(boot)}")
log(f"  sanity (SP400 test slices, top-picks pool): {json.dumps(san_stats)}")
log(f"T1: {'PASS' if T1 else 'FAIL'}")

with open(OUT / "gauntlet.json", "w") as f:
    json.dump({"transfer": s_t, "boot": boot, "T1_pass": T1,
               "sanity_sp400": san_stats,
               "truncation_spotcheck": {"n": int(len(sample)), "fails": int(fails)},
               "config": {"n_boot": N_BOOT, "seed": SEED, "thresh": THRESH},
               "seconds": round(time.time() - t0)}, f, indent=2, default=str)
if len(ex):
    keep = [c for c in ("permaTicker", "fold", "entry_date", "exit_date", "pregap_return",
                        "score", "pead_pass", "pass_g1") if c in ex.columns]
    ex[keep].to_hdf(OUT / "executed.h5", key="transfer", format="table")
log(f"wrote {OUT / 'gauntlet.json'} ({round(time.time() - t0)}s)")

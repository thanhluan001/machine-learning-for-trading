"""Feature-set audit (back to basics): does ANY feature carry signal?

For every feature in the deploy set (22 retired + F1_sb_h3 + A2), measured
against (a) pass_g1 = car_10d > +3%  [the label] and (b) the 5-session
harvest pregap_return  [what a trade actually earns]:

  - coverage
  - univariate AUC (binary targets)
  - spearman with the continuous targets
  - top-minus-bottom quintile spread on pregap_return, with week-block
    bootstrap CI (10k, seed 20260807) — respects cross-sectional correlation
  - same spread in the first vs second half of the sample (era stability)
  - Benjamini-Hochberg FDR across the family of tests

Plus: feature-feature Spearman redundancy, and XGBoost gain importance vs
univariate signal (does the model rely on signal-less features?).

No model selection here — pure measurement. Full sample; OOS test windows
reported separately for comparison.
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

OUT = HERE / "archive" / "experiments" / "feature_audit"
OUT.mkdir(parents=True, exist_ok=True)
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
N_BOOT, SEED = 10_000, 20260807
TOPK = 10


def log(*a):
    print(*a, flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = load("bt_audit", HERE / "51_hp_theta_sweep_23feat.py")
p0 = load("p0_audit", HERE / "rc18_p0_firewall.py")
DB = bt.DB
FEATURES22 = [f for f in bt.DEPLOY_FEATURES if f != "car_drift_historical_q1"]

t0 = time.time()
# ---------------- recompute F1_sb_h3 / A2 on the combined universe ----------------
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

v7 = pd.read_hdf(DB, "/features/train_matrix_v7c")
v7["report_date"] = pd.to_datetime(v7["report_date"])
v7["label_end"] = pd.to_datetime(v7["label_end"])

ev6 = v7[v7["is_sp400"] == 0][["permaTicker", "report_date", "is_sp400"]].copy()
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

v6 = v7[v7["is_sp400"] == 1].copy()
v6c = pd.read_hdf(DB, "/features/train_matrix_v6c")
v6c["report_date"] = pd.to_datetime(v6c["report_date"])
j = v6c.set_index(["permaTicker", "report_date"])[["T", "is_bmo", "sue_score"]]
sel4 = v6["is_sp400"] == 1
v6 = v6.join(j, on=["permaTicker", "report_date"])
ev4 = v6[["permaTicker", "report_date", "T", "is_bmo", "sue_score"]].copy()
ev4["beatflag"] = pd.to_numeric(ev4["sue_score"], errors="coerce") > 0
ev4 = p0.compute_drifts(ev4.reset_index(drop=True).copy(), prices, cal, ijh_close, horizons=(1, 3, 11))
ev4.index = v6.index
ev4["is_sp400"] = 1
ev6c["is_sp400"] = 0


def prep_clocks(evD):
    pos = np.searchsorted(cal, evD["T"].to_numpy().astype("datetime64[D]"), side="left")
    evD["pos"] = pos
    back = np.where(pd.isna(evD["is_bmo"]), -1, np.where(evD["is_bmo"] == 1, 2, 1))
    cut = cal[np.clip(pos - back, 0, len(cal) - 1)]
    cut[pd.isna(evD["is_bmo"])] = np.datetime64("NaT", "D")
    evD["cut_date"] = cut
    return evD


ev4 = prep_clocks(ev4)
ev6c = prep_clocks(ev6c)
evC = pd.concat([ev4, ev6c], ignore_index=True)


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


cpsC = build_embedding(evC[(evC["beatflag"].to_numpy()) & evC["r1"].notna() & evC["mat1"].notna()])


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


F1 = compute_F1(evC, cpsC)
A2 = compute_A2(evC)
kC = pd.MultiIndex.from_arrays([evC["permaTicker"], evC["report_date"]])
v7["F1_sb_h3"] = pd.Series(F1, index=kC).reindex(
    pd.MultiIndex.from_arrays([v7["permaTicker"], v7["report_date"]])).to_numpy()
v7["A2"] = pd.Series(A2, index=kC).reindex(
    pd.MultiIndex.from_arrays([v7["permaTicker"], v7["report_date"]])).to_numpy()
log(f"features ready ({time.time()-t0:.0f}s) | F1 nan {v7.F1_sb_h3.isna().mean():.1%} | A2 nan {v7.A2.isna().mean():.1%}")

# ---------------- audit ----------------
FEATS = [f for f in FEATURES22 if f in v7.columns] + ["F1_sb_h3", "A2"]
for extra in ("adv20",):
    if extra in v7.columns:
        FEATS.append(extra)
d = v7.copy()
d["pregap_return"] = pd.to_numeric(d["pregap_return"], errors="coerce")
d["car_10d"] = pd.to_numeric(d["car_10d"], errors="coerce")
d["pass_g1"] = pd.to_numeric(d["pass_g1"], errors="coerce")
d = d[~d["sector"].isin(bt.EXCLUDE_SECTORS)]
iso = d["report_date"].dt.isocalendar()
d["wk"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
med = d["report_date"].median()
d["era"] = np.where(d["report_date"] <= med, "first_half", "second_half")
oos = np.zeros(len(d), dtype=bool)
for te, sve, tse in bt.DEFAULT_FOLDS:
    oos |= (d["label_end"] > pd.Timestamp(sve)) & (d["label_end"] <= pd.Timestamp(tse))
d["oos"] = oos
log(f"audit sample {len(d):,} (oos {int(oos.sum()):,})")

from sklearn.metrics import roc_auc_score


def spread_boot(sub, feat, target, n_boot=N_BOOT, seed=SEED):
    s = sub[[feat, target, "wk"]].dropna()
    if len(s) < 200:
        return None
    s = s.assign(q=pd.qcut(s[feat].rank(method="first"), 5, labels=False))
    top, bot = s[s.q == 4], s[s.q == 0]
    obs = float(top[target].mean() - bot[target].mean())
    tw = top.groupby("wk")[target].agg(["sum", "count"])
    bw = bot.groupby("wk")[target].agg(["sum", "count"])
    wks = tw.index.union(bw.index)
    ts = tw.reindex(wks).fillna(0.0)
    bs = bw.reindex(wks).fillna(0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wks), size=(n_boot, len(wks)))
    tm = ts["sum"].to_numpy()[idx].sum(1) / np.maximum(ts["count"].to_numpy()[idx].sum(1), 1)
    bm = bs["sum"].to_numpy()[idx].sum(1) / np.maximum(bs["count"].to_numpy()[idx].sum(1), 1)
    dd = tm - bm
    se = float(dd.std(ddof=1))
    return {"spread_pct": round(obs * 100, 3),
            "ci95_pct": [round(float(np.percentile(dd, 2.5)) * 100, 3),
                         round(float(np.percentile(dd, 97.5)) * 100, 3)],
            "se_pct": round(se * 100, 3),
            "t": round(obs / se, 2) if se > 0 else None,
            "n": int(len(s))}


rows = []
for f in FEATS:
    if f not in d.columns:
        continue
    x = pd.to_numeric(d[f], errors="coerce")
    cov = float(x.notna().mean())
    r = {"feature": f, "coverage": round(cov, 3)}
    for tgt, tname in (("pass_g1", "g1"), ("pregap_return", "ret")):
        s = pd.DataFrame({"x": x, "y": pd.to_numeric(d[tgt], errors="coerce")}).dropna()
        if len(s) > 200 and s["y"].nunique() > 1:
            if tname == "g1":
                r["auc_g1"] = round(float(roc_auc_score(s["y"].astype(int), s["x"])), 4)
            else:
                r["auc_ret_pos"] = round(float(roc_auc_score((s["y"] > 0).astype(int), s["x"])), 4)
            r[f"spearman_{tname}"] = round(float(s["x"].corr(s["y"], method="spearman")), 4)
    sb = spread_boot(d, f, "pregap_return")
    if sb:
        r["spread_ret_pct"] = sb["spread_pct"]
        r["spread_ret_ci"] = sb["ci95_pct"]
        r["spread_ret_t"] = sb["t"]
    sc = spread_boot(d, f, "car_10d")
    if sc:
        r["spread_car_pct"] = sc["spread_pct"]
        r["spread_car_t"] = sc["t"]
    for era in ("first_half", "second_half"):
        se_ = spread_boot(d[d["era"] == era], f, "pregap_return")
        if se_:
            r[f"spread_{era}"] = se_["spread_pct"]
    rows.append(r)
    log(f"  {f:38} cov={cov:.2f} auc_g1={r.get('auc_g1')} auc_ret+={r.get('auc_ret_pos')} "
        f"spread={r.get('spread_ret_pct')}% t={r.get('spread_ret_t')}")

aud = pd.DataFrame(rows)

# FDR (BH) across the pregap-spread t-stats (two-sided)
from scipy import stats
tt = aud["spread_ret_t"].astype(float)
p = pd.Series(2 * (1 - stats.norm.cdf(np.abs(tt.to_numpy(dtype=float)))), index=aud.index)
order = np.argsort(p.fillna(1.0))
m = int(p.notna().sum())
q = np.full(len(p), np.nan)
prev = 1.0
for rank, oi in enumerate(order[::-1], start=1):
    k = m - rank + 1
    val = min(prev, p.iloc[oi] * m / k)
    q[oi] = val
    prev = val
aud["q_bh"] = q
aud = aud.sort_values("spread_ret_t", key=lambda s: s.abs(), ascending=False)
log("\n=== FEATURE AUDIT (sorted by |t| of 5-session quintile spread) ===")
log(aud[["feature", "coverage", "auc_g1", "auc_ret_pos", "spearman_g1", "spearman_ret",
         "spread_ret_pct", "spread_ret_t", "spread_car_pct", "spread_first_half",
         "spread_second_half", "q_bh"]].to_string(index=False))

# redundancy
corr = d[FEATS].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
pairs = []
for i, a in enumerate(FEATS):
    for b in FEATS[i + 1:]:
        if abs(corr.loc[a, b]) > 0.7:
            pairs.append((a, b, round(float(corr.loc[a, b]), 3)))
log(f"\nredundant pairs |rho|>0.7: {len(pairs)}")
for a, b, c in pairs:
    log(f"  {a} ~ {b}: {c}")

# model gain importance vs univariate signal
import xgboost as xgb
tr = d[d["label_end"] <= pd.Timestamp(bt.DEFAULT_FOLDS[-1][1])]
clf = xgb.XGBClassifier(objective="binary:logistic", n_estimators=300, learning_rate=0.05,
                        max_depth=3, min_child_weight=20, gamma=8, reg_lambda=1.0,
                        subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=-1)
clf.fit(tr[FEATS], tr["pass_g1"].astype(int))
gain = pd.Series(clf.feature_importances_, index=FEATS).sort_values(ascending=False)
log("\n=== XGBoost gain importance (24 features, pass_g1) ===")
for f, v in gain.items():
    a = aud[aud.feature == f]
    log(f"  {f:38} gain={v:.3f}  auc_g1={a.auc_g1.iloc[0] if len(a) else None}  "
        f"spread_t={a.spread_ret_t.iloc[0] if len(a) else None}")

with open(OUT / "audit.json", "w") as fh:
    json.dump({"features": aud.to_dict("records"), "redundant_pairs": pairs,
               "gain": gain.round(4).to_dict(), "n": int(len(d)), "oos_n": int(oos.sum()),
               "seconds": round(time.time() - t0)}, fh, indent=2, default=str)
aud.to_csv(OUT / "audit.csv", index=False)
log(f"\nwrote {OUT}/audit.json ({round(time.time()-t0)}s)")

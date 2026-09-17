"""RC-22 — simulator alignment. Re-runs v6n/rc19/v7n/rc20/rc21 under BOTH
selectors from identical fitted scores. Registered 6be0476; gates unchanged.
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

OUT = HERE / "archive" / "experiments" / "rc22"
OUT.mkdir(parents=True, exist_ok=True)
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
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


bt = load("bt_rc22", HERE / "51_hp_theta_sweep_23feat.py")
p0 = load("p0_rc22", HERE / "rc18_p0_firewall.py")
DB = bt.DB
FEATURES22 = [f for f in bt.DEPLOY_FEATURES if f != "car_drift_historical_q1"]
FEATS24 = FEATURES22 + ["F1_sb_h3", "A2"]

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


# ---------------- frames ----------------
v6c = pd.read_hdf(DB, "/features/train_matrix_v6c")
v6c["report_date"] = pd.to_datetime(v6c["report_date"])
v6c["label_end"] = pd.to_datetime(v6c["label_end"])
old = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
old["report_date"] = pd.to_datetime(old["report_date"])
ko = set(zip(old.permaTicker, old.report_date))
common_mask = v6c.set_index(["permaTicker", "report_date"]).index.isin(ko)

ev4 = v6c[["permaTicker", "report_date", "T", "is_bmo", "sue_score"]].copy()
ev4["beatflag"] = ev4["sue_score"] > 0
ev4 = p0.compute_drifts(ev4.reset_index(drop=True).copy(), prices, cal, ijh_close, horizons=(1, 3, 11))
ev4 = prep_clocks(ev4)
cps4 = build_embedding(ev4[(ev4["beatflag"].to_numpy()) & ev4["r1"].notna() & ev4["mat1"].notna()])
F1_4 = compute_F1(ev4, cps4)
A2_4 = compute_A2(ev4)
log(f"SP400 events {len(ev4):,} | F1 nan {pd.isna(F1_4).mean():.1%} ({time.time()-t0:.0f}s)")

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
ev6c = prep_clocks(ev6c)
ev6c["is_sp400"] = 0
ev4c = ev4.copy()
ev4c["is_sp400"] = 1
evC = pd.concat([ev4c, ev6c], ignore_index=True)
cpsC = build_embedding(evC[(evC["beatflag"].to_numpy()) & evC["r1"].notna() & evC["mat1"].notna()])
evCp = evC.reset_index(drop=True)
n4c = int((evCp["is_sp400"] == 1).sum())
F1_C = compute_F1(evCp, cpsC)
A2_C = compute_A2(evCp)
log(f"combined events {len(evCp):,} | F1 nan {pd.isna(F1_C).mean():.1%} ({time.time()-t0:.0f}s)")

# key-indexed feature series
k4 = pd.MultiIndex.from_arrays([ev4["permaTicker"], ev4["report_date"]])
ser_f1_4 = pd.Series(F1_4, index=k4)
ser_a2_4 = pd.Series(A2_4, index=k4)
kC = pd.MultiIndex.from_arrays([evCp["permaTicker"], evCp["report_date"]])
ser_f1_C = pd.Series(F1_C, index=kC)
ser_a2_C = pd.Series(A2_C, index=kC)

sp400 = v6c[common_mask].copy()
sp400 = sp400.join(pd.DataFrame({"F1_sb_h3": ser_f1_4.reindex(pd.MultiIndex.from_arrays([sp400.permaTicker, sp400.report_date])).to_numpy(),
                                 "A2": ser_a2_4.reindex(pd.MultiIndex.from_arrays([sp400.permaTicker, sp400.report_date])).to_numpy()},
                                index=sp400.index))
v7f = v7.copy()
v7f = v7f.join(pd.DataFrame({"F1_sb_h3": ser_f1_C.reindex(pd.MultiIndex.from_arrays([v7f.permaTicker, v7f.report_date])).to_numpy(),
                             "A2": ser_a2_C.reindex(pd.MultiIndex.from_arrays([v7f.permaTicker, v7f.report_date])).to_numpy()},
                            index=v7f.index))
log(f"frames: sp400 {len(sp400):,} | v7f {len(v7f):,} ({time.time()-t0:.0f}s)")


# ---------------- selectors ----------------
def select_weekly_old(picks, n_slots=4, sort_col="p"):
    if picks.empty:
        return picks
    pk = picks.copy()
    pk["entry_date"] = pd.to_datetime(pk["entry_date"])
    pk["exit_date"] = pd.to_datetime(pk["exit_date"])
    iso = pk["entry_date"].dt.isocalendar()
    pk["_week_key"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    sel = []
    active = []
    for wk, wdf in pk.groupby("_week_key", sort=True):
        ws = wdf["entry_date"].min()
        active = [ex for ex in active if ex >= ws]
        free = n_slots - len(active)
        if free <= 0:
            continue
        taken = wdf.sort_values(sort_col, ascending=False).head(min(free, len(wdf)))
        sel.append(taken)
        for _, r in taken.iterrows():
            active.append(r["exit_date"])
    return pd.concat(sel).sort_values("entry_date").reset_index(drop=True) if sel else pd.DataFrame()


def select_continuous(picks, n_slots=4, sort_col="p"):
    if picks.empty:
        return picks
    pk = picks.copy()
    pk["entry_date"] = pd.to_datetime(pk["entry_date"])
    pk["exit_date"] = pd.to_datetime(pk["exit_date"])
    sel = []
    active_exits = []
    for d, ddf in pk.groupby("entry_date", sort=True):
        active_exits = [x for x in active_exits if x > d]
        free = n_slots - len(active_exits)
        if free <= 0:
            continue
        taken = ddf.sort_values(sort_col, ascending=False).head(min(free, len(ddf)))
        sel.append(taken)
        active_exits.extend(pd.to_datetime(taken["exit_date"]).tolist())
    return pd.concat(sel).sort_values("entry_date").reset_index(drop=True) if sel else pd.DataFrame()


def fit_gate(X, y, Xe, ye, hp):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=300, learning_rate=0.05, max_depth=hp["max_depth"],
        min_child_weight=hp["min_child_weight"], gamma=hp["gamma"],
        reg_lambda=1.0, subsample=0.7, colsample_bytree=0.7,
        random_state=42, n_jobs=-1,
    ).fit(X, y, eval_set=[(Xe, ye)], verbose=False)


def run_arm(df, feats, gates_list, hp_map, score="min", apply_df=None):
    rd = df["label_end"]
    out = []
    for fi, (te, sve, tse) in enumerate(bt.DEFAULT_FOLDS, 1):
        lo, hi = pd.Timestamp(sve), pd.Timestamp(tse)
        tr = df[rd <= lo]
        ts = (apply_df if apply_df is not None else df)
        ts = ts[(ts["label_end"] > lo) & (ts["label_end"] <= hi)].copy()
        for i, g in enumerate(gates_list, 1):
            clf = fit_gate(tr[feats], tr[g].astype(int).to_numpy(),
                           ts[feats], ts[g].astype(int).to_numpy(), hp_map[g])
            ts[f"p{i}"] = clf.predict_proba(ts[feats])[:, 1]
        ts["score"] = ts["p1"] if score == "single" else ts[[f"p{i}" for i in range(1, len(gates_list) + 1)]].min(axis=1)
        ts["fold"] = fi
        out.append(ts)
    return pd.concat(out, ignore_index=True)


def candidates(pred):
    msk = ((pred["score"] >= THRESH) & (~pred["sector"].isin(bt.EXCLUDE_SECTORS))
           & pred["pregap_return"].notna())
    raw = pred[msk].copy()
    raw["p"] = raw["score"]
    ec = "entry_date" if "entry_date" in raw.columns else "pregap_entry_date"
    xc = "exit_date" if "exit_date" in raw.columns else "pregap_exit_date"
    raw["entry_date"] = pd.to_datetime(raw[ec])
    raw["exit_date"] = pd.to_datetime(raw[xc])
    return raw


def weekly(ex):
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
            "avg_trade_pct": round(float(r.mean() * 100), 3), "weeks": int(len(w)),
            "nav_pct": round(float(((1 + w).prod() - 1) * 100), 2),
            "raw_precision": round(float(ex["pead_pass"].mean() * 100), 1)}


def boot_mean(w):
    rng = np.random.default_rng(SEED)
    v = w.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    mm = v[idx].mean(axis=1)
    return {"mean_pct": round(float(v.mean() * 100), 4),
            "ci95_pct": [round(float(np.percentile(mm, 2.5)) * 100, 4),
                         round(float(np.percentile(mm, 97.5)) * 100, 4)]}


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


# ---------------- arms ----------------
arms = {
    "v6n": run_arm(sp400, FEATURES22, GATES3, V6_HP, "min"),
    "rc19": run_arm(sp400, FEATS24, ["pass_g1"], V6_HP, "single"),
    "v7n": run_arm(v7f, FEATURES22, GATES3, V6_HP, "min"),
    "rc20": run_arm(v7f, FEATS24, ["pass_g1"], V6_HP, "single"),
    "rc21": run_arm(sp400, FEATS24, ["pass_g1"], V6_HP, "single",
                    apply_df=v7f[v7f["is_sp400"] == 0].copy()),
}
log(f"arms fitted ({time.time()-t0:.0f}s)")

res = {}
for nm, pred in arms.items():
    raw = candidates(pred)
    for sel_nm, sel_fn in (("old", select_weekly_old), ("new", select_continuous)):
        pieces = [sel_fn(raw[raw["fold"] == fi], bt.N_SLOTS) for fi in range(1, 5)
                  if not raw[raw["fold"] == fi].empty]
        ex = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        res.setdefault(nm, {})[sel_nm] = {"stats": stats(ex), "boot": boot_mean(weekly(ex)),
                                          "w": weekly(ex), "ex": ex}
    log(f"{nm}: OLD {json.dumps(res[nm]['old']['stats'])} | NEW {json.dumps(res[nm]['new']['stats'])}")

gates = {}
for nm, base in (("rc19", "v6n"), ("rc20", "v7n"), ("rc21", None)):
    for sel_nm in ("old", "new"):
        s = res[nm][sel_nm]["stats"]
        ci = res[nm][sel_nm]["boot"]["ci95_pct"]
        g1 = bool(s.get("avg_trade_pct", 0) > 0 and ci[0] > 0)
        d = {"G1_or_T1": g1, "boot": res[nm][sel_nm]["boot"]}
        if base:
            p = paired(res[nm][sel_nm]["w"], res[base][sel_nm]["w"])
            d["G2_beats_" + base] = bool(s.get("avg_trade_pct", 0) > res[base][sel_nm]["stats"].get("avg_trade_pct", -9e9)
                                         and p["ci95_pct"][0] > 0)
            d["paired"] = p
        gates[f"{nm}_{sel_nm}"] = d
log("GATES: " + json.dumps(gates, indent=1, default=str))

with open(OUT / "report.json", "w") as f:
    json.dump({"arms": {nm: {s: {"stats": v[s]["stats"], "boot": v[s]["boot"]} for s in ("old", "new")}
                        for nm, v in res.items()},
               "gates": gates, "config": {"n_boot": N_BOOT, "seed": SEED, "thresh": THRESH},
               "seconds": round(time.time() - t0)}, f, indent=2, default=str)
with pd.HDFStore(OUT / "executed.h5", "w") as st:
    for nm, v in res.items():
        for s in ("old", "new"):
            ex = v[s]["ex"]
            if len(ex):
                keep = [c for c in ("permaTicker", "fold", "entry_date", "exit_date", "pregap_return",
                                    "score", "pead_pass", "pass_g1") if c in ex.columns]
                st.put(f"/{nm}_{s}", ex[keep], format="table")
log(f"wrote {OUT / 'report.json'} ({round(time.time()-t0)}s)")

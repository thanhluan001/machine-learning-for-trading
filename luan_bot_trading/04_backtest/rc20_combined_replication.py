"""RC-20 — combined-universe replication of the frozen RC-19 stack.

Registered spec (ca93391). Arms: rc20 (24 feats, single pass_g1, thr 0.33,
frozen g1 HPs) vs v7n (22 feats, 3-gate min-gate), both on FULL
/features/train_matrix_v7c. SP600 feature-side deviations declared in the
registration: T:=report_date, is_bmo from earnings_full time, per-universe
beat flags, IJR drift bench for SP600. 30-event truncation spot-check
(prices truncated at consumer cutoff; F1 must reproduce); abort on any
mismatch. Gates G1/G2 unamendable.
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

OUT = HERE / "archive" / "experiments" / "rc20"
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


bt = load("bt_rc20", HERE / "51_hp_theta_sweep_23feat.py")
p0 = load("rc18_p0_mod_20", HERE / "rc18_p0_firewall.py")
DB = bt.DB
FEATURES22 = [f for f in bt.DEPLOY_FEATURES if f != "car_drift_historical_q1"]
NEW = ["F1_sb_h3", "A2"]

t0 = time.time()
# ---------------- event frame from v7c ----------------
m = pd.read_hdf(DB, "/features/train_matrix_v7c")
m["report_date"] = pd.to_datetime(m["report_date"])
m["label_end"] = pd.to_datetime(m["label_end"])
ev = m[["permaTicker", "report_date", "is_sp400"]].copy()
ev["T"] = ev["report_date"]
ev["is_bmo"] = np.nan

v6c = pd.read_hdf(DB, "/features/train_matrix_v6c")
v6c["report_date"] = pd.to_datetime(v6c["report_date"])
j400 = v6c.set_index(["permaTicker", "report_date"])[["T", "is_bmo", "sue_score"]]
sel4 = ev["is_sp400"] == 1
ev.loc[sel4, ["T", "is_bmo", "sue_score"]] = (
    j400.reindex(pd.MultiIndex.from_arrays([ev.loc[sel4, "permaTicker"],
                                            ev.loc[sel4, "report_date"]])).to_numpy())
log(f"sp400 rows {int(sel4.sum()):,} | T join miss "
    f"{pd.isna(ev.loc[sel4,'T']).sum()}")

# SP600: ticker -> earnings_full(time, actual, estimate)
with pd.HDFStore(DB_SP600, "r") as st:
    s6 = st["/features/train_matrix_sp600_pt_v7c"]
    s6["report_date"] = pd.to_datetime(s6["report_date"])
    tk = s6.set_index(["permaTicker", "report_date"])["ticker"]
    earn = {}
    for k in st.keys():
        if k.startswith("/sp600/earnings_full/"):
            earn[k.split("/")[-1]] = st[k]
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

sel6 = ~sel4
tkr = tk.reindex(pd.MultiIndex.from_arrays([ev.loc[sel6, "permaTicker"],
                                            ev.loc[sel6, "report_date"]])).to_numpy()
rows_time, rows_beat = [], []
for pt, rd, t_ in zip(ev.loc[sel6, "permaTicker"], ev.loc[sel6, "report_date"], tkr):
    if isinstance(t_, str) and t_ in earn:
        e = earn[t_]
        e = e[pd.to_datetime(e["report_date"]) == rd]
        if len(e):
            r0 = e.iloc[-1]
            tm = 1.0 if str(r0["time"]).lower() == "bmo" else (0.0 if str(r0["time"]).lower() == "amc" else np.nan)
            bt_ = (np.isfinite(float(r0["actual"])) and np.isfinite(float(r0["estimate"]))
                   and float(r0["actual"]) > float(r0["estimate"]))
            rows_time.append(tm)
            rows_beat.append(bt_)
            continue
    rows_time.append(np.nan)
    rows_beat.append(False)
ev.loc[sel6, "is_bmo"] = rows_time
ev.loc[sel6, "sue_score"] = np.where(rows_beat, 1.0, np.where(pd.isna(rows_time), np.nan, -1.0))
log(f"sp600 rows {int(sel6.sum()):,} | time known {np.isfinite(ev.loc[sel6,'is_bmo']).mean():.1%} "
    f"| beat rate {(ev.loc[sel6,'sue_score']>0).mean():.1%} ({time.time()-t0:.0f}s)")

ev["beatflag"] = np.where(ev["sue_score"] > 0, True, np.where(pd.isna(ev["sue_score"]), False, False))
ev["T"] = pd.to_datetime(ev["T"])

# ---------------- prices / benches / drifts ----------------
prices = p0.load_prices()
n_overlap = sum(1 for k in p6 if k in prices)
prices.update({k: v for k, v in p6.items() if k not in prices})
log(f"prices: {len(prices):,} (sp600 added, overlap skipped {n_overlap})")

cal, ijh_close = p0.load_calendar()
b6 = pd.to_datetime(ijr["Date"]).to_numpy().astype("datetime64[D]")
o6 = np.argsort(b6)
cal6, close6 = b6[o6], ijr["Adj_Close"].to_numpy(dtype=float)[o6]

ev4 = p0.compute_drifts(ev[sel4].reset_index(drop=True).copy(), prices, cal, ijh_close,
                          horizons=(1, 3, 11))
ev4.index = ev.index[sel4]
ev6 = p0.compute_drifts(ev[sel6].reset_index(drop=True).copy(), prices, cal6, close6,
                          horizons=(1, 3, 11))
ev6.index = ev.index[sel6]
ev = pd.concat([ev4, ev6]).sort_index()
log(f"drifts done ({time.time()-t0:.0f}s) | r3 nan {pd.isna(ev['r3']).mean():.1%} "
    f"| r11 nan {pd.isna(ev['r11']).mean():.1%}")

pos = np.searchsorted(cal, ev["T"].to_numpy().astype("datetime64[D]"), side="left")
ev["pos"] = pos
back = np.where(pd.isna(ev["is_bmo"]), -1, np.where(ev["is_bmo"] == 1, 2, 1))
cut = cal[np.clip(pos - back, 0, len(cal) - 1)]
cut[pd.isna(ev["is_bmo"])] = np.datetime64("NaT", "D")
ev["cut_date"] = cut
log(f"cutoffs set ({time.time()-t0:.0f}s) | is_bmo unknown {pd.isna(ev['is_bmo']).mean():.1%}")

# ---------------- feature computation (shared arrays) ----------------
pt_arr = ev["permaTicker"].to_numpy()
sue = ev["beatflag"].to_numpy()
r1 = ev["r1"].to_numpy()
r3 = ev["r3"].to_numpy()
mat3 = ev["mat3"].to_numpy()
r11 = ev["r11"].to_numpy()
mat11 = ev["mat11"].to_numpy()
mat1 = ev["mat1"].to_numpy()
n = len(ev)

A2 = np.full(n, np.nan)
tick_groups = {}
for pt, sub in ev.groupby("permaTicker", sort=False):
    s = sub.sort_values("pos")
    tick_groups[pt] = (s["pos"].to_numpy(), s.index.to_numpy())
for i in range(n):
    if pd.isna(cut[i]):
        continue
    p_, ix_ = tick_groups[pt_arr[i]]
    k_ = bisect.bisect_left(p_, pos[i])
    vals = []
    for jx in ix_[:k_][::-1]:
        mj = mat11[jx]
        if pd.isna(mj) or mj > cut[i] or not np.isfinite(r11[jx]) or not np.isfinite(r1[jx]):
            continue
        vals.append(r11[jx] * np.sign(r1[jx]))
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
    mm = (pt_arr[w] != pt_arr[i_idx]) & sue[w]
    mm &= ~pd.isna(mat3[w]) & (mat3[w] <= cut[i_idx]) & np.isfinite(r3[w])
    return w[mm]


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
beat_ok = ev[sue & ev["r1"].notna() & ev["mat1"].notna()]
Tb = pd.to_datetime(beat_ok["T"]).dt.isocalendar()
wkb = (Tb.year.astype(str) + "-W" + Tb.week.astype(str).str.zfill(2)).to_numpy()
ptb = beat_ok["permaTicker"].to_numpy()
r1b = beat_ok["r1"].to_numpy()
matb = beat_ok["mat1"].to_numpy()
svd_cps = []
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
    svd_cps.append((cp, pd.DataFrame(E / np.where(nrm > 0, nrm, 1.0), index=M.index)))
cp_list = [c for c, _ in svd_cps]
log(f"SVD checkpoints {len(svd_cps)} ({time.time()-t0:.0f}s)")

F1 = np.full(n, np.nan)
for i in range(n):
    if pd.isna(cut[i]):
        continue
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
    mm = ws > 0
    if not mm.any():
        continue
    ws, rs = ws[mm], rs[mm]
    F1[i] = float((ws * rs).sum() / ws.sum())
log(f"F1_sb_h3 defined {np.isfinite(F1).mean():.1%} ({time.time()-t0:.0f}s)")

# ---------------- truncation spot-check (30 events, full recompute) ----------------
rng_tc = np.random.default_rng(SEED)
cand = np.where(np.isfinite(F1) & (pos > 400) & (pos < len(cal) - 50))[0]
sample = rng_tc.choice(cand, size=min(30, len(cand)), replace=False)
fails = 0
bench4 = pd.Series(ijh_close, index=pd.DatetimeIndex(cal))
bench6 = pd.Series(close6, index=pd.DatetimeIndex(cal6))
for i in sample:
    cut_i = cut[i]
    ptr = {k: (d[d <= cut_i], c[d <= cut_i]) for k, (d, c) in prices.items()}
    lo = bisect.bisect_left(pos_sorted, pos[i] - p0.K_WINDOW)
    hi = bisect.bisect_left(pos_sorted, pos[i])
    mini = ev.iloc[order[lo:hi]].copy()
    mini = pd.concat([mini, ev.iloc[[i]]])
    m4 = p0.compute_drifts(mini[mini["is_sp400"] == 1].reset_index(drop=True).copy(), ptr, cal,
                           ijh_close, horizons=(3,))
    m4.index = mini.index[mini["is_sp400"] == 1]
    m6 = p0.compute_drifts(mini[mini["is_sp400"] == 0].reset_index(drop=True).copy(), ptr, cal6,
                           close6, horizons=(3,))
    m6.index = mini.index[mini["is_sp400"] == 0]
    mini = pd.concat([m4, m6]).sort_index()
    mv3 = mini["mat3"].to_numpy()
    rv3 = mini["r3"].to_numpy()
    w_all = order[lo:hi]
    mm = (pt_arr[w_all] != pt_arr[i]) & sue[w_all]
    mm &= ~pd.isna(mv3[:-1]) & (mv3[:-1] <= cut_i) & np.isfinite(rv3[:-1])
    w_idx = w_all[mm]
    if len(w_idx) == 0:
        fails += 0 if np.isnan(F1[i]) else 1
        continue
    k = bisect.bisect_right(cp_list, cut_i) - 1
    emb = svd_cps[k][1]
    if pt_arr[i] not in emb.index:
        fails += 0 if np.isnan(F1[i]) else 1
        continue
    u = emb.loc[pt_arr[i]].to_numpy()
    sims = (emb @ u).clip(lower=0.0)
    ws = np.array([float(sims.get(o, 0.0)) for o in pt_arr[w_idx]])
    if len(ws) > TOPK:
        keep = np.argsort(-ws)[:TOPK]
        w_idx, ws = w_idx[keep], ws[keep]
    r3g = pd.Series(rv3, index=mini.index.to_numpy())
    rs = r3g.reindex(w_idx).to_numpy(dtype=float)
    mkeep = ws > 0
    if not mkeep.any():
        fails += 0 if np.isnan(F1[i]) else 1
        continue
    f1t = float((ws[mkeep] * rs[mkeep]).sum() / ws[mkeep].sum())
    if not (np.isnan(f1t) and np.isnan(F1[i])) and not np.isclose(f1t, F1[i], equal_nan=True):
        fails += 1
        log(f"  TRUNC MISMATCH ev {i}: full={F1[i]:.6f} trunc={f1t:.6f}")
log(f"truncation spot-check: {len(sample)} events, {fails} mismatch")
if fails:
    log("ABORT per registration: firewall mismatch — no verdict")
    with open(OUT / "gauntlet.json", "w") as f:
        json.dump({"status": "ABORTED_TRUNCATION_FAIL", "fails": fails}, f)
    sys.exit(2)

# ---------------- assemble + arms ----------------
m = m.join(pd.DataFrame({"F1_sb_h3": F1, "A2": A2}, index=ev.index))
log(f"matrix {len(m):,} | F1 nan {m.F1_sb_h3.isna().mean():.1%} | A2 nan {m.A2.isna().mean():.1%}")


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
    raw["entry_date"] = pd.to_datetime(raw["entry_date"])
    raw["exit_date"] = pd.to_datetime(raw["exit_date"])
    pieces = [bt.select_weekly(raw[raw["fold"] == fi], bt.N_SLOTS) for fi in range(1, 5)
              if not raw[raw["fold"] == fi].empty]
    ex = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return raw, ex


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
    d = {"executed": int(len(r)), "win_rate_pct": round(float((r > 0).mean() * 100), 1),
         "avg_trade_pct": round(float(r.mean() * 100), 3), "weeks": int(len(w)),
         "nav_pct": round(float(((1 + w).prod() - 1) * 100), 2),
         "raw_precision": round(float(ex["pead_pass"].mean() * 100), 1),
         "g1_rate": round(float(ex["pass_g1"].mean() * 100), 1)}
    if "is_sp400" in ex.columns:
        for tag, msk in (("sp400", ex["is_sp400"] == 1), ("sp600", ex["is_sp400"] == 0)):
            rr = ex.loc[msk, "pregap_return"].astype(float)
            d[f"{tag}_n"] = int(len(rr))
            d[f"{tag}_avg"] = round(float(rr.mean() * 100), 3) if len(rr) else None
    return d


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


runs = {}
_, ex7 = run_arm(m, FEATURES22, GATES3, V6_HP, score="min")
_, ex20 = run_arm(m, FEATURES22 + NEW, ["pass_g1"], V6_HP, score="single")
runs["v7n"] = {"stats": stats(ex7), "boot": boot_mean(weekly(ex7)), "ex": ex7}
runs["rc20"] = {"stats": stats(ex20), "boot": boot_mean(weekly(ex20)), "ex": ex20}
for k, v in runs.items():
    log(f"{k}: {json.dumps(v['stats'], default=str)}")
    log(f"  CI: {json.dumps(v['boot'])}")

g2p = paired(weekly(runs["rc20"]["ex"]), weekly(runs["v7n"]["ex"]))
s20, s7n = runs["rc20"]["stats"], runs["v7n"]["stats"]
ci = runs["rc20"]["boot"]["ci95_pct"]
G1 = bool(s20.get("avg_trade_pct", 0) > 0 and ci and (ci[0] > 0 or ci[1] < 0))
G2 = bool(s20.get("avg_trade_pct", 0) > s7n.get("avg_trade_pct", -9e9)
          and g2p["ci95_pct"] and (g2p["ci95_pct"][0] > 0 or g2p["ci95_pct"][1] < 0))
verdict = {"G1_rc20_positive": {"pass": G1, "boot": runs["rc20"]["boot"]},
           "G2_beats_v7n": {"pass": G2, "paired": g2p},
           "truncation_spotcheck": {"n": int(len(sample)), "fails": int(fails)}}
log("VERDICT: " + json.dumps(verdict, indent=1, default=str))

with open(OUT / "gauntlet.json", "w") as f:
    json.dump({"runs": {k: v["stats"] for k, v in runs.items()},
               "boot": {k: v["boot"] for k, v in runs.items()},
               "verdict": verdict,
               "config": {"n_boot": N_BOOT, "seed": SEED, "thresh": THRESH,
                          "matrix": "train_matrix_v7c full"},
               "seconds": round(time.time() - t0)}, f, indent=2, default=str)
with pd.HDFStore(OUT / "executed.h5", "w") as st:
    for k, v in runs.items():
        if len(v["ex"]):
            keep = [c for c in ("permaTicker", "fold", "entry_date", "exit_date",
                                "pregap_return", "score", "pead_pass", "pass_g1",
                                "is_sp400") if c in v["ex"].columns]
            st.put(f"/{k}", v["ex"][keep], format="table")
log(f"wrote {OUT / 'gauntlet.json'} ({round(time.time() - t0)}s)")

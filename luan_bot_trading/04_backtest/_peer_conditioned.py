"""Diagnostic (exploratory): beat-confidence-conditioned peer-embedding signal.

User hypothesis: the 22 features cannot predict drift; the beat model built
on them just repackages that. The embedding (F1_sb_h3) is anchored on
peers' post-beat drift — genuinely different information. For LOW
beat-confidence events (where the model has no view), lean on the peer
embedding.

Tests (all EXCESS returns, week-block bootstrap):
  1. grid: excess harvest by p_beat quintile x F1_sb_h3 quintile
  2. within each p_beat region: F1 quintile spread on excess harvest
  3. does F1 predict the BEAT within low-p_beat events (incremental AUC)?
  4. the user's proposed cell: low p_beat x high F1 — full payoff structure
  5. same for A2 (secondary)
"""
from __future__ import annotations

import bisect
import importlib.util
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
TOPK = 10
N_BOOT, SEED = 10_000, 20260807


def log(*a):
    print(*a, flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = load("bt_cond", HERE / "51_hp_theta_sweep_23feat.py")
p0 = load("p0_cond", HERE / "rc18_p0_firewall.py")
DB = bt.DB
t0 = time.time()

# ---------------- prices + benchmarks ----------------
cal, ijh_close = p0.load_calendar()
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
prices_all = dict(p0.load_prices())
prices_all.update(p6)
b6 = pd.to_datetime(ijr["Date"]).to_numpy().astype("datetime64[D]")
o6 = np.argsort(b6)
cal6, close6 = b6[o6], ijr["Adj_Close"].to_numpy(dtype=float)[o6]

cp_dates = []
last_m, prev_d = None, None
for dd in cal:
    mth = dd.astype("M8[M]")
    if last_m is not None and mth != last_m:
        cp_dates.append(prev_d)
    prev_d = dd
    last_m = mth
cp_dates.append(cal[-1])
cp_dates = np.array([dd for dd in cp_dates if dd >= np.datetime64("2016-01-01", "D")])


# ---------------- events + features (combined pipeline) ----------------
v7 = pd.read_hdf(DB, "/features/train_matrix_v7c")
v7["report_date"] = pd.to_datetime(v7["report_date"])
v6 = pd.read_hdf(DB, "/features/train_matrix_v6c")
v6["report_date"] = pd.to_datetime(v6["report_date"])
js = v6.set_index(["permaTicker", "report_date"])[["T", "is_bmo", "sue_score"]]
sel4 = v7["is_sp400"] == 1
ev4 = v7[sel4][["permaTicker", "report_date"]].join(js, on=["permaTicker", "report_date"])
ev4["beatflag"] = pd.to_numeric(ev4["sue_score"], errors="coerce") > 0
ev4 = p0.compute_drifts(ev4.reset_index(drop=True).copy(), prices_all, cal, ijh_close, horizons=(1, 3, 11))
ev4["is_sp400"] = 1

ev6 = v7[~sel4][["permaTicker", "report_date"]].copy()
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
evCp = evC.reset_index(drop=True)
n4c = int((evCp["is_sp400"] == 1).sum())
log(f"events {len(evCp):,} | embedding checkpoints {len(cpsC)} ({time.time()-t0:.0f}s)")

pta, posa, cuta = evCp["permaTicker"].to_numpy(), evCp["pos"].to_numpy(), evCp["cut_date"].to_numpy()
beat, r3, mat3 = evCp["beatflag"].to_numpy(), evCp["r3"].to_numpy(), evCp["mat3"].to_numpy()
r1, r11, m11 = evCp["r1"].to_numpy(), evCp["r11"].to_numpy(), evCp["mat11"].to_numpy()
cp_list = [c for c, _ in cpsC]
order = np.argsort(posa, kind="stable")
pos_sorted = posa[order]

F1 = np.full(len(evCp), np.nan)
for i in range(len(evCp)):
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
    emb = cpsC[k][1]
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
    F1[i] = float((ws * rs).sum() / ws.sum())

A2 = np.full(len(evCp), np.nan)
tg = {}
for pt, sub in evCp.groupby("permaTicker", sort=False):
    s = sub.sort_values("pos")
    tg[pt] = (s["pos"].to_numpy(), s.index.to_numpy())
for i in range(len(evCp)):
    if pd.isna(cuta[i]):
        continue
    p_, ix_ = tg[pta[i]]
    k_ = bisect.bisect_left(p_, posa[i])
    vals = []
    for jx in ix_[:k_][::-1]:
        mj = m11[jx]
        if pd.isna(mj) or mj > cuta[i] or not np.isfinite(r11[jx]) or not np.isfinite(r1[jx]):
            continue
        vals.append(r11[jx] * np.sign(r1[jx]))
        if len(vals) == 8:
            break
    if vals:
        A2[i] = float(np.mean(vals))
log(f"F1 defined {np.isfinite(F1).mean():.1%} | A2 defined {np.isfinite(A2).mean():.1%} ({time.time()-t0:.0f}s)")

# ---------------- join with OOS p_beat + excess returns ----------------
P = pd.read_hdf(HERE / "archive" / "experiments" / "beat_prediction" / "oos_predictions.h5", "p")
v7e = v7[["permaTicker", "report_date", "entry_date", "exit_date", "is_sp400"]].copy()
v7e["entry_date"] = pd.to_datetime(v7e["entry_date"])
v7e["exit_date"] = pd.to_datetime(v7e["exit_date"])
P = P.merge(v7e, on=["permaTicker", "report_date"], how="left", suffixes=("", "_v7"))
kC = pd.MultiIndex.from_arrays([evCp["permaTicker"], evCp["report_date"]])
P["F1"] = pd.Series(F1, index=kC).reindex(pd.MultiIndex.from_arrays([P["permaTicker"], P["report_date"]])).to_numpy()
P["A2"] = pd.Series(A2, index=kC).reindex(pd.MultiIndex.from_arrays([P["permaTicker"], P["report_date"]])).to_numpy()


def excess_ret(pt, ed, xd, sp4):
    if pt not in prices_all or pd.isna(ed) or pd.isna(xd):
        return np.nan
    d, c = prices_all[pt]
    i = np.searchsorted(d, ed.to_datetime64().astype("datetime64[D]"))
    j = np.searchsorted(d, xd.to_datetime64().astype("datetime64[D]"))
    if i >= len(d) or j >= len(d) or i >= j or c[i] <= 0:
        return np.nan
    sr = c[j] / c[i] - 1.0
    bdd, bcc = (cal, ijh_close) if sp4 == 1 else (cal6, close6)
    bi = np.searchsorted(bdd, ed.to_datetime64().astype("datetime64[D]"))
    bj = np.searchsorted(bdd, xd.to_datetime64().astype("datetime64[D]"))
    if bi >= len(bdd) or bj >= len(bdd) or bcc[bi] <= 0:
        return np.nan
    return sr - (bcc[bj] / bcc[bi] - 1.0)


P["ex"] = [excess_ret(pt, ed, xd, s) for pt, ed, xd, s in
           zip(P.permaTicker, P.entry_date, P.exit_date, P.is_sp400)]
M = P[P["ex"].notna() & P["F1"].notna() & P["p_beat"].notna()].copy()
M["ex"] = M["ex"] * 100
iso = M["report_date"].dt.isocalendar()
M["wk"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
log(f"joined analysis set: {len(M):,} ({time.time()-t0:.0f}s)")

M["pq"] = pd.qcut(M.p_beat.rank(method="first"), 5, labels=False) + 1
M["fq"] = pd.qcut(M.F1.rank(method="first"), 5, labels=False) + 1

print("\n=== 1. EXCESS HARVEST grid: p_beat quintile x F1 quintile (%) ===")
g = M.pivot_table(index="pq", columns="fq", values="ex", aggfunc="mean")
n = M.pivot_table(index="pq", columns="fq", values="ex", aggfunc="size")
for i in g.index:
    print(f"  p_beat q{i}: " + "  ".join(f"{g.loc[i,j]:+6.2f}[{int(n.loc[i,j]):>3}]" for j in g.columns))

rng = np.random.default_rng(SEED)


def spread_ci(sub, col="fq", lo=1, hi=5):
    a = sub[sub[col] == hi]["ex"]
    b = sub[sub[col] == lo]["ex"]
    if len(a) < 20 or len(b) < 20:
        return None
    obs = a.mean() - b.mean()
    wa = sub[sub[col] == hi].groupby("wk")["ex"].agg(["sum", "count"])
    wb = sub[sub[col] == lo].groupby("wk")["ex"].agg(["sum", "count"])
    wks = wa.index.union(wb.index)
    ta = wa.reindex(wks).fillna(0.0)
    tb = wb.reindex(wks).fillna(0.0)
    idx = rng.integers(0, len(wks), size=(N_BOOT, len(wks)))
    ma = ta["sum"].to_numpy()[idx].sum(1) / np.maximum(ta["count"].to_numpy()[idx].sum(1), 1)
    mb = tb["sum"].to_numpy()[idx].sum(1) / np.maximum(tb["count"].to_numpy()[idx].sum(1), 1)
    d = ma - mb
    return obs, d.std(ddof=1), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


print("\n=== 2. F1 quintile spread on excess harvest, WITHIN p_beat regions ===")
print(f"{'p_beat region':16} {'n':>5} {'q5 mean':>8} {'q1 mean':>8} {'spread':>8} {'SE':>5} {'t':>5} {'CI95':>16}")
for lab, sub in (("bottom 2 quintiles", M[M.pq <= 2]), ("q3", M[M.pq == 3]),
                 ("top 2 quintiles", M[M.pq >= 4]), ("all", M)):
    r_ = spread_ci(sub)
    if r_:
        obs, se, lo5, hi95 = r_
        print(f"{lab:16} {len(sub):>5} {sub[sub.fq==5].ex.mean():>+8.2f} {sub[sub.fq==1].ex.mean():>+8.2f} "
              f"{obs:>+8.2f} {se:>5.2f} {obs/se:>5.2f} [{lo5:>+6.2f},{hi95:>+6.2f}]")

print("=== USER MECHANISM: E[drift | BEAT] by F1 — is drift proportional to peers? ===")
rng2 = np.random.default_rng(SEED)
def cond_beat_table(lab, sub):
    B_ = sub[sub.beat==1].copy()
    if len(B_)<100: return
    B_ = B_.assign(fq=pd.qcut(B_.F1.rank(method="first"),5,labels=False)+1)
    parts=[]
    for q,g in B_.groupby("fq"):
        parts.append(f"q{int(q)}:{g.ex.mean():+.2f}[{len(g)}]")
    print(f"  {lab:26} beats n={len(B_):>4}: " + "  ".join(parts))
    # regression slope: ex ~ F1 (slope near 1 = drift proportional to peer drift)
    x,y = B_.F1.to_numpy()*100, B_.ex.to_numpy()
    ok=np.isfinite(x)&np.isfinite(y); x,y=x[ok],y[ok]
    sl,ic = np.polyfit(x,y,1)
    # week-clustered slope SE
    df_=B_.iloc[ok]
    bo=[]
    wks=df_.wk.unique()
    for _ in range(2000):
        sel=rng2.choice(wks,len(wks),replace=True)
        parts2=[df_[df_.wk==w] for w in sel]
        dd=pd.concat(parts2)
        xx,yy=dd.F1.to_numpy()*100, dd.ex.to_numpy()
        if xx.std()>0: bo.append(np.polyfit(xx,yy,1)[0])
    se_sl=np.std(bo,ddof=1) if len(bo)>10 else np.nan
    print(f"    slope(dex/dF1) = {sl:.2f}  SE {se_sl:.2f}  t {sl/se_sl:+.2f}   (1.0 = proportional)")

cond_beat_table("low p_beat (bottom half)", M[M.p_beat<=M.p_beat.quantile(0.5)])
cond_beat_table("high p_beat (top half)", M[M.p_beat>=M.p_beat.quantile(0.5)])
print()
print("=== and the miss side: E[drift | MISS] by F1 within low-p_beat ===")
L=M[(M.p_beat<=M.p_beat.quantile(0.5))&(M.beat==0)].copy()
L=L.assign(fq=pd.qcut(L.F1.rank(method="first"),5,labels=False)+1)
print("  " + "  ".join(f"q{int(q)}:{g.ex.mean():+.2f}[{len(g)}]" for q,g in L.groupby("fq")))
print()
print("=== tradeable slice: low-p x top F1 tercile — full EV decomposition ===")
T=M[(M.p_beat<=M.p_beat.quantile(0.5))&(M.F1>=M.F1.quantile(0.67))].copy()
print(f"  n={len(T)}  beat_rate={T.beat.mean()*100:.1f}%  E[ex|beat]={T[T.beat==1].ex.mean():+.2f}%  E[ex|miss]={T[T.beat==0].ex.mean():+.2f}%  EV={T.ex.mean():+.3f}%")
T=M[(M.p_beat<=M.p_beat.quantile(0.4))&(M.F1>=M.F1.quantile(0.8))].copy()
if len(T)>50: print(f"  stricter (p<40%, F1 top20%): n={len(T)} beat_rate={T.beat.mean()*100:.1f}%  E[ex|beat]={T[T.beat==1].ex.mean():+.2f}%  E[ex|miss]={T[T.beat==0].ex.mean():+.2f}%  EV={T.ex.mean():+.3f}%")


print("\n=== 3. does F1 predict the BEAT within low-p_beat events? ===")
from sklearn.metrics import roc_auc_score
for lab, sub in (("p_beat bottom 40%", M[M.p_beat <= M.p_beat.quantile(0.4)]),
                 ("p_beat top 40%", M[M.p_beat >= M.p_beat.quantile(0.6)])):
    if sub.beat.nunique() > 1:
        print(f"  {lab}: n={len(sub)}  AUC(F1 -> beat) = {roc_auc_score(sub.beat.astype(int), sub.F1):.4f}  "
              f"beat rate F1q1={sub[sub.fq==1].beat.mean()*100:.1f}% F1q5={sub[sub.fq==5].beat.mean()*100:.1f}%")

print("\n=== 4. THE USER'S CELL: low p_beat x high F1 (payoff structure) ===")
lo_half = M[M.p_beat <= M.p_beat.quantile(0.5)]
cell = lo_half[lo_half.fq >= 4]
rest_lo = lo_half[lo_half.fq <= 2]
b = cell[cell.beat == 1].ex
m_ = cell[cell.beat == 0].ex
print(f"  cell: n={len(cell)} beat_rate={cell.beat.mean()*100:.1f}%  excess={cell.ex.mean():+.3f}%")
print(f"    E[ex|beat]={b.mean():+.2f}% (n={len(b)})   E[ex|miss]={m_.mean():+.2f}% (n={len(m_)})")
r_ = spread_ci(lo_half)
if r_:
    obs, se, lo5, hi95 = r_
    print(f"  within low-p_beat: F1 q5-q1 spread={obs:+.2f} SE={se:.2f} CI[{lo5:+.2f},{hi95:+.2f}]")
print("=== ROBUSTNESS: F1 spread within low-p_beat, by universe and time ===")
med_date = M.report_date.median()
for lab, sub in (("SP400", M[(M.is_sp400==1)&(M.p_beat<=M.p_beat.quantile(0.5))]),
                 ("SP600", M[(M.is_sp400==0)&(M.p_beat<=M.p_beat.quantile(0.5))]),
                 ("first half", M[(M.report_date<=med_date)&(M.p_beat<=M.p_beat.quantile(0.5))]),
                 ("second half", M[(M.report_date>med_date)&(M.p_beat<=M.p_beat.quantile(0.5))])):
    r_ = spread_ci(sub)
    if r_:
        obs, se, lo5, hi95 = r_
        print(f"  {lab:12} n={len(sub):>4}  spread={obs:+.2f} SE={se:.2f} t={obs/se:+.2f} CI[{lo5:+.2f},{hi95:+.2f}]")


print("\n=== 5. A2 (secondary): spread within low-p_beat ===")
M["aq"] = pd.qcut(M.A2.rank(method="first"), 5, labels=False) + 1
r_ = spread_ci(lo_half.assign(fq=lo_half.A2.map(
    lambda v: 5 if v >= lo_half.A2.quantile(0.8) else (1 if v <= lo_half.A2.quantile(0.2) else 3))) if "fq" in lo_half else lo_half)
lo2 = M[M.p_beat <= M.p_beat.quantile(0.5)].copy()
lo2["fq"] = np.where(lo2.A2 >= lo2.A2.quantile(0.8), 5, np.where(lo2.A2 <= lo2.A2.quantile(0.2), 1, 3))
r_ = spread_ci(lo2)
if r_:
    obs, se, lo5, hi95 = r_
    print(f"  within low-p_beat: A2 q5-q1 spread={obs:+.2f} SE={se:.2f} CI[{lo5:+.2f},{hi95:+.2f}]")

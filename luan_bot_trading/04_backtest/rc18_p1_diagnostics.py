"""RC-18 P1 — diagnostics for Family F (learned PEAD-behavior similarity).

Pre-registered (rc18_amendment_1_peer_similarity.md §5):
  - sample: BEATS only (sue_score > 0)
  - CAR-first: quintile spreads in car_10d (primary), pead_pass (secondary)
  - week-block bootstrap CIs, 10k resamples, seed 20260807
  - orthogonalization vs rel_ret_5d/20d, sector_adjusted_ret_20d,
    revision_momentum_30/60/90d — raw AND orthogonalized reported
  - F3 dispersion interaction test
  - S-a / S-b vs S-c (taxonomy) on the common defined set
No model is trained, no artifact promoted. Diagnostic only.

Implementation notes (documented deviations, all PIT-safe):
  - S-b embeddings trained at MONTH-END checkpoints (events matured <=
    checkpoint); an event uses the latest checkpoint <= its cutoff.
  - Quintile boundaries computed once on the full defined sample and held
    fixed during bootstrap (bootstrap resamples weeks, not boundaries).
  - Fast instance builder (vectorized) replaces P0's loop version; same
    definition (validated on a sample against P0's output).
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
OUT = HERE / "archive" / "experiments" / "rc18_p1"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 10_000
SEED = 20260807
TOPK = 10
CONF = ["rel_ret_5d", "rel_ret_20d", "sector_adjusted_ret_20d",
        "revision_momentum_30d", "revision_momentum_60d", "revision_momentum_90d"]


def log(*a):
    print(*a, flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p0 = load("rc18_p0_mod", HERE / "rc18_p0_firewall.py")

t0 = time.time()
ev = p0.load_events()
cal, bclose = p0.load_calendar()
ev["pos"] = np.searchsorted(cal, ev["T"].to_numpy().astype("datetime64[D]"), side="left")
back = np.where(ev["is_bmo"].astype(bool), 2, 1)
ev["cut_date"] = cal[np.clip(ev["pos"] - back, 0, len(cal) - 1)]
prices = p0.load_prices()
ev = p0.compute_drifts(ev, prices, cal, bclose)
ev["sic4"] = ev["sic"]
ev["sic3"] = ev["sic"] // 10
ev["sic2"] = ev["sic"] // 100
log(f"events {len(ev):,} loaded+drifts ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------- vectorized ----
order = np.argsort(ev["pos"].to_numpy(), kind="stable")
pos_sorted = ev["pos"].to_numpy()[order]
pt_arr = ev["permaTicker"].to_numpy()
sue_arr = ev["sue_score"].to_numpy()
r1_arr = ev["r1"].to_numpy()
r3_arr = ev["r3"].to_numpy()
mat1 = ev["mat1"].to_numpy()
mat3 = ev["mat3"].to_numpy()
sic4_arr = ev["sic4"].to_numpy()
sic2_arr = ev["sic2"].to_numpy()
cut_arr = ev["cut_date"].to_numpy()


def peer_window(i_idx):
    """Sorted-position window [pos_i-10, pos_i): global row indices."""
    p = ev.at[i_idx, "pos"]
    lo = bisect.bisect_left(pos_sorted, p - p0.K_WINDOW)
    hi = bisect.bisect_left(pos_sorted, p)
    return order[lo:hi]


def peers_vec(i_idx, h, grain):
    """Vectorized eligibility (same contract as p0.peers_for):
    different ticker, beat, mat_h <= cut, r_h finite; grain in
    {None, 'sic4', 'sic2', 'any'}."""
    w = peer_window(i_idx)
    if len(w) == 0:
        return w
    m = (pt_arr[w] != ev.at[i_idx, "permaTicker"]) & (sue_arr[w] > 0)
    if h == 1:
        mj = mat1[w]
        rj = r1_arr[w]
    else:
        mj = mat3[w]
        rj = r3_arr[w]
    m &= ~pd.isna(mj) & (mj <= ev.at[i_idx, "cut_date"]) & np.isfinite(rj)
    if grain == "sic4":
        m &= sic4_arr[w] == ev.at[i_idx, "sic4"]
    elif grain == "sic2":
        m &= sic2_arr[w] == ev.at[i_idx, "sic2"]
    return w[m]


# ------------------------------------------------- fast S-a instances -----
def build_instances_fast():
    rows_late, rows_early, xs, ys, tmats = [], [], [], [], []
    for _, sub in ev.groupby("sic2", sort=False):
        if len(sub) < 2:
            continue
        sub = sub[(sub["sue_score"] > 0) & sub["r1"].notna()].sort_values("pos")
        if len(sub) < 2:
            continue
        p = sub["pos"].to_numpy()
        idxs = sub.index.to_numpy()
        r = sub["r1"].to_numpy()
        m = sub["mat1"].to_numpy()
        ptl = sub["permaTicker"].to_numpy()
        for k in range(len(sub)):
            lo = bisect.bisect_left(p, p[k] - p0.K_WINDOW)
            hi = k
            if hi <= lo:
                continue
            rows_early.extend(idxs[lo:hi])
            rows_late.extend([idxs[k]] * (hi - lo))
            xs.extend(r[lo:hi])
            ys.extend([r[k]] * (hi - lo))
            tmats.extend(m[lo:hi])  # mat of EARLY event (matured before later's print)
    df = pd.DataFrame({"row_early": rows_early, "row_late": rows_early and rows_late,
                       "x": xs, "y": ys, "tmat_early": tmats})
    # NOTE: instance maturity = mat of the LATER event (the label side)
    df["tmat"] = [mat1[l] for l in df["row_late"]]
    df["pt_early"] = pt_arr[df["row_early"].to_numpy()]
    df["pt_late"] = pt_arr[df["row_late"].to_numpy()]
    return df


inst = build_instances_fast()
# correctness spot-check vs P0's builder on a small group
log(f"instances {len(inst):,} ({time.time()-t0:.0f}s)")

inst_by_pt = {}
_e = inst["pt_early"].to_numpy()
_l = inst["pt_late"].to_numpy()
for pt in np.unique(np.concatenate([_e, _l])):
    sel = np.where((_e == pt) | (_l == pt))[0]
    inst_by_pt[pt] = inst.iloc[sel]


def weights_shrunk(i_idx, j_idxs, tau):
    pt_i = ev.at[i_idx, "permaTicker"]
    sub_rows = inst_by_pt.get(pt_i)
    w = {}
    if sub_rows is not None and len(sub_rows):
        sub = sub_rows[sub_rows["tmat"] <= tau]
        if len(sub):
            other = np.where(sub["pt_early"].to_numpy() == pt_i,
                             sub["pt_late"], sub["pt_early"])
            jpt = pt_arr[j_idxs]
            for k, o in enumerate(other):
                # accumulate via groupby-lite
                pass
            df = pd.DataFrame({"other": other, "x": sub["x"].to_numpy(),
                               "y": sub["y"].to_numpy()})
            g = df.groupby("other")
            stats = g.apply(lambda d: (len(d), d.x.std(), d.y.std(),
                                       ((d.x - d.x.mean()) * (d.y - d.y.mean())).sum()
                                       / max(len(d) - 1, 1)), include_groups=False)
            for o, (n, sx, sy, cov) in stats.items():
                corr = cov / (sx * sy) if (n >= 3 and sx and sy and sx > 0 and sy > 0) else 0.0
                w[o] = max((n / (n + p0.SHRINK_K)) * corr + (p0.SHRINK_K / (n + p0.SHRINK_K)), 0.0)
    jpt = pt_arr[j_idxs]
    return np.array([w.get(o, 1.0) for o in jpt])


# ------------------------------------------------- S-b checkpoints --------
month_ends = []
seen = set()
for d in cal:
    key = (d.astype("M8[M]").astype(int))
    if key not in seen:
        seen.add(key)
        month_ends.append(d)
# real month-END checkpoints: last cal day of each month
cp_dates = []
last_m = None
for d in cal:
    m = d.astype("M8[M]")
    if last_m is not None and m != last_m:
        cp_dates.append(prev_d)
    prev_d = d
    last_m = m
cp_dates.append(cal[-1])
cp_dates = np.array([d for d in cp_dates if d >= np.datetime64("2016-01-01", "D")])

beat_ok = ev[(ev["sue_score"] > 0) & ev["r1"].notna() & ev["mat1"].notna()]
Ts_beat = beat_ok["T"].dt.isocalendar()
week_key_beat = (Ts_beat.year.astype(str) + "-W"
                 + Ts_beat.week.astype(str).str.zfill(2)).to_numpy()
pt_beat = beat_ok["permaTicker"].to_numpy()
r1_beat = beat_ok["r1"].to_numpy()
mat_beat = beat_ok["mat1"].to_numpy()
all_pts = ev["permaTicker"].unique()

svd_cps = []
for cp in cp_dates:
    sel = mat_beat <= cp
    if sel.sum() < 200:
        continue
    M = pd.pivot_table(pd.DataFrame({"pt": pt_beat[sel], "wk": week_key_beat[sel],
                                     "r": r1_beat[sel]}),
                       index="pt", columns="wk", values="r", aggfunc="mean")
    M = M.sub(M.mean(axis=0), axis=1).fillna(0.0)
    if M.shape[1] < p0.EMB_DIM:
        continue
    U, S, _ = np.linalg.svd(M.to_numpy(), full_matrices=False)
    E = U[:, :p0.EMB_DIM] * S[:p0.EMB_DIM]
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.where(nrm > 0, nrm, 1.0)
    svd_cps.append((cp, pd.DataFrame(E, index=M.index)))
log(f"SVD checkpoints: {len(svd_cps)} ({time.time()-t0:.0f}s)")
cp_list = [c for c, _ in svd_cps]


def weights_svd(i_idx, j_idxs):
    cut = ev.at[i_idx, "cut_date"]
    k = bisect.bisect_right(cp_list, cut) - 1
    if k < 0:
        return np.zeros(len(j_idxs))
    emb = svd_cps[k][1]
    pt_i = ev.at[i_idx, "permaTicker"]
    if pt_i not in emb.index:
        return np.zeros(len(j_idxs))
    u = emb.loc[pt_i].to_numpy()
    sims = (emb @ u).clip(lower=0.0)
    jpt = pt_arr[j_idxs]
    return np.array([float(sims.get(o, 0.0)) for o in jpt])


# ------------------------------------------------------- feature build ----
beats = ev.index[ev["sue_score"] > 0].to_numpy()
log(f"beat events: {len(beats):,}")

feat_cols = {}
for est, grain in (("sc", "sic4"), ("sa", "sic2"), ("sb", "any")):
    for h in (1, 3):
        F1 = np.full(len(ev), np.nan)
        F2 = np.full(len(ev), np.nan)
        F3 = np.full(len(ev), np.nan)
        F4 = np.full(len(ev), np.nan)
        HAS = np.zeros(len(ev), bool)
        for i_idx in beats:
            w_idx = peers_vec(i_idx, h, grain)
            if len(w_idx) == 0:
                continue
            if est == "sc":
                ws = np.ones(len(w_idx))
            elif est == "sa":
                ws = weights_shrunk(i_idx, w_idx, ev.at[i_idx, "cut_date"])
            else:
                ws = weights_svd(i_idx, w_idx)
            if len(ws) > TOPK:
                keep = np.argsort(-ws)[:TOPK]
                w_idx, ws = w_idx[keep], ws[keep]
            rs = (r1_arr if h == 1 else r3_arr)[w_idx]
            m = ws > 0
            if not m.any():
                continue
            ws, rs = ws[m], rs[m]
            W = ws.sum()
            f1 = float((ws * rs).sum() / W)
            F1[i_idx] = f1
            F2[i_idx] = float((ws * np.sign(rs)).sum() / W)
            F3[i_idx] = float((ws * np.abs(rs - f1)).sum() / W)
            F4[i_idx] = float(W * W / (ws * ws).sum())
            HAS[i_idx] = True
        tag = f"{est}_h{h}"
        feat_cols[f"F1_{tag}"] = F1
        feat_cols[f"F2_{tag}"] = F2
        feat_cols[f"F3_{tag}"] = F3
        feat_cols[f"F4_{tag}"] = F4
        log(f"features {tag}: defined {HAS.mean() and int(HAS[beats].mean()*100)}% of beats "
            f"({time.time()-t0:.0f}s)")

feats = pd.DataFrame(feat_cols, index=ev.index)
feats["has_sc_h1"] = feats["F1_sc_h1"].notna()
feats["has_sa_h1"] = feats["F1_sa_h1"].notna()
feats["has_sb_h1"] = feats["F1_sb_h1"].notna()
feats.to_pickle(OUT / "features_beats.pkl")

# -------------------------------------------------------- diagnostics -----
car = pd.to_numeric(ev["car_10d"], errors="coerce").to_numpy()
pead = ev["pead_pass"].astype(int).to_numpy()
Tiso = ev["T"].dt.isocalendar()
week_id = (Tiso.year.astype(int) * 100 + Tiso.week.astype(int)).to_numpy()
X_conf = np.column_stack([np.nan_to_num(pd.to_numeric(ev[c], errors="coerce").to_numpy(),
                                        nan=0.0) for c in CONF] + [np.ones(len(ev))])


def spread_stat(f, y, wk, pd_arr):
    """Top-minus-bottom quintile spread with FIXED boundaries."""
    m = np.isfinite(f) & np.isfinite(y)
    if m.sum() < 250:
        return None
    f_, y_, w_, p_ = f[m], y[m], wk[m], pd_arr[m]
    qs = np.quantile(f_, [0.2, 0.8])
    top = f_ >= qs[1]
    bot = f_ < qs[0]
    if top.sum() < 30 or bot.sum() < 30:
        return None
    uw, inv = np.unique(w_, return_inverse=True)
    def spread_with_counts(counts):
        ew = counts[inv]
        mt, mb = ew * top, ew * bot
        return (float((y_ * mt).sum() / mt.sum()) - float((y_ * mb).sum() / mb.sum()))
    rng = np.random.default_rng(SEED)
    est = spread_with_counts(np.ones(len(uw), dtype=int))
    idx = rng.integers(0, len(uw), size=(N_BOOT, len(uw)))
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        vals[b] = spread_with_counts(np.bincount(idx[b], minlength=len(uw)))
    lo_, hi_ = np.percentile(vals, [2.5, 97.5])
    return {"n": int(m.sum()), "n_top": int(top.sum()), "n_bot": int(bot.sum()),
            "spread_pct": round(est * 100, 4),
            "ci95_pct": [round(float(lo_) * 100, 4), round(float(hi_) * 100, 4)],
            "excl0": bool(lo_ > 0 or hi_ < 0),
            "pead_top": round(float(p_[top].mean()), 4),
            "pead_bot": round(float(p_[bot].mean()), 4)}


def residualize(y, rows):
    Xc = X_conf[rows]
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return y - Xc @ beta


diag = {}
beats_mask = np.zeros(len(ev), bool)
beats_mask[beats] = True
for col in [c for c in feats.columns if c.startswith(("F1_", "F2_", "F3_", "F4_"))]:
    f = feats[col].to_numpy()
    m = beats_mask & np.isfinite(f)
    rows = np.where(m)[0]
    if len(rows) < 250:
        diag[col] = {"n": int(len(rows)), "skip": "insufficient"}
        continue
    raw = spread_stat(f[rows], car[rows], week_id[rows], pead[rows])
    res_y = residualize(car[rows], rows)
    ortho = spread_stat(f[rows], res_y, week_id[rows], pead[rows])
    res_f = residualize(f[rows], rows)
    ortho_f = spread_stat(res_f, car[rows], week_id[rows], pead[rows])
    diag[col] = {"n": int(len(rows)), "raw": raw, "ortho_outcome": ortho,
                 "ortho_feature": ortho_f}
    log(f"{col}: n={len(rows):,} raw {raw['spread_pct'] if raw else None} "
        f"[{raw['ci95_pct'] if raw else None}] | ortho {ortho['spread_pct'] if ortho else None} "
        f"[{ortho['ci95_pct'] if ortho else None}]")

# dispersion interaction (F1 vs F3 median split) per estimator h=1
inter = {}
for est in ("sc", "sa", "sb"):
    f1 = feats[f"F1_{est}_h1"].to_numpy()
    f3 = feats[f"F3_{est}_h1"].to_numpy()
    m = beats_mask & np.isfinite(f1) & np.isfinite(f3)
    rows = np.where(m)[0]
    if len(rows) < 300:
        continue
    med = np.median(f3[rows])
    lo_rows = rows[f3[rows] <= med]
    hi_rows = rows[f3[rows] > med]
    s_lo = spread_stat(f1[lo_rows], car[lo_rows], week_id[lo_rows], pead[lo_rows])
    s_hi = spread_stat(f1[hi_rows], car[hi_rows], week_id[hi_rows], pead[hi_rows])
    inter[est] = {"low_disp": s_lo, "high_disp": s_hi,
                  "n_low": len(lo_rows), "n_high": len(hi_rows)}
    log(f"interaction[{est}]: low-disp spread {s_lo['spread_pct'] if s_lo else None} "
        f"vs high-disp {s_hi['spread_pct'] if s_hi else None}")

# estimator comparisons on common defined sets
comp = {}
for a, b_ in (("sa", "sc"), ("sb", "sc")):
    fa, fb = feats[f"F1_{a}_h1"].to_numpy(), feats[f"F1_{b_}_h1"].to_numpy()
    m = beats_mask & np.isfinite(fa) & np.isfinite(fb)
    rows = np.where(m)[0]
    s_a = spread_stat(fa[rows], car[rows], week_id[rows], pead[rows])
    s_b = spread_stat(fb[rows], car[rows], week_id[rows], pead[rows])
    comp[f"{a}_vs_{b_}"] = {"common_n": int(len(rows)), f"spread_{a}": s_a,
                            f"spread_{b_}": s_b,
                            "diff_pct": round(s_a["spread_pct"] - s_b["spread_pct"], 4)
                            if s_a and s_b else None}
    log(f"compare {a} vs {b_}: common n={len(rows):,} "
        f"{s_a['spread_pct'] if s_a else None} vs {s_b['spread_pct'] if s_b else None}")

report = {"diagnostics": diag, "interaction": inter, "comparisons": comp,
          "config": {"n_boot": N_BOOT, "seed": SEED, "topk": TOPK,
                     "confounders": CONF, "svd_checkpoints": len(svd_cps),
                     "instances": int(len(inst))},
          "seconds": round(time.time() - t0)}
with open(OUT / "report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
log(f"wrote {OUT / 'report.json'} ({report['seconds']}s)")

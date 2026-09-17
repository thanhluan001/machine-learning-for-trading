"""RC-18 P1(b) — families A (honest drift personality) and D (label-matured
sector context): completing the registered >=2-family P2 test.

Both families were pre-registered in rc18_new_features_pre_registration.md
(the original registration, before any diagnostic). Same firewall and
diagnostics machinery as P1 (week-block bootstrap 10k, seed 20260807,
orthogonalized vs the six confounders).

F0a provenance specs (exact):
  A1 drift_prop_beat_m4
    source: own-ticker prior events (v6c event set), 45-session excess CAR
            (r45, full-window-or-NaN on the stock's calendar, IJH-adjusted).
    eligibility: prior events j of the SAME permaTicker with pos_j < pos_i,
            sue_j > 0 (beat), mat45_j <= cut_i. Last <=4 by pos.
    value: mean(r45_j). No episodes -> NaN.
  A2 gap_followthrough_hist
    source: own-ticker prior episodes, r11 (10-session CAR, label horizon)
            and r1 (day-0 excess reaction).
    eligibility: pos_j < pos_i, mat11_j <= cut_i, r1/r11 finite.
            Last <=8 by pos (recent personality).
    value: mean(r11_j * sign(r1_j))  [positive = this ticker tends to
            CONTINUE in the direction of its day-0 reaction; negative =
            fades it].
  D1 sector_pead_rate_8w
    source: same-ETF-sector peer events (v6c.sector), label column
            pead_pass (builder-computed; consumption guarded by maturity).
    eligibility: sector equal, permaTicker different, pos_j in
            [pos_i - 40, pos_i), mat11_j <= cut_i.
    value: mean(pead_pass_j); none -> NaN.
  D2 sector_peer_car_8w
    same eligibility; value: mean(r11_j).

F0b: truncation-equivalence run at the end (sampled events, truncated
prices, identical features required). F0c: all windows full-or-NaN; D's
use of builder labels is maturity-guarded at consumption.
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
OUT = HERE / "archive" / "experiments" / "rc18_p1b"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 10_000
SEED = 20260807
CONF = ["rel_ret_5d", "rel_ret_20d", "sector_adjusted_ret_20d",
        "revision_momentum_30d", "revision_momentum_60d", "revision_momentum_90d"]


def log(*a):
    print(*a, flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p0 = load("rc18_p0_mod_b", HERE / "rc18_p0_firewall.py")

t0 = time.time()
ev = p0.load_events()
cal, bclose = p0.load_calendar()
ev["pos"] = np.searchsorted(cal, ev["T"].to_numpy().astype("datetime64[D]"), side="left")
back = np.where(ev["is_bmo"].astype(bool), 2, 1)
ev["cut_date"] = cal[np.clip(ev["pos"] - back, 0, len(cal) - 1)]
prices = p0.load_prices()
ev = p0.compute_drifts(ev, prices, cal, bclose, horizons=(1, 11, 45))
log(f"events {len(ev):,} | drifts r1/r11/r45 ({time.time()-t0:.0f}s) | "
    f"r45 NaN {ev.r45.isna().mean():.1%}")

pos = ev["pos"].to_numpy()
pt_arr = ev["permaTicker"].to_numpy()
sec_arr = ev["sector"].to_numpy()
sue = ev["sue_score"].to_numpy()
cut = ev["cut_date"].to_numpy()
r1 = ev["r1"].to_numpy()
r11 = ev["r11"].to_numpy()
r45 = ev["r45"].to_numpy()
mat1 = ev["mat1"].to_numpy()
mat11 = ev["mat11"].to_numpy()
mat45 = ev["mat45"].to_numpy()
pead = ev["pead_pass"].astype(int).to_numpy()

# per-ticker and per-sector sorted structures
tick_groups = {}
for pt, sub in ev.groupby("permaTicker", sort=False):
    s = sub.sort_values("pos")
    tick_groups[pt] = (s["pos"].to_numpy(), s.index.to_numpy())
sec_groups = {}
for sc, sub in ev.groupby("sector", sort=False):
    s = sub.sort_values("pos")
    sec_groups[sc] = (s["pos"].to_numpy(), s.index.to_numpy())

n = len(ev)
A1 = np.full(n, np.nan)
A2 = np.full(n, np.nan)
D1 = np.full(n, np.nan)
D2 = np.full(n, np.nan)
ND = np.zeros(n, int)

for i in range(n):
    pt_i, cut_i, pos_i = pt_arr[i], cut[i], pos[i]
    p, ix = tick_groups[pt_i]
    k = bisect.bisect_left(p, pos_i)   # events strictly before pos_i are [0:k)
    # A1: last <=4 matured beat episodes (45-session CAR)
    vals45 = []
    for j in ix[:k][::-1]:
        if j >= n:
            continue
        mj = mat45[j]
        if pd.isna(mj) or mj > cut_i or not np.isfinite(r45[j]) or not (sue[j] > 0):
            continue
        vals45.append(r45[j])
        if len(vals45) == 4:
            break
    if vals45:
        A1[i] = float(np.mean(vals45))
    # A2: last <=8 matured episodes, sign-adjusted follow-through
    vals11 = []
    for j in ix[:k][::-1]:
        mj = mat11[j]
        if pd.isna(mj) or mj > cut_i or not np.isfinite(r11[j]) or not np.isfinite(r1[j]):
            continue
        vals11.append(r11[j] * np.sign(r1[j]))
        if len(vals11) == 8:
            break
    if vals11:
        A2[i] = float(np.mean(vals11))
    # D: same-sector peers, 40-session window, label-matured
    sp, six = sec_groups[sec_arr[i]]
    lo = bisect.bisect_left(sp, pos_i - 40)
    hi = bisect.bisect_left(sp, pos_i)
    pr = []
    for j in six[lo:hi]:
        if pt_arr[j] == pt_i:
            continue
        mj = mat11[j]
        if pd.isna(mj) or mj > cut_i or not np.isfinite(r11[j]):
            continue
        pr.append(j)
    if pr:
        D1[i] = float(pead[pr].mean())
        D2[i] = float(r11[pr].mean())
        ND[i] = len(pr)

feats = pd.DataFrame({"A1": A1, "A2": A2, "D1": D1, "D2": D2, "n_sector_peers": ND},
                     index=ev.index)
beats = sue > 0
log(f"defined on beats: A1 {np.isfinite(A1[beats]).mean():.1%} | "
    f"A2 {np.isfinite(A2[beats]).mean():.1%} | D1 {np.isfinite(D1[beats]).mean():.1%} | "
    f"D2 {np.isfinite(D2[beats]).mean():.1%} ({time.time()-t0:.0f}s)")
feats.to_pickle(OUT / "features_AD.pkl")

# ---------------------------------------------------- F0b truncation -----
rng = np.random.default_rng(SEED)
sample = list(rng.choice(np.where(beats & np.isfinite(A1) & np.isfinite(D1))[0],
                         size=60, replace=False))
res = {"pass": 0, "fail": 0}
for i in sample:
    cp, bdt, bct = p0.truncate(prices, cal, bclose, cut[i])
    evt = p0.compute_drifts(ev.copy(), cp, bdt, bct, horizons=(1, 11, 45))
    # recompute A1/A2/D1/D2 for event i on truncated view
    r11t = evt["r11"].to_numpy(); r45t = evt["r45"].to_numpy()
    r1t = evt["r1"].to_numpy(); mat45t = evt["mat45"].to_numpy()
    mat11t = evt["mat11"].to_numpy()
    p_, ix_ = tick_groups[pt_arr[i]]
    k_ = bisect.bisect_left(p_, pos[i])
    v45 = []
    for j in ix_[:k_][::-1]:
        mj = mat45t[j]
        if pd.isna(mj) or mj > cut[i] or not np.isfinite(r45t[j]) or not (sue[j] > 0):
            continue
        v45.append(r45t[j])
        if len(v45) == 4:
            break
    a1t = float(np.mean(v45)) if v45 else np.nan
    v11 = []
    for j in ix_[:k_][::-1]:
        mj = mat11t[j]
        if pd.isna(mj) or mj > cut[i] or not np.isfinite(r11t[j]) or not np.isfinite(r1t[j]):
            continue
        v11.append(r11t[j] * np.sign(r1t[j]))
        if len(v11) == 8:
            break
    a2t = float(np.mean(v11)) if v11 else np.nan
    sp_, six_ = sec_groups[sec_arr[i]]
    lo_ = bisect.bisect_left(sp_, pos[i] - 40)
    hi_ = bisect.bisect_left(sp_, pos[i])
    prt = [j for j in six_[lo_:hi_]
           if pt_arr[j] != pt_arr[i] and not pd.isna(mat11t[j])
           and mat11t[j] <= cut[i] and np.isfinite(r11t[j])]
    d1t = float(pead[prt].mean()) if prt else np.nan
    d2t = float(r11t[prt].mean()) if prt else np.nan
    for nm, full_, tr in (("A1", A1[i], a1t), ("A2", A2[i], a2t),
                          ("D1", D1[i], d1t), ("D2", D2[i], d2t)):
        ok = (np.isnan(full_) and np.isnan(tr)) or abs(full_ - tr) < 1e-12
        res["pass" if ok else "fail"] += 1
log(f"truncation-equivalence (A/D): {res} ({time.time()-t0:.0f}s)")

# ------------------------------------------------------- diagnostics -----
car = pd.to_numeric(ev["car_10d"], errors="coerce").to_numpy()
Ts = pd.to_datetime(ev["T"]); iso = Ts.dt.isocalendar()
week_id = (iso.year.astype(int) * 100 + iso.week.astype(int)).to_numpy()
X_conf = np.column_stack([np.nan_to_num(pd.to_numeric(ev[c], errors="coerce").to_numpy(),
                                        nan=0.0) for c in CONF] + [np.ones(n)])


def spread_stat(f, y, wk, pd_arr):
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

    def s(c):
        ew = c[inv]
        mt, mb = ew * top, ew * bot
        return (y_ * mt).sum() / mt.sum() - (y_ * mb).sum() / mb.sum()

    rg = np.random.default_rng(SEED)
    est = s(np.ones(len(uw), int))
    idx = rg.integers(0, len(uw), size=(N_BOOT, len(uw)))
    vals = np.array([s(np.bincount(idx[b], minlength=len(uw))) for b in range(N_BOOT)])
    lo_, hi_ = np.percentile(vals, [2.5, 97.5])
    return {"n": int(m.sum()), "spread_pct": round(est * 100, 4),
            "ci95_pct": [round(float(lo_) * 100, 4), round(float(hi_) * 100, 4)],
            "excl0": bool(lo_ > 0 or hi_ < 0),
            "pead_top": round(float(p_[top].mean()), 4),
            "pead_bot": round(float(p_[bot].mean()), 4)}


def residualize(y, rows):
    Xc = X_conf[rows]
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return y - Xc @ beta


diag = {}
for col in ("A1", "A2", "D1", "D2"):
    f = feats[col].to_numpy()
    rows = np.where(beats & np.isfinite(f))[0]
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

report = {"diagnostics": diag, "truncation": res,
          "coverage_beats": {c: float(np.isfinite(feats[c].to_numpy()[beats]).mean())
                             for c in ("A1", "A2", "D1", "D2")},
          "config": {"n_boot": N_BOOT, "seed": SEED, "confounders": CONF,
                     "A1": "mean r45 of last<=4 matured beat episodes",
                     "A2": "mean r11*sign(r1) of last<=8 matured episodes",
                     "D": "same-sector peers, 40-session window, mat11<=cut"},
          "seconds": round(time.time() - t0)}
with open(OUT / "report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
log(f"wrote {OUT / 'report.json'} ({report['seconds']}s)")

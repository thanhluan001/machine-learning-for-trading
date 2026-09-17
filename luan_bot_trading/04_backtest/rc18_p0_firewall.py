"""RC-18 P0 — leak firewall implementation + self-test (Family F).

Implements F0a/F0b/F0c from rc18_amendment_1_peer_similarity.md:

  F0a  provenance contracts are in each function's docstring (source,
       window, observability inequality, maturity rule).
  F0b  truncation-equivalence harness: features recomputed from data
       truncated at each event's cutoff must equal the full-data values.
       Includes a NEGATIVE CONTROL: a deliberately leaky drift variant
       that the harness must FAIL (proves the test detects leaks).
  F0c  maturity by construction: all windows full-or-NaN; the only
       partial windows are the declared h=1 / h=3 horizons.

Two clocks (never conflated):
  peer signal  fresh: r_j over [T_j+1, T_j+h], usable iff
               dates_j[pos_j + h] <= cut_date_i   (observability)
  similarity   matured: every training instance/cell satisfies
               mat <= tau (= cut_date of the consuming event)

Spec constants (frozen by the amendment): K=10, h in {1,3}, shrink k=5,
top-k=10, SVD dim=8. No outcome relationship is measured here (P1).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB = ROOT / "01_data" / "db.h5"
OUT = HERE / "archive" / "experiments" / "rc18_p0"
OUT.mkdir(parents=True, exist_ok=True)

K_WINDOW = 10          # trailing peer window, sessions (IJH positions)
SHRINK_K = 5           # shrinkage strength for S-a
TOPK = 10              # similarity sparsity per stock
EMB_DIM = 8            # S-b embedding dimension
SEED = 20260807

DAY = np.timedelta64(1, "D")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- data ----
def load_events() -> pd.DataFrame:
    """Event universe = /features/train_matrix_v6c (the canonical scored set).

    Columns used: permaTicker, report_date, T (stock-calendar session of the
    print), is_bmo, sue_score (current-quarter SUE), sector. SIC joined from
    /metadata/sp400_permatickers (string dtype -> to_numeric; covers
    delisted names -- the metadata-ticker join loses 40% of history).
    """
    m = pd.read_hdf(DB, "/features/train_matrix_v6c")
    pm = pd.read_hdf(DB, "/metadata/sp400_permatickers")[["permaTicker", "sic"]]
    pm["sic"] = pd.to_numeric(pm["sic"], errors="coerce")
    m = m.merge(pm, on="permaTicker", how="left")
    m["report_date"] = pd.to_datetime(m["report_date"])
    m["T"] = pd.to_datetime(m["T"])
    m["sue_score"] = pd.to_numeric(m["sue_score"], errors="coerce")
    return m


def load_calendar():
    ijh = pd.read_hdf(DB, "/macros/IJH")
    cal = pd.to_datetime(ijh["Date"]).to_numpy().astype("datetime64[D]")
    close = ijh["Close"].to_numpy(dtype=float)   # builder convention: Close
    o = np.argsort(cal)
    return cal[o], close[o]


def load_prices():
    """One pass over /sp400/* -> pt -> (dates[D], adj_close)."""
    cache = {}
    with pd.HDFStore(DB, "r") as st:
        keys = [k for k in st.keys() if k.startswith("/sp400/")]
        for k in keys:
            df = st[k]
            pt = k.split("/")[-1]
            col = "Adj_Close" if "Adj_Close" in df.columns else "Close"
            d = pd.to_datetime(df["Date"]).to_numpy().astype("datetime64[D]")
            o = np.argsort(d)
            cache[pt] = (d[o], df[col].to_numpy(dtype=float)[o])
    return cache


# ------------------------------------------------------------- drifts ----
def compute_drifts(ev: pd.DataFrame, prices, bench_dates, bench_close,
                   horizons=(1, 3)):
    """Per event: r_h = excess CAR over [T+1, T+h] on the STOCK's calendar
    (log stock ret - log IJH ret, benchmark date-aligned ffill).

    F0c: FULL-window-or-NaN per horizon. Maturity date of the h-window =
    dates[pos + h]. No window may extend past the consuming cutoff -- that
    check lives in peer eligibility (see peers_for), not here, because this
    function is also the training-side label computer.
    """
    bench = pd.Series(bench_close, index=pd.DatetimeIndex(bench_dates))
    cols = {}
    for h in horizons:
        cols[f"r{h}"] = np.full(len(ev), np.nan)
        cols[f"mat{h}"] = np.full(len(ev), np.datetime64("NaT", "D"))
    pos_s = np.full(len(ev), -1)

    for pt, sub in ev.groupby("permaTicker", sort=False):
        if pt not in prices:
            continue
        d, c = prices[pt]
        ok = np.isfinite(c) & (c > 0)
        logc = np.where(ok, np.log(np.where(c > 0, c, 1.0)), np.nan)
        b = bench.reindex(pd.DatetimeIndex(d), method="ffill").to_numpy(float)
        logb = np.log(np.where(np.isfinite(b) & (b > 0), b, np.nan))
        # session excess return at position p uses [p-1 -> p]
        ex = logc[1:] - logc[:-1] - (logb[1:] - logb[:-1])   # length n-1, index p-1
        Ts = sub["T"].to_numpy().astype("datetime64[D]")
        p = np.searchsorted(d, Ts, side="left")
        idx = sub.index
        for k_i, t_i in zip(range(len(idx)), p):
            if t_i >= len(d) or d[t_i] != Ts[k_i]:
                continue
            pos_s[idx[k_i] if False else sub.index[k_i]] = t_i
            for h in horizons:
                e = t_i + h
                if e < len(d):
                    cols[f"r{h}"][sub.index[k_i]] = float(np.nansum(ex[t_i:e]))
                    cols[f"mat{h}"][sub.index[k_i]] = d[e]
    ev = ev.copy()
    ev["pos_s"] = pos_s
    for h in horizons:
        ev[f"r{h}"] = cols[f"r{h}"]
        ev[f"mat{h}"] = cols[f"mat{h}"]
    return ev


# ---------------------------------------------------------- eligibility ----
def peers_for(ev, i_idx, h, grain, grain_val_i, cut_date_i, pos_i, pt_i,
              require_beat=True):
    """F0a CONTRACT (peer eligibility for event i, horizon h):
      j != i (different permaTicker)
      gap: pos_i - pos_j in [1, K_WINDOW]            (IJH positions)
      beat: sue_j > 0                                  (if require_beat)
      observability: mat{h}[j] <= cut_date_i           (peer's h-th session
                                                        printed by my cutoff)
      r{h}[j] is not NaN                               (full-window-or-NaN)
    Returns list of (j_idx, r_j).
    """
    out = []
    g = grain.iloc if hasattr(grain, "iloc") else grain
    lo = pos_i - K_WINDOW
    cand = ev.index[(ev["pos"] >= lo) & (ev["pos"] < pos_i)]
    for j in cand:
        if ev.at[j, "permaTicker"] == pt_i:
            continue
        if grain is not None and g[j] != grain_val_i:
            continue
        if require_beat and not (ev.at[j, "sue_score"] > 0):
            continue
        mj = ev.at[j, f"mat{h}"]
        if pd.isna(mj) or mj > cut_date_i:
            continue
        rj = ev.at[j, f"r{h}"]
        if np.isnan(rj):
            continue
        out.append((j, float(rj)))
    return out


# ------------------------------------------------- similarity estimators ----
def weights_taxonomy(ev, i_idx, j_idxs):
    """S-c: w = 1{same SIC-4}. No estimation -> no leak surface."""
    s = ev.at[i_idx, "sic"]
    return {j: 1.0 for j in j_idxs if pd.notna(s) and ev.at[j, "sic"] == s}


def build_instances(ev, grain_col="sic2"):
    """S-a training instances: ordered ticker-pairs within the grain, both
    beats, gap in [1,K], both r1 defined. Instance maturity = mat1 of the
    LATER event (the label side). F0a: an instance is usable at tau iff
    mat1_later <= tau. x = earlier's r1, y = later's r1 (pooled both
    orderings at the PAIR level by the estimator).
    """
    rows = []
    for _, sub in ev.groupby(grain_col, sort=False):
        if len(sub) < 2:
            continue
        sub = sub[(sub["sue_score"] > 0) & sub["r1"].notna()]
        p = sub["pos"].to_numpy()
        idxs = sub.index.to_numpy()
        for a in range(len(sub)):
            lo = p[a] - K_WINDOW
            for b in range(len(sub)):
                gap = p[b] - p[a]
                if gap < 1 or gap > K_WINDOW:
                    continue
                i_late, j_early = idxs[b], idxs[a]
                rows.append((ev.at[i_late, "permaTicker"],
                             ev.at[j_early, "permaTicker"],
                             float(ev.at[j_early, "r1"]),
                             float(ev.at[i_late, "r1"]),
                             ev.at[i_late, "mat1"]))
    df = pd.DataFrame(rows, columns=["pt_late", "pt_early", "x", "y", "tmat"])
    df["pair"] = list(zip(np.minimum(df.pt_late, df.pt_early),
                          np.maximum(df.pt_late, df.pt_early)))
    return df


def weights_shrunk(ev, i_idx, j_idxs, inst, tau, pt_i):
    """S-a: pooled pair correlation of r1 (x=earlier, y=late), shrunk to the
    prior 1{same SIC-2} with strength SHRINK_K; w clipped to [0, inf);
    candidates = same SIC-2 (instances built within SIC-2)."""
    sub = inst[inst["tmat"] <= tau]
    if sub.empty:
        base = 1.0
        return {j: base for j in j_idxs}
    rel = sub[(sub.pt_late == pt_i) | (sub.pt_early == pt_i)]
    stats = {}
    for pair, gdf in rel.groupby("pair"):
        other = pair[0] if pair[1] == pt_i else pair[1]
        n = len(gdf)
        x, y = gdf.x.to_numpy(), gdf.y.to_numpy()
        if n >= 3 and x.std() > 0 and y.std() > 0:
            corr = float(np.corrcoef(x, y)[0, 1])
        else:
            corr = 0.0
        w = (n / (n + SHRINK_K)) * corr + (SHRINK_K / (n + SHRINK_K)) * 1.0
        stats[other] = max(w, 0.0)
    return {j: stats.get(ev.at[j, "permaTicker"], 1.0) for j in j_idxs}


def weights_svd(ev, i_idx, j_idxs, tau, pt_i, dim=EMB_DIM):
    """S-b: SVD embedding of the (stock x ISO-week) beat-drift matrix,
    cells only from events with mat1 <= tau (PIT). w = clip(cosine,0,1)."""
    ok = ev[(ev["sue_score"] > 0) & ev["r1"].notna()
            & (ev["mat1"] <= tau) & ev["mat1"].notna()]
    if ok.empty:
        return {j: 0.0 for j in j_idxs}
    wk = pd.to_datetime(ok["T"]).dt.isocalendar()
    ok = ok.assign(week=(wk.year.astype(str) + "-W"
                         + wk.week.astype(str).str.zfill(2)))
    M = ok.pivot_table(index="permaTicker", columns="week", values="r1", aggfunc="mean")
    M = M.sub(M.mean(axis=0), axis=1).fillna(0.0)
    if M.shape[1] < dim:
        return {j: 0.0 for j in j_idxs}
    U, S, Vt = np.linalg.svd(M.to_numpy(), full_matrices=False)
    E = U[:, :dim] * S[:dim]
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.where(nrm > 0, nrm, 1.0)
    emb = pd.DataFrame(E, index=M.index)
    if pt_i not in emb.index:
        return {j: 0.0 for j in j_idxs}
    u = emb.loc[pt_i].to_numpy()
    sims = (emb @ u).clip(lower=0.0)
    return {j: float(sims.get(ev.at[j, "permaTicker"], 0.0)) for j in j_idxs}


# ------------------------------------------------------------- features ----
def features_row(ev, i_idx, h, grain_col, estimator, inst=None,
                 prices=None, bench_dates=None, bench_close=None,
                 grain=None, leaky=False):
    """F1..F4 for one event. `leaky=True` injects a deliberate RC-16-style
    bug (peer window extended past the cutoff) for the negative control."""
    pt_i = ev.at[i_idx, "permaTicker"]
    cut_date_i = ev.at[i_idx, "cut_date"]
    pos_i = ev.at[i_idx, "pos"]
    gv = None if grain_col == "any" else ev.at[i_idx, grain_col]
    h_eff = h + 5 if leaky else h
    cand = peers_for(ev, i_idx, h_eff, None if grain_col == "any" else ev[grain_col],
                     gv, cut_date_i, pos_i, pt_i)
    if leaky:
        cand = [(j, rj) for (j, rj) in cand
                if not np.isnan(ev.at[j, f"r{h}"])]
        cand = [(j, float(ev.at[j, f"r{h + 5}"])
                 if f"r{h + 5}" in ev.columns and not np.isnan(ev.at[j, f"r{h + 5}"])
                 else rj) for (j, rj) in cand]
    j_idxs = [j for j, _ in cand]
    r = {j: rj for j, rj in cand}
    if estimator == "taxonomy":
        w = weights_taxonomy(ev, i_idx, j_idxs)
    elif estimator == "shrunk":
        w = weights_shrunk(ev, i_idx, j_idxs, inst, cut_date_i, pt_i)
    elif estimator == "svd":
        w = weights_svd(ev, i_idx, j_idxs, cut_date_i, pt_i)
    else:
        raise ValueError(estimator)
    pairs = [(w[j], r[j]) for j in j_idxs if j in w and w.get(j, 0) > 0]
    if not pairs:
        return {"F1": np.nan, "F2": np.nan, "F3": np.nan, "F4": 0.0,
                "has_peer": False, "n_peers": 0}
    ws = np.array([p[0] for p in pairs])
    rs = np.array([p[1] for p in pairs])
    W = ws.sum()
    F1 = float((ws * rs).sum() / W)
    F2 = float((ws * np.sign(rs)).sum() / W)
    F3 = float((ws * np.abs(rs - F1)).sum() / W)
    F4 = float(W * W / (ws * ws).sum())
    return {"F1": F1, "F2": F2, "F3": F3, "F4": F4,
            "has_peer": True, "n_peers": len(pairs)}


# ------------------------------------------------------- truncation F0b ----
def truncate(prices, bench_dates, bench_close, cut_date):
    """Truncate BARS at the cutoff. Event ROWS are kept: their T/BMO/SIC
    are decision-time metadata; peer eligibility (mat <= cut) and truncated
    prices enforce observability. The target event's own drift becomes NaN
    under truncation (window beyond cut) -- correct and harmless (features
    never read the target's drift)."""
    cp = {}
    for pt, (d, c) in prices.items():
        m = d <= cut_date
        cp[pt] = (d[m], c[m])
    m = bench_dates <= cut_date
    return cp, bench_dates[m], bench_close[m]


def truncation_test(ev_full, prices, bench_dates, bench_close, inst_full,
                    n=100, seed=SEED):
    rng = np.random.default_rng(seed)
    ev_any = ev_full.copy()
    res = {"pass": 0, "fail": 0, "fail_detail": []}
    # sample: events with same-SIC-4 peers (60), without (40) incl. NaN checks
    cand_all = ev_full.index.tolist()
    sample = list(rng.choice(cand_all, size=min(n, len(cand_all)), replace=False))
    for i_idx in sample:
        cut = ev_full.at[i_idx, "cut_date"]
        cp, bd_t, bc_t = truncate(prices, bench_dates, bench_close, cut)
        ev_t = compute_drifts(ev_full.copy(), cp, bd_t, bc_t)
        for h, est in ((1, "taxonomy"), (1, "shrunk"), (1, "svd")):
            a = features_row(ev_full, i_idx, h, "sic4", est, inst_full)
            b = features_row(ev_t, i_idx, h, "sic4", est, inst_full)
            ok = all(
                (np.isnan(a[k]) and np.isnan(b[k]))
                or (isinstance(a[k], float) and abs(a[k] - b[k]) < 1e-12)
                if isinstance(a[k], float) or isinstance(b[k], float)
                else a[k] == b[k]
                for k in ("F1", "F2", "F3", "F4", "has_peer", "n_peers"))
            if ok:
                res["pass"] += 1
            else:
                res["fail"] += 1
                if len(res["fail_detail"]) < 10:
                    res["fail_detail"].append(
                        {"idx": int(i_idx), "h": h, "est": est, "full": a, "trunc": b})
    return res


def negative_control(ev_full, i_idx_list, prices, bench_dates, bench_close,
                     n=60):
    """The harness must FAIL the deliberately leaky variant (peer window
    h+5, ignoring observability). If it passes the leak, the harness is
    broken."""
    rng = np.random.default_rng(SEED + 1)
    sample = list(rng.choice(i_idx_list, size=min(n, len(i_idx_list)), replace=False))
    caught = 0
    tested = 0
    ev_l = ev_full.copy()
    ev_l = compute_drifts(ev_l, prices, bench_dates, bench_close, horizons=(6,))
    for i_idx in sample:
        cut = ev_l.at[i_idx, "cut_date"]
        cp, bd_t, bc_t = truncate(prices, bench_dates, bench_close, cut)
        ev_t = compute_drifts(ev_l.copy(), cp, bd_t, bc_t, horizons=(6,))
        a = features_row(ev_l, i_idx, 1, "sic4", "taxonomy", leaky=True)
        b = features_row(ev_t, i_idx, 1, "sic4", "taxonomy", leaky=True)
        tested += 1
        if a["F1"] != b["F1"] or a["n_peers"] != b["n_peers"]:
            caught += 1
    return {"tested": tested, "leak_caught": caught}


# ---------------------------------------------------------------- main ----
if __name__ == "__main__":
    t0 = time.time()
    log("RC-18 P0 — leak firewall self-test")
    ev = load_events()
    cal, bclose = load_calendar()
    ev["pos"] = np.searchsorted(cal, ev["T"].to_numpy().astype("datetime64[D]"), side="left")
    back = np.where(ev["is_bmo"].astype(bool), 2, 1)
    ev["cut_date"] = cal[np.clip(ev["pos"] - back, 0, len(cal) - 1)]
    log(f"events {len(ev):,} | tickers {ev.permaTicker.nunique()}")

    prices_cache = load_prices()
    log(f"price nodes {len(prices_cache):,} ({time.time()-t0:.0f}s)")

    ev = compute_drifts(ev, prices_cache, cal, bclose)
    ev["sic4"] = ev["sic"]
    ev["sic3"] = ev["sic"] // 10
    ev["sic2"] = ev["sic"] // 100
    log(f"drifts computed ({time.time()-t0:.0f}s) | r1 NaN {ev.r1.isna().mean():.1%} "
        f"| r3 NaN {ev.r3.isna().mean():.1%}")

    # ---- coverage at exact gap requirements (amendment section 2) ----
    cov = {}
    for h in (1, 3):
        for grain, colname in (("SIC-4", "sic4"), ("SIC-3", "sic3"),
                               ("SIC-2", "sic2"), ("sector", "sector"),
                               ("any", None)):
            n_any = 0
            for i_idx, row in ev.iterrows():
                if peers_for(ev, i_idx, h, None if colname is None else ev[colname],
                             None if colname is None else row[colname],
                             row["cut_date"], row["pos"], row["permaTicker"]):
                    n_any += 1
            cov[f"h={h}|{grain}"] = round(n_any / len(ev), 4)
        log(f"coverage h={h}: " + ", ".join(
            f"{g} {cov[f'h={h}|{g}']:.1%}" for g in ("SIC-4", "SIC-3", "SIC-2", "sector", "any")))

    inst = build_instances(ev)
    log(f"instances {len(inst):,} ({time.time()-t0:.0f}s)")

    tt = truncation_test(ev, prices_cache, cal, bclose, inst, n=100)
    log(f"truncation-equivalence: pass {tt['pass']} / fail {tt['fail']}")
    if tt["fail_detail"]:
        log("FAIL DETAIL:", json.dumps(tt["fail_detail"][:2], default=str))

    sub_idx = ev.index[:2000]
    with_peer = [i for i in sub_idx
                 if peers_for(ev, i, 1, ev["sic4"], ev.at[i, "sic4"],
                              ev.at[i, "cut_date"], ev.at[i, "pos"],
                              ev.at[i, "permaTicker"])]
    nc = negative_control(ev, with_peer, prices_cache, cal, bclose)
    log(f"negative control (leaky variant must be caught): "
        f"{nc['leak_caught']}/{nc['tested']} caught")

    # worked example: GWRE 2026-06-04
    gw = ev[(ev["permaTicker"] == ev.loc[ev["canonical_ticker"] == "GWRE",
                                         "permaTicker"].iloc[-1])
            & (ev["report_date"] == pd.Timestamp("2026-06-04"))]
    ex = None
    if len(gw):
        i_idx = gw.index[0]
        ex = {"taxonomy": features_row(ev, i_idx, 1, "sic4", "taxonomy"),
              "shrunk": features_row(ev, i_idx, 1, "sic4", "shrunk", inst)}
        peers = peers_for(ev, i_idx, 1, ev["sic4"], ev.at[i_idx, "sic4"],
                          ev.at[i_idx, "cut_date"], ev.at[i_idx, "pos"],
                          ev.at[i_idx, "permaTicker"])
        ex["peers"] = [{ "ticker": ev.at[j, "canonical_ticker"],
                         "report": str(ev.at[j, "report_date"].date()),
                         "r1": round(rj, 5)} for j, rj in peers]
    log(f"GWRE example: {json.dumps(ex, default=str)}")

    report = {"coverage": cov, "instances": len(inst),
              "truncation": {k: v for k, v in tt.items() if k != "fail_detail"},
              "truncation_fail_detail": tt["fail_detail"],
              "negative_control": nc, "gwre_example": ex,
              "config": {"K": K_WINDOW, "SHRINK_K": SHRINK_K, "TOPK": TOPK,
                         "EMB_DIM": EMB_DIM, "SEED": SEED},
              "seconds": round(time.time() - t0)}
    with open(OUT / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"wrote {OUT / 'report.json'} ({report['seconds']}s)")

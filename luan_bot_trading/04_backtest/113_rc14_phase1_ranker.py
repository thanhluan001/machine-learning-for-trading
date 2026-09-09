"""RC-14 Phase 1 — LambdaMART ranker, arms P (24 pure) / G (27 with gate
probs). SELF-CONTAINED (no import of 51/63/104/108: script-51 double-
import stdout crash). Gates G1-G3 per rc14_phase1_ranker_registration.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB = ROOT / "01_data" / "db.h5"
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
MATRIX_KEY = "/features/train_matrix_combined"

FEATURES = ["sue_lag_1", "sue_lag_2", "car_drift_historical_q1",
            "pre_event_idiosyncratic_vol", "pre_event_volume_trend",
            "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d", "rel_ret_30d",
            "sector_adjusted_ret_20d", "revision_momentum_30d", "revision_momentum_60d",
            "revision_momentum_90d", "revision_ordinal_momentum_90d", "revision_intensity_90d",
            "grade_dispersion_90d", "n_analysts_covering", "last_action_days_before_earnings",
            "consecutive_surprises_pre", "unemployment_roc21", "fed_funds", "vix",
            "is_sp400"]
GATE_COLS = ["p_pass_g1", "p_pass_g2", "p_pass_g3"]
GATES = ["pass_g1", "pass_g2", "pass_g3"]
XLF = {"XLF"}
FOLDS = [(1, "2024-07-01", "2024-12-31"), (2, "2025-01-01", "2025-06-30"),
         (3, "2025-07-01", "2025-12-31"), (4, "2026-01-01", "2026-06-30")]
STOP = 0.10
N_SLOTS = 4

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_PCACHE: dict = {}


def _bulk_load():
    with pd.HDFStore(DB_SP600, "r") as s6:
        for k in s6.keys():
            if k.startswith("/sp600/") and k.count("/") == 2:
                p = s6[k]
                _PCACHE[k.split("/")[-1]] = (
                    pd.to_datetime(p["Date"]).dt.tz_localize(None).dt.normalize().values,
                    p["Adj_Close"].to_numpy(float))
    with pd.HDFStore(DB, "r") as sp:
        for k in sp.keys():
            if k.startswith("/sp400/") and k.count("/") == 2:
                pt = k.split("/")[-1]
                if pt not in _PCACHE:
                    p = sp[k]
                    _PCACHE[pt] = (
                        pd.to_datetime(p["Date"]).dt.tz_localize(None).dt.normalize().values,
                        p["Adj_Close"].to_numpy(float))


def ret_with_stop(closes, e, x, stop=STOP):
    path = closes[e:x + 1]
    peak = path[0]
    for c in path:
        if c / peak - 1.0 <= -stop:
            return -stop
        peak = max(peak, c)
    return float(path[-1] / path[0] - 1.0)


def iso_week(ts):
    ts = pd.Timestamp(ts)
    iso = ts.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def fit(X_train, y_train, X_eval, y_eval):
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=300, learning_rate=0.05, max_depth=2,
        min_child_weight=100, gamma=3, reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1,
    ).fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)


def folds_for(df):
    rd = pd.to_datetime(df["report_date"])
    for fi, (te, sve, tse) in enumerate(FOLDS, 1):
        tr = df[rd <= pd.Timestamp(te)]
        sv = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))]
        ts = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        yield fi, tr, sv, ts


def simulate(slate):
    slots, trades = [], []
    for ev in slate:
        ed = ev["entry_date"]
        kept = []
        for s in slots:
            if s["exit_date"] <= ed:
                trades.append({**s, "return": s["ret5"], "ret_cost": s["ret_cost"],
                               "exit_reason": "natural"})
            else:
                kept.append(s)
        slots = kept
        if len(slots) < N_SLOTS:
            slots.append(ev)
        else:
            scored = []
            for s in slots:
                if iso_week(s["entry_date"]) >= iso_week(ed):
                    continue
                pl = _PCACHE.get(s["permaTicker"])
                if pl is None:
                    continue
                vdates, vcloses = pl
                sidx = int(np.searchsorted(vdates, np.datetime64(ed), side="right")) - 1
                if sidx - s["entry_idx"] < 4:
                    continue
                scored.append((s, sidx, vcloses))
            if scored:
                victim, sidx, vcloses = min(scored, key=lambda z: z[0]["entry_date"])
                part = ret_with_stop(vcloses, victim["entry_idx"], sidx)
                rc = (part if part is not None else victim["ret5"]) - victim["cost"]
                trades.append({**victim, "return": part, "ret_cost": rc,
                               "exit_reason": "force_refresh"})
                slots.remove(victim)
                slots.append(ev)
    for s in slots:
        trades.append({**s, "return": s["ret5"], "ret_cost": s["ret_cost"],
                       "exit_reason": "end"})
    return trades


def spear(x, y):
    if len(x) < 2:
        return np.nan
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    print("=" * 100)
    print("RC-14 PHASE 1 — LambdaMART arms P / G (self-contained)")
    print("=" * 100, flush=True)
    d = pd.read_hdf(DB, MATRIX_KEY)
    s6 = pd.read_hdf(DB_SP600, "/features/train_matrix_sp600_pt",
                     columns=["permaTicker", "report_date", "adv20"])
    s6["report_date"] = pd.to_datetime(s6.report_date)
    d = d.merge(s6, on=["permaTicker", "report_date"], how="left")
    d["adv_pass"] = ~(d.adv20.notna() & (d.adv20 < 1e7))

    frames = []
    for fi, tr, sv, ts in folds_for(d):
        train = pd.concat([tr, sv], ignore_index=True)
        for g in GATES:
            y = train[g].astype(int).to_numpy()
            yt = ts[g].astype(int).to_numpy()
            clf = fit(train[FEATURES], y, ts[FEATURES], yt)
            ts[f"p_{g}"] = clf.predict_proba(ts[FEATURES])[:, 1]
        ts["fold"] = fi
        ts["score"] = ts[[f"p_{g}" for g in GATES]].min(axis=1)
        frames.append(ts)
        print(f"  gate fold {fi} scored", flush=True)
    oos = pd.concat(frames, ignore_index=True)

    _bulk_load()
    m = ((oos.score >= 0.33) & (~oos.sector.isin(XLF)) & oos.pregap_return.notna()
         & oos.adv_pass)
    elig = oos[m].copy()
    elig["entry_date"] = pd.to_datetime(elig.entry_date)
    iso = elig.entry_date.dt.isocalendar()
    elig["wk"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    rows = []
    for r in elig.itertuples(index=False):
        pl = _PCACHE.get(r.permaTicker)
        if pl is None:
            continue
        dates, closes = pl
        e = int(np.searchsorted(dates, np.datetime64(r.entry_date), side="left"))
        x = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(r.exit_date)), side="left"))
        if e >= len(closes) or x >= len(closes) or x <= e:
            continue
        ret = ret_with_stop(closes, e, x)
        if ret is None:
            continue
        rows.append({"permaTicker": r.permaTicker, "entry_date": r.entry_date,
                     "exit_date": pd.Timestamp(dates[x]), "score": r.score,
                     "score_r": np.nan, "is_sp400": int(r.is_sp400),
                     "cost": float(r.cost), "entry_idx": e, "ret5": ret,
                     "ret_cost": ret - float(r.cost), "wk": r.wk, "fold": r.fold,
                     "report_date": r.report_date})
    E = pd.DataFrame(rows)
    print(f"eligible events with rets: {len(E)} | weeks {E.wk.nunique()}", flush=True)

    def eval_scores(df, score_col, label, window):
        sub = df[df[score_col].notna()]
        comp = sub.groupby("wk").filter(lambda g: len(g) >= 2)
        sp = comp.groupby("wk").apply(lambda g: spear(g[score_col].values, g.ret_cost.values)).dropna()
        top1 = comp.sort_values(["wk", score_col], ascending=[True, False]).groupby("wk").head(1)
        print(f"  [{window} | {label}] comp-weeks {comp.wk.nunique()} "
              f"Spearman {sp.mean():+.3f} (n={len(sp)}) | top-1 {top1.ret_cost.mean():+.2%}", flush=True)
        return sp.mean(), top1.ret_cost.mean()

    def sim_with(score_col, fis):
        trades = []
        for fi in fis:
            sub = E[E.fold == fi].copy()
            if score_col != "score":
                sub = sub[sub[score_col].notna()]
            slate = sub.sort_values(["entry_date", score_col],
                                    ascending=[True, False]).rename(
                columns={score_col: "_ord"}).sort_values(
                ["entry_date", "_ord"], ascending=[True, False])
            trades.extend(simulate([r._asdict() for r in slate.itertuples(index=False)]))
        t = pd.DataFrame(trades)
        return t.ret_cost.astype(float).mean() if len(t) else float("nan")

    results = {}
    for arm in ("P", "G"):
        print(f"\n===== ARM {arm} =====", flush=True)
        cols = FEATURES if arm == "P" else FEATURES + GATE_COLS
        for fi in (2, 3, 4):
            tw = E[(E.fold < fi)]
            tw = tw.groupby("wk").filter(lambda g: len(g) >= 2)
            if tw.empty:
                continue
            dd = tw.sort_values("wk").reset_index(drop=True)
            groups = dd.groupby("wk", sort=True).size().to_numpy()
            feat = elig.set_index(["permaTicker", "report_date"])
            Xtr = dd.merge(elig[["permaTicker", "report_date"] + cols],
                           on=["permaTicker", "report_date"], how="left",
                           suffixes=("", "_f"))
            Xc = Xtr[[c for c in cols]].astype(float)
            rk = xgb.XGBRanker(objective="rank:pairwise", max_depth=3,
                               learning_rate=0.05, n_estimators=300,
                               subsample=0.7, colsample_bytree=0.7,
                               random_state=42, n_jobs=-1)
            rk.fit(Xc, dd.ret_cost.astype(float), group=groups, verbose=False)
            test = E[E.fold == fi].merge(
                elig[["permaTicker", "report_date"] + cols],
                on=["permaTicker", "report_date"], how="left", suffixes=("", "_f"))
            test_idx = E.fold == fi
            Xt = test[[c for c in cols]].astype(float)
            E.loc[test_idx, "score_r"] = rk.predict(Xt)
            print(f"  fold {fi}: trained {len(dd)} events / {len(groups)} weeks, "
                  f"scored {int(test_idx.sum())}", flush=True)

        dev = E[E.fold.isin([2, 3])]
        ho = E[E.fold == 4]
        sp_dev, t1_dev = eval_scores(dev, "score_r", f"ranker {arm}", "DEV 2-3")
        sp_ho, t1_ho = eval_scores(ho, "score_r", f"ranker {arm}", "HOLDOUT")
        sp_b_dev, t1_b_dev = eval_scores(dev, "score", "min-gate", "DEV 2-3")
        sp_b_ho, t1_b_ho = eval_scores(ho, "score", "min-gate", "HOLDOUT")
        sim_r_dev, sim_b_dev = sim_with("score_r", [2, 3]), sim_with("score", [2, 3])
        sim_r_ho, sim_b_ho = sim_with("score_r", [4]), sim_with("score", [4])
        print(f"  G3 sim: DEV2-3 ranker {sim_r_dev:+.2%} vs baseline {sim_b_dev:+.2%} | "
              f"HOLDOUT ranker {sim_r_ho:+.2%} vs baseline {sim_b_ho:+.2%}", flush=True)
        g1 = sp_ho >= 0.10
        g2 = (t1_ho >= t1_b_ho + 0.010) and (t1_dev >= t1_b_dev)
        g3 = (sim_r_dev >= sim_b_dev) and (sim_r_ho >= sim_b_ho - 0.005)
        print(f"  G1 {'PASS' if g1 else 'FAIL'} ({sp_ho:+.3f}) | "
              f"G2 {'PASS' if g2 else 'FAIL'} (ho {t1_ho:+.2%} vs {t1_b_ho:+.2%}+1pp; "
              f"dev {t1_dev:+.2%} vs {t1_b_dev:+.2%}) | G3 {'PASS' if g3 else 'FAIL'}", flush=True)
        results[arm] = {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3),
                        "sp_ho": float(sp_ho), "t1_ho": float(t1_ho),
                        "sim_r_dev": float(sim_r_dev), "sim_b_dev": float(sim_b_dev)}

    p_ok = all(results["P"][k] for k in ("G1", "G2", "G3"))
    g_ok = all(results["G"][k] for k in ("G1", "G2", "G3"))
    if p_ok and g_ok:
        verdict = "BOTH PASS -> ARM P (parsimony, pre-stated)"
    elif p_ok:
        verdict = "ARM P only -> P"
    elif g_ok:
        verdict = "ARM G only -> G"
    else:
        verdict = "NEITHER -> RC-14 CLOSED, min-gate ranking retained"
    print(f"\nRC-14 PHASE 1 VERDICT: {verdict}", flush=True)

    out = HERE / "archive" / "experiments" / "gate_decomposition_v6" / "rc14_phase1.json"
    out.write_text(json.dumps({"arms": results, "verdict": verdict}, indent=2),
                   encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()

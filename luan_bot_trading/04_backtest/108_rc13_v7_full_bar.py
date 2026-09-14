"""RC-13 — V7 full-bar validation: 4-slot portfolio sim + stop +
force-refresh on fold-OOS scores; per-universe decomposition; bootstrap
CIs; formal G1/G2/G3 verdicts per rc13_combined_pre_registration.md.

Protocol identical to the validated transfer machinery (104), with:
- candidates = V7 fold-OOS scores (no leakage: each fold scored by a
  model trained strictly on prior data, same folds_for as V6)
- threshold 0.33 (DEV-selected in 107)
- per-event costs (10bp SP400 / 30bp SP600) subtracted from the
  stop-truncated return
- ADV>=10M hard filter on SP600 rows; XLF exclusion throughout
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


bt = load("bt_gate", HERE / "51_hp_theta_sweep_23feat.py")
v104 = load("v104", HERE / "104_rc12b_phase3_transfer.py")
fr = v104.fr

DB = bt.DB
MATRIX_KEY = "/features/train_matrix_combined"
FEATURES = list(bt.DEPLOY_FEATURES) + ["is_sp400"]
GATES = ["pass_g1", "pass_g2", "pass_g3"]
V4_HP = {"gamma": 3, "min_child_weight": 100, "max_depth": 2, "n_estimators": 300}
THRESH = 0.33
XLF = {"XLF"}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def fit(X_train, y_train, X_eval, y_eval):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=V4_HP["n_estimators"], learning_rate=0.05,
        max_depth=V4_HP["max_depth"], min_child_weight=V4_HP["min_child_weight"],
        gamma=V4_HP["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1,
    ).fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)


def folds_for(df):
    rd = df["label_end"] if "label_end" in df.columns else pd.to_datetime(df["report_date"])  # RC-16 F3
    for fi, (te, sve, tse) in enumerate(bt.DEFAULT_FOLDS, 1):
        tr = df[rd <= pd.Timestamp(te)]
        sv = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))]
        ts = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        yield fi, tr, sv, ts


def build_cands(oos, w0, w1):
    m = ((oos.score >= THRESH) & (~oos.sector.isin(XLF)) & oos.pregap_return.notna()
         & oos.adv_pass
         & (pd.to_datetime(oos.report_date) >= pd.Timestamp(w0))
         & (pd.to_datetime(oos.report_date) <= pd.Timestamp(w1)))
    c = oos[m].copy()
    c["entry_date"] = pd.to_datetime(c.entry_date)
    rows = []
    for r in c.itertuples(index=False):
        pl = v104.get_prices(r.permaTicker)
        if pl is None:
            continue
        dates, closes = pl
        e = int(np.searchsorted(dates, np.datetime64(r.entry_date), side="left"))
        x = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(r.exit_date)), side="left"))
        if e >= len(closes) or x >= len(closes) or x <= e:
            continue
        ret = fr.ret_with_stop(closes, e, x, stop=v104.STOP)
        if ret is None:
            continue
        rows.append({"permaTicker": r.permaTicker, "entry_date": r.entry_date,
                     "exit_date": pd.Timestamp(dates[x]), "score": r.score,
                     "is_sp400": int(r.is_sp400), "cost": float(r.cost),
                     "entry_idx": e, "ret5": ret, "ret_cost": ret - float(r.cost)})
    out = pd.DataFrame(rows)
    return out.sort_values(["entry_date", "score"], ascending=[True, False]).reset_index(drop=True)


def simulate(slate):
    slots, trades = [], []
    for ev in slate.itertuples(index=False):
        ed = ev.entry_date
        evd = ev._asdict()
        kept = []
        for s in slots:
            if s["exit_date"] <= ed:
                trades.append({**s, "return": s["ret5"], "ret_cost": s["ret_cost"],
                               "exit_reason": "natural"})
            else:
                kept.append(s)
        slots = kept
        if len(slots) < v104.N_SLOTS:
            slots.append(evd)
        else:
            scored = []
            for s in slots:
                if fr.iso_week(s["entry_date"]) >= fr.iso_week(ed):
                    continue
                pl = v104._PCACHE.get(s["permaTicker"])
                if pl is None:
                    continue
                vdates, vcloses = pl
                sidx = int(np.searchsorted(vdates, np.datetime64(ed), side="right")) - 1
                if sidx - s["entry_idx"] < 4:
                    continue
                scored.append((s, sidx, vcloses))
            if scored:
                victim, sidx, vcloses = min(scored, key=lambda z: z[0]["entry_date"])
                part = fr.ret_with_stop(vcloses, victim["entry_idx"], sidx, stop=v104.STOP)
                rc = (part if part is not None else victim["ret5"]) - victim["cost"]
                trades.append({**victim, "return": part if part is not None else victim["ret5"],
                               "ret_cost": rc, "exit_reason": "force_refresh"})
                slots.remove(victim)
                slots.append(evd)
    for s in slots:
        trades.append({**s, "return": s["ret5"], "ret_cost": s["ret_cost"],
                       "exit_reason": "end"})
    return trades


def report(label, trades):
    df = pd.DataFrame(trades)
    if df.empty:
        print(f"\n[{label}] NO TRADES")
        return df
    r = df["ret_cost"].astype(float)
    nav = float(np.prod(1 + r.sort_index(kind="stable") / v104.N_SLOTS))
    print(f"\n[{label}] trades {len(df)} | win {(r > 0).mean():.0%} | "
          f"cost-adj mean {r.mean():+.3%} | NAV {nav:.3f}x")
    for u, lab in [(1, "SP400"), (0, "SP600")]:
        ru = df[df.is_sp400 == u]["ret_cost"].astype(float)
        if len(ru):
            print(f"    {lab:6s} n={len(ru):3d}  mean {ru.mean():+.3%}  win {(ru > 0).mean():.0%}")
    return df


def boot_ci(x, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return (float("nan"), float("nan"))
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main() -> None:
    print("=" * 100)
    print("RC-13 V7 FULL BAR — portfolio sim on fold-OOS scores, G1-G3 verdicts")
    print("=" * 100)
    d = pd.read_hdf(DB, MATRIX_KEY)
    s6 = pd.read_hdf(v104.DB_SP600, "/features/train_matrix_sp600_pt",
                     columns=["permaTicker", "report_date", "adv20"])
    s6["report_date"] = pd.to_datetime(s6.report_date)
    d = d.merge(s6, on=["permaTicker", "report_date"], how="left")
    d["adv_pass"] = ~(d.adv20.notna() & (d.adv20 < 1e7))
    print(f"events: {len(d):,}")

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
        print(f"  fold {fi} scored: {len(ts):,}")
    oos = pd.concat(frames, ignore_index=True)

    v104._bulk_load()

    dev_all = []
    for i, w0, w1 in v104.FOLDS[:3]:
        t = simulate(fr.weekly_slate(build_cands(oos, w0, w1)))
        report(f"fold {i} {w0[:7]}..{w1[:7]}", t)
        dev_all.extend(t)
    report("DEV folds 1-3 (V6 home +3.08% raw-adj; transfer +1.90%)", dev_all)
    t4 = simulate(fr.weekly_slate(build_cands(oos, "2026-01-01", "2026-06-30")))
    ho = report("HOLDOUT 2026H1 (V6 home +4.15% 1.84x; transfer +3.68% 1.76x)", t4)

    # ---- formal verdicts
    print("\n" + "=" * 70)
    if ho is None or ho.empty:
        print("VERDICT: inconclusive (no holdout trades)")
        return
    hr = ho["ret_cost"].astype(float).to_numpy()
    ci = boot_ci(hr)
    g1 = (hr.mean() > 0) and (ci[0] > 0)
    u4 = ho[ho.is_sp400 == 1]["ret_cost"].astype(float)
    u6 = ho[ho.is_sp400 == 0]["ret_cost"].astype(float)
    g2 = len(u4) and u4.mean() >= 0.021
    g3 = (not len(u6)) or u6.mean() >= 0.0
    print(f"G1 holdout cost-adj {hr.mean():+.3%} CI95 [{ci[0]:+.3%}, {ci[1]:+.3%}] "
          f"-> {'PASS' if g1 else 'FAIL'}")
    print(f"G2 SP400 subset mean {u4.mean() if len(u4) else float('nan'):+.3%} "
          f"(bar +2.1%) n={len(u4)} -> {'PASS' if g2 else 'FAIL'}")
    print(f"G3 SP600 subset mean {u6.mean() if len(u6) else float('nan'):+.3%} "
          f"(bar 0) n={len(u6)} -> {'PASS' if g3 else 'FAIL'}")
    verdict = "PASS" if (g1 and g2 and g3) else "FAIL"
    print(f"\nRC-13 V7 FULL-BAR VERDICT: {verdict}")

    out = HERE / "archive" / "experiments" / "gate_decomposition_v6" / "rc13_v7_full_bar.json"
    res = {"threshold": THRESH,
           "holdout": {"n": int(len(hr)), "mean": float(hr.mean()), "ci95": ci,
                       "sp400_mean": float(u4.mean()) if len(u4) else None,
                       "sp600_mean": float(u6.mean()) if len(u6) else None},
           "verdict": {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3),
                       "overall": verdict}}
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()

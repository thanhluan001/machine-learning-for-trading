"""RC-13 — V7 first-pass gate training on the combined universe.

Adaptation of 53_gate_decomposition_v6 to /features/train_matrix_combined
(33,604 events; FEATURES + is_sp400). Walk-forward folds identical to V6
(bt.DEFAULT_FOLDS). Gate classifiers use V6's frozen hyperparameters
(V4_HP) — a fair challenger inherits the same capacity; tuning comes later
if the first pass clears.

Outputs (rc13_v7_first_pass.json + stdout):
  - per-gate OOS F1/precision/recall, split by universe (is_sp400)
  - FEATURE IMPORTANCES per gate (the is_sp400 readout — pre-registered
    as the SUFFICIENT information probe)
  - min-gate score threshold curve on DEV folds (cost-adjusted returns;
    ADV>=10M filter on SP600 rows where adv20 known)
  - holdout 2026H1 at the DEV-selected threshold -> gates G1/G2/G3
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

DB = bt.DB
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
MATRIX_KEY = "/features/train_matrix_combined"
FEATURES = list(bt.DEPLOY_FEATURES) + ["is_sp400"]
GATES = ["pass_g1", "pass_g2", "pass_g3"]
V4_HP = {"gamma": 3, "min_child_weight": 100, "max_depth": 2, "n_estimators": 300}
XLF = {"XLF"}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def fit(X_train, y_train, X_eval, y_eval, hp=V4_HP):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=hp["n_estimators"], learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1,
    ).fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)


def folds_for(df):
    rd = pd.to_datetime(df["report_date"])
    for fi, (te, sve, tse) in enumerate(bt.DEFAULT_FOLDS, 1):
        tr = df[rd <= pd.Timestamp(te)]
        sv = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))]
        ts = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        yield fi, tr, sv, ts


def main() -> None:
    print("=" * 100)
    print("RC-13 V7 FIRST PASS — combined-universe gate decomposition")
    print("=" * 100)
    d = pd.read_hdf(DB, MATRIX_KEY)
    need = FEATURES + GATES + ["pead_pass", "pregap_return", "cost", "sector",
                               "entry_date", "exit_date", "permaTicker", "report_date"]
    missing = [c for c in need if c not in d]
    assert not missing, f"missing columns: {missing}"
    # adv20 join for SP600 rows (execution filter)
    s6 = pd.read_hdf(DB_SP600, "/features/train_matrix_sp600_pt",
                     columns=["permaTicker", "report_date", "adv20"])
    s6["report_date"] = pd.to_datetime(s6.report_date)
    d = d.merge(s6.rename(columns={"adv20": "adv20"}), on=["permaTicker", "report_date"],
                how="left")
    d["adv_pass"] = ~(d.adv20.notna() & (d.adv20 < 1e7))
    d["ret_cost"] = d.pregap_return - d.cost
    d["entry_date"] = pd.to_datetime(d.entry_date)
    print(f"events: {len(d):,} | SP400 {int((d.is_sp400==1).sum()):,} "
          f"| SP600 {int((d.is_sp400==0).sum()):,}")

    # ---- train the 3 gates walk-forward, collect OOS scores + importances
    imps = {g: [] for g in GATES}
    frames = []
    for fi, tr, sv, ts in folds_for(d):
        train = pd.concat([tr, sv], ignore_index=True)
        for g in GATES:
            y = train[g].astype(int).to_numpy()
            yt = ts[g].astype(int).to_numpy()
            clf = fit(train[FEATURES], y, ts[FEATURES], yt)
            ts[f"p_{g}"] = clf.predict_proba(ts[FEATURES])[:, 1]
            imps[g].append(pd.Series(clf.feature_importances_, index=FEATURES))
        ts["fold"] = fi
        ts["score"] = ts[[f"p_{g}" for g in GATES]].min(axis=1)
        frames.append(ts)
    oos = pd.concat(frames, ignore_index=True)
    print(f"OOS events scored: {len(oos):,}")
    # ---- per-gate stats by universe
    for g in GATES:
        for u, lab in [(1, "SP400"), (0, "SP600")]:
            sub = oos[(oos.is_sp400 == u) & oos[f"p_{g}"].notna()]
            pick = sub[f"p_{g}"] >= 0.20
            y = sub[g].astype(int).to_numpy()
            tp = int((pick.to_numpy() & (y == 1)).sum())
            fp = int((pick.to_numpy() & (y == 0)).sum())
            prec = tp / (tp + fp) * 100 if tp + fp else 0.0
            print(f"  {g} [{lab}] n={len(sub):,} picks={int(pick.sum()):,} "
                  f"precision@0.20 {prec:.1f}% base {y.mean()*100:.1f}%")
    # ---- importances (mean over folds)
    print("\nFEATURE IMPORTANCE (mean across folds), top 10 + is_sp400:")
    for g in GATES:
        mi = pd.concat(imps[g], axis=1).mean(axis=1).sort_values(ascending=False)
        line = ", ".join(f"{k}={v:.3f}" for k, v in mi.head(10).items())
        print(f"  {g}: {line}")
        print(f"      is_sp400 rank {int((mi > mi['is_sp400']).sum())+1}/24 "
              f"importance {mi['is_sp400']:.4f}")

    # ---- threshold curve on DEV (folds 1-3), cost-adjusted, ADV filter
    dev = oos[(oos.fold <= 3) & oos.adv_pass & ~oos.sector.isin(XLF)].copy()
    print(f"\nDEV pool (XLF-excl, ADV-pass): {len(dev):,}")
    best = None
    for th in [0.30, 0.32, 0.33, 0.35, 0.38, 0.40, 0.43, 0.45]:
        ex = dev[dev.score >= th]
        if not len(ex):
            continue
        nav, iso = 1.0, ex.entry_date.dt.isocalendar()
        ex = ex.assign(wk=iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2))
        for _, w in ex.groupby("wk", sort=True):
            nav *= 1.0 + float((w["ret_cost"] / bt.N_SLOTS).sum())
        m = ex.ret_cost.mean()
        u4 = ex[ex.is_sp400 == 1].ret_cost
        u6 = ex[ex.is_sp400 == 0].ret_cost
        print(f"  th={th:.2f}: n={len(ex):3d} mean {m:+.2%} win {(ex.ret_cost>0).mean():.0%} "
              f"NAV {nav:.2f}x | SP400 n={len(u4)} {u4.mean() if len(u4) else float('nan'):+.2%} "
              f"| SP600 n={len(u6)} {u6.mean() if len(u6) else float('nan'):+.2%}")
        if len(ex) >= 50 and (best is None or nav > best[1]):
            best = (th, nav)
    th_sel = best[0] if best else 0.33
    print(f"\nDEV-selected threshold: {th_sel:.2f} (max NAV, >=50 trades)")

    # ---- holdout 2026H1 at selected threshold -> pre-registered gates
    ho = oos[(oos.fold == 4) & oos.adv_pass & ~oos.sector.isin(XLF)]
    ex = ho[ho.score >= th_sel]
    nav, iso = 1.0, ex.entry_date.dt.isocalendar()
    ex = ex.assign(wk=iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2))
    for _, w in ex.groupby("wk", sort=True):
        nav *= 1.0 + float((w["ret_cost"] / bt.N_SLOTS).sum())
    u4 = ex[ex.is_sp400 == 1].ret_cost
    u6 = ex[ex.is_sp400 == 0].ret_cost
    print(f"\nHOLDOUT 2026H1 @ {th_sel:.2f}: n={len(ex)} mean {ex.ret_cost.mean():+.2%} "
          f"win {(ex.ret_cost>0).mean():.0%} NAV {nav:.2f}x")
    print(f"  SP400 subset: n={len(u4)} mean {u4.mean() if len(u4) else float('nan'):+.2%} (G2 >= +2.1%)")
    print(f"  SP600 subset: n={len(u6)} mean {u6.mean() if len(u6) else float('nan'):+.2%} (G3 >= 0)")
    print(f"  G1: holdout mean {ex.ret_cost.mean():+.2%} > 0 (bootstrap CI next stage)")

    out = HERE / "archive" / "experiments" / "gate_decomposition_v6" / "rc13_v7_first_pass.json"
    res = {
        "matrix": MATRIX_KEY, "threshold": th_sel,
        "dev_nav": float(best[1]) if best else None,
        "holdout": {"n": int(len(ex)), "mean": float(ex.ret_cost.mean()) if len(ex) else None,
                    "sp400_mean": float(u4.mean()) if len(u4) else None,
                    "sp600_mean": float(u6.mean()) if len(u6) else None},
        "importance_is_sp400": {g: float(pd.concat(imps[g], axis=1).mean(axis=1)["is_sp400"])
                                for g in GATES},
    }
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nsaved", out)


if __name__ == "__main__":
    main()

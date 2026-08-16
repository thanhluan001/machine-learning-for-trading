#!/usr/bin/env python3
"""Nested walk-forward validation for the v6 three-gate PEAD decomposition.

For each outer fold:
  1. Select each gate model's HP on TRAIN -> SWEEP validation.
  2. Select ensemble rule/threshold on the same SWEEP predictions.
  3. Refit each selected gate model on TRAIN + SWEEP.
  4. Evaluate exactly once on the untouched outer TEST window.

No production artifacts or HDF5 nodes are written.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parents[2])
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent

def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bt = load("bt_nested", HERE / "51_hp_theta_sweep_23feat.py")
DB = bt.DB
MATRIX = "/features/train_matrix_v4_timing_correct"
FEATURES = bt.DEPLOY_FEATURES
GATES = ["pass_g1", "pass_g2", "pass_g3"]
V4_HP = {"gamma": 3, "min_child_weight": 100, "max_depth": 2, "n_estimators": 300}
GRID = list(product([1, 3, 5, 8, 12], [20, 50, 100, 200], [2, 3, 4]))
THRESHOLDS = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def fit(X, y, Xe, ye, hp):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=hp["n_estimators"], learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1,
    ).fit(X, y, eval_set=[(Xe, ye)], verbose=False)


def f1_for(y, p, threshold):
    y = np.asarray(y, dtype=int); take = np.asarray(p) >= threshold
    tp = int((take & (y == 1)).sum()); fp = int((take & (y == 0)).sum())
    fn = int((~take & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"f1": f1 * 100, "precision": prec * 100, "recall": rec * 100,
            "picks": int(take.sum()), "tp": tp}


def split(df, fi):
    te, sve, tse = bt.DEFAULT_FOLDS[fi - 1]
    rd = pd.to_datetime(df["report_date"])
    return (
        df[rd <= pd.Timestamp(te)].copy(),
        df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy(),
        df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy(),
    )


def choose_ensemble(sweep):
    best = None
    for rule in ["product", "minimum", "hard"]:
        score = sweep[["p1", "p2", "p3"]].prod(axis=1) if rule == "product" else sweep[["p1", "p2", "p3"]].min(axis=1)
        for threshold in THRESHOLDS:
            s = f1_for(sweep["pead_pass"], score if rule != "hard" else sweep["p1"], threshold)
            if rule == "hard":
                take = (sweep[["p1", "p2", "p3"]] >= threshold).all(axis=1)
                y = sweep["pead_pass"].to_numpy(int)
                tp = int((take.to_numpy() & (y == 1)).sum()); fp = int((take.to_numpy() & (y == 0)).sum()); fn = int((~take.to_numpy() & (y == 1)).sum())
                pr = tp / (tp + fp) if tp + fp else 0.; re = tp / (tp + fn) if tp + fn else 0.
                s = {"f1": 2 * pr * re / (pr + re) * 100 if pr + re else 0., "precision": pr * 100, "recall": re * 100, "picks": int(take.sum()), "tp": tp}
            # Tie-break toward higher precision, then fewer picks.
            key = (s["f1"], s["precision"], -s["picks"])
            if best is None or key > best["key"]:
                best = {"rule": rule, "threshold": threshold, "sweep_metric": s, "key": key}
    best.pop("key")
    return best


def portfolio(ex):
    if ex.empty:
        return {"executed": 0, "wins": 0, "losses": 0, "wr": 0., "avg": 0., "nav": 0., "min_fold": 0., "fold_navs": [], "precision": 0.}
    r = ex["pregap_return"].astype(float)
    z = ex.copy(); iso = z["entry_date"].dt.isocalendar(); z["wk"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    nav = 1.
    for _, w in z.groupby("wk", sort=True): nav *= 1. + float((w["pregap_return"] / bt.N_SLOTS).sum())
    fold_navs = []
    for fi in range(1, 5):
        q = z[z["fold"] == fi]
        if q.empty: fold_navs.append(0.); continue
        nf = 1.;
        for _, w in q.groupby("wk", sort=True): nf *= 1. + float((w["pregap_return"] / bt.N_SLOTS).sum())
        fold_navs.append((nf - 1) * 100)
    return {"executed": int(len(r)), "wins": int((r > 0).sum()), "losses": int((r <= 0).sum()),
            "wr": float((r > 0).mean() * 100), "avg": float(r.mean() * 100), "nav": float((nav - 1) * 100),
            "min_fold": float(min(fold_navs)), "fold_navs": fold_navs,
            "precision": float(ex["pead_pass"].mean() * 100)}


def evaluate_selected(test, choice):
    if choice["rule"] == "product": score = test[["p1", "p2", "p3"]].prod(axis=1); take = score >= choice["threshold"]
    elif choice["rule"] == "minimum": score = test[["p1", "p2", "p3"]].min(axis=1); take = score >= choice["threshold"]
    else: take = (test[["p1", "p2", "p3"]] >= choice["threshold"]).all(axis=1)
    take &= ~test["sector"].isin(bt.EXCLUDE_SECTORS)
    raw = test[take & test["pregap_return"].notna()].copy()
    if raw.empty: return raw, {"raw_picks": 0, "raw_precision": 0., "raw_recall": 0.}
    raw["p"] = (raw[["p1", "p2", "p3"]].prod(axis=1) if choice["rule"] == "product" else raw[["p1", "p2", "p3"]].min(axis=1))
    raw["entry_date"] = pd.to_datetime(raw["pregap_entry_date"]); raw["exit_date"] = pd.to_datetime(raw["pregap_exit_date"])
    raw_recall = float(raw["pead_pass"].sum() / test["pead_pass"].sum() * 100) if test["pead_pass"].sum() else 0.
    return raw, {"raw_picks": int(len(raw)), "raw_precision": float(raw["pead_pass"].mean() * 100), "raw_recall": raw_recall}


def json_default(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main():
    print("=" * 100); print("NESTED GATE DECOMPOSITION: HP + ensemble selected before each outer test"); print("=" * 100)
    df = pd.read_hdf(DB, MATRIX).reset_index(drop=False).rename(columns={"index": "_row_id"})
    required = FEATURES + GATES + ["pead_pass", "pregap_return", "pregap_entry_date", "pregap_exit_date", "sector"]
    missing = [c for c in required if c not in df]
    if missing: raise RuntimeError(missing)
    fold_reports = []; all_raw = []
    for fi in range(1, 5):
        tr, sv, te = split(df, fi)
        print(f"\nOUTER FOLD {fi}: train={len(tr)} sweep={len(sv)} test={len(te)}")
        gate_hp = {}; sweep = sv[["_row_id", "pead_pass"]].copy(); test = te.copy()
        for gate in GATES:
            rows = []
            best_pred = None
            for gamma, mcw, md in GRID:
                hp = {"gamma": gamma, "min_child_weight": mcw, "max_depth": md, "n_estimators": 300}
                model = fit(tr[FEATURES], tr[gate].astype(int), sv[FEATURES], sv[gate].astype(int), hp)
                p = model.predict_proba(sv[FEATURES])[:, 1]
                m = f1_for(sv[gate], p, .20)
                rows.append((m["f1"], m["precision"], -m["picks"], hp, p))
            rows.sort(key=lambda x: x[:3], reverse=True)
            _, _, _, hp, p_sv = rows[0]
            gate_hp[gate] = hp
            sweep["p" + gate[-1]] = p_sv
            final = fit(pd.concat([tr[FEATURES], sv[FEATURES]]), pd.concat([tr[gate], sv[gate]]).astype(int), te[FEATURES], te[gate].astype(int), hp)
            test["p" + gate[-1]] = final.predict_proba(te[FEATURES])[:, 1]
            print(f"  {gate}: HP={hp}, sweep F1={rows[0][0]:.2f}, sweep precision={rows[0][1]:.2f}")
        choice = choose_ensemble(sweep)
        raw, raw_stats = evaluate_selected(test, choice)
        if not raw.empty:
            raw["fold"] = fi; all_raw.append(raw)
        report = {"fold": fi, "train": len(tr), "sweep": len(sv), "test": len(te), "gate_hp": gate_hp, "choice": choice, **raw_stats}
        fold_reports.append(report)
        print(f"  choice={choice['rule']} threshold={choice['threshold']}; sweep F1={choice['sweep_metric']['f1']:.2f}; test raw={raw_stats}")
    ex_parts = []
    if all_raw:
        raw = pd.concat(all_raw, ignore_index=True)
        for fi in range(1, 5):
            q = raw[raw["fold"] == fi]
            if not q.empty: ex_parts.append(bt.select_weekly(q, bt.N_SLOTS))
    ex = pd.concat(ex_parts, ignore_index=True) if ex_parts else pd.DataFrame()
    stats = portfolio(ex)
    # Baseline is fixed v4, computed by the same implementation.
    base = bt.run_cv(df, FEATURES, V4_HP, .20)
    print("\nNESTED GATE PORTFOLIO:", stats)
    print("FIXED V4 BASELINE:", base)
    print("FOLD REPORTS:")
    for r in fold_reports: print(r)
    out = HERE / "archive" / "experiments" / "gate_decomposition_v6"
    out.mkdir(parents=True, exist_ok=True)
    result = {"model_version": "phase_g_v6_gate_decomposition", "matrix": MATRIX, "features": FEATURES, "outer_folds": fold_reports, "nested_portfolio": stats, "fixed_v4_baseline": base, "hp_grid_size": len(GRID), "thresholds": THRESHOLDS,
              "executed_trades": ex.to_dict(orient="records") if not ex.empty else []}
    with open(out / "nested_results.json", "w", encoding="utf-8") as f: json.dump(result, f, indent=2, default=json_default)
    print(f"Saved {out / 'nested_results.json'}")

if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Research experiment: v6 decomposes PEAD into three independently predicted gates.

Uses the persisted timing-correct v4 matrix and its exact gate labels. This is
research-only: no production artifacts or HDF nodes are written.

Models:
  g1: CAR > +3%
  g2: event volume ratio > 2x baseline
  g3: market-adjusted MaxDD > -1.5%

The script first compares fixed v4 HP, then runs a descriptive 60-point HP
sweep per gate. HPs selected from the pooled OOS results are explicitly marked
exploratory; they must not be promoted without a nested validation rerun.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
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
ROOT = HERE.parent

def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bt = load("bt_gate", HERE / "51_hp_theta_sweep_23feat.py")

DB = bt.DB
MATRIX_KEY = "/features/train_matrix_v4_timing_correct"
FEATURES = bt.DEPLOY_FEATURES
GATES = ["pass_g1", "pass_g2", "pass_g3"]
GATE_NAMES = {
    "pass_g1": "CAR > +3%",
    "pass_g2": "Volume > 2x",
    "pass_g3": "MaxDD > -1.5%",
}
V4_HP = {"gamma": 3, "min_child_weight": 100, "max_depth": 2, "n_estimators": 300}
THETA = 0.20
HP_GRID = list(product([1, 3, 5, 8, 12], [20, 50, 100, 200], [2, 3, 4]))


def fit(X_train, y_train, X_eval, y_eval, hp):
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


def predictions(df, target, hp):
    out = []
    for fi, tr, sv, ts in folds_for(df):
        train = pd.concat([tr, sv], ignore_index=True)
        y = train[target].astype(int).to_numpy()
        yt = ts[target].astype(int).to_numpy()
        clf = fit(train[FEATURES], y, ts[FEATURES], yt, hp)
        ts["p"] = clf.predict_proba(ts[FEATURES])[:, 1]
        ts["fold"] = fi
        out.append(ts)
    return pd.concat(out, ignore_index=True)


def binary_stats(pred, target, threshold=.20):
    pick = pred["p"] >= threshold
    y = pred[target].astype(int).to_numpy()
    p = pick.to_numpy()
    tp = int((p & (y == 1)).sum())
    fp = int((p & (y == 0)).sum())
    fn = int((~p & (y == 1)).sum())
    tn = int((~p & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": int(len(pred)), "picks": int(p.sum()), "tp": tp, "fp": fp,
            "tn": tn, "fn": fn, "precision": precision * 100,
            "recall": recall * 100, "f1": f1 * 100,
            "positive_rate": float(y.mean() * 100)}


def _nav_stats(ex):
    if ex.empty:
        return {"executed": 0, "wins": 0, "losses": 0, "wr": 0., "avg": 0.,
                "nav": 0., "min_fold": 0., "fold_navs": []}
    r = ex["pregap_return"].astype(float)
    nav = 1.0
    fold_navs = []
    for fi in range(1, 5):
        sub = ex[ex["fold"] == fi].copy()
        if sub.empty:
            fold_navs.append(0.0)
            continue
        iso = sub["entry_date"].dt.isocalendar()
        sub["wk"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
        nf = 1.0
        for _, w in sub.groupby("wk", sort=True):
            nf *= 1.0 + float((w["pregap_return"] / bt.N_SLOTS).sum())
        fold_navs.append((nf - 1) * 100)
    iso = ex["entry_date"].dt.isocalendar()
    z = ex.copy()
    z["wk"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    for _, w in z.groupby("wk", sort=True):
        nav *= 1.0 + float((w["pregap_return"] / bt.N_SLOTS).sum())
    wins = r[r > 0]
    return {"executed": int(len(r)), "wins": int((r > 0).sum()),
            "losses": int((r <= 0).sum()), "wr": float((r > 0).mean() * 100),
            "avg": float(r.mean() * 100), "nav": float((nav - 1) * 100),
            "min_fold": float(min(fold_navs)), "fold_navs": fold_navs,
            "precision": float(ex["pead_pass"].mean() * 100)}


def portfolio_stats(pred, score, rule):
    x = pred.copy()
    if rule == "product":
        mask = x[score] >= THETA
    elif rule == "min":
        mask = x[score] >= THETA
    elif rule == "hard":
        mask = (x["p1"] >= THETA) & (x["p2"] >= THETA) & (x["p3"] >= THETA)
    elif rule == "single":
        mask = x["p"] >= THETA
    else:
        raise ValueError(rule)
    mask &= ~x["sector"].isin(bt.EXCLUDE_SECTORS)
    raw = x[mask].copy()
    # select_weekly expects a ranking column named `p`; use the ensemble
    # score for product/minimum and a neutral score for hard conjunction.
    if rule == "product":
        raw["p"] = raw["product"]
    elif rule == "min":
        raw["p"] = raw["minimum"]
    elif rule == "hard":
        raw["p"] = raw[["p1", "p2", "p3"]].min(axis=1)
    raw["entry_date"] = pd.to_datetime(raw["pregap_entry_date"])
    raw["exit_date"] = pd.to_datetime(raw["pregap_exit_date"])
    raw = raw[raw["pregap_return"].notna()]
    pieces = []
    for fi in range(1, 5):
        q = raw[raw["fold"] == fi]
        if not q.empty:
            pieces.append(bt.select_weekly(q, bt.N_SLOTS))
    ex = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    out = _nav_stats(ex)
    out["raw_picks"] = int(len(raw))
    out["raw_precision"] = float(raw["pead_pass"].mean() * 100) if len(raw) else 0.
    out["raw_recall"] = float(raw["pead_pass"].sum() / pred["pead_pass"].sum() * 100) if pred["pead_pass"].sum() else 0.
    return out


def ensemble_threshold_sweep(pred):
    rows = []
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    for t in thresholds:
        for rule in ["product", "min", "hard"]:
            if rule == "product": score = "product"
            elif rule == "min": score = "minimum"
            else: score = "hard"
            x = pred.copy()
            x["product"] = x["p1"] * x["p2"] * x["p3"]
            x["minimum"] = x[["p1", "p2", "p3"]].min(axis=1)
            old = THETA
            # portfolio_stats uses the module threshold; set locally for sweep.
            globals()["THETA"] = t
            s = portfolio_stats(x, score, rule)
            globals()["THETA"] = old
            s.update({"rule": rule, "threshold": t})
            rows.append(s)
    return rows


def main():
    print("=" * 100)
    print("GATE DECOMPOSITION: three classifiers on persisted v4 gate labels")
    print("=" * 100)
    df = pd.read_hdf(DB, MATRIX_KEY)
    missing = [c for c in FEATURES + GATES + ["pead_pass", "pregap_return", "pregap_entry_date", "pregap_exit_date", "sector"] if c not in df]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    df["report_date"] = pd.to_datetime(df["report_date"])
    print(f"Rows={len(df):,}; features={len(FEATURES)}")
    for g in GATES + ["pead_pass"]:
        print(f"  {g}: {int(df[g].sum()):,} ({df[g].mean()*100:.2f}%)")
    print("Label identity:", bool((df["pead_pass"] == df[GATES].all(axis=1).astype(int)).all()))

    # Fixed HP predictions for exact v4-style comparison.
    print("\n[1] Fixed-HP gate models ...")
    # Rebuild robustly with a stable row key because duplicate ticker/date rows
    # can exist. Start from the single-model OOS frame, not the full matrix:
    # all metrics must use only the four held-out test windows.
    base = df.reset_index(drop=False).rename(columns={"index": "_row_id"})
    single_oos = predictions(base, "pead_pass", V4_HP)
    pred = single_oos.rename(columns={"p": "p_single"}).copy()
    for gate, prob_col in zip(GATES, ["p1", "p2", "p3"]):
        q = predictions(base, gate, V4_HP)[["_row_id", "p"]].rename(columns={"p": prob_col})
        pred = pred.merge(q, on="_row_id", how="inner")
    pred["p"] = pred["p_single"]
    pred["product"] = pred[["p1", "p2", "p3"]].prod(axis=1)
    pred["minimum"] = pred[["p1", "p2", "p3"]].min(axis=1)
    print("Fixed gate model metrics:")
    for g, prob_col in zip(GATES, ["p1", "p2", "p3"]):
        z = pred.copy(); z["p"] = z[prob_col]
        print(g, binary_stats(z, g, THETA))
    z = pred.copy(); z["p"] = z["p_single"]
    print("v4 single", binary_stats(z, "pead_pass", THETA))
    print("Fixed ensemble portfolio:")
    fixed_portfolio = {}
    for rule, score in [("product", "product"), ("min", "minimum"), ("hard", "hard")]:
        fixed_portfolio[rule] = portfolio_stats(pred, score, rule)
        print(rule, fixed_portfolio[rule])
    single_pred = pred.copy(); single_pred["p"] = single_pred["p_single"]
    fixed_portfolio["single"] = portfolio_stats(single_pred, "p", "single")
    print("single", fixed_portfolio["single"])

    # Descriptive HP sweep for each gate. Store only aggregate rows, not models.
    print("\n[2] Descriptive 60-combination HP sweep per gate ...")
    sweep = {}
    for gate in GATES:
        rows = []
        start = time.time()
        for i, (gamma, mcw, md) in enumerate(HP_GRID, 1):
            hp = {"gamma": gamma, "min_child_weight": mcw, "max_depth": md, "n_estimators": 300}
            p = predictions(df, gate, hp)
            st = binary_stats(p, gate, THETA)
            rows.append({"gate": gate, "gamma": gamma, "mcw": mcw, "md": md, **st})
            if i % 20 == 0:
                print(f"  {gate} [{i}/{len(HP_GRID)}] {time.time()-start:.1f}s")
        rows.sort(key=lambda r: (r["f1"], r["precision"]), reverse=True)
        sweep[gate] = rows
        print(f"Top {gate} by pooled OOS F1:")
        for r in rows[:5]: print(r)

    selected = {g: {"gamma": sweep[g][0]["gamma"], "min_child_weight": sweep[g][0]["mcw"], "max_depth": sweep[g][0]["md"], "n_estimators": 300} for g in GATES}
    print("\n[3] Refit selected exploratory HPs and evaluate ensembles ...")
    tuned = predictions(base, "pead_pass", V4_HP).rename(columns={"p": "p_single"}).copy()
    for gate, prob_col in zip(GATES, ["p1", "p2", "p3"]):
        q = predictions(base, gate, selected[gate])[["_row_id", "p"]].rename(columns={"p": prob_col})
        tuned = tuned.merge(q, on="_row_id", how="left")
    tuned["p"] = tuned["p_single"]
    tuned["product"] = tuned[["p1", "p2", "p3"]].prod(axis=1)
    tuned["minimum"] = tuned[["p1", "p2", "p3"]].min(axis=1)
    print("Selected HP:", selected)
    tuned_results = ensemble_threshold_sweep(tuned)
    for rule in ["product", "min", "hard"]:
        candidates = [r for r in tuned_results if r["rule"] == rule]
        print(f"Top {rule} by raw precision:")
        for r in sorted(candidates, key=lambda x: (x["raw_precision"], x["nav"]), reverse=True)[:3]: print(r)
        print(f"Top {rule} by NAV:")
        for r in sorted(candidates, key=lambda x: (x["nav"], x["raw_precision"]), reverse=True)[:3]: print(r)

    fixed_gate_metrics = {}
    for g, prob_col in zip(GATES, ["p1", "p2", "p3"]):
        z = pred.copy(); z["p"] = z[prob_col]
        fixed_gate_metrics[g] = binary_stats(z, g, THETA)
    single_z = pred.copy(); single_z["p"] = single_z["p_single"]
    output = {
        "matrix": MATRIX_KEY, "features": FEATURES, "gate_labels": GATE_NAMES,
        "v4_hp": V4_HP, "theta": 0.20, "hp_grid_size": len(HP_GRID),
        "label_identity": True,
        "oos_rows": int(len(pred)),
        "fixed_gate_metrics": fixed_gate_metrics,
        "fixed_single_metrics": binary_stats(single_z, "pead_pass", THETA),
        "fixed_portfolio": fixed_portfolio,
        "selected_exploratory_hp": selected,
        "hp_sweep": sweep,
        "tuned_ensemble_thresholds": tuned_results,
    }
    out = HERE / "archive" / "experiments" / "gate_decomposition_v6"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved research results: {out / 'results.json'}")
    print("No production model or matrix was written.")


if __name__ == "__main__":
    main()

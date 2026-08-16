#!/usr/bin/env python3
"""Freeze the three V6 gate classifiers for shadow paper trading.

This is a shadow candidate artifact, not the live V4 model. It uses the frozen
V6 policy HPs and the complete persisted v4 timing-correct training matrix.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "01_data" / "db.h5"
MATRIX = "/features/train_matrix_v4_timing_correct"
OUT = HERE / "models" / "phase_g_v6_gate_decomposition"
FEATURES = [
    "sue_lag_1", "sue_lag_2", "car_drift_historical_q1",
    "pre_event_idiosyncratic_vol", "pre_event_volume_trend",
    "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d", "rel_ret_30d",
    "sector_adjusted_ret_20d", "revision_momentum_30d", "revision_momentum_60d",
    "revision_momentum_90d", "revision_ordinal_momentum_90d", "revision_intensity_90d",
    "grade_dispersion_90d", "n_analysts_covering", "last_action_days_before_earnings",
    "consecutive_surprises_pre", "unemployment_roc21", "fed_funds", "vix",
]
POLICY = {
    "pass_g1": {"label": "CAR > +3%", "gamma": 8, "min_child_weight": 20, "max_depth": 3, "n_estimators": 300},
    "pass_g2": {"label": "event volume ratio > 2x baseline", "gamma": 12, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300},
    "pass_g3": {"label": "market-adjusted MaxDD > -1.5%", "gamma": 1, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300},
}
COMMON = {"learning_rate": 0.05, "reg_lambda": 1.0, "subsample": 0.7, "colsample_bytree": 0.7, "random_state": 42, "n_jobs": -1}


def main():
    with pd.HDFStore(DB, "r") as store:
        df = store[MATRIX]
    missing = [c for c in FEATURES + list(POLICY) if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    artifact = {"model_version": "phase_g_v6_gate_decomposition", "status": "shadow_candidate_not_live", "matrix": MATRIX, "features": FEATURES, "rows": int(len(df)), "gates": {}}
    for gate, hp in POLICY.items():
        params = {**COMMON, **{k: hp[k] for k in ["gamma", "min_child_weight", "max_depth", "n_estimators"]}}
        model = xgb.XGBClassifier(objective="binary:logistic", eval_metric=["logloss", "auc"], **params)
        model.fit(df[FEATURES], df[gate].astype(int), eval_set=[(df[FEATURES], df[gate].astype(int))], verbose=False)
        path = OUT / gate / "classifier.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))
        artifact["gates"][gate] = {**hp, **COMMON, "classifier": str(path.relative_to(HERE.parent.parent))}
        print(f"Saved {path}")
    with open(OUT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"Saved {OUT / 'meta.json'}")
    print("V4 production artifact was not modified.")

if __name__ == "__main__":
    main()

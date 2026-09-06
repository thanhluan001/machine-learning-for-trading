#!/usr/bin/env python3
"""Freeze the three V7 combined-universe gate classifiers for shadow
paper trading.

Mirrors 08_freeze_v6_gate_models conventions. Deployment retrain on the
FULL combined matrix (validation was fold-based; once passed, the
deployment artifact retrains on all available data — the V6 precedent).

Configuration = EXACTLY what was validated (rc13 pre-registration):
- uniform V4_HP for all three gates (inherited, no tuning)
- features: V6's 23 + is_sp400 (pt-in-time membership flag)
- labels: net-of-cost thresholds (10bp SP400 / 30bp SP600, in-matrix)

Validation dossier: rc13_v7_full_bar.json (G1-G3 PASS),
rc13_v7_threshold_sweep.json (0.33 robust, plateau 0.28-0.43).
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "01_data" / "db.h5"
MATRIX = "/features/train_matrix_combined"
OUT = HERE / "models" / "phase_g_v7_combined"
FEATURES = [
    "sue_lag_1", "sue_lag_2", "car_drift_historical_q1",
    "pre_event_idiosyncratic_vol", "pre_event_volume_trend",
    "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d", "rel_ret_30d",
    "sector_adjusted_ret_20d", "revision_momentum_30d", "revision_momentum_60d",
    "revision_momentum_90d", "revision_ordinal_momentum_90d", "revision_intensity_90d",
    "grade_dispersion_90d", "n_analysts_covering", "last_action_days_before_earnings",
    "consecutive_surprises_pre", "unemployment_roc21", "fed_funds", "vix",
    "is_sp400",
]
UNIFORM_HP = {"gamma": 3, "min_child_weight": 100, "max_depth": 2, "n_estimators": 300}
GATE_LABELS = {
    "pass_g1": "CAR > +3% (net of 10/30bp cost)",
    "pass_g2": "event volume ratio > 2x baseline",
    "pass_g3": "market-adjusted MaxDD > -1.5% (net of cost)",
}
COMMON = {"learning_rate": 0.05, "reg_lambda": 1.0, "subsample": 0.7,
          "colsample_bytree": 0.7, "random_state": 42, "n_jobs": -1}
VALIDATION = {
    "full_bar": "rc13_v7_full_bar.json: G1 PASS (+4.26% CI[+1.41,+7.40]) "
                "G2 PASS (SP400 +3.54%>=2.1%) G3 PASS (SP600 +5.50%>=0)",
    "threshold_sweep": "rc13_v7_threshold_sweep.json: 0.33 robust; all "
                       "thresholds 0.28-0.43 positive both windows",
    "dev": "+4.39% / 5.05x (159 trades, 62% win)",
    "pre_registration": "04_backtest/rc13_combined_pre_registration.md",
}


def main():
    with pd.HDFStore(DB, "r") as store:
        df = store[MATRIX]
    missing = [c for c in FEATURES + list(GATE_LABELS) if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_version": "phase_g_v7_combined",
        "status": "shadow_candidate_not_live",
        "matrix": MATRIX,
        "features": FEATURES,
        "rows": int(len(df)),
        "universe": "combined pt-in-time SP400+SP600 (16,696 + 16,908)",
        "labels": "net-of-cost thresholds (SP400 10bp / SP600 30bp)",
        "threshold": 0.33,
        "validation": VALIDATION,
        "frozen": "2026-09-06",
        "gates": {},
    }
    for gate, label in GATE_LABELS.items():
        params = {**COMMON, **UNIFORM_HP}
        model = xgb.XGBClassifier(objective="binary:logistic",
                                  eval_metric=["logloss", "auc"], **params)
        model.fit(df[FEATURES], df[gate].astype(int),
                  eval_set=[(df[FEATURES], df[gate].astype(int))], verbose=False)
        path = OUT / gate / "classifier.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))
        artifact["gates"][gate] = {"label": label, **UNIFORM_HP, **COMMON,
                                   "classifier": str(path.relative_to(HERE.parent.parent))}
        print(f"Saved {path}")
    with open(OUT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"Saved {OUT / 'meta.json'}")
    print("V6 production artifact was not modified.")


if __name__ == "__main__":
    main()

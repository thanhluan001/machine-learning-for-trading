#!/usr/bin/env python3
"""
Freeze the deployable binary classifier (Phase G v2 final).

Replaces the 3-class softprob model. The deep comparison
(34_binary_vs_3class_deep.py) showed binary theta=0.20 beats 3-class
P(any)>=0.20 on total return (+636% vs +607%) because:
  - Better marginal trade selection (75% win rate on binary-only trades
    vs 50% on 3-class-only)
  - More trades (109 vs 99)
  - The 3-class degenerate argmax slightly muddied probability calibration

The 2-stage test (33_two_stage_model.py) proved CAR magnitude is
unpredictable from Sunday-safe features (regression correlation ~0),
so separating small/large PEAD adds no value.

Label: binary pead_pass (0/1) from 3 PEAD verification gates.
Inference: P(PEAD) >= 0.20, sort by P(PEAD) for weekly batch selection.

Entry: pre-gap (Close[T-1] BMO / Close[T] AMC)
Exit:  Close[T+5] (5-day hold)
No stop-loss.
"""
from __future__ import annotations
import sys, importlib.util, json, time
from pathlib import Path
import numpy as np, pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tm", HERE / "01_train_model.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
DB = tm.DB_FILE

v3_spec = importlib.util.spec_from_file_location(
    "v3", HERE.parent / "04_backtest" / "_pead_target_retrain.py")
v3 = importlib.util.module_from_spec(v3_spec); v3_spec.loader.exec_module(v3)

SUNDAY_SAFE_FEATURES = [
    "sue_score", "eps_surprise_pct", "consecutive_surprises",
    "sue_acceleration", "sue_lag_1", "sue_lag_2",
    "car_drift_historical_q1",
    "pre_event_idiosyncratic_vol", "pre_event_volume_trend",
    "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d",
    "rel_ret_30d", "sector_adjusted_ret_20d",
    "sue_abs_x_inverse_vol",
    "revision_momentum_30d", "revision_momentum_60d",
    "revision_momentum_90d", "revision_ordinal_momentum_90d",
    "revision_intensity_90d", "grade_dispersion_90d",
    "n_analysts_covering", "last_action_days_before_earnings",
]
assert len(SUNDAY_SAFE_FEATURES) == 24

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 50,
    "gamma": 3.0,
    "reg_lambda": 1.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "random_state": 42,
    "n_jobs": -1,
}

THETA = 0.20
OUT_DIR = HERE / "models" / "phase_g_v2_binary"


def main():
    print("=" * 80)
    print("FREEZE BINARY CLASSIFIER (Phase G v2 final)")
    print("=" * 80)
    print(f"  Features: {len(SUNDAY_SAFE_FEATURES)} Sunday-safe")
    print(f"  Objective: binary:logistic")
    print(f"  Target: pead_pass (0/1 from 3 PEAD gates)")
    print(f"  Deployable theta: P(PEAD) >= {THETA}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 80)

    print("\n[1] Loading train_matrix + priming + gates ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    print(f"    rows: {len(df)}, pead_pass: {int(df['pead_pass'].sum())} "
          f"({df['pead_pass'].mean()*100:.1f}%)")

    print("\n[2] Splitting TRAIN/VAL (last 12 months = VAL) ...")
    rd = pd.to_datetime(df["report_date"])
    val_start = rd.max() - pd.DateOffset(months=12)
    train_df = df[rd <= val_start].copy()
    val_df = df[rd > val_start].copy()
    print(f"    TRAIN: {len(train_df)} rows (up to {val_start.date()})")
    print(f"    VAL:   {len(val_df)} rows (after {val_start.date()})")

    print("\n[3] Training binary XGBClassifier ...")
    print(f"    xgb params: {XGB_PARAMS}")
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, average_precision_score

    X_train = train_df[SUNDAY_SAFE_FEATURES].copy()
    y_train = train_df["pead_pass"].astype(int).values
    X_val = val_df[SUNDAY_SAFE_FEATURES].copy()
    y_val = val_df["pead_pass"].astype(int).values

    t0 = time.time()
    clf = xgb.XGBClassifier(**XGB_PARAMS)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_s = time.time() - t0
    print(f"    Trained in {train_s:.1f}s")

    print("\n[4] Evaluating ...")
    train_proba = clf.predict_proba(X_train)[:, 1]
    val_proba = clf.predict_proba(X_val)[:, 1]
    auc_train = roc_auc_score(y_train, train_proba)
    auc_val = roc_auc_score(y_val, val_proba)
    ap_train = average_precision_score(y_train, train_proba)
    ap_val = average_precision_score(y_val, val_proba)
    print(f"    TRAIN AUC: {auc_train:.4f}  AP: {ap_train:.4f}")
    print(f"    VAL   AUC: {auc_val:.4f}  AP: {ap_val:.4f}")

    print(f"\n[5] Saving frozen artifact to {OUT_DIR} ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(OUT_DIR / "classifier.json"))

    meta = {
        "name": "phase_g_v2_binary",
        "objective": "binary:logistic",
        "target_label": "pead_pass (binary 0/1 from 3 PEAD verification gates)",
        "feature_set": "sunday_safe_24",
        "feature_columns": SUNDAY_SAFE_FEATURES,
        "xgb_params": XGB_PARAMS,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "train_pead_pos": int(y_train.sum()),
        "val_pead_pos": int(y_val.sum()),
        "auc_train": float(auc_train),
        "auc_val": float(auc_val),
        "ap_train": float(ap_train),
        "ap_val": float(ap_val),
        "gate_thresholds": {
            "GATE1_CAR_MIN": v3.GATE1_CAR_MIN,
            "GATE2_VOL_RATIO_MIN": v3.GATE2_VOL_RATIO_MIN,
            "GATE3_MAXDD_MIN": v3.GATE3_MAXDD_MIN,
        },
        "deployable_operating_point": {
            "theta": THETA,
            "entry": "pre-gap: Close[T-1] (BMO) / Close[T] (AMC)",
            "exit": "Close[T+5] (5 trading days from report date)",
            "stop_loss": "-10% delayed (skip gap day, check days 1+)",
            "exclude_sectors": ["XLF"],
            "exclude_rationale": "Financials have 13% PEAD precision vs 41% for rest. Inference-only filter (model trains on all sectors). Structural: financial earnings are more macro-driven, less surprise-driven.",
            "max_slots": 4,
            "sizing": "equal-weight 1/4 NAV",
            "selection": "weekly batch (per-week sort by P(PEAD), top N = free slots)",
        },
        "oos_stats_4fold_nested_cv": {
            "n_trades": 101,
            "win_rate_pct": 75.2,
            "avg_pnl_per_trade_pct": 6.66,
            "avg_win_pct": 12.36,
            "avg_loss_pct": -6.30,
            "payoff": 1.36,
            "total_pnl_pct": 672.4,
            "pead_precision_pct": 38.6,
            "large_pead_n": 21,
            "large_pead_win_rate_pct": 90.5,
            "large_pead_avg_return_pct": 19.62,
        },
        "why_binary_over_3class": "Deep comparison (34_binary_vs_3class_deep.py) showed binary theta=0.20 beats 3-class P(any)>=0.20 on total return (+636% vs +607%) and win rate (69.7% vs 64.6%). The 2-stage test (33_two_stage_model.py) proved CAR magnitude is unpredictable from Sunday-safe features (regression corr ~0), so the 3-class small/large split adds no value. Binary has better-calibrated probabilities for marginal 20-25% band trades.",
        "created_at": pd.Timestamp.now().isoformat(),
    }

    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"    Saved: classifier.json")
    print(f"    Saved: meta.json")

    print("\n[6] Verifying saved model loads ...")
    clf_check = xgb.XGBClassifier()
    clf_check.load_model(str(OUT_DIR / "classifier.json"))
    check_proba = clf_check.predict_proba(X_val[:5])[:, 1]
    orig_proba = clf.predict_proba(X_val[:5])[:, 1]
    max_diff = np.abs(check_proba - orig_proba).max()
    print(f"    Max proba diff (loaded vs original): {max_diff:.8f}")
    assert max_diff < 1e-6, "Model verification failed!"
    print(f"    VERIFIED: saved model produces identical predictions")

    print(f"\n{'='*80}")
    print("FROZEN BINARY CLASSIFIER SAVED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"  Artifact:  {OUT_DIR / 'classifier.json'}")
    print(f"  Meta:      {OUT_DIR / 'meta.json'}")
    print(f"  Features:  {len(SUNDAY_SAFE_FEATURES)} Sunday-safe")
    print(f"  Theta:     P(PEAD) >= {THETA}")
    print(f"  Entry:     pre-gap (Close[T-1] BMO / Close[T] AMC)")
    print(f"  Exit:      Close[T+5] (5-day hold)")
    print(f"  Stop:      -10% delayed (skip gap day)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

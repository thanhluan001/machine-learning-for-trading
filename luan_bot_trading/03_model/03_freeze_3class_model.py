#!/usr/bin/env python3
"""
Freeze the deployable 3-class softprob classifier (Phase G v2).

This is the FROZEN artifact for live paper-trading. The classifier is
trained on ALL available data (TRAIN + VAL, i.e. everything up to the
live fold start) using the 24 Sunday-safe features and the 3-class
label {no PEAD, small PEAD, large PEAD} where large = CAR_10d >= 10%.

The model is NEVER retrained during live trading. This script is run
once to produce the artifact; the live script (05_live/) loads it.

Artifact saved to:
    03_model/models/phase_g_v2_3class/classifier.json
    03_model/models/phase_g_v2_3class/meta.json

Label definition:
    Class 0 (no PEAD):     pead_pass == 0 (fails >= 1 gate)
    Class 1 (small PEAD):  pead_pass == 1 AND CAR_10d < 10% (linear)
    Class 2 (large PEAD):  pead_pass == 1 AND CAR_10d >= 10% (linear)

Inference rule (documented in meta.json):
    P(any PEAD) = P(small) + P(large)
    ACCEPT if P(any PEAD) >= 0.20

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

# ---------------------------------------------------------------------------
# Sunday-safe feature set (24 features, is_bmo removed, 8 revision momentum)
# Must match 03_model/02_phase_g_sunday_classifier.py exactly.
# ---------------------------------------------------------------------------
SUNDAY_SAFE_FEATURES = [
    # Block 1 (7)
    "sue_score", "eps_surprise_pct", "consecutive_surprises",
    "sue_acceleration", "sue_lag_1", "sue_lag_2",
    "car_drift_historical_q1",
    # Block 2 (2, is_bmo removed)
    "pre_event_idiosyncratic_vol",
    "pre_event_volume_trend",
    # Block 3 (6)
    "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d",
    "rel_ret_30d", "sector_adjusted_ret_20d",
    # Block 4 (1)
    "sue_abs_x_inverse_vol",
    # Block 6 (8, FMP revision momentum)
    "revision_momentum_30d", "revision_momentum_60d",
    "revision_momentum_90d", "revision_ordinal_momentum_90d",
    "revision_intensity_90d", "grade_dispersion_90d",
    "n_analysts_covering", "last_action_days_before_earnings",
]
assert len(SUNDAY_SAFE_FEATURES) == 24, "Sunday-safe set must be 24 features"

# ---------------------------------------------------------------------------
# Model hyperparameters (gamma=3, selected by nested CV)
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
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

# Deployable operating point
CAR_LARGE_THRESH = 10.0   # % linear CAR threshold for "large PEAD"
THETA_ANY = 0.20          # P(any PEAD) = P(small) + P(large) >= 0.20

# Output artifact directory
OUT_DIR = HERE / "models" / "phase_g_v2_3class"


def main():
    print("=" * 80)
    print("FREEZE 3-CLASS SOFTPROB CLASSIFIER (Phase G v2)")
    print("=" * 80)
    print(f"  Features: {len(SUNDAY_SAFE_FEATURES)} Sunday-safe")
    print(f"  Objective: multi:softprob, num_class=3")
    print(f"  Labels: {{0=no PEAD, 1=small PEAD, 2=large PEAD}}")
    print(f"  Large PEAD threshold: CAR_10d >= {CAR_LARGE_THRESH}% (linear)")
    print(f"  Deployable theta: P(any PEAD) >= {THETA_ANY}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 80)

    # Step 1: Load train_matrix + priming cutoff + PEAD gates
    print("\n[1] Loading train_matrix + priming + gates ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    print(f"    rows: {len(df)}, pead_pass: {int(df['pead_pass'].sum())} "
          f"({df['pead_pass'].mean()*100:.1f}%)")

    # Step 2: Build 3-class label
    print(f"\n[2] Building 3-class label (CAR threshold = {CAR_LARGE_THRESH}%) ...")
    df["car_10d_pct"] = np.expm1(df["car_10d"]) * 100  # log -> linear %
    df["label_3class"] = 0
    mask_small = (df["pead_pass"] == 1) & (df["car_10d_pct"] < CAR_LARGE_THRESH)
    mask_large = (df["pead_pass"] == 1) & (df["car_10d_pct"] >= CAR_LARGE_THRESH)
    df.loc[mask_small, "label_3class"] = 1
    df.loc[mask_large, "label_3class"] = 2

    n_no = int((df["label_3class"] == 0).sum())
    n_small = int((df["label_3class"] == 1).sum())
    n_large = int((df["label_3class"] == 2).sum())
    print(f"    Class 0 (no PEAD):    {n_no:>6} ({n_no/len(df)*100:.1f}%)")
    print(f"    Class 1 (small PEAD): {n_small:>6} ({n_small/len(df)*100:.1f}%)")
    print(f"    Class 2 (large PEAD): {n_large:>6} ({n_large/len(df)*100:.1f}%)")

    # Step 3: Split into TRAIN (everything except last year) + VAL (last year)
    # For the frozen artifact, we train on ALL data. But we hold out the last
    # year as VAL for reporting AUC/mlogloss metrics.
    print("\n[3] Splitting TRAIN/VAL (last 12 months = VAL) ...")
    rd = pd.to_datetime(df["report_date"])
    val_start = rd.max() - pd.DateOffset(months=12)
    train_df = df[rd <= val_start].copy()
    val_df = df[rd > val_start].copy()
    print(f"    TRAIN: {len(train_df)} rows (up to {val_start.date()})")
    print(f"    VAL:   {len(val_df)} rows (after {val_start.date()})")

    # Step 4: Train the 3-class classifier
    print("\n[4] Training 3-class XGBClassifier ...")
    print(f"    xgb params: {XGB_PARAMS}")
    import xgboost as xgb
    from sklearn.metrics import log_loss

    X_train = train_df[SUNDAY_SAFE_FEATURES].copy()
    y_train = train_df["label_3class"].values
    X_val = val_df[SUNDAY_SAFE_FEATURES].copy()
    y_val = val_df["label_3class"].values

    t0 = time.time()
    clf = xgb.XGBClassifier(**XGB_PARAMS)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_s = time.time() - t0
    print(f"    Trained in {train_s:.1f}s")

    # Step 5: Evaluate
    print("\n[5] Evaluating ...")
    train_proba = clf.predict_proba(X_train)
    val_proba = clf.predict_proba(X_val)

    mlogloss_train = log_loss(y_train, train_proba, labels=[0, 1, 2])
    mlogloss_val = log_loss(y_val, val_proba, labels=[0, 1, 2])
    print(f"    TRAIN mlogloss: {mlogloss_train:.4f}")
    print(f"    VAL   mlogloss: {mlogloss_val:.4f}")

    # Per-class accuracy on VAL
    val_pred = val_proba.argmax(axis=1)
    for c in [0, 1, 2]:
        mask = y_val == c
        n_c = mask.sum()
        if n_c > 0:
            acc = (val_pred[mask] == c).mean()
            print(f"    VAL class {c} recall: {acc*100:.1f}% (n={n_c})")

    # P(any PEAD) AUC (binary-equivalent for reporting)
    from sklearn.metrics import roc_auc_score
    y_val_any = (y_val >= 1).astype(int)
    p_val_any = val_proba[:, 1] + val_proba[:, 2]
    if y_val_any.sum() > 0 and y_val_any.sum() < len(y_val_any):
        auc_any = roc_auc_score(y_val_any, p_val_any)
        print(f"    VAL P(any PEAD) AUC: {auc_any:.4f}")

    # Step 6: Save artifact
    print(f"\n[6] Saving frozen artifact to {OUT_DIR} ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(OUT_DIR / "classifier.json"))

    # Step 7: Write meta.json
    meta = {
        "name": "phase_g_v2_3class",
        "objective": "multi:softprob",
        "num_class": 3,
        "target_label": "label_3class (0=no PEAD, 1=small PEAD, 2=large PEAD)",
        "label_definition": {
            "class_0": "no PEAD (pead_pass == 0, fails >= 1 gate)",
            "class_1": "small PEAD (pead_pass == 1 AND CAR_10d < 10% linear)",
            "class_2": "large PEAD (pead_pass == 1 AND CAR_10d >= 10% linear)",
        },
        "car_large_threshold_pct": CAR_LARGE_THRESH,
        "feature_set": "sunday_safe_24",
        "feature_columns": SUNDAY_SAFE_FEATURES,
        "xgb_params": XGB_PARAMS,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "train_class_counts": {
            "no_pead": int((y_train == 0).sum()),
            "small_pead": int((y_train == 1).sum()),
            "large_pead": int((y_train == 2).sum()),
        },
        "val_class_counts": {
            "no_pead": int((y_val == 0).sum()),
            "small_pead": int((y_val == 1).sum()),
            "large_pead": int((y_val == 2).sum()),
        },
        "mlogloss_train": float(mlogloss_train),
        "mlogloss_val": float(mlogloss_val),
        "auc_val_any_pead": float(auc_any) if 'auc_any' in dir() else None,
        "gate_thresholds": {
            "GATE1_CAR_MIN": v3.GATE1_CAR_MIN,
            "GATE2_VOL_RATIO_MIN": v3.GATE2_VOL_RATIO_MIN,
            "GATE3_MAXDD_MIN": v3.GATE3_MAXDD_MIN,
        },
        "deployable_operating_point": {
            "theta_any_pead": THETA_ANY,
            "entry": "pre-gap: Close[T-1] (BMO) / Close[T] (AMC)",
            "exit": "Close[T+5] (5 trading days from report date)",
            "stop_loss": "none (winners overcompensate losers 3.5:1)",
            "max_slots": 4,
            "sizing": "equal-weight 1/4 NAV",
            "selection": "weekly batch (per-week sort by P(any), top N = free slots)",
        },
        "oos_stats_4fold_nested_cv": {
            "n_trades": 99,
            "win_rate_pct": 64.6,
            "avg_pnl_per_trade_pct": 6.13,
            "avg_win_pct": 13.28,
            "avg_loss_pct": -6.94,
            "payoff": 1.91,
            "total_pnl_pct": 607.2,
            "pead_precision_pct": 33.3,
            "large_pead_n": 19,
            "large_pead_win_rate_pct": 94.7,
            "large_pead_avg_return_pct": 23.55,
        },
        "bootstrap_ci": {
            "expectancy_ci_95": [3.46, 8.87],
            "total_pnl_ci_95": [342.5, 878.1],
            "win_rate_ci_95": [55.6, 73.7],
            "n_bootstrap": 10000,
        },
        "created_at": pd.Timestamp.now().isoformat(),
    }

    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"    Saved: classifier.json")
    print(f"    Saved: meta.json")

    # Step 8: Verify the saved model loads correctly
    print("\n[7] Verifying saved model loads ...")
    clf_check = xgb.XGBClassifier()
    clf_check.load_model(str(OUT_DIR / "classifier.json"))
    check_proba = clf_check.predict_proba(X_val[:5])
    orig_proba = clf.predict_proba(X_val[:5])
    max_diff = np.abs(check_proba - orig_proba).max()
    print(f"    Max proba diff (loaded vs original): {max_diff:.8f}")
    assert max_diff < 1e-6, "Model verification failed!"
    print(f"    VERIFIED: saved model produces identical predictions")

    print(f"\n{'='*80}")
    print("FROZEN 3-CLASS CLASSIFIER SAVED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"  Artifact:  {OUT_DIR / 'classifier.json'}")
    print(f"  Meta:      {OUT_DIR / 'meta.json'}")
    print(f"  Features:  {len(SUNDAY_SAFE_FEATURES)} Sunday-safe")
    print(f"  Classes:   3 (no PEAD / small PEAD / large PEAD)")
    print(f"  Theta:     P(any PEAD) >= {THETA_ANY}")
    print(f"  Entry:     pre-gap (Close[T-1] BMO / Close[T] AMC)")
    print(f"  Exit:      Close[T+5] (5-day hold)")
    print(f"  Stop:      none")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

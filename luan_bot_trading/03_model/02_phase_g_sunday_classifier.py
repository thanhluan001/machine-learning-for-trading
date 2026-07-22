"""
Phase G -- Sunday-safe classifier retrain + T+1 gap confirmation sweep.

Per `04_backtest/pead_target_findings.md` §8. The most important
question hanging over the §6 classifier result: does the classifier
retain alpha when we remove the 4 leak/Day-T features that are NOT
available at Sunday planning time?

Sunday-safe feature set (17 features):
  - sue_score, eps_surprise_pct, consecutive_surprises, sue_acceleration,
    sue_lag_1, sue_lag_2, car_drift_historical_q1, is_bmo,
    pre_event_idiosyncratic_vol, pre_event_volume_trend,
    rel_ret_3d, rel_ret_5d, rel_ret_10d, rel_ret_20d, rel_ret_30d,
    sector_adjusted_ret_20d, sue_abs_x_inverse_vol

DROPPED (4 leak/Day-T features):
  - opening_gap_t1 (forward-looking, uses Open[T+1])
  - intraday_range_t, volume_vma20_ratio_pre_event, suv_day_1 (Day-T,
    available only post-T-close)

Protocol:
  Step 1 -- load /features/train_matrix, §12 cut, split TRAIN/VAL.
  Step 2 -- compute the 3 PEAD gates (pead_pass label) on all rows.
  Step 3 -- train XGBClassifier with the 17-feature Sunday set, binary
            target = pead_pass. Report AUC + AP on TRAIN and VAL.
  Step 4 -- compute realized entry PnL = log(Close[T+11] / Open[T+1]).
  Step 5 -- threshold sweep on P(PEAD) alone (no gap filter) to
            establish the Sunday-only baseline operating point.
  Step 6 -- two-stage filter sweep:
            enter iff (P(PEAD) >= theta_screen) AND
                     (opening_gap_t1 in [gap_lo, gap_hi])
            Sweep theta_screen in {0.10, 0.15, 0.20, 0.25, 0.30} and
            (gap_lo, gap_hi) in a grid of bounds. Report per-event
            PnL, hit rate, recall on PEADs, and N-trades per cell.
  Step 7 -- Save the best Sunday-only classifier artifact under
            03_model/models/phase_g_v1_sunday_classifier/.

NO DB WRITES. Read-only on db.h5.
"""
from __future__ import annotations
import sys, importlib.util, json, pickle, time
from pathlib import Path
import numpy as np, pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

HERE = Path(__file__).resolve().parent
# Reuse the train module helpers (load_train_matrix, splits, etc.)
spec = importlib.util.spec_from_file_location(
    "tm", HERE / "01_train_model.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
DB = tm.DB_FILE

# Reuse compute_pead_gates_full from the v3 retrain script (val gate logic
# + oracle arith metrics). We import it as a module and reuse its helper.
v3_spec = importlib.util.spec_from_file_location(
    "v3", HERE.parent / "04_backtest" / "_pead_target_retrain.py")
v3 = importlib.util.module_from_spec(v3_spec); v3_spec.loader.exec_module(v3)

# ---------------------------------------------------------------------------
# Sunday-safe feature set
# ---------------------------------------------------------------------------
SUNDAY_SAFE_FEATURES = [
    # Block 1 (7)
    "sue_score", "eps_surprise_pct", "consecutive_surprises",
    "sue_acceleration", "sue_lag_1", "sue_lag_2",
    "car_drift_historical_q1",
    # Block 2 (5 of 7 -- drop volume_vma20_ratio_pre_event, suv_day_1,
    #           intraday_range_t, opening_gap_t1)
    "is_bmo",
    "pre_event_idiosyncratic_vol",
    "pre_event_volume_trend",
    # Block 3 (6)
    "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d",
    "rel_ret_30d", "sector_adjusted_ret_20d",
    # Block 4 (1)
    "sue_abs_x_inverse_vol",
]
assert len(SUNDAY_SAFE_FEATURES) == 17, "Sunday-safe set must be 17 features"
# Defensive cross-check vs the master list:
_MISSING = [c for c in SUNDAY_SAFE_FEATURES if c not in tm.FEATURE_COLUMNS]
assert not _MISSING, f"Sunday features not in master FEATURE_COLUMNS: {_MISSING}"

# 4 features we are deliberately dropping:
DROPPED_LEAK_FEATURES = [
    "opening_gap_t1", "intraday_range_t",
    "volume_vma20_ratio_pre_event", "suv_day_1",
]

# T+1 gap confirmation -- we'll filter on the (forward-looking)
# opening_gap_t1 column in the train matrix. This col exists in
# train_matrix even though it's a leak feature -- it's just a record
# of what Open[T+1] realized, used as the confirmation signal at the
# T+1 morning stage (where it IS available by then).
GAP_COL = "opening_gap_t1"


def compute_entry_pnl(val_df: pd.DataFrame) -> pd.DataFrame:
    """Add col `ret_open_t1_close_t11` = log(Close[T+11] / Open[T+1]).
    Returns the same val_df with the new column added."""
    val_df = val_df.copy()
    val_df["ret_open_t1_close_t11"] = np.nan
    with pd.HDFStore(DB, mode="r") as s:
        pts = val_df["permaTicker"].unique()
        n_done = 0
        for pt in pts:
            key = f"/sp400/{pt}"
            if key not in s:
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_open = p["Adj_Open"].values
            p_close = p["Adj_Close"].values
            sub = val_df[val_df["permaTicker"] == pt]
            for idx, row in sub.iterrows():
                rdate = pd.to_datetime(row["report_date"]).to_datetime64()
                t_mask = p_index >= rdate
                if not t_mask.any():
                    continue
                t_idx = int(np.argmax(t_mask))
                if t_idx + 11 >= len(p_close):
                    continue
                o_t1 = p_open[t_idx + 1]
                c_t11 = p_close[t_idx + 11]
                if pd.isna(o_t1) or pd.isna(c_t11) or o_t1 <= 0:
                    continue
                val_df.loc[idx, "ret_open_t1_close_t11"] = \
                    float(np.log(c_t11 / o_t1))
            n_done += 1
            if n_done % 100 == 0:
                print(f"    [entry-pnl] {n_done}/{len(pts)} permaTickers")
    return val_df


def main():
    print("=" * 78)
    print("PHASE G v1 -- Sunday-safe classifier + T+1 gap confirmation sweep")
    print("=" * 78)

    # Step 1 -- Load + cut + split
    print("\n[1] Loading train_matrix + applying §12 cutoff + walk-forward split")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    print(f"    rows after §12 cut: {len(df)}")

    # Step 2 -- Compute the 3 PEAD gates (label pead_pass + diagnostics).
    print("\n[2] Computing 3 PEAD gates across all primed rows ...")
    df = v3.compute_pead_gates_full(df)
    n_total = len(df)
    n_pead = int(df["pead_pass"].sum())
    print(f"    pead_pass positives: {n_pead} ({n_pead/n_total*100:.2f}%)")

    train_df, val_df = tm.split_walk_forward(df, tm.DEFAULT_SPLIT_DATE)
    train_df, _ = tm.drop_sparse_weeks(train_df, tm.DEFAULT_MIN_GROUP_SIZE)
    val_df, _ = tm.drop_sparse_weeks(val_df, tm.DEFAULT_MIN_GROUP_SIZE)
    train_df = train_df.sort_values(
        ["calendar_week_group", "permaTicker", "report_date"]
    ).reset_index(drop=True)
    val_df = val_df.sort_values(
        ["calendar_week_group", "permaTicker", "report_date"]
    ).reset_index(drop=True)
    print(f"    TRAIN rows: {len(train_df)}  pead_pos: {int(train_df['pead_pass'].sum())} "
          f"({train_df['pead_pass'].mean()*100:.2f}%)")
    print(f"    VAL   rows: {len(val_df)}  pead_pos: {int(val_df['pead_pass'].sum())} "
          f"({val_df['pead_pass'].mean()*100:.2f}%)")

    # Step 3 -- Train Sunday-safe classifier
    print("\n[3] Training Sunday-safe XGBClassifier (17 features, NO leak/Day-T)")
    print(f"    Dropped leak/Day-T features: {DROPPED_LEAK_FEATURES}")
    print(f"    Sunday-safe feature set ({len(SUNDAY_SAFE_FEATURES)}): {SUNDAY_SAFE_FEATURES}")
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    X_train = train_df[SUNDAY_SAFE_FEATURES].copy()
    y_train = train_df["pead_pass"].astype(int).values
    X_val = val_df[SUNDAY_SAFE_FEATURES].copy()
    y_val = val_df["pead_pass"].astype(int).values

    params = dict(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=50,
        gamma=5.0,
        reg_lambda=1.0,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
    )
    print(f"    xgb params: {params}")
    t0 = time.time()
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False)
    train_s = time.time() - t0
    print(f"    Trained in {train_s:.1f}s")

    train_proba = clf.predict_proba(X_train)[:, 1]
    val_proba = clf.predict_proba(X_val)[:, 1]
    auc_train = roc_auc_score(y_train, train_proba)
    auc_val = roc_auc_score(y_val, val_proba)
    ap_train = average_precision_score(y_train, train_proba)
    ap_val = average_precision_score(y_val, val_proba)
    print(f"    TRAIN AUC: {auc_train:.4f}  AP: {ap_train:.4f}")
    print(f"    VAL   AUC: {auc_val:.4f}  AP: {ap_val:.4f}")

    # Compare against the v3 (21-feature) classifier from §6:
    print(f"\n    [vs §6 21-feature classifier: AUC_train=0.877, AUC_val=0.860, AP=0.526]")
    print(f"    AUC loss from dropping 4 leak features: "
          f"{(0.8596 - auc_val)*100:+.2f} pp (VAL), "
          f"{(0.8770 - auc_train)*100:+.2f} pp (TRAIN)")

    val_df["pead_proba"] = val_proba

    # Step 4 -- Compute realized entry PnL (Open[T+1] -> Close[T+11])
    print("\n[4] Computing realized entry PnL Open[T+1] -> Close[T+11] ...")
    val_df = compute_entry_pnl(val_df)
    valid_pnl_mask = val_df["ret_open_t1_close_t11"].notna()
    print(f"    coverage: {int(valid_pnl_mask.sum())}/{len(val_df)} val rows")

    # Universe baseline
    univ_arith = np.expm1(val_df.loc[valid_pnl_mask, "ret_open_t1_close_t11"])
    univ_mean = float(univ_arith.mean()) * 100
    univ_hit = float((univ_arith > 0).mean()) * 100
    print(f"    Universe baseline: mean={univ_mean:+.4f}%/event  hit={univ_hit:.1f}%")

    # Oracle (true PEAD events)
    oracle_mask = (val_df["pead_pass"] == 1) & valid_pnl_mask
    oracle_arith = np.expm1(val_df.loc[oracle_mask, "ret_open_t1_close_t11"])
    print(f"    Oracle (pead_pass==1): mean={float(oracle_arith.mean())*100:+.4f}%/event  "
          f"hit={float((oracle_arith > 0).mean())*100:.1f}%  n={int(oracle_mask.sum())}")

    # Step 5 -- Sunday-only threshold sweep (no gap filter yet)
    n_total_pead_val = int((val_df["pead_pass"] == 1).sum())
    print(f"\n[5] SUNDAY-ONLY threshold sweep (no gap filter) -- "
          f"({n_total_pead_val} true PEADs in val)")
    print(f"\n    {'thresh':>7s} {'n':>6s} {'recall%':>9s} {'precision%':>11s} "
          f"{'avg_pnl%':>10s} {'hit%':>7s} {'sharpe_liq':>10s}")
    print("    " + "-" * 70)
    best_sunday_thresh = None
    best_sunday_pnl = -1e9
    for thresh in [0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]:
        pick_mask = (val_df["pead_proba"] >= thresh) & valid_pnl_mask
        picks = val_df[pick_mask]
        if len(picks) < 1:
            continue
        n_picks = len(picks)
        n_pos = int((picks["pead_pass"] == 1).sum())
        recall = n_pos / n_total_pead_val * 100
        precision = n_pos / n_picks * 100
        arith = np.expm1(picks["ret_open_t1_close_t11"])
        avg_pnl = float(arith.mean()) * 100
        hit_rate = float((arith > 0).mean()) * 100
        # Approximate annualized Sharpe (107 weeks of independent events):
        std_arith = float(arith.std())
        sharpe_liq = (avg_pnl / 100) / (std_arith + 1e-9) * np.sqrt(52) if std_arith > 0 else 0.0
        print(f"    >={thresh:>5.2f}  {n_picks:>5d}   {recall:>7.2f}%   {precision:>9.2f}%   "
              f"{avg_pnl:>+8.3f}%   {hit_rate:>5.1f}%   {sharpe_liq:>+8.2f}")
        if avg_pnl > best_sunday_pnl:
            best_sunday_pnl = avg_pnl
            best_sunday_thresh = thresh

    print(f"\n    Best Sunday-only operating point: P>={best_sunday_thresh}"
          f"  avg_pnl={best_sunday_pnl:+.3f}%/event")

    # Step 6 -- Two-stage Sunday classifier + T+1 gap confirmation filter
    print("\n[6] TWO-STAGE filter sweep -- Sunday classifier P(PEAD) >= theta_screen")
    print("                            AND opening_gap_t1 in [gap_lo, gap_hi]")
    print(f"    (opening_gap_t1 used as a T+1-MORNING CONFIRMATION only -- ")
    print(f"     NOT a Sunday predictor. It is observed AT T+1 open, AFTER the")
    print(f"     Sunday ranker has named its watchlist.)")

    # Build the grid
    theta_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
    gap_grid = [
        # (gap_lo, gap_hi, label)
        (-1.0, 1.0,  "all gaps (passthrough)"),
        (0.00, 1.0,  "gap in [0%,+100%]  (any positive gap)"),
        (0.02, 0.15, "gap in [+2%,+15%]   (mild-moderate positive gap)"),
        (0.03, 0.10, "gap in [+3%,+10%]   (moderate positive gap)"),
        (0.05, 0.15, "gap in [+5%,+15%]   (large positive gap)"),
        (-0.10, 0.0, "gap in [-10%,0%]    (any small negative gap)"),
        (-0.15, -0.02, "gap in [-15%,-2%]  (mild-moderate negative gap)"),
        # Long-only filter intended to catch upward drift continuation:
        (0.02, 0.08, "gap in [+2%,+8%]    (sweet spot, used in classic PEAD)"),
    ]

    print(f"\n    {'theta':>6s} {'gap_label':>42s} {'n_trades':>10s} "
          f"{'recall%':>9s} {'avg_pnl%':>10s} {'hit%':>7s} {'sharpe':>8s}")
    print("    " + "-" * 100)
    best_combo = None
    best_combo_pnl = -1e9
    rows = []
    for theta in theta_grid:
        for gap_lo, gap_hi, label in gap_grid:
            # Drop NaN gaps (no realized Open[T+1])
            mask = (
                (val_df["pead_proba"] >= theta)
                & (val_df[GAP_COL] >= gap_lo)
                & (val_df[GAP_COL] <= gap_hi)
                & valid_pnl_mask
            )
            picks = val_df[mask]
            if len(picks) < 1:
                continue
            n_picks = len(picks)
            n_pos = int((picks["pead_pass"] == 1).sum())
            recall = n_pos / n_total_pead_val * 100
            arith = np.expm1(picks["ret_open_t1_close_t11"])
            avg_pnl = float(arith.mean()) * 100
            hit_rate = float((arith > 0).mean()) * 100
            std_arith = float(arith.std())
            sharpe = (avg_pnl / 100) / (std_arith + 1e-9) * np.sqrt(52) if std_arith > 0 else 0.0
            print(f"    >={theta:>4.2f}  {label:>42s}  {n_picks:>7d}  "
                  f"{recall:>7.2f}%  {avg_pnl:>+8.3f}%  {hit_rate:>5.1f}%  {sharpe:>+6.2f}")
            rows.append({
                "theta_screen": theta, "gap_lo": gap_lo, "gap_hi": gap_hi,
                "gap_label": label, "n_trades": n_picks,
                "pead_recall": recall, "avg_pnl_pct": avg_pnl,
                "hit_pct": hit_rate, "sharpe": sharpe,
            })
            if avg_pnl > best_combo_pnl and n_picks >= 20:
                best_combo_pnl = avg_pnl
                best_combo = (theta, gap_lo, gap_hi, label, n_picks,
                              recall, avg_pnl, hit_rate, sharpe)

    print("\n" + "=" * 100)
    print("BEST COMBO (min 20 trades for stability):")
    if best_combo is None:
        print("  No combo meets the min-20-trades threshold.")
    else:
        theta, glo, ghi, lbl, n, rec, pnl, hit, shp = best_combo
        print(f"  theta_screen >= {theta}")
        print(f"  opening_gap_t1 in [{glo*100:+.1f}%, {ghi*100:+.1f}%]  ({lbl})")
        print(f"  n_trades: {n}")
        print(f"  PEAD-recall: {rec:.2f}%")
        print(f"  avg PnL/event: {pnl:+.4f}%")
        print(f"  hit rate: {hit:.1f}%")
        print(f"  Sharpe (liq, ann.): {shp:+.2f}")
    print("=" * 100)

    # Step 7 -- Persist the Sunday classifier as Phase G v1 artifact
    out_dir = HERE / "models" / "phase_g_v1_sunday_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(out_dir / "classifier.json"))
    # Calibrator: model's P(PEAD) -> realized arith CAR
    calib_mask = valid_pnl_mask
    calib = IsotonicRegression(out_of_bounds="clip")
    calib.fit(val_proba[calib_mask.values],
              np.expm1(val_df.loc[calib_mask, "ret_open_t1_close_t11"]).values)
    with open(out_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(calib, f)

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    meta = {
        "name": "phase_g_v1_sunday_classifier",
        "objective": "binary:logistic",
        "target_label": "pead_pass (binary 0/1 from 3 PEAD verification gates)",
        "feature_set": "sunday_safe_17",
        "feature_columns": SUNDAY_SAFE_FEATURES,
        "dropped_features": DROPPED_LEAK_FEATURES,
        "xgb_params": params,
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
        "universe_baseline_pnl_pct": univ_mean,
        "oracle_pnl_pct": float(oracle_arith.mean()) * 100,
        "best_sunday_only_thresh": best_sunday_thresh,
        "best_sunday_only_pnl_pct": best_sunday_pnl,
        "best_combo": {
            "theta_screen": best_combo[0] if best_combo else None,
            "gap_lo": best_combo[1] if best_combo else None,
            "gap_hi": best_combo[2] if best_combo else None,
            "gap_label": best_combo[3] if best_combo else None,
            "n_trades": best_combo[4] if best_combo else None,
            "pead_recall_pct": best_combo[5] if best_combo else None,
            "avg_pnl_pct": best_combo[6] if best_combo else None,
            "hit_pct": best_combo[7] if best_combo else None,
            "sharpe": best_combo[8] if best_combo else None,
        },
        "created_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n[7] Saved Phase G v1 Sunday classifier to {out_dir}")
    print(f"    classifier.json, calibrator.pkl, meta.json, threshold_sweep.csv")
    print("\n" + "=" * 78)
    print("PHASE G v1 DONE -- review the threshold sweep output above.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

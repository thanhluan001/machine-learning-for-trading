#!/usr/bin/env python3
"""
2-stage model test (instructor feedback -> fix P(large) calibration).

Stage 1: Binary classifier (PEAD vs no PEAD) — gate on P(PEAD) >= theta1
Stage 2: Among Stage 1 positives, regression on CAR_10d magnitude —
         rank by predicted CAR, take top N per week

The 3-class softprob had degenerate argmax (predicts class 0 for 100%
of events) because class imbalance is extreme (89/6/4%). A 2-stage
model decouples the two questions:
  Stage 1: "Is this a PEAD event?" (binary, well-calibrated)
  Stage 2: "How big is the drift?" (regression on CAR magnitude)

This should give better P(large) calibration because:
  - Stage 1 binary is balanced enough (10.7% positive) to train well
  - Stage 2 regression only sees PEAD events, so the CAR magnitude
    signal isn't drowned out by the 89% no-PEAD majority

Test 3 configurations:
  A. Stage 1 only (binary P(PEAD) >= theta1) — baseline
  B. 2-stage: Stage 1 gate + Stage 2 rank by predicted CAR
  C. 3-class P(any) >= 0.20 — current model for comparison
"""
import sys, io, importlib.util
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tm", HERE.parent / "03_model" / "01_train_model.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
pg_spec = importlib.util.spec_from_file_location("pg", HERE.parent / "03_model" / "02_phase_g_sunday_classifier.py")
pg = importlib.util.module_from_spec(pg_spec); pg_spec.loader.exec_module(pg)
v3_spec = importlib.util.spec_from_file_location("v3", HERE / "_pead_target_retrain.py")
v3 = importlib.util.module_from_spec(v3_spec); v3_spec.loader.exec_module(v3)

DB = tm.DB_FILE
SUNDAY_SAFE = pg.SUNDAY_SAFE_FEATURES
N_SLOTS = 4
EXIT_SNAP = 5
CAR_LARGE_THRESH = 10.0

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]

# Stage 1 thresholds to sweep (binary P(PEAD))
THETAS_STAGE1 = [0.15, 0.20, 0.25, 0.30]


def fit_clf_binary(X_tr, y_tr, X_val, y_val, hp):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=hp["n_estimators"], learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def fit_reg_car(X_tr, y_tr, X_val, y_val, hp):
    """Stage 2: XGBRegressor on CAR_10d (log units) for PEAD events only."""
    import xgboost as xgb
    reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=hp["n_estimators"], learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1)
    reg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return reg


def fit_clf_3class(X_tr, y_tr, X_val, y_val, hp):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss",
        n_estimators=hp["n_estimators"], learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def select_weekly_top_n(picks, n_slots=4, sort_col="p"):
    if picks.empty:
        return picks
    pk = picks.copy()
    pk["entry_date"] = pd.to_datetime(pk["entry_date"])
    pk["exit_date"] = pd.to_datetime(pk["exit_date"])
    iso = pk["entry_date"].dt.isocalendar()
    pk["_week_key"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    selected_rows = []
    active_positions = []
    for week_key, week_df in pk.groupby("_week_key", sort=True):
        week_start = week_df["entry_date"].min()
        active_positions = [ex for ex in active_positions if ex >= week_start]
        free_slots = n_slots - len(active_positions)
        if free_slots <= 0:
            continue
        week_sorted = week_df.sort_values(sort_col, ascending=False)
        taken = week_sorted.head(min(free_slots, len(week_sorted)))
        selected_rows.append(taken)
        for _, row in taken.iterrows():
            active_positions.append(row["exit_date"])
    if selected_rows:
        return pd.concat(selected_rows).sort_values("entry_date").reset_index(drop=True)
    return pd.DataFrame(columns=picks.columns)


def compute_pregap_returns(df, db_path, hold_days=5):
    df = df.copy()
    df["pregap_return"] = np.nan
    df["pregap_entry_date"] = pd.NaT
    df["pregap_exit_date"] = pd.NaT
    with pd.HDFStore(db_path, mode="r") as s:
        for idx, row in df.iterrows():
            pt = row["permaTicker"]
            key = f"/sp400/{pt}"
            if key not in s:
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_close = p["Adj_Close"].values
            rdate = pd.to_datetime(row["report_date"]).to_datetime64()
            t_mask = p_index >= rdate
            if not t_mask.any():
                continue
            t_idx = int(np.argmax(t_mask))
            is_bmo = bool(row.get("is_bmo", False))
            entry_t = t_idx - 1 if is_bmo else t_idx
            exit_t = t_idx + hold_days
            if entry_t < 0 or exit_t >= len(p_close):
                continue
            entry_price = p_close[entry_t]
            exit_price = p_close[exit_t]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue
            df.at[idx, "pregap_return"] = float(exit_price / entry_price - 1.0)
            df.at[idx, "pregap_entry_date"] = pd.Timestamp(p_index[entry_t])
            df.at[idx, "pregap_exit_date"] = pd.Timestamp(p_index[exit_t])
    return df


def stats_row(exec_df, label, sort_col=None):
    if len(exec_df) == 0:
        print(f"  {label:<45} N=0")
        return
    pnls = exec_df["pregap_return"].dropna()
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / n * 100
    avg = pnls.mean() * 100
    aw = wins.mean() * 100 if len(wins) > 0 else 0
    al = losses.mean() * 100 if len(losses) > 0 else 0
    payoff = aw / abs(al) if al != 0 else float('inf')
    total = pnls.sum() * 100
    n_pead = int((exec_df["label_3class"] >= 1).sum())
    n_large = int((exec_df["label_3class"] == 2).sum())
    prec = n_pead / n * 100
    prec_l = n_large / n * 100
    print(f"  {label:<45} N={n:>3} Win={wr:>5.1f}% Avg={avg:>+6.2f}% "
          f"Tot={total:>+7.1f}% Pay={payoff:>4.2f} "
          f"PEAD={n_pead}({prec:.0f}%) Lg={n_large}({prec_l:.0f}%)")


def main():
    print("=" * 110)
    print("2-STAGE MODEL TEST (Stage 1: binary PEAD gate, Stage 2: CAR regression ranker)")
    print("=" * 110)

    print("\n[1] Loading + priming + gates ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)

    df["car_10d_pct"] = np.expm1(df["car_10d"]) * 100
    df["label_3class"] = 0
    df.loc[(df["pead_pass"] == 1) & (df["car_10d_pct"] < CAR_LARGE_THRESH), "label_3class"] = 1
    df.loc[(df["pead_pass"] == 1) & (df["car_10d_pct"] >= CAR_LARGE_THRESH), "label_3class"] = 2

    print("[2] Computing pre-gap returns ...")
    df = compute_pregap_returns(df, DB, EXIT_SNAP)

    print("[3] Training 3 models per fold (binary + CAR regressor + 3-class) ...")
    fold_data = {}
    hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}

    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_tr = train_df[SUNDAY_SAFE]; X_sv = sweep_df[SUNDAY_SAFE]
        X_te = test_df[SUNDAY_SAFE]
        X_ts = pd.concat([X_tr, X_sv])
        X_ts_pead = pd.concat([X_tr, X_sv])

        # Binary labels
        y_ts_b = pd.concat([train_df, sweep_df])["pead_pass"].astype(int).values
        y_te_b = test_df["pead_pass"].astype(int).values

        # Stage 1: binary classifier
        clf_b = fit_clf_binary(X_ts, y_ts_b, X_te, y_te_b, hp)

        # Stage 2: CAR regressor — trained ONLY on PEAD events (pead_pass == 1)
        pead_mask_ts = pd.concat([train_df, sweep_df])["pead_pass"] == 1
        X_ts_pead_only = X_ts[pead_mask_ts.values]
        car_ts = pd.concat([train_df, sweep_df]).loc[pead_mask_ts, "car_10d"].values  # log units
        # VAL for stage 2: PEAD events in test set
        pead_mask_te = test_df["pead_pass"] == 1
        X_te_pead_only = X_te[pead_mask_te]
        car_te = test_df.loc[pead_mask_te, "car_10d"].values

        if len(car_ts) > 10 and len(car_te) > 2:
            reg_car = fit_reg_car(X_ts_pead_only, car_ts, X_te_pead_only, car_te, hp)
        else:
            reg_car = None

        # 3-class (for comparison)
        y_ts_3 = pd.concat([train_df, sweep_df])["label_3class"].values
        y_te_3 = test_df["label_3class"].values
        clf_3 = fit_clf_3class(X_ts, y_ts_3, X_te, y_te_3, hp)

        # Predictions
        test_df = test_df.copy()
        test_df["p_binary"] = clf_b.predict_proba(X_te)[:, 1]
        if reg_car is not None:
            test_df["pred_car"] = reg_car.predict(X_te)
        else:
            test_df["pred_car"] = 0.0
        proba_3 = clf_3.predict_proba(X_te)
        test_df["p_any_pead"] = proba_3[:, 1] + proba_3[:, 2]
        test_df["p_large"] = proba_3[:, 2]

        fold_data[fi] = {"test_df": test_df, "reg_car": reg_car}
        n_pead_train = int(pead_mask_ts.sum())
        print(f"    Fold {fi}: TEST={len(test_df)}, "
              f"PEAD_train={n_pead_train}, "
              f"reg_car={'OK' if reg_car else 'SKIP'}")

    # ===== STAGE 2 CALIBRATION CHECK =====
    print(f"\n{'='*110}")
    print("1. STAGE 2 (CAR REGRESSOR) CALIBRATION CHECK")
    print(f"{'='*110}")

    # Does pred_car correlate with actual returns / CAR?
    all_preds = []
    for fi in range(1, 5):
        td = fold_data[fi]["test_df"]
        td = td[td["pred_car"].notna() & td["pregap_return"].notna()].copy()
        all_preds.append(td)
    all_df = pd.concat(all_preds)

    corr_car = all_df["pred_car"].corr(all_df["pregap_return"])
    corr_pead = all_df["p_binary"].corr(all_df["pregap_return"])
    from scipy.stats import spearmanr
    sp_car, _ = spearmanr(all_df["pred_car"], all_df["pregap_return"])
    sp_pead, _ = spearmanr(all_df["p_binary"], all_df["pregap_return"])

    print(f"\n  Correlation with pre-gap return ({len(all_df)} events):")
    print(f"    pred_car (Stage 2)  Pearson: {corr_car:+.4f}  Spearman: {sp_car:+.4f}")
    print(f"    p_binary (Stage 1)  Pearson: {corr_pead:+.4f}  Spearman: {sp_pead:+.4f}")

    # pred_car quintile analysis
    print(f"\n  pred_car quintile analysis (all {len(all_df)} test events):")
    # Use rank-based quintiles (handles duplicates)
    all_df["_car_rank"] = all_df["pred_car"].rank(method="first")
    all_df["car_quintile"] = pd.qcut(all_df["_car_rank"], 5, labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"])
    for q in ["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"]:
        sub = all_df[all_df["car_quintile"] == q]
        if len(sub) > 0:
            n_large = int((sub["label_3class"] == 2).sum())
            print(f"    {q}: n={len(sub):>4}, avg_ret={sub['pregap_return'].mean()*100:>+6.2f}%, "
                  f"win={((sub['pregap_return']>0).mean())*100:.0f}%, "
                  f"large_pead={n_large}({n_large/len(sub)*100:.0f}%), "
                  f"pred_car=[{sub['pred_car'].min():.3f}, {sub['pred_car'].max():.3f}]")

    # ===== SCENARIO COMPARISON =====
    print(f"\n{'='*110}")
    print("2. SCENARIO COMPARISON (weekly batch, 4 slots, pre-gap entry)")
    print(f"{'='*110}")

    print(f"\n  {'Scenario':<45} {'N':>4} {'Win%':>6} {'Avg':>7} {'Total':>8} {'Pay':>5} {'PEAD':>10} {'Large':>10}")
    print("  " + "-" * 100)

    # C. 3-class P(any) >= 0.20 (baseline)
    all_exec_c = []
    for fi in range(1, 5):
        td = fold_data[fi]["test_df"]
        mask = (td["p_any_pead"] >= 0.20) & (td["pregap_return"].notna())
        picks = td[mask].copy()
        if len(picks) == 0:
            continue
        picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
        picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
        picks["fold"] = fi
        sel = select_weekly_top_n(picks, N_SLOTS, sort_col="p_any_pead")
        if len(sel) > 0:
            all_exec_c.append(sel)
    exec_c = pd.concat(all_exec_c) if all_exec_c else pd.DataFrame()
    stats_row(exec_c, "C. 3-class P(any)>=0.20 (current)")

    # A. Stage 1 only (binary P(PEAD) >= theta1), sort by P(PEAD)
    for theta1 in THETAS_STAGE1:
        all_exec = []
        for fi in range(1, 5):
            td = fold_data[fi]["test_df"]
            mask = (td["p_binary"] >= theta1) & (td["pregap_return"].notna())
            picks = td[mask].copy()
            if len(picks) == 0:
                continue
            picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
            picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
            picks["fold"] = fi
            sel = select_weekly_top_n(picks, N_SLOTS, sort_col="p_binary")
            if len(sel) > 0:
                all_exec.append(sel)
        exec_a = pd.concat(all_exec) if all_exec else pd.DataFrame()
        stats_row(exec_a, f"A. Stage1 only P(PEAD)>={theta1:.2f}, sort=P(PEAD)")

    # B. 2-stage: Stage 1 gate (various theta1) + Stage 2 sort by pred_car
    print()
    for theta1 in THETAS_STAGE1:
        all_exec = []
        for fi in range(1, 5):
            td = fold_data[fi]["test_df"]
            mask = (td["p_binary"] >= theta1) & (td["pregap_return"].notna())
            picks = td[mask].copy()
            if len(picks) == 0:
                continue
            picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
            picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
            picks["fold"] = fi
            # Sort by pred_car (Stage 2 regression)
            sel = select_weekly_top_n(picks, N_SLOTS, sort_col="pred_car")
            if len(sel) > 0:
                all_exec.append(sel)
        exec_b = pd.concat(all_exec) if all_exec else pd.DataFrame()
        stats_row(exec_b, f"B. 2-stage P(PEAD)>={theta1:.2f} + CAR rank")

    # ===== PER-FOLD FOR BEST 2-STAGE =====
    print(f"\n{'='*110}")
    print("3. PER-FOLD BREAKDOWN (best 2-stage config)")
    print(f"{'='*110}")

    # Find best 2-stage by total PnL
    best_theta = None
    best_total = -999
    for theta1 in THETAS_STAGE1:
        all_exec = []
        for fi in range(1, 5):
            td = fold_data[fi]["test_df"]
            mask = (td["p_binary"] >= theta1) & (td["pregap_return"].notna())
            picks = td[mask].copy()
            if len(picks) == 0:
                continue
            picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
            picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
            picks["fold"] = fi
            sel = select_weekly_top_n(picks, N_SLOTS, sort_col="pred_car")
            if len(sel) > 0:
                all_exec.append(sel)
        exec_b = pd.concat(all_exec) if all_exec else pd.DataFrame()
        if len(exec_b) > 0:
            total = exec_b["pregap_return"].sum() * 100
            if total > best_total:
                best_total = total
                best_theta = theta1

    print(f"\n  Best 2-stage: theta1={best_theta:.2f} (total={best_total:+.1f}%)")
    print(f"\n  {'Config':<20} {'Fold':>5} {'N':>4} {'Win%':>6} {'Avg':>8} {'Total':>8} {'Large':>6}")
    for label, theta1, sort_col in [("3-class 0.20", 0.20, "p_any_pead"),
                                     (f"2-stage {best_theta:.2f}", best_theta, "pred_car")]:
        for fi in range(1, 5):
            td = fold_data[fi]["test_df"]
            if label.startswith("3-class"):
                mask = (td["p_any_pead"] >= 0.20) & (td["pregap_return"].notna())
            else:
                mask = (td["p_binary"] >= theta1) & (td["pregap_return"].notna())
            picks = td[mask].copy()
            if len(picks) == 0:
                continue
            picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
            picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
            picks["fold"] = fi
            sel = select_weekly_top_n(picks, N_SLOTS, sort_col=sort_col)
            if len(sel) == 0:
                continue
            pnls = sel["pregap_return"].dropna()
            n = len(pnls)
            wr = (pnls > 0).mean() * 100
            avg = pnls.mean() * 100
            total = pnls.sum() * 100
            n_large = int((sel["label_3class"] == 2).sum())
            print(f"  {label:<20} {fi:>5} {n:>4} {wr:>5.1f}% {avg:>+7.2f}% {total:>+7.1f}% {n_large:>6}")
        print()

    print(f"{'='*110}")


if __name__ == "__main__":
    main()

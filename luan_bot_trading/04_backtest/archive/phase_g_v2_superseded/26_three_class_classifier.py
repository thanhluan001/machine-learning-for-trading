#!/usr/bin/env python3
"""
Lever 4: 3-class softprob classifier targeting {no PEAD, small PEAD, large PEAD}.

Label definition (CAR threshold = 10% linear):
  Class 0 (no PEAD):    pead_pass == 0 (fails >= 1 gate)
  Class 1 (small PEAD): pead_pass == 1 AND CAR_10d < 10%
  Class 2 (large PEAD):  pead_pass == 1 AND CAR_10d >= 10%

Inference logic:
  - P(large) = P(class 2)
  - P(any PEAD) = P(class 1) + P(class 2)
  - Trade if P(large) >= theta_large  → targets high-precision large-PEAD capture
  - Also report P(any PEAD) sweep for binary-equivalent comparison

Uses pre-gap entry (Close[T-1] BMO / Close[T] AMC), 5-day hold,
weekly batch selection, 4 slots, same 24 Sunday-safe features.
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
ps_spec = importlib.util.spec_from_file_location("ps", HERE / "04_phase_g_portfolio.py")
ps = importlib.util.module_from_spec(ps_spec); ps_spec.loader.exec_module(ps)
rb_spec = importlib.util.spec_from_file_location("rb", HERE / "_phase_g_random_baseline.py")
rb = importlib.util.module_from_spec(rb_spec); rb_spec.loader.exec_module(rb)

DB = tm.DB_FILE
SUNDAY_SAFE = pg.SUNDAY_SAFE_FEATURES
N_SLOTS = 4
EXIT_SNAP = 5
CAR_LARGE_THRESH = 10.0  # % linear CAR threshold for "large PEAD"

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]

# Thresholds to sweep for P(large PEAD)
THETAS_LARGE = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
# Thresholds to sweep for P(any PEAD) — binary-equivalent comparison
THETAS_ANY = [0.20, 0.25, 0.30, 0.35]


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


def select_weekly_top_n(picks, n_slots=4, sort_col="p_large"):
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
        n_take = min(free_slots, len(week_sorted))
        taken = week_sorted.head(n_take)
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


def main():
    print("=" * 100)
    print(f"3-CLASS SOFTPROB CLASSIFIER (Lever 4)")
    print(f"  CAR threshold for 'large PEAD': {CAR_LARGE_THRESH}%")
    print(f"  Entry: pre-gap, Exit: Close[T+5], {N_SLOTS} slots, weekly batch")
    print(f"  Features: {len(SUNDAY_SAFE)} Sunday-safe, gamma=3 fixed")
    print("=" * 100)

    # Load + prime + gates
    print("\n[1] Loading + priming + gates ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)

    # Build 3-class label
    df["car_10d_pct"] = np.expm1(df["car_10d"]) * 100
    df["label_3class"] = 0  # default: no PEAD
    mask_small = (df["pead_pass"] == 1) & (df["car_10d_pct"] < CAR_LARGE_THRESH)
    mask_large = (df["pead_pass"] == 1) & (df["car_10d_pct"] >= CAR_LARGE_THRESH)
    df.loc[mask_small, "label_3class"] = 1
    df.loc[mask_large, "label_3class"] = 2

    n_no = int((df["label_3class"] == 0).sum())
    n_small = int((df["label_3class"] == 1).sum())
    n_large = int((df["label_3class"] == 2).sum())
    print(f"    3-class labels: no_PEAD={n_no} ({n_no/len(df)*100:.1f}%), "
          f"small_PEAD={n_small} ({n_small/len(df)*100:.1f}%), "
          f"large_PEAD={n_large} ({n_large/len(df)*100:.1f}%)")

    # Compute pre-gap returns
    print("[2] Computing pre-gap returns ...")
    df = compute_pregap_returns(df, DB, EXIT_SNAP)
    n_valid = df["pregap_return"].notna().sum()
    print(f"    {n_valid}/{len(df)} rows have valid pre-gap returns")

    # Pre-train 3-class + binary classifiers per fold
    print("\n[3] Pre-training classifiers per fold ...")
    fold_data = {}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()

        X_tr = train_df[SUNDAY_SAFE]
        X_sv = sweep_df[SUNDAY_SAFE]
        X_te = test_df[SUNDAY_SAFE]
        X_ts = pd.concat([X_tr, X_sv])

        # 3-class labels
        y_tr_3 = train_df["label_3class"].values
        y_sv_3 = sweep_df["label_3class"].values
        y_te_3 = test_df["label_3class"].values
        y_ts_3 = np.concatenate([y_tr_3, y_sv_3])

        # Binary labels (for comparison)
        y_tr_b = train_df["pead_pass"].astype(int).values
        y_sv_b = sweep_df["pead_pass"].astype(int).values
        y_te_b = test_df["pead_pass"].astype(int).values
        y_ts_b = np.concatenate([y_tr_b, y_sv_b])

        hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}

        # Train 3-class
        clf_3 = fit_clf_3class(X_ts, y_ts_3, X_te, y_te_3, hp)
        proba_3 = clf_3.predict_proba(X_te)  # shape (n, 3)
        test_df = test_df.copy()
        test_df["p_no_pead"] = proba_3[:, 0]
        test_df["p_small"] = proba_3[:, 1]
        test_df["p_large"] = proba_3[:, 2]
        test_df["p_any_pead"] = proba_3[:, 1] + proba_3[:, 2]

        # Train binary (for comparison)
        clf_b = fit_clf_binary(X_ts, y_ts_b, X_te, y_te_b, hp)
        test_df["p_binary"] = clf_b.predict_proba(X_te)[:, 1]

        fold_data[fi] = {"test_df": test_df}
        print(f"    Fold {fi}: TEST={len(test_df)}, "
              f"large_PEAD={int((y_te_3==2).sum())}, "
              f"any_PEAD={int((y_te_3>=1).sum())}")

    # ===== SWEEP: P(large PEAD) threshold =====
    print(f"\n[4] Sweeping P(large PEAD) threshold ({len(THETAS_LARGE)} values) ...")
    print(f"\n{'theta_l':>7s}  {'Picks':>5s} {'Exec':>5s} {'L_PEAD':>6s} {'Any_PEAD':>8s} "
          f"{'L_Prec':>7s} {'AnyPrec':>7s} {'Recall_L':>8s} "
          f"{'AvgPnL':>8s} {'Win%':>6s} {'Payoff':>7s}")
    print("-" * 100)

    results_large = []
    for theta_l in THETAS_LARGE:
        all_picks_list = []
        all_exec_list = []
        total_large_in_test = 0
        total_any_in_test = 0

        for fi in range(1, 5):
            test_df = fold_data[fi]["test_df"]
            total_large_in_test += int((test_df["label_3class"] == 2).sum())
            total_any_in_test += int((test_df["label_3class"] >= 1).sum())

            mask = (test_df["p_large"] >= theta_l) & (test_df["pregap_return"].notna())
            picks = test_df[mask].copy()
            if len(picks) == 0:
                continue
            picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
            picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
            all_picks_list.append(picks)

            selected = select_weekly_top_n(picks, N_SLOTS)
            if len(selected) > 0:
                all_exec_list.append(selected)

        all_picks = pd.concat(all_picks_list) if all_picks_list else pd.DataFrame()
        all_exec = pd.concat(all_exec_list) if all_exec_list else pd.DataFrame()

        n_picks = len(all_picks)
        n_exec = len(all_exec)
        n_large_picks = int((all_picks["label_3class"] == 2).sum()) if n_picks > 0 else 0
        n_any_picks = int((all_picks["label_3class"] >= 1).sum()) if n_picks > 0 else 0
        n_large_exec = int((all_exec["label_3class"] == 2).sum()) if n_exec > 0 else 0
        n_any_exec = int((all_exec["label_3class"] >= 1).sum()) if n_exec > 0 else 0

        prec_large = n_large_picks / n_picks * 100 if n_picks > 0 else 0
        prec_any = n_any_picks / n_picks * 100 if n_picks > 0 else 0
        recall_large = n_large_picks / total_large_in_test * 100 if total_large_in_test > 0 else 0

        if n_exec > 0:
            pnls = all_exec["pregap_return"].dropna()
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            win_rate = len(wins) / len(pnls) * 100
            avg_pnl = pnls.mean() * 100
            avg_win = wins.mean() * 100 if len(wins) > 0 else 0
            avg_loss = losses.mean() * 100 if len(losses) > 0 else 0
            payoff = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
            prec_large_exec = n_large_exec / n_exec * 100
            prec_any_exec = n_any_exec / n_exec * 100
        else:
            win_rate = avg_pnl = avg_win = avg_loss = payoff = 0
            prec_large_exec = prec_any_exec = 0

        results_large.append({
            "theta": theta_l, "n_picks": n_picks, "n_exec": n_exec,
            "n_large": n_large_picks, "n_any": n_any_picks,
            "prec_large": prec_large, "prec_any": prec_any,
            "recall_large": recall_large, "avg_pnl": avg_pnl,
            "win_rate": win_rate, "payoff": payoff,
            "prec_large_exec": prec_large_exec, "prec_any_exec": prec_any_exec,
        })

        print(f"{theta_l:>7.2f}  {n_picks:>5d} {n_exec:>5d} {n_large_picks:>6d} {n_any_picks:>8d} "
              f"{prec_large:>6.1f}% {prec_any:>6.1f}% {recall_large:>7.1f}% "
              f"{avg_pnl:>+7.2f}% {win_rate:>5.1f}% {payoff:>6.2f}")

    # ===== SWEEP: P(any PEAD) threshold (binary-equivalent comparison) =====
    print(f"\n[5] Sweeping P(any PEAD) threshold — binary-equivalent comparison ({len(THETAS_ANY)} values) ...")
    print(f"\n{'theta_a':>7s}  {'Picks':>5s} {'Exec':>5s} {'Any_PEAD':>8s} {'AnyPrec':>7s} "
          f"{'L_in_exec':>9s} {'L_Prec%':>7s} "
          f"{'AvgPnL':>8s} {'Win%':>6s} {'Payoff':>7s}")
    print("-" * 100)

    results_any = []
    for theta_a in THETAS_ANY:
        all_picks_list = []
        all_exec_list = []
        total_any_in_test = 0

        for fi in range(1, 5):
            test_df = fold_data[fi]["test_df"]
            total_any_in_test += int((test_df["label_3class"] >= 1).sum())

            # Sort by p_any_pead for weekly selection
            mask = (test_df["p_any_pead"] >= theta_a) & (test_df["pregap_return"].notna())
            picks = test_df[mask].copy()
            if len(picks) == 0:
                continue
            picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
            picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
            all_picks_list.append(picks)

            selected = select_weekly_top_n(picks, N_SLOTS, sort_col="p_any_pead")
            if len(selected) > 0:
                all_exec_list.append(selected)

        all_picks = pd.concat(all_picks_list) if all_picks_list else pd.DataFrame()
        all_exec = pd.concat(all_exec_list) if all_exec_list else pd.DataFrame()

        n_picks = len(all_picks)
        n_exec = len(all_exec)
        n_any_picks = int((all_picks["label_3class"] >= 1).sum()) if n_picks > 0 else 0
        n_large_exec = int((all_exec["label_3class"] == 2).sum()) if n_exec > 0 else 0
        n_any_exec = int((all_exec["label_3class"] >= 1).sum()) if n_exec > 0 else 0

        prec_any = n_any_picks / n_picks * 100 if n_picks > 0 else 0
        recall_any = n_any_picks / total_any_in_test * 100 if total_any_in_test > 0 else 0
        prec_large_in_exec = n_large_exec / n_exec * 100 if n_exec > 0 else 0

        if n_exec > 0:
            pnls = all_exec["pregap_return"].dropna()
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            win_rate = len(wins) / len(pnls) * 100
            avg_pnl = pnls.mean() * 100
            payoff = wins.mean() / abs(losses.mean()) * 100 / 100 if len(losses) > 0 and losses.mean() != 0 else float('inf')
            payoff = wins.mean() * 100 / abs(losses.mean() * 100) if len(losses) > 0 and losses.mean() != 0 else float('inf')
        else:
            win_rate = avg_pnl = payoff = 0

        results_any.append({
            "theta": theta_a, "n_picks": n_picks, "n_exec": n_exec,
            "n_any": n_any_picks, "prec_any": prec_any, "recall_any": recall_any,
            "n_large_exec": n_large_exec, "prec_large_in_exec": prec_large_in_exec,
            "avg_pnl": avg_pnl, "win_rate": win_rate, "payoff": payoff,
        })

        print(f"{theta_a:>7.2f}  {n_picks:>5d} {n_exec:>5d} {n_any_picks:>8d} {prec_any:>6.1f}% "
              f"{n_large_exec:>9d} {prec_large_in_exec:>6.1f}% "
              f"{avg_pnl:>+7.2f}% {win_rate:>5.1f}% {payoff:>6.2f}")

    # ===== COMPARISON: Binary vs 3-class =====
    print(f"\n{'='*100}")
    print("COMPARISON: Binary classifier (theta=0.25) vs 3-class P(large) sweep")
    print(f"{'='*100}")

    # Binary at theta=0.25 (from the theta sweep we already ran)
    print(f"\n  Binary classifier (P(PEAD) >= 0.25):")
    print(f"    Refer to 25_theta_sweep_pregap.py results:")
    print(f"    ~86 picks, ~69 executed, 34.9% precision, +7.61% PnL, 65.2% win")

    # Find best 3-class P(large) threshold
    print(f"\n  3-class P(large PEAD) sweep highlights:")
    for r in results_large:
        if r["n_exec"] >= 15:  # only show thresholds with enough trades
            print(f"    theta_l={r['theta']:.2f}: picks={r['n_picks']}, exec={r['n_exec']}, "
                  f"large_prec={r['prec_large']:.1f}%, any_prec={r['prec_any']:.1f}%, "
                  f"pnl={r['avg_pnl']:+.2f}%, win={r['win_rate']:.1f}%, payoff={r['payoff']:.2f}")

    # Find best 3-class P(any) threshold
    print(f"\n  3-class P(any PEAD) sweep highlights:")
    for r in results_any:
        if r["n_exec"] >= 15:
            print(f"    theta_a={r['theta']:.2f}: picks={r['n_picks']}, exec={r['n_exec']}, "
                  f"any_prec={r['prec_any']:.1f}%, large_in_exec={r['prec_large_in_exec']:.1f}%, "
                  f"pnl={r['avg_pnl']:+.2f}%, win={r['win_rate']:.1f}%, payoff={r['payoff']:.2f}")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

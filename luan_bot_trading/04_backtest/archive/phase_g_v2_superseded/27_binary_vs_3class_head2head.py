#!/usr/bin/env python3
"""
Head-to-head comparison: Binary theta=0.25 vs 3-class P(any)>=0.25.

Both operating points use:
  - Pre-gap entry (Close[T-1] BMO / Close[T] AMC)
  - 5-day hold (exit Close[T+5])
  - Weekly batch selection, 4 slots
  - Same 24 Sunday-safe features, gamma=3

Reports per-fold breakdown, total return (sum of all trades),
trade-level stats, and PEAD precision for a fair comparison.
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
CAR_LARGE_THRESH = 10.0

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]


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


def run_scenario(fold_data, theta, prob_col, label_col="label_3class"):
    """Run weekly batch selection + collect executed trades for a given theta."""
    all_exec_list = []
    for fi in range(1, 5):
        test_df = fold_data[fi]["test_df"]
        mask = (test_df[prob_col] >= theta) & (test_df["pregap_return"].notna())
        picks = test_df[mask].copy()
        if len(picks) == 0:
            continue
        picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
        picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
        picks["fold"] = fi
        selected = select_weekly_top_n(picks, N_SLOTS, sort_col=prob_col)
        if len(selected) > 0:
            all_exec_list.append(selected)
    return pd.concat(all_exec_list) if all_exec_list else pd.DataFrame()


def print_trade_stats(df, label):
    if len(df) == 0:
        print(f"  {label:<40} N=0")
        return
    pnls = df["pregap_return"].dropna()
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n = len(pnls)
    wr = len(wins) / n * 100
    avg = pnls.mean() * 100
    aw = wins.mean() * 100 if len(wins) > 0 else 0
    al = losses.mean() * 100 if len(losses) > 0 else 0
    payoff = aw / abs(al) if al != 0 else float('inf')
    total = pnls.sum() * 100
    # PEAD precision
    n_any = int((df["label_3class"] >= 1).sum())
    n_large = int((df["label_3class"] == 2).sum())
    prec_any = n_any / n * 100
    prec_large = n_large / n * 100
    print(f"  {label:<40} N={n:>3}  Win={wr:>5.1f}%  Avg={avg:>+6.2f}%  "
          f"Total={total:>+7.1f}%  Payoff={payoff:>5.2f}")
    print(f"  {'':40} AnyPEAD={n_any:>3} ({prec_any:>4.1f}%)  "
          f"LargePEAD={n_large:>3} ({prec_large:>4.1f}%)")


def main():
    print("=" * 100)
    print("HEAD-TO-HEAD: Binary theta=0.25 vs 3-class P(any)>=0.25")
    print(f"  Pre-gap entry, 5-day hold, {N_SLOTS} slots, weekly batch")
    print("=" * 100)

    print("\n[1] Loading + priming + gates ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)

    # 3-class label
    df["car_10d_pct"] = np.expm1(df["car_10d"]) * 100
    df["label_3class"] = 0
    df.loc[(df["pead_pass"] == 1) & (df["car_10d_pct"] < CAR_LARGE_THRESH), "label_3class"] = 1
    df.loc[(df["pead_pass"] == 1) & (df["car_10d_pct"] >= CAR_LARGE_THRESH), "label_3class"] = 2

    print("[2] Computing pre-gap returns ...")
    df = compute_pregap_returns(df, DB, EXIT_SNAP)

    print("[3] Training classifiers per fold ...")
    fold_data = {}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_tr = train_df[SUNDAY_SAFE]; X_sv = sweep_df[SUNDAY_SAFE]
        X_te = test_df[SUNDAY_SAFE]
        X_ts = pd.concat([X_tr, X_sv])
        y_tr_b = train_df["pead_pass"].astype(int).values
        y_sv_b = sweep_df["pead_pass"].astype(int).values
        y_te_b = test_df["pead_pass"].astype(int).values
        y_ts_b = np.concatenate([y_tr_b, y_sv_b])
        y_tr_3 = train_df["label_3class"].values
        y_sv_3 = sweep_df["label_3class"].values
        y_te_3 = test_df["label_3class"].values
        y_ts_3 = np.concatenate([y_tr_3, y_sv_3])
        hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}

        clf_b = fit_clf_binary(X_ts, y_ts_b, X_te, y_te_b, hp)
        clf_3 = fit_clf_3class(X_ts, y_ts_3, X_te, y_te_3, hp)
        proba_3 = clf_3.predict_proba(X_te)
        test_df = test_df.copy()
        test_df["p_binary"] = clf_b.predict_proba(X_te)[:, 1]
        test_df["p_any_pead"] = proba_3[:, 1] + proba_3[:, 2]
        test_df["p_large"] = proba_3[:, 2]
        fold_data[fi] = {"test_df": test_df}

    # Run both scenarios
    print("\n[4] Running both scenarios ...")
    binary_exec = run_scenario(fold_data, 0.25, "p_binary")
    threeclass_exec = run_scenario(fold_data, 0.25, "p_any_pead")

    # ===== COMPARISON =====
    print(f"\n{'='*100}")
    print("HEAD-TO-HEAD COMPARISON")
    print(f"{'='*100}")

    print(f"\n1. AGGREGATE TRADE-LEVEL STATS")
    print(f"\n  {'Scenario':<40} {'N':>4}  {'Win%':>6}  {'Avg':>7}  {'Total':>8}  {'Payoff':>6}")
    print_trade_stats(binary_exec, "A. Binary theta=0.25 (P(PEAD))")
    print()
    print_trade_stats(threeclass_exec, "B. 3-class P(any)>=0.25")

    # Per-fold breakdown
    print(f"\n2. PER-FOLD BREAKDOWN")
    print(f"\n  {'Scenario':<20} {'Fold':>5} {'N':>4} {'PEAD':>5} {'Prec':>6} {'Win%':>6} {'Avg':>8} {'Total':>8}")
    for label, exec_df, prob_col in [("Binary", binary_exec, "p_binary"),
                                      ("3-class", threeclass_exec, "p_any_pead")]:
        for fi in range(1, 5):
            sub = exec_df[exec_df["fold"] == fi]
            if len(sub) == 0:
                print(f"  {label:<20} {fi:>5} {0:>4}")
                continue
            pnls = sub["pregap_return"].dropna()
            n = len(pnls)
            n_pead = int((sub["label_3class"] >= 1).sum())
            prec = n_pead / n * 100 if n > 0 else 0
            wr = (pnls > 0).mean() * 100
            avg = pnls.mean() * 100
            total = pnls.sum() * 100
            print(f"  {label:<20} {fi:>5} {n:>4} {n_pead:>5} {prec:>5.1f}% {wr:>5.1f}% {avg:>+7.2f}% {total:>+7.1f}%")

    # Summary comparison
    print(f"\n3. SUMMARY COMPARISON")
    b_pnls = binary_exec["pregap_return"].dropna()
    t_pnls = threeclass_exec["pregap_return"].dropna()
    b_pead = int((binary_exec["label_3class"] >= 1).sum())
    t_pead = int((threeclass_exec["label_3class"] >= 1).sum())
    b_large = int((binary_exec["label_3class"] == 2).sum())
    t_large = int((threeclass_exec["label_3class"] == 2).sum())

    print(f"\n  {'Metric':<30} {'Binary 0.25':>15} {'3-class 0.25':>15} {'Delta':>10}")
    print(f"  {'-'*70}")
    print(f"  {'Executed trades':<30} {len(b_pnls):>15} {len(t_pnls):>15} {len(t_pnls)-len(b_pnls):>+10}")
    print(f"  {'Any PEAD in executed':<30} {b_pead:>15} {t_pead:>15} {t_pead-b_pead:>+10}")
    print(f"  {'Large PEAD in executed':<30} {b_large:>15} {t_large:>15} {t_large-b_large:>+10}")
    print(f"  {'PEAD precision':<30} {b_pead/len(b_pnls)*100:>14.1f}% {t_pead/len(t_pnls)*100:>14.1f}% {(t_pead/len(t_pnls)-b_pead/len(b_pnls))*100:>+9.1f}pp")
    print(f"  {'Large PEAD precision':<30} {b_large/len(b_pnls)*100:>14.1f}% {t_large/len(t_pnls)*100:>14.1f}% {(t_large/len(t_pnls)-b_large/len(b_pnls))*100:>+9.1f}pp")
    print(f"  {'Win rate':<30} {(b_pnls>0).mean()*100:>14.1f}% {(t_pnls>0).mean()*100:>14.1f}% {((t_pnls>0).mean()-(b_pnls>0).mean())*100:>+9.1f}pp")
    print(f"  {'Avg win':<30} {b_pnls[b_pnls>0].mean()*100:>+14.2f}% {t_pnls[t_pnls>0].mean()*100:>+14.2f}% {(t_pnls[t_pnls>0].mean()-b_pnls[b_pnls>0].mean())*100:>+9.2f}pp")
    print(f"  {'Avg loss':<30} {b_pnls[b_pnls<=0].mean()*100:>+14.2f}% {t_pnls[t_pnls<=0].mean()*100:>+14.2f}% {(t_pnls[t_pnls<=0].mean()-b_pnls[b_pnls<=0].mean())*100:>+9.2f}pp")
    print(f"  {'Avg PnL per trade':<30} {b_pnls.mean()*100:>+14.2f}% {t_pnls.mean()*100:>+14.2f}% {(t_pnls.mean()-b_pnls.mean())*100:>+9.2f}pp")
    print(f"  {'TOTAL PnL (sum)':<30} {b_pnls.sum()*100:>+14.1f}% {t_pnls.sum()*100:>+14.1f}% {(t_pnls.sum()-b_pnls.sum())*100:>+9.1f}pp")
    print(f"  {'Std per trade':<30} {b_pnls.std()*100:>14.2f}% {t_pnls.std()*100:>14.2f}%")

    # Total return analysis (the user's key metric)
    print(f"\n4. TOTAL RETURN ANALYSIS (the user's key metric)")
    b_total = b_pnls.sum() * 100
    t_total = t_pnls.sum() * 100
    b_per_trade = b_pnls.mean() * 100
    t_per_trade = t_pnls.mean() * 100
    print(f"\n  Binary theta=0.25:   {len(b_pnls)} trades x {b_per_trade:+.2f}% = {b_total:+.1f}% total")
    print(f"  3-class P(any)>=0.25: {len(t_pnls)} trades x {t_per_trade:+.2f}% = {t_total:+.1f}% total")
    print(f"  Delta:                {len(t_pnls)-len(b_pnls):+d} trades, {t_per_trade-b_per_trade:+.2f}%/trade, {t_total-b_total:+.1f}% total")
    if b_total > 0:
        print(f"  3-class total / Binary total = {t_total/b_total:.2f}x")
    print(f"\n  => 3-class is {'BETTER' if t_total > b_total else 'WORSE'} on total return "
          f"by {abs(t_total-b_total):.1f}% ({abs(t_total-b_total)/max(abs(b_total),0.01)*100:.1f}%)")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

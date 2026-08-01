#!/usr/bin/env python3
"""
Test 5-day vs 10-day hold with 3-class model + pre-gap entry.
Also compute bootstrap CIs to verify robustness.

Addresses two user concerns:
  1. "Is this too good?" -> bootstrap CI on the operating point
  2. "Should we increase to 10-day hold?" -> test both hold periods
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
CAR_LARGE_THRESH = 10.0
THETA = 0.20

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]


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


def compute_pregap_returns(df, db_path, hold_days):
    """Compute pre-gap returns for a given hold period."""
    df = df.copy()
    col = f"pregap_return_{hold_days}d"
    entry_col = f"pregap_entry_date_{hold_days}d"
    exit_col = f"pregap_exit_date_{hold_days}d"
    df[col] = np.nan
    df[entry_col] = pd.NaT
    df[exit_col] = pd.NaT
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
            df.at[idx, col] = float(exit_price / entry_price - 1.0)
            df.at[idx, entry_col] = pd.Timestamp(p_index[entry_t])
            df.at[idx, exit_col] = pd.Timestamp(p_index[exit_t])
    return df


def run_scenario(fold_data, theta, hold_days):
    """Run weekly batch selection for a given hold period."""
    prob_col = "p_any_pead"
    ret_col = f"pregap_return_{hold_days}d"
    entry_col = f"pregap_entry_date_{hold_days}d"
    exit_col = f"pregap_exit_date_{hold_days}d"
    all_exec_list = []
    for fi in range(1, 5):
        test_df = fold_data[fi]["test_df"]
        mask = (test_df[prob_col] >= theta) & (test_df[ret_col].notna())
        picks = test_df[mask].copy()
        if len(picks) == 0:
            continue
        picks["entry_date"] = pd.to_datetime(picks[entry_col])
        picks["exit_date"] = pd.to_datetime(picks[exit_col])
        picks["pregap_return"] = picks[ret_col]
        picks["fold"] = fi
        selected = select_weekly_top_n(picks, N_SLOTS, sort_col=prob_col)
        if len(selected) > 0:
            all_exec_list.append(selected)
    return pd.concat(all_exec_list) if all_exec_list else pd.DataFrame()


def print_stats(exec_df, label):
    if len(exec_df) == 0:
        print(f"  {label}: N=0")
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
    print(f"  {label}")
    print(f"    N={n}, Win={wr:.1f}%, Avg={avg:+.2f}%, Total={total:+.1f}%, Payoff={payoff:.2f}")
    print(f"    PEAD={n_pead} ({prec:.1f}%), Large={n_large} ({n_large/n*100:.1f}%)")
    print(f"    AvgWin={aw:+.2f}%, AvgLoss={al:+.2f}%, Std={pnls.std()*100:.2f}%")


def main():
    print("=" * 95)
    print(f"5-DAY vs 10-DAY HOLD (3-class P(any)>={THETA}) + BOOTSTRAP CI")
    print("=" * 95)

    print("\n[1] Loading + priming + gates ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)

    df["car_10d_pct"] = np.expm1(df["car_10d"]) * 100
    df["label_3class"] = 0
    df.loc[(df["pead_pass"] == 1) & (df["car_10d_pct"] < CAR_LARGE_THRESH), "label_3class"] = 1
    df.loc[(df["pead_pass"] == 1) & (df["car_10d_pct"] >= CAR_LARGE_THRESH), "label_3class"] = 2

    # Compute returns for BOTH hold periods
    print("[2] Computing pre-gap returns for 5-day and 10-day holds ...")
    df = compute_pregap_returns(df, DB, 5)
    df = compute_pregap_returns(df, DB, 10)
    print(f"    5-day valid: {df['pregap_return_5d'].notna().sum()}")
    print(f"    10-day valid: {df['pregap_return_10d'].notna().sum()}")

    print("[3] Training 3-class classifiers per fold ...")
    fold_data = {}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_tr = train_df[SUNDAY_SAFE]; X_sv = sweep_df[SUNDAY_SAFE]
        X_te = test_df[SUNDAY_SAFE]
        X_ts = pd.concat([X_tr, X_sv])
        y_tr_3 = train_df["label_3class"].values
        y_sv_3 = sweep_df["label_3class"].values
        y_te_3 = test_df["label_3class"].values
        y_ts_3 = np.concatenate([y_tr_3, y_sv_3])
        hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
        clf_3 = fit_clf_3class(X_ts, y_ts_3, X_te, y_te_3, hp)
        proba_3 = clf_3.predict_proba(X_te)
        test_df = test_df.copy()
        test_df["p_any_pead"] = proba_3[:, 1] + proba_3[:, 2]
        fold_data[fi] = {"test_df": test_df}

    # ===== 5-DAY vs 10-DAY HOLD COMPARISON =====
    print(f"\n{'='*95}")
    print("HOLD PERIOD COMPARISON (3-class P(any)>=0.20, pre-gap entry)")
    print(f"{'='*95}")

    exec_5d = run_scenario(fold_data, THETA, 5)
    exec_10d = run_scenario(fold_data, THETA, 10)

    print(f"\n1. AGGREGATE STATS")
    print_stats(exec_5d, "5-day hold (Close[T+5] exit)")
    print()
    print_stats(exec_10d, "10-day hold (Close[T+10] exit)")

    # Per-fold
    print(f"\n2. PER-FOLD BREAKDOWN")
    print(f"  {'Hold':>5} {'Fold':>5} {'N':>4} {'Win%':>6} {'Avg':>8} {'Total':>8}")
    for label, exec_df in [("5d", exec_5d), ("10d", exec_10d)]:
        for fi in range(1, 5):
            sub = exec_df[exec_df["fold"] == fi]
            if len(sub) == 0:
                print(f"  {label:>5} {fi:>5} {0:>4}")
                continue
            pnls = sub["pregap_return"].dropna()
            n = len(pnls)
            wr = (pnls > 0).mean() * 100
            avg = pnls.mean() * 100
            total = pnls.sum() * 100
            print(f"  {label:>5} {fi:>5} {n:>4} {wr:>5.1f}% {avg:>+7.2f}% {total:>+7.1f}%")
        print()

    # ===== BOOTSTRAP CI (5-day hold) =====
    print(f"\n{'='*95}")
    print(f"BOOTSTRAP CI (5-day hold, 3-class P(any)>={THETA})")
    print(f"{'='*95}")

    pnls_5d = exec_5d["pregap_return"].dropna().values
    n_trades = len(pnls_5d)

    # Bootstrap
    rng = np.random.default_rng(42)
    N_BOOT = 10000
    boot_means = np.zeros(N_BOOT)
    boot_winrates = np.zeros(N_BOOT)
    boot_totals = np.zeros(N_BOOT)
    boot_precisions = np.zeros(N_BOOT)
    labels = exec_5d["label_3class"].values

    for b in range(N_BOOT):
        idx = rng.integers(0, n_trades, size=n_trades)
        sample = pnls_5d[idx]
        boot_means[b] = sample.mean()
        boot_winrates[b] = (sample > 0).mean()
        boot_totals[b] = sample.sum()
        boot_precisions[b] = (labels[idx] >= 1).mean()

    print(f"\n  Bootstrap: {N_BOOT} resamples of {n_trades} trades")
    print(f"\n  {'Metric':<25} {'Mean':>10} {'95% CI':>20}")
    print(f"  {'-'*55}")
    print(f"  {'Expectancy/trade':<25} {boot_means.mean()*100:>+9.2f}% "
          f"[{np.percentile(boot_means,2.5)*100:+.2f}%, {np.percentile(boot_means,97.5)*100:+.2f}%]")
    print(f"  {'Win rate':<25} {boot_winrates.mean()*100:>9.1f}% "
          f"[{np.percentile(boot_winrates,2.5)*100:.1f}%, {np.percentile(boot_winrates,97.5)*100:.1f}%]")
    print(f"  {'Total PnL (sum)':<25} {boot_totals.mean()*100:>+9.1f}% "
          f"[{np.percentile(boot_totals,2.5)*100:+.1f}%, {np.percentile(boot_totals,97.5)*100:+.1f}%]")
    print(f"  {'PEAD precision':<25} {boot_precisions.mean()*100:>9.1f}% "
          f"[{np.percentile(boot_precisions,2.5)*100:.1f}%, {np.percentile(boot_precisions,97.5)*100:.1f}%]")

    # Check if CIs exclude 0
    ci_low = np.percentile(boot_means, 2.5) * 100
    ci_high = np.percentile(boot_means, 97.5) * 100
    print(f"\n  Expectancy CI excludes 0? {'YES' if ci_low > 0 else 'NO'} (CI: [{ci_low:+.2f}%, {ci_high:+.2f}%])")

    ci_low_total = np.percentile(boot_totals, 2.5) * 100
    ci_high_total = np.percentile(boot_totals, 97.5) * 100
    print(f"  Total PnL CI excludes 0? {'YES' if ci_low_total > 0 else 'NO'} (CI: [{ci_low_total:+.1f}%, {ci_high_total:+.1f}%])")

    # Large PEAD bootstrap
    large_pnls = exec_5d[exec_5d["label_3class"] == 2]["pregap_return"].dropna().values
    if len(large_pnls) > 0:
        boot_large_wr = np.zeros(N_BOOT)
        boot_large_avg = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, len(large_pnls), size=len(large_pnls))
            sample = large_pnls[idx]
            boot_large_wr[b] = (sample > 0).mean()
            boot_large_avg[b] = sample.mean()
        print(f"\n  Large PEAD class ({len(large_pnls)} trades) bootstrap:")
        print(f"    Win rate:    {boot_large_wr.mean()*100:.1f}% "
              f"[{np.percentile(boot_large_wr,2.5)*100:.1f}%, {np.percentile(boot_large_wr,97.5)*100:.1f}%]")
        print(f"    Avg return:  {boot_large_avg.mean()*100:+.2f}% "
              f"[{np.percentile(boot_large_avg,2.5)*100:+.2f}%, {np.percentile(boot_large_avg,97.5)*100:+.2f}%]")
        print(f"    (n={len(large_pnls)} is small -- CI is wide)")

    # ===== SLOT UTILIZATION =====
    print(f"\n{'='*95}")
    print("SLOT UTILIZATION ANALYSIS")
    print(f"{'='*95}")

    for label, exec_df, hold_days in [("5-day", exec_5d, 5), ("10-day", exec_10d, 10)]:
        exec_df = exec_df.copy()
        exec_df["entry_date"] = pd.to_datetime(exec_df["entry_date"])
        exec_df["exit_date"] = pd.to_datetime(exec_df["exit_date"])
        # Count trades per week
        exec_df["year_week"] = exec_df["entry_date"].dt.strftime("%G-W%V")
        weekly = exec_df.groupby("year_week").size()
        # Estimate slot utilization: each trade occupies a slot for hold_days trading days
        # ~5 trading days per week, so 5-day hold = 1 week, 10-day hold = 2 weeks
        weeks_per_hold = hold_days / 5
        avg_trades_per_week = weekly.mean()
        # If we have avg_trades_per_week trades entering each week, and each holds weeks_per_hold weeks,
        # avg open positions = avg_trades_per_week * weeks_per_hold
        avg_open = avg_trades_per_week * weeks_per_hold
        util = avg_open / N_SLOTS * 100
        print(f"\n  {label} hold:")
        print(f"    Trades/week:       {avg_trades_per_week:.1f}")
        print(f"    Weeks per hold:    {weeks_per_hold:.1f}")
        print(f"    Avg open positions: {avg_open:.1f}")
        print(f"    Slot utilization:  {util:.0f}% ({avg_open:.1f}/{N_SLOTS} slots)")

    print(f"\n{'='*95}")


if __name__ == "__main__":
    main()

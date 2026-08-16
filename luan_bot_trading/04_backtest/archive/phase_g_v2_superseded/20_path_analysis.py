#!/usr/bin/env python3
"""
Path analysis: max drawdown during hold for winning vs losing trades.

Hypothesis: PEAD events lean positive (institutional drift). Even wrong
picks should have PEAD-like characteristics (high volume, good ratings),
so losses should be small (-2%). The -4% to -7% avg losses seem too
negative. This script analyzes the intra-hold path to understand why.

Questions:
  1. Max drawdown during hold for winning trades (do winners dip first?)
  2. Max drawdown during hold for losing trades (do losers crash or drift?)
  3. Path shape: immediate gap-down vs gradual decline
  4. Final return vs max drawdown relationship
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
THETA = 0.25
N_SLOTS = 4
HOLD_DAYS = 5
EXIT_SNAP = 5

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]


def select_weekly_top_n(picks, n_slots=4):
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
        week_sorted = week_df.sort_values("p", ascending=False)
        n_take = min(free_slots, len(week_sorted))
        taken = week_sorted.head(n_take)
        selected_rows.append(taken)
        for _, row in taken.iterrows():
            active_positions.append(row["exit_date"])
    if selected_rows:
        result = pd.concat(selected_rows).sort_values("entry_date").reset_index(drop=True)
    else:
        result = pd.DataFrame(columns=picks.columns)
    return result


def fit_clf(X_tr, y_tr, X_val, y_val, hp):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=hp["n_estimators"], learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def compute_max_drawdown(path_values):
    """Max drawdown during hold from the path of cumulative returns.
    path_values = [r_t0, r_t1, ..., r_tN] where r_t0=0 (entry at open).
    Drawdown at time t = path[t] - max(path[0..t]).
    Returns the most negative drawdown (worst intra-hold pain)."""
    path = np.array(path_values, dtype=float)
    running_max = np.maximum.accumulate(path)
    drawdowns = path - running_max
    return float(np.min(drawdowns))


def main():
    print("=" * 80)
    print(f"PATH ANALYSIS — theta={THETA}, {N_SLOTS} slots, {HOLD_DAYS}-day hold")
    print("=" * 80)

    # Load + prime + gates + paths
    print("\n[1] Loading + priming + gates + paths ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)
    print(f"    rows: {len(df)}, pead: {int(df['pead_pass'].sum())}")

    # Cache calendar + remap exit to 5-day
    print("[2] Caching trading calendar ...")
    with pd.HDFStore(DB, mode="r") as sstore:
        all_dates = set()
        for key in sstore.keys():
            if not key.startswith("/sp400/"):
                continue
            d = sstore[key]
            all_dates.update(pd.to_datetime(d["Date"]).tolist())
    calendar = sorted(all_dates)
    print(f"    {len(calendar)} trading days")

    cal_idx = {d: i for i, d in enumerate(calendar)}
    df["_entry_idx"] = pd.to_datetime(df["entry_date"]).map(cal_idx)
    df["_exit_idx_new"] = df["_entry_idx"] + (EXIT_SNAP - 1)
    valid = df["_exit_idx_new"].notna() & (df["_exit_idx_new"] < len(calendar))
    df.loc[valid, "exit_date"] = df.loc[valid, "_exit_idx_new"].astype(int).map(lambda i: calendar[i])
    df["path_pnl_t11_pct"] = df[f"path_pnl_t{EXIT_SNAP}_pct"]
    df = df.drop(columns=["_entry_idx", "_exit_idx_new"])

    # Pre-train classifiers
    print("\n[3] Pre-training 4 fold classifiers (gamma=3) ...")
    fold_data = {}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_tr = train_df[SUNDAY_SAFE]
        y_tr = train_df["pead_pass"].astype(int).values
        X_sv = sweep_df[SUNDAY_SAFE]
        y_sv = sweep_df["pead_pass"].astype(int).values
        X_te = test_df[SUNDAY_SAFE]
        y_te = test_df["pead_pass"].astype(int).values
        X_ts = pd.concat([X_tr, X_sv])
        y_ts = np.concatenate([y_tr, y_sv])
        hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
        clf = fit_clf(X_ts, y_ts, X_te, y_te, hp)
        proba = clf.predict_proba(test_df[SUNDAY_SAFE])[:, 1]
        test_df = test_df.copy()
        test_df["p"] = proba
        mask = (test_df["p"] >= THETA) & (test_df["path_pnl_t11_pct"].notna())
        picks = test_df[mask].copy()
        fold_data[fi] = {"test_df": test_df, "picks": picks}
        print(f"    Fold {fi}: raw picks={len(picks)}, pead={int(picks['pead_pass'].sum())}")

    # Run simulation + collect paths
    print(f"\n[4] Running {N_SLOTS}-slot portfolio simulation per fold ...")
    all_trades = []

    for fi in range(1, 5):
        picks = fold_data[fi]["picks"]
        selected = select_weekly_top_n(picks, N_SLOTS)
        result = rb._simulate_with_cached_calendar(selected, N_SLOTS, 100_000.0, calendar)
        executed = result.get("trades_done", pd.DataFrame())
        if len(executed) == 0:
            continue

        # Join back to selected picks to get full path columns
        pead_lookup = selected[["permaTicker", "entry_date", "pead_pass", "p"]].copy()
        pead_lookup["entry_date"] = pd.to_datetime(pead_lookup["entry_date"])
        executed["entry_date"] = pd.to_datetime(executed["entry_date"])
        executed = executed.merge(pead_lookup, on=["permaTicker", "entry_date"], how="left")

        # Also join the path columns from selected picks
        path_cols = [f"path_pnl_t{t}_pct" for t in range(EXIT_SNAP + 1)]
        path_lookup = selected[["permaTicker", "entry_date"] + path_cols].copy()
        path_lookup["entry_date"] = pd.to_datetime(path_lookup["entry_date"])
        executed = executed.merge(path_lookup, on=["permaTicker", "entry_date"], how="left")

        executed["fold"] = fi
        all_trades.append(executed)

    all_trades_df = pd.concat(all_trades).reset_index(drop=True)
    print(f"\n  Total executed trades: {len(all_trades_df)}")

    # Compute max drawdown during hold for each trade
    print(f"\n[5] Computing max drawdown during hold for each trade ...")
    path_cols = [f"path_pnl_t{t}_pct" for t in range(EXIT_SNAP + 1)]
    path_arrays = all_trades_df[path_cols].values  # shape (n_trades, EXIT_SNAP+1)
    all_trades_df["max_dd_during_hold"] = [compute_max_drawdown(row) for row in path_arrays]
    # Also: the lowest point during the hold (min of path)
    all_trades_df["min_path"] = path_arrays.min(axis=1)
    # Final return
    all_trades_df["final_return"] = all_trades_df["realized_arith_pct"]
    # Win/loss classification
    all_trades_df["is_win"] = all_trades_df["final_return"] > 0
    all_trades_df["is_pead"] = all_trades_df["pead_pass"] == 1

    # ===== ANALYSIS =====
    print(f"\n{'='*80}")
    print("PATH ANALYSIS RESULTS")
    print(f"{'='*80}")

    n = len(all_trades_df)
    wins = all_trades_df[all_trades_df["is_win"]]
    losses = all_trades_df[~all_trades_df["is_win"]]

    print(f"\n1. OVERVIEW ({n} trades, {HOLD_DAYS}-day hold)")
    print(f"   Wins:   {len(wins)} ({len(wins)/n*100:.1f}%)")
    print(f"   Losses: {len(losses)} ({len(losses)/n*100:.1f}%)")

    print(f"\n2. FINAL RETURN DISTRIBUTION")
    print(f"   All trades:    mean={all_trades_df['final_return'].mean()*100:+.2f}%, "
          f"med={all_trades_df['final_return'].median()*100:+.2f}%")
    print(f"   Wins:          mean={wins['final_return'].mean()*100:+.2f}%, "
          f"med={wins['final_return'].median()*100:+.2f}%, "
          f"max={wins['final_return'].max()*100:+.2f}%")
    print(f"   Losses:        mean={losses['final_return'].mean()*100:+.2f}%, "
          f"med={losses['final_return'].median()*100:+.2f}%, "
          f"min={losses['final_return'].min()*100:+.2f}%")

    print(f"\n3. MAX DRAWDOWN DURING HOLD (worst intra-hold pain)")
    print(f"   All trades:    mean={all_trades_df['max_dd_during_hold'].mean()*100:+.2f}%, "
          f"med={all_trades_df['max_dd_during_hold'].median()*100:+.2f}%")
    print(f"   Wins:          mean={wins['max_dd_during_hold'].mean()*100:+.2f}%, "
          f"med={wins['max_dd_during_hold'].median()*100:+.2f}%")
    print(f"   Losses:        mean={losses['max_dd_during_hold'].mean()*100:+.2f}%, "
          f"med={losses['max_dd_during_hold'].median()*100:+.2f}%")

    print(f"\n4. MIN PATH (lowest point during hold)")
    print(f"   All trades:    mean={all_trades_df['min_path'].mean()*100:+.2f}%, "
          f"med={all_trades_df['min_path'].median()*100:+.2f}%")
    print(f"   Wins:          mean={wins['min_path'].mean()*100:+.2f}%, "
          f"med={wins['min_path'].median()*100:+.2f}%")
    print(f"   Losses:        mean={losses['min_path'].mean()*100:+.2f}%, "
          f"med={losses['min_path'].median()*100:+.2f}%")

    print(f"\n5. PEAD vs NON-PEAD BREAKDOWN")
    for label, mask in [("PEAD", all_trades_df["is_pead"]),
                        ("non-PEAD", ~all_trades_df["is_pead"])]:
        sub = all_trades_df[mask]
        sub_wins = sub[sub["is_win"]]
        sub_losses = sub[~sub["is_win"]]
        print(f"\n   {label} ({len(sub)} trades):")
        print(f"     Win rate:  {len(sub_wins)/len(sub)*100:.1f}%")
        print(f"     Final ret: mean={sub['final_return'].mean()*100:+.2f}%, "
              f"med={sub['final_return'].median()*100:+.2f}%")
        if len(sub_wins) > 0:
            print(f"     Wins:      mean={sub_wins['final_return'].mean()*100:+.2f}%, "
                  f"max={sub_wins['final_return'].max()*100:+.2f}%, "
                  f"maxDD={sub_wins['max_dd_during_hold'].mean()*100:+.2f}%")
        if len(sub_losses) > 0:
            print(f"     Losses:    mean={sub_losses['final_return'].mean()*100:+.2f}%, "
                  f"min={sub_losses['final_return'].min()*100:+.2f}%, "
                  f"maxDD={sub_losses['max_dd_during_hold'].mean()*100:+.2f}%")

    print(f"\n6. LOSS DISTRIBUTION (losing trades only, {len(losses)} trades)")
    loss_rets = losses["final_return"].values * 100
    print(f"   min={loss_rets.min():+.2f}%, "
          f"p10={np.percentile(loss_rets,10):+.2f}%, "
          f"p25={np.percentile(loss_rets,25):+.2f}%, "
          f"p50={np.percentile(loss_rets,50):+.2f}%, "
          f"p75={np.percentile(loss_rets,75):+.2f}%, "
          f"p90={np.percentile(loss_rets,90):+.2f}%, "
          f"max={loss_rets.max():+.2f}%")
    print(f"   Losses < -2%:  {(loss_rets < -2).sum()}")
    print(f"   Losses < -5%:  {(loss_rets < -5).sum()}")
    print(f"   Losses < -8%:  {(loss_rets < -8).sum()}")
    print(f"   Losses < -10%: {(loss_rets < -10).sum()}")

    print(f"\n7. PATH SHAPE ANALYSIS (losing trades)")
    # Did losers gap down immediately (t1 < -2%) or drift down gradually?
    loss_t1 = losses["path_pnl_t1_pct"].values * 100 if "path_pnl_t1_pct" in losses.columns else None
    loss_final = losses["final_return"].values * 100
    if loss_t1 is not None:
        print(f"   Day-1 return (t1):  mean={loss_t1.mean():+.2f}%, med={np.median(loss_t1):+.2f}%")
        print(f"   Day-1 < -2%:        {(loss_t1 < -2).sum()}/{len(losses)} "
              f"({(loss_t1 < -2).sum()/len(losses)*100:.1f}%)")
        print(f"   Day-1 < -5%:        {(loss_t1 < -5).sum()}/{len(losses)} "
              f"({(loss_t1 < -5).sum()/len(losses)*100:.1f}%)")
        # Did the loss widen from t1 to final?
        widened = (loss_final < loss_t1 - 0.5).sum()
        recovered = (loss_final > loss_t1 + 0.5).sum()
        same = len(losses) - widened - recovered
        print(f"\n   From t1 to final exit:")
        print(f"     Widened (got worse):  {widened} ({widened/len(losses)*100:.1f}%)")
        print(f"     Recovered (got better): {recovered} ({recovered/len(losses)*100:.1f}%)")
        print(f"     Same (~unchanged):    {same} ({same/len(losses)*100:.1f}%)")

    print(f"\n8. AVERAGE PATH BY OUTCOME (cumulative return at each day)")
    print(f"   {'Day':<6} {'All':>10} {'Wins':>10} {'Losses':>10} {'PEAD':>10} {'non-PEAD':>10}")
    for t in range(EXIT_SNAP + 1):
        col = f"path_pnl_t{t}_pct"
        all_v = all_trades_df[col].mean() * 100
        win_v = wins[col].mean() * 100 if col in wins.columns else 0
        loss_v = losses[col].mean() * 100 if col in losses.columns else 0
        pead_v = all_trades_df[all_trades_df["is_pead"]][col].mean() * 100
        npead_v = all_trades_df[~all_trades_df["is_pead"]][col].mean() * 100
        print(f"   t{t:<5} {all_v:>+9.2f}% {win_v:>+9.2f}% {loss_v:>+9.2f}% {pead_v:>+9.2f}% {npead_v:>+9.2f}%")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

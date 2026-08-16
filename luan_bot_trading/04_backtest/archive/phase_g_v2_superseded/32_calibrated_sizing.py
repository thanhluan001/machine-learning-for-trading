#!/usr/bin/env python3
"""
Test calibrated position sizing (instructor feedback #2).

Instead of equal-weight 1/4 NAV, size by P(large):
  w_i = P(large)_i / sum(P(large)_j for j in week's batch)

Also tests: does P(large) correlate with actual returns?
If not, calibrated sizing won't help (and may hurt).

Compares 3 sizing schemes:
  A. Equal-weight 1/4 NAV (current)
  B. P(large)-weighted (within each weekly batch)
  C. P(any)-weighted (within each weekly batch)
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
    print("=" * 95)
    print("CALIBRATED POSITION SIZING TEST (instructor feedback #2)")
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

    print("[2] Computing pre-gap returns ...")
    df = compute_pregap_returns(df, DB, EXIT_SNAP)

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
        test_df["p_large"] = proba_3[:, 2]
        fold_data[fi] = {"test_df": test_df}

    print("[4] Running weekly batch selection ...")
    all_exec_list = []
    for fi in range(1, 5):
        test_df = fold_data[fi]["test_df"]
        mask = (test_df["p_any_pead"] >= THETA) & (test_df["pregap_return"].notna())
        picks = test_df[mask].copy()
        if len(picks) == 0:
            continue
        picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
        picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
        picks["fold"] = fi
        selected = select_weekly_top_n(picks, N_SLOTS, sort_col="p_any_pead")
        if len(selected) > 0:
            all_exec_list.append(selected)

    exec_df = pd.concat(all_exec_list).reset_index(drop=True)
    print(f"  Total executed: {len(exec_df)}")

    # ===== CORRELATION CHECK =====
    print(f"\n{'='*95}")
    print("1. DOES P(large) CORRELATE WITH RETURNS?")
    print(f"{'='*95}")

    # Overall correlation
    valid = exec_df["pregap_return"].notna()
    corr_large = exec_df.loc[valid, "p_large"].corr(exec_df.loc[valid, "pregap_return"])
    corr_any = exec_df.loc[valid, "p_any_pead"].corr(exec_df.loc[valid, "pregap_return"])
    print(f"\n  Pearson correlation (all {valid.sum()} trades):")
    print(f"    P(large) vs return:     {corr_large:+.4f}")
    print(f"    P(any)  vs return:      {corr_any:+.4f}")

    # For large PEAD trades only
    large = exec_df[exec_df["label_3class"] == 2]
    corr_large_only = large["p_large"].corr(large["pregap_return"])
    print(f"\n  Pearson correlation (large PEAD only, {len(large)} trades):")
    print(f"    P(large) vs return:     {corr_large_only:+.4f}")

    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    sp_large, _ = spearmanr(exec_df.loc[valid, "p_large"], exec_df.loc[valid, "pregap_return"])
    sp_any, _ = spearmanr(exec_df.loc[valid, "p_any_pead"], exec_df.loc[valid, "pregap_return"])
    print(f"\n  Spearman rank correlation (all trades):")
    print(f"    P(large) vs return:     {sp_large:+.4f}")
    print(f"    P(any)  vs return:      {sp_any:+.4f}")

    # Bucket analysis: sort by P(large) into quintiles, check avg return
    print(f"\n  P(large) quintile analysis:")
    exec_v = exec_df[valid].copy()
    exec_v["p_large_quintile"] = pd.qcut(exec_v["p_large"], 5, labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"])
    for q in ["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"]:
        sub = exec_v[exec_v["p_large_quintile"] == q]
        if len(sub) > 0:
            print(f"    {q}: n={len(sub):>3}, avg_ret={sub['pregap_return'].mean()*100:>+6.2f}%, "
                  f"win={((sub['pregap_return']>0).mean())*100:.0f}%, "
                  f"P(large) range=[{sub['p_large'].min():.3f}, {sub['p_large'].max():.3f}]")

    # ===== SIZING COMPARISON =====
    print(f"\n{'='*95}")
    print("2. SIZING SCHEME COMPARISON")
    print(f"{'='*95}")

    # For sizing, we need to group trades by week and allocate within each week
    exec_sized = exec_df[valid].copy()
    exec_sized["entry_date"] = pd.to_datetime(exec_sized["pregap_entry_date"])
    iso = exec_sized["entry_date"].dt.isocalendar()
    exec_sized["_week_key"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    # Scheme A: equal weight 1/N per trade (N = trades in that week, capped at N_SLOTS)
    # Scheme B: P(large)-weighted within week
    # Scheme C: P(any)-weighted within week

    for scheme, weight_col in [("A. Equal-weight", None), ("B. P(large)-weighted", "p_large"), ("C. P(any)-weighted", "p_any_pead")]:
        exec_sized["weight"] = 0.0
        for week, group in exec_sized.groupby("_week_key"):
            n = len(group)
            if weight_col is None:
                w = np.ones(n) / n  # equal weight within week
            else:
                vals = group[weight_col].values.clip(min=0.001)  # avoid zero
                w = vals / vals.sum()
            exec_sized.loc[group.index, "weight"] = w

        # Weighted return per trade = weight * return
        # But each week uses at most N_SLOTS of capital, so scale by min(n, N_SLOTS)/N_SLOTS
        exec_sized["weighted_return"] = exec_sized["weight"] * exec_sized["pregap_return"]
        # Total portfolio return = sum of weighted returns * (slots used / N_SLOTS)
        # Actually simpler: each week contributes weight*return, summed across all trades
        total_weighted = exec_sized["weighted_return"].sum() * 100

        # Per-trade stats (unweighted, for comparison)
        pnls = exec_sized["pregap_return"]
        n = len(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        wr = len(wins) / n * 100

        # Weighted avg return per trade
        w_avg = (exec_sized["weight"] * exec_sized["pregap_return"]).sum() / exec_sized["weight"].sum() * 100

        print(f"\n  {scheme}:")
        print(f"    N trades:           {n}")
        print(f"    Win rate (unwtd):   {wr:.1f}%")
        print(f"    Avg return (unwtd): {pnls.mean()*100:+.2f}%")
        print(f"    Weighted avg/trade: {w_avg:+.2f}%")
        print(f"    TOTAL weighted PnL: {total_weighted:+.1f}%")
        # Max single-trade weight
        print(f"    Max weight:         {exec_sized['weight'].max()*100:.1f}% (concentration risk)")
        print(f"    Avg weight:         {exec_sized['weight'].mean()*100:.1f}%")

    # Per-week detail for top weeks
    print(f"\n{'='*95}")
    print("3. PER-WEEK DETAIL (top 5 weeks by total return)")
    print(f"{'='*95}")

    weekly = exec_sized.groupby("_week_key").agg(
        n_trades=("pregap_return", "count"),
        eq_total=("pregap_return", "sum"),
        large_total=("p_large", lambda x: (exec_sized.loc[x.index, "weight"] * exec_sized.loc[x.index, "pregap_return"] * (exec_sized.loc[x.index, "p_large"] / exec_sized.loc[x.index, "p_any_pead"])).sum()),
    ).reset_index()
    # Compute P(large)-weighted total per week
    pl_wt = []
    for week, group in exec_sized.groupby("_week_key"):
        vals = group["p_large"].values.clip(min=0.001)
        w = vals / vals.sum()
        pl_wt.append((week, (w * group["pregap_return"].values).sum()))
    pl_wt_df = pd.DataFrame(pl_wt, columns=["_week_key", "plarge_total"])

    weekly = weekly.merge(pl_wt_df, on="_week_key")
    weekly["eq_total_pct"] = weekly["eq_total"] * 100
    weekly["plarge_total_pct"] = weekly["plarge_total"] * 100
    weekly["delta"] = weekly["plarge_total_pct"] - weekly["eq_total_pct"]

    top5 = weekly.nlargest(5, "eq_total_pct")
    print(f"\n  {'Week':<10} {'N':>3} {'Eq-weighted':>12} {'P(large)-wtd':>13} {'Delta':>8}")
    for _, r in top5.iterrows():
        print(f"  {r['_week_key']:<10} {r['n_trades']:>3} {r['eq_total_pct']:>+11.1f}% "
              f"{r['plarge_total_pct']:>+12.1f}% {r['delta']:>+7.1f}%")

    bottom5 = weekly.nsmallest(5, "eq_total_pct")
    print(f"\n  Bottom 5 weeks:")
    for _, r in bottom5.iterrows():
        print(f"  {r['_week_key']:<10} {r['n_trades']:>3} {r['eq_total_pct']:>+11.1f}% "
              f"{r['plarge_total_pct']:>+12.1f}% {r['delta']:>+7.1f}%")

    print(f"\n{'='*95}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Deep comparison: Binary theta=0.20 vs 3-class P(any)>=0.20.

Answers:
  1. Why does binary outperform 3-class on total return?
  2. Are the 2 extra large PEAD trades the sole reason, or is it broader?
  3. Are the picks the SAME trades or DIFFERENT trades?
  4. Per-fold, per-week, per-class breakdown
  5. Calibration comparison (predicted prob vs actual PEAD rate)
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


def run_scenario(fold_data, theta, prob_col):
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


def main():
    print("=" * 105)
    print("DEEP COMPARISON: Binary theta=0.20 vs 3-class P(any)>=0.20")
    print("=" * 105)

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

    print("[3] Training both classifiers per fold ...")
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
        y_ts_b = pd.concat([train_df, sweep_df])["pead_pass"].astype(int).values
        y_te_b = test_df["pead_pass"].astype(int).values
        y_ts_3 = pd.concat([train_df, sweep_df])["label_3class"].values
        y_te_3 = test_df["label_3class"].values

        clf_b = fit_clf_binary(X_ts, y_ts_b, X_te, y_te_b, hp)
        clf_3 = fit_clf_3class(X_ts, y_ts_3, X_te, y_te_3, hp)
        proba_3 = clf_3.predict_proba(X_te)

        test_df = test_df.copy()
        test_df["p_binary"] = clf_b.predict_proba(X_te)[:, 1]
        test_df["p_any_pead"] = proba_3[:, 1] + proba_3[:, 2]
        test_df["p_large"] = proba_3[:, 2]
        fold_data[fi] = {"test_df": test_df}

    # Run both scenarios
    print("[4] Running both scenarios ...")
    binary_exec = run_scenario(fold_data, 0.20, "p_binary")
    threeclass_exec = run_scenario(fold_data, 0.20, "p_any_pead")
    print(f"  Binary executed:    {len(binary_exec)}")
    print(f"  3-class executed:   {len(threeclass_exec)}")

    # ===== 1. OVERALL COMPARISON =====
    print(f"\n{'='*105}")
    print("1. OVERALL COMPARISON")
    print(f"{'='*105}")
    for label, exec_df in [("Binary 0.20", binary_exec), ("3-class 0.20", threeclass_exec)]:
        pnls = exec_df["pregap_return"].dropna()
        n = len(pnls)
        wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
        wr = len(wins) / n * 100
        avg = pnls.mean() * 100
        aw = wins.mean() * 100 if len(wins) > 0 else 0
        al = losses.mean() * 100 if len(losses) > 0 else 0
        payoff = aw / abs(al) if al != 0 else float('inf')
        total = pnls.sum() * 100
        n_pead = int((exec_df["label_3class"] >= 1).sum())
        n_large = int((exec_df["label_3class"] == 2).sum())
        n_small = int((exec_df["label_3class"] == 1).sum())
        n_no = int((exec_df["label_3class"] == 0).sum())
        print(f"\n  {label}:")
        print(f"    N={n}, Win={wr:.1f}%, Avg={avg:+.2f}%, Total={total:+.1f}%, Payoff={payoff:.2f}")
        print(f"    No PEAD={n_no}({n_no/n*100:.0f}%), Small={n_small}({n_small/n*100:.0f}%), Large={n_large}({n_large/n*100:.0f}%)")

    # ===== 2. ARE THE PICKS THE SAME OR DIFFERENT? =====
    print(f"\n{'='*105}")
    print("2. ARE THE PICKS THE SAME OR DIFFERENT?")
    print(f"{'='*105}")

    # Create keys for matching
    for exec_df in [binary_exec, threeclass_exec]:
        exec_df["_key"] = exec_df["permaTicker"] + "_" + exec_df["report_date"].astype(str)

    b_keys = set(binary_exec["_key"])
    t_keys = set(threeclass_exec["_key"])
    both = b_keys & t_keys
    b_only = b_keys - t_keys
    t_only = t_keys - b_keys

    print(f"\n  Binary-only trades:    {len(b_only)}")
    print(f"  3-class-only trades:   {len(t_only)}")
    print(f"  In both:               {len(both)}")
    print(f"  Total unique:          {len(b_keys | t_keys)}")

    # PnL from each group
    b_only_df = binary_exec[binary_exec["_key"].isin(b_only)]
    t_only_df = threeclass_exec[threeclass_exec["_key"].isin(t_only)]
    both_b = binary_exec[binary_exec["_key"].isin(both)]
    both_t = threeclass_exec[threeclass_exec["_key"].isin(both)]

    print(f"\n  PnL breakdown by group:")
    if len(b_only_df) > 0:
        print(f"    Binary-only ({len(b_only_df)} trades):  avg={b_only_df['pregap_return'].mean()*100:+.2f}%, "
              f"total={b_only_df['pregap_return'].sum()*100:+.1f}%, "
              f"win={((b_only_df['pregap_return']>0).mean())*100:.0f}%")
    if len(t_only_df) > 0:
        print(f"    3-class-only ({len(t_only_df)} trades): avg={t_only_df['pregap_return'].mean()*100:+.2f}%, "
              f"total={t_only_df['pregap_return'].sum()*100:+.1f}%, "
              f"win={((t_only_df['pregap_return']>0).mean())*100:.0f}%")
    if len(both_b) > 0:
        print(f"    In both ({len(both_b)} trades):       avg={both_b['pregap_return'].mean()*100:+.2f}%, "
              f"total={both_b['pregap_return'].sum()*100:+.1f}%, "
              f"win={((both_b['pregap_return']>0).mean())*100:.0f}%")

    # ===== 3. THE 2 EXTRA LARGE PEAD — DO THEY SAVE BINARY? =====
    print(f"\n{'='*105}")
    print("3. THE EXTRA LARGE PEAD TRADES — IMPACT ANALYSIS")
    print(f"{'='*105}")

    b_large = binary_exec[binary_exec["label_3class"] == 2]
    t_large = threeclass_exec[threeclass_exec["label_3class"] == 2]

    b_large_keys = set(b_large["_key"])
    t_large_keys = set(t_large["_key"])
    large_both = b_large_keys & t_large_keys
    large_b_only = b_large_keys - t_large_keys
    large_t_only = t_large_keys - b_large_keys

    print(f"\n  Large PEAD in binary:    {len(b_large)} trades, total={b_large['pregap_return'].sum()*100:+.1f}%")
    print(f"  Large PEAD in 3-class:   {len(t_large)} trades, total={t_large['pregap_return'].sum()*100:+.1f}%")
    print(f"  Large PEAD in both:      {len(large_both)}")
    print(f"  Large PEAD binary-only:  {len(large_b_only)}")
    print(f"  Large PEAD 3-class-only: {len(large_t_only)}")

    # What if we remove the 2 extra large PEAD from binary?
    b_no_extra_large = binary_exec[~((binary_exec["label_3class"] == 2) & (binary_exec["_key"].isin(large_b_only)))]
    if len(b_no_extra_large) > 0:
        print(f"\n  Binary WITHOUT the {len(large_b_only)} extra large PEAD:")
        pnls = b_no_extra_large["pregap_return"].dropna()
        print(f"    N={len(pnls)}, Total={pnls.sum()*100:+.1f}%, Avg={pnls.mean()*100:+.2f}%")
        print(f"    (vs 3-class total={threeclass_exec['pregap_return'].sum()*100:+.1f}%)")
        print(f"    => Without extras, binary {'STILL BEATS' if pnls.sum() > threeclass_exec['pregap_return'].sum() else 'LOSES TO'} 3-class")

    # Detail of the extra large PEAD trades
    if len(large_b_only) > 0:
        print(f"\n  The {len(large_b_only)} binary-only Large PEAD trades:")
        for _, t in b_large[b_large["_key"].isin(large_b_only)].sort_values("report_date").iterrows():
            ticker = t.get("canonical_ticker", t["permaTicker"])
            rd = str(pd.Timestamp(t["report_date"]).date())
            ret = t["pregap_return"] * 100
            p_b = t["p_binary"]
            p_any = t["p_any_pead"]
            print(f"    {ticker:10s}  report={rd}  ret={ret:+.2f}%  P(binary)={p_b:.3f}  P(any 3cls)={p_any:.3f}")

    # ===== 4. PER-CLASS BREAKDOWN =====
    print(f"\n{'='*105}")
    print("4. PER-CLASS BREAKDOWN (same trades, grouped by label)")
    print(f"{'='*105}")

    for label_name, exec_df in [("Binary 0.20", binary_exec), ("3-class 0.20", threeclass_exec)]:
        print(f"\n  {label_name}:")
        for cls_name, mask in [("No PEAD", exec_df["label_3class"] == 0),
                                ("Small PEAD", exec_df["label_3class"] == 1),
                                ("Large PEAD", exec_df["label_3class"] == 2)]:
            sub = exec_df[mask]
            if len(sub) == 0:
                continue
            pnls = sub["pregap_return"].dropna()
            n = len(pnls)
            wr = (pnls > 0).mean() * 100
            avg = pnls.mean() * 100
            total = pnls.sum() * 100
            print(f"    {cls_name:<12} N={n:>3}  Win={wr:>5.1f}%  Avg={avg:>+6.2f}%  Total={total:>+7.1f}%")

    # ===== 5. CALIBRATION COMPARISON =====
    print(f"\n{'='*105}")
    print("5. CALIBRATION COMPARISON (predicted prob vs actual PEAD rate)")
    print(f"{'='*105}")

    for label_name, prob_col in [("Binary (p_binary)", "p_binary"), ("3-class (p_any)", "p_any_pead")]:
        print(f"\n  {label_name}:")
        all_test = []
        for fi in range(1, 5):
            all_test.append(fold_data[fi]["test_df"])
        all_df = pd.concat(all_test)
        valid = all_df[prob_col].notna() & all_df["pregap_return"].notna()
        all_df = all_df[valid].copy()

        # Bin by predicted probability
        bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.01]
        labels = ["<5%", "5-10%", "10-15%", "15-20%", "20-25%", "25-30%", "30-40%", "40-50%", ">50%"]
        all_df["_bin"] = pd.cut(all_df[prob_col], bins=bins, labels=labels, right=False)

        print(f"    {'Bin':<10} {'N':>5} {'PEAD%':>7} {'LgPEAD%':>8} {'AvgRet':>8} {'Win%':>6}")
        for b in labels:
            sub = all_df[all_df["_bin"] == b]
            if len(sub) == 0:
                continue
            n = len(sub)
            pead_rate = (sub["label_3class"] >= 1).mean() * 100
            large_rate = (sub["label_3class"] == 2).mean() * 100
            avg_ret = sub["pregap_return"].mean() * 100
            wr = (sub["pregap_return"] > 0).mean() * 100
            print(f"    {b:<10} {n:>5} {pead_rate:>6.1f}% {large_rate:>7.1f}% {avg_ret:>+7.2f}% {wr:>5.1f}%")

    # ===== 6. PROBABILITY CORRELATION =====
    print(f"\n{'='*105}")
    print("6. PROBABILITY CORRELATION (binary vs 3-class)")
    print(f"{'='*105}")

    all_test = []
    for fi in range(1, 5):
        all_test.append(fold_data[fi]["test_df"])
    all_df = pd.concat(all_test)
    valid = all_df["p_binary"].notna() & all_df["p_any_pead"].notna() & all_df["pregap_return"].notna()
    all_df = all_df[valid].copy()

    corr = all_df["p_binary"].corr(all_df["p_any_pead"])
    print(f"\n  Correlation between p_binary and p_any_pead: {corr:+.4f}")

    from scipy.stats import spearmanr
    sp, _ = spearmanr(all_df["p_binary"], all_df["p_any_pead"])
    print(f"  Spearman rank correlation: {sp:+.4f}")

    # Correlation with return
    corr_b_ret = all_df["p_binary"].corr(all_df["pregap_return"])
    corr_t_ret = all_df["p_any_pead"].corr(all_df["pregap_return"])
    sp_b, _ = spearmanr(all_df["p_binary"], all_df["pregap_return"])
    sp_t, _ = spearmanr(all_df["p_any_pead"], all_df["pregap_return"])
    print(f"\n  Correlation with pre-gap return ({len(all_df)} events):")
    print(f"    p_binary:   Pearson={corr_b_ret:+.4f}  Spearman={sp_b:+.4f}")
    print(f"    p_any_pead: Pearson={corr_t_ret:+.4f}  Spearman={sp_t:+.4f}")

    # ===== 7. PER-FOLD BREAKDOWN =====
    print(f"\n{'='*105}")
    print("7. PER-FOLD BREAKDOWN")
    print(f"{'='*105}")
    print(f"\n  {'Model':<15} {'Fold':>5} {'N':>4} {'Win%':>6} {'Avg':>8} {'Total':>8} {'Large':>6}")
    for label, exec_df in [("Binary 0.20", binary_exec), ("3-class 0.20", threeclass_exec)]:
        for fi in range(1, 5):
            sub = exec_df[exec_df["fold"] == fi]
            if len(sub) == 0:
                continue
            pnls = sub["pregap_return"].dropna()
            n = len(pnls)
            wr = (pnls > 0).mean() * 100
            avg = pnls.mean() * 100
            total = pnls.sum() * 100
            n_large = int((sub["label_3class"] == 2).sum())
            print(f"  {label:<15} {fi:>5} {n:>4} {wr:>5.1f}% {avg:>+7.2f}% {total:>+7.1f}% {n_large:>6}")
        print()

    print(f"{'='*105}")


if __name__ == "__main__":
    main()

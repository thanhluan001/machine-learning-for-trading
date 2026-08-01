#!/usr/bin/env python3
"""
Investigate why PEAD precision is low (35.8%) and find improvement levers.

Questions:
  1. Which PEAD gate do the false positives fail? (CAR, volume, or MaxDD?)
  2. How CLOSE are false positives to passing? (CAR just under 3%?)
  3. Which features separate true PEAD from false positives?
  4. Single-feature secondary filters: which feature, at what threshold,
     best lifts precision without losing too many true PEAD?
  5. Feature importance from the model vs actual PEAD correlation
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
THETA = 0.20

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]


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


def compute_pregap(df, db_path, hold_days=5):
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
            ep = p_close[entry_t]; xp = p_close[exit_t]
            if pd.isna(ep) or pd.isna(xp) or ep <= 0:
                continue
            df.at[idx, "pregap_return"] = float(xp / ep - 1.0)
            df.at[idx, "pregap_entry_date"] = pd.Timestamp(p_index[entry_t])
            df.at[idx, "pregap_exit_date"] = pd.Timestamp(p_index[exit_t])
    return df


def main():
    print("=" * 100)
    print("PEAD PRECISION INVESTIGATION: Why is precision 35.8%?")
    print("=" * 100)

    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df["car_10d_pct"] = np.expm1(df["car_10d"]) * 100
    df = compute_pregap(df, DB, EXIT_SNAP)

    # Train + select
    fold_data = {}
    hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_ts = pd.concat([train_df[SUNDAY_SAFE], sweep_df[SUNDAY_SAFE]])
        y_ts = pd.concat([train_df, sweep_df])["pead_pass"].astype(int).values
        y_te = test_df["pead_pass"].astype(int).values
        clf = fit_clf(X_ts, y_ts, test_df[SUNDAY_SAFE], y_te, hp)
        test_df = test_df.copy()
        test_df["p"] = clf.predict_proba(test_df[SUNDAY_SAFE])[:, 1]
        fold_data[fi] = {"test_df": test_df, "clf": clf}

    all_exec = []
    for fi in range(1, 5):
        td = fold_data[fi]["test_df"]
        mask = (td["p"] >= THETA) & (td["pregap_return"].notna())
        picks = td[mask].copy()
        if len(picks) == 0:
            continue
        picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
        picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
        picks["fold"] = fi
        sel = select_weekly_top_n(picks, N_SLOTS, sort_col="p")
        if len(sel) > 0:
            all_exec.append(sel)
    exec_df = pd.concat(all_exec).reset_index(drop=True)

    pead = exec_df[exec_df["pead_pass"] == 1].copy()
    nonpead = exec_df[exec_df["pead_pass"] == 0].copy()
    n_total = len(exec_df)
    n_pead = len(pead)
    n_nonpead = len(nonpead)

    print(f"\n  Executed: {n_total}, PEAD: {n_pead} ({n_pead/n_total*100:.1f}%), "
          f"Non-PEAD: {n_nonpead} ({n_nonpead/n_total*100:.1f}%)")

    # ===== 1. WHICH GATE DO FALSE POSITIVES FAIL? =====
    print(f"\n{'='*100}")
    print("1. WHICH PEAD GATE DO THE FALSE POSITIVES FAIL?")
    print(f"{'='*100}")

    # Check each gate for non-PEAD picks
    for gate_name, gate_col in [("Gate 1 (CAR > 3%)", "pass_g1"),
                                  ("Gate 2 (vol > 2x)", "pass_g2"),
                                  ("Gate 3 (MaxDD_MA > -1.5%)", "pass_g3")]:
        if gate_col in nonpead.columns:
            pass_rate = nonpead[gate_col].sum() / len(nonpead) * 100
            print(f"  {gate_name}: {int(nonpead[gate_col].sum())}/{n_nonpead} pass ({pass_rate:.1f}%)")

    # How many gates does each false positive fail?
    gate_cols = ["pass_g1", "pass_g2", "pass_g3"]
    for col in gate_cols:
        if col not in nonpead.columns:
            gate_cols.remove(col)
    if gate_cols:
        nonpead_gates = nonpead[gate_cols].sum(axis=1)
        print(f"\n  Gates passed by false positives:")
        for n_pass in range(4):
            n = int((nonpead_gates == n_pass).sum())
            print(f"    {n_pass} gates passed: {n} ({n/n_nonpead*100:.1f}%)")

    # ===== 2. HOW CLOSE ARE FALSE POSITIVES TO PASSING? =====
    print(f"\n{'='*100}")
    print("2. HOW CLOSE ARE FALSE POSITIVES TO PASSING?")
    print(f"{'='*100}")

    # CAR distribution for false positives
    print(f"\n  CAR_10d distribution for non-PEAD picks:")
    car_np = nonpead["car_10d_pct"].dropna()
    print(f"    min={car_np.min():.2f}%, p10={np.percentile(car_np,10):.2f}%, "
          f"p25={np.percentile(car_np,25):.2f}%, p50={np.percentile(car_np,50):.2f}%, "
          f"p75={np.percentile(car_np,75):.2f}%, p90={np.percentile(car_np,90):.2f}%, "
          f"max={car_np.max():.2f}%")
    # How many are close to the 3% threshold?
    print(f"\n  CAR proximity to gate 1 threshold (3%):")
    for lo, hi in [(-100, 0), (0, 1), (1, 2), (2, 3), (3, 5), (5, 100)]:
        n = int(((car_np >= lo) & (car_np < hi)).sum())
        print(f"    CAR in [{lo}%, {hi}%): {n} ({n/n_nonpead*100:.1f}%)")

    # Vol ratio distribution
    if "inst_vol_ratio" in nonpead.columns:
        print(f"\n  Volume ratio distribution for non-PEAD picks:")
        vol_np = nonpead["inst_vol_ratio"].dropna()
        print(f"    min={vol_np.min():.2f}x, p25={np.percentile(vol_np,25):.2f}x, "
              f"p50={np.percentile(vol_np,50):.2f}x, p75={np.percentile(vol_np,75):.2f}x, "
              f"max={vol_np.max():.2f}x")
        print(f"    Gate 2 threshold = 2.0x")
        close_vol = int(((vol_np >= 1.5) & (vol_np < 2.0)).sum())
        print(f"    Close (1.5x-2.0x): {close_vol} ({close_vol/n_nonpead*100:.1f}%)")

    # MaxDD_MA distribution
    if "maxdd_ma" in nonpead.columns:
        print(f"\n  MaxDD_MA distribution for non-PEAD picks:")
        dd_np = nonpead["maxdd_ma"].dropna() * 100
        print(f"    min={dd_np.min():.2f}%, p25={np.percentile(dd_np,25):.2f}%, "
              f"p50={np.percentile(dd_np,50):.2f}%, p75={np.percentile(dd_np,75):.2f}%, "
              f"max={dd_np.max():.2f}%")
        print(f"    Gate 3 threshold = -1.5%")
        close_dd = int(((dd_np >= -3.0) & (dd_np < -1.5)).sum())
        print(f"    Close (-3.0% to -1.5%): {close_dd} ({close_dd/n_nonpead*100:.1f}%)")

    # ===== 3. FEATURE DISTRIBUTIONS: PEAD vs NON-PEAD =====
    print(f"\n{'='*100}")
    print("3. FEATURE DISTRIBUTIONS: PEAD vs NON-PEAD picks")
    print(f"{'='*100}")

    print(f"\n  {'Feature':<35} {'PEAD mean':>10} {'NonPEAD mean':>12} {'Delta':>8} {'Sep?':>5}")
    print("  " + "-" * 75)
    for feat in SUNDAY_SAFE:
        if feat not in exec_df.columns:
            continue
        pead_vals = pead[feat].dropna()
        np_vals = nonpead[feat].dropna()
        if len(pead_vals) == 0 or len(np_vals) == 0:
            continue
        pm = pead_vals.mean()
        nm = np_vals.mean()
        delta = pm - nm
        # Simple separation score: Cohen's d
        pooled_std = np.sqrt((pead_vals.std()**2 + np_vals.std()**2) / 2)
        cohens_d = delta / pooled_std if pooled_std > 0 else 0
        sep = "***" if abs(cohens_d) > 0.5 else ("**" if abs(cohens_d) > 0.3 else ("*" if abs(cohens_d) > 0.15 else ""))
        print(f"  {feat:<35} {pm:>+9.3f} {nm:>+11.3f} {delta:>+7.3f} {sep:>5}")

    # ===== 4. SINGLE-FEATURE SECONDARY FILTERS =====
    print(f"\n{'='*100}")
    print("4. SINGLE-FEATURE SECONDARY FILTERS")
    print("   (If we add a filter on feature X >= threshold, how does precision change?)")
    print(f"{'='*100}")

    # For each feature, find the threshold that maximizes precision while keeping >= 80% of true PEAD
    print(f"\n  {'Feature':<35} {'Best thresh':>11} {'Prec':>6} {'Recall':>7} {'N':>4} {'Lift':>6}")
    print("  " + "-" * 75)

    best_filters = []
    for feat in SUNDAY_SAFE:
        if feat not in exec_df.columns:
            continue
        vals = exec_df[feat].dropna()
        pead_mask = exec_df["pead_pass"] == 1
        if len(vals) == 0:
            continue

        # Try different thresholds
        best_prec = 0
        best_thresh = None
        best_n = 0
        best_recall = 0
        for pct in [10, 20, 25, 30, 33, 40, 50, 60, 67, 70, 75, 80, 90]:
            thresh = np.percentile(vals, pct)
            if abs(thresh) > 1e10:
                continue
            # Filter: keep picks where feat >= thresh
            filtered_mask = exec_df[feat] >= thresh
            filtered = exec_df[filtered_mask.fillna(False)]
            if len(filtered) == 0:
                continue
            n_filt_pead = int(filtered["pead_pass"].sum())
            prec = n_filt_pead / len(filtered) * 100
            recall = n_filt_pead / n_pead * 100 if n_pead > 0 else 0
            # We want precision improvement while keeping >= 70% of PEAD
            if recall >= 70 and prec > best_prec:
                best_prec = prec
                best_thresh = thresh
                best_n = len(filtered)
                best_recall = recall

        if best_thresh is not None:
            lift = best_prec - n_pead/n_total*100
            print(f"  {feat:<35} {best_thresh:>+10.3f} {best_prec:>5.1f}% {best_recall:>6.1f}% "
                  f"{best_n:>4} {lift:>+5.1f}pp")
            best_filters.append((feat, best_thresh, best_prec, best_recall, best_n))

    # ===== 5. TOP FALSE POSITIVES ANALYSIS =====
    print(f"\n{'='*100}")
    print("5. TOP 10 FALSE POSITIVES (highest P(PEAD) but not PEAD)")
    print(f"{'='*100}")

    fp = nonpead.sort_values("p", ascending=False).head(10)
    print(f"\n  {'Ticker':<10} {'Date':<12} {'P(PEAD)':>8} {'CAR%':>7} {'Ret%':>7} {'g1':>4} {'g2':>4} {'g3':>4} {'VolR':>6} {'EPS%':>7}")
    for _, t in fp.iterrows():
        ticker = t.get("canonical_ticker", t["permaTicker"])
        rd = str(pd.Timestamp(t["report_date"]).date())
        p = t["p"]
        car = t.get("car_10d_pct", float('nan'))
        ret = t["pregap_return"] * 100
        g1 = int(t.get("pass_g1", 0))
        g2 = int(t.get("pass_g2", 0))
        g3 = int(t.get("pass_g3", 0))
        vol = t.get("inst_vol_ratio", float('nan'))
        eps = t.get("eps_surprise_pct", float('nan'))
        print(f"  {ticker:<10} {rd:<12} {p:>7.3f} {car:>+6.1f}% {ret:>+6.1f}% "
              f"{g1:>4} {g2:>4} {g3:>4} {vol:>5.1f}x {eps:>+6.1f}%")

    # ===== 6. MODEL CONFIDENCE CALIBRATION =====
    print(f"\n{'='*100}")
    print("6. MODEL CONFIDENCE CALIBRATION (P(PEAD) bins vs actual PEAD rate)")
    print(f"{'='*100}")

    print(f"\n  {'P(PEAD) bin':<12} {'N':>5} {'PEAD':>5} {'Prec':>6} {'AvgRet':>8} {'Win%':>6}")
    bins = [(0.20, 0.22), (0.22, 0.25), (0.25, 0.28), (0.28, 0.32), (0.32, 0.40), (0.40, 1.0)]
    for lo, hi in bins:
        sub = exec_df[(exec_df["p"] >= lo) & (exec_df["p"] < hi)]
        if len(sub) == 0:
            continue
        n = len(sub)
        n_p = int(sub["pead_pass"].sum())
        prec = n_p / n * 100
        avg_ret = sub["pregap_return"].mean() * 100
        wr = (sub["pregap_return"] > 0).mean() * 100
        print(f"  [{lo:.2f}, {hi:.2f})   {n:>5} {n_p:>5} {prec:>5.1f}% {avg_ret:>+7.2f}% {wr:>5.1f}%")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

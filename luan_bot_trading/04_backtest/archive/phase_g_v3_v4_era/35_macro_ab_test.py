#!/usr/bin/env python3
"""
A/B test: does adding macro features help or hurt the binary classifier?

The original LTR model excluded macros because a ranker only cares about
relative ordering — macro features are constant across all candidates at
a given time, so they add zero to the rank gradient. But a binary
classifier predicts an ABSOLUTE probability (P(PEAD) >= 0.20), so macro
regime could matter.

Tests:
  A. 24 Sunday-safe features (baseline, current deployable)
  B. 24 + 12 macro features (6 raw levels + 6 rate-of-change)

Same 4-fold nested CV, pre-gap entry, 5-day hold, weekly batch, theta=0.20.

Macro features joined by report_date (forward-fill, no look-ahead):
  - fred_vix_close (VIX level + 20d ROC)
  - fred_fed_funds_rate (level + 20d ROC)
  - fred_yield_curve_spread (level + 20d ROC)
  - fred_wti_oil (level + 20d ROC)
  - fred_cpi (level + ROC, monthly)
  - fred_unemployment_rate (level + ROC, monthly)
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

# Macro series to join (key, value_column)
MACRO_SERIES = [
    ("/macros/fred_vix_close", "vix_close"),
    ("/macros/fred_fed_funds_rate", "fed_funds_rate"),
    ("/macros/fred_yield_curve_spread", "yield_curve_spread"),
    ("/macros/fred_wti_oil", "wti_oil"),
    ("/macros/fred_cpi", "cpi"),
    ("/macros/fred_unemployment_rate", "unemployment_rate"),
]
ROC_WINDOW = 20  # 20 trading-day rate of change


def join_macros(df, db_path):
    """Join macro features to df by report_date (forward-fill, no look-ahead).
    Adds: for each macro, the level and 20d rate-of-change."""
    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])

    with pd.HDFStore(db_path, mode="r") as s:
        for key, val_col in MACRO_SERIES:
            if key not in s.keys():
                continue
            macro = s[key].copy()
            macro["Date"] = pd.to_datetime(macro["Date"])
            macro = macro.sort_values("Date").drop_duplicates("Date", keep="last")

            # Compute 20-day ROC (pct change)
            short_name = val_col
            roc_name = f"{short_name}_roc20"
            macro[roc_name] = macro[val_col].pct_change(ROC_WINDOW).replace([np.inf, -np.inf], np.nan)

            # Forward-fill: for each report_date, get the most recent macro value
            # (closest Date <= report_date, no look-ahead)
            macro = macro.rename(columns={"Date": "_macro_date", val_col: f"macro_{short_name}"})
            # Merge_asof: requires both sorted by the join key
            df_sorted = df.sort_values("report_date").reset_index()
            merged = pd.merge_asof(
                df_sorted,
                macro[["_macro_date", f"macro_{short_name}", roc_name]],
                left_on="report_date",
                right_on="_macro_date",
                direction="backward"
            )
            # Rename roc column
            merged = merged.rename(columns={roc_name: f"macro_{short_name}_roc20"})
            # Put back in original order
            df = merged.sort_values("index").drop(columns=["index", "_macro_date"]).reset_index(drop=True)

    return df


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


def run_scenario(fold_data, theta, proba_col):
    """Run weekly batch selection with given proba column."""
    all_exec_list = []
    for fi in range(1, 5):
        test_df = fold_data[fi]["test_df"]
        mask = (test_df[proba_col] >= theta) & (test_df["pregap_return"].notna())
        picks = test_df[mask].copy()
        if len(picks) == 0:
            continue
        picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
        picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
        picks["fold"] = fi
        selected = select_weekly_top_n(picks, N_SLOTS, sort_col=proba_col)
        if len(selected) > 0:
            all_exec_list.append(selected)
    return pd.concat(all_exec_list) if all_exec_list else pd.DataFrame()


def stats_row(exec_df, label):
    if len(exec_df) == 0:
        print(f"  {label:<50} N=0")
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
    n_pead = int(exec_df["pead_pass"].sum())
    prec = n_pead / n * 100
    print(f"  {label}")
    print(f"    N={n}, Win={wr:.1f}%, Avg={avg:+.2f}%, Total={total:+.1f}%, Payoff={payoff:.2f}, PEAD={n_pead}({prec:.0f}%)")


def main():
    print("=" * 105)
    print("A/B TEST: Macro features — help or hurt the binary classifier?")
    print("=" * 105)

    print("\n[1] Loading + priming + gates ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)

    print("[2] Joining macro features ...")
    df = join_macros(df, DB)
    macro_cols = [c for c in df.columns if c.startswith("macro_")]
    print(f"    Macro features added: {macro_cols}")
    print(f"    Coverage check:")
    for c in macro_cols:
        n_valid = df[c].notna().sum()
        print(f"      {c}: {n_valid}/{len(df)} ({n_valid/len(df)*100:.1f}%)")

    print("[3] Computing pre-gap returns ...")
    df = compute_pregap_returns(df, DB, EXIT_SNAP)

    # Define feature sets
    features_A = SUNDAY_SAFE  # 24 Sunday-safe (baseline)
    features_B = SUNDAY_SAFE + macro_cols  # 24 + 12 macro = 36 total
    print(f"\n[4] Feature sets:")
    print(f"    A (baseline): {len(features_A)} Sunday-safe")
    print(f"    B (with macros): {len(features_B)} = {len(features_A)} Sunday-safe + {len(macro_cols)} macro")

    # Train classifiers per fold for BOTH feature sets
    print("\n[5] Training classifiers per fold (both A and B) ...")
    hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
    fold_data = {}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_ts = pd.concat([train_df, sweep_df])
        y_ts = X_ts["pead_pass"].astype(int).values
        y_te = test_df["pead_pass"].astype(int).values

        # Model A: Sunday-safe only
        X_ts_A = X_ts[features_A]
        X_te_A = test_df[features_A]
        clf_A = fit_clf(X_ts_A, y_ts, X_te_A, y_te, hp)
        test_df = test_df.copy()
        test_df["p_A"] = clf_A.predict_proba(X_te_A)[:, 1]

        # Model B: Sunday-safe + macros
        X_ts_B = X_ts[features_B]
        X_te_B = test_df[features_B]
        clf_B = fit_clf(X_ts_B, y_ts, X_te_B, y_te, hp)
        test_df["p_B"] = clf_B.predict_proba(X_te_B)[:, 1]

        fold_data[fi] = {"test_df": test_df, "proba_A": "p_A", "proba_B": "p_B"}

        # Quick AUC check
        from sklearn.metrics import roc_auc_score
        auc_A = roc_auc_score(y_te, test_df["p_A"])
        auc_B = roc_auc_score(y_te, test_df["p_B"])
        print(f"    Fold {fi}: AUC_A={auc_A:.4f}  AUC_B={auc_B:.4f}  delta={auc_B-auc_A:+.4f}")

    # Run both scenarios
    print("\n[6] Running weekly batch selection ...")
    exec_A = run_scenario(fold_data, THETA, "p_A")
    exec_B = run_scenario(fold_data, THETA, "p_B")

    # ===== COMPARISON =====
    print(f"\n{'='*105}")
    print("RESULTS: A (Sunday-safe) vs B (Sunday-safe + macros)")
    print(f"{'='*105}")

    print(f"\n1. AGGREGATE TRADE-LEVEL STATS")
    stats_row(exec_A, "A. Sunday-safe only (24 features)")
    print()
    stats_row(exec_B, "B. Sunday-safe + macros (36 features)")

    # Per-fold
    print(f"\n2. PER-FOLD BREAKDOWN")
    print(f"  {'Model':<5} {'Fold':>5} {'N':>4} {'Win%':>6} {'Avg':>8} {'Total':>8} {'PEAD':>5}")
    for label, exec_df, proba_col in [("A", exec_A, "p_A"), ("B", exec_B, "p_B")]:
        for fi in range(1, 5):
            sub = exec_df[exec_df["fold"] == fi]
            if len(sub) == 0:
                continue
            pnls = sub["pregap_return"].dropna()
            n = len(pnls)
            wr = (pnls > 0).mean() * 100
            avg = pnls.mean() * 100
            total = pnls.sum() * 100
            n_pead = int(sub["pead_pass"].sum())
            print(f"  {label:<5} {fi:>5} {n:>4} {wr:>5.1f}% {avg:>+7.2f}% {total:>+7.1f}% {n_pead:>5}")
        print()

    # Are the picks different?
    print(f"\n3. PICK OVERLAP")
    for exec_df in [exec_A, exec_B]:
        exec_df["_key"] = exec_df["permaTicker"] + "_" + exec_df["report_date"].astype(str)
    a_keys = set(exec_A["_key"])
    b_keys = set(exec_B["_key"])
    both = a_keys & b_keys
    a_only = a_keys - b_keys
    b_only = b_keys - a_keys
    print(f"  A-only: {len(a_only)}, B-only: {len(b_only)}, Both: {len(both)}")

    # Feature importance for model B (macro features)
    print(f"\n4. FEATURE IMPORTANCE (model B, fold 4)")
    import xgboost as xgb
    rd = pd.to_datetime(df["report_date"])
    train_df = df[rd <= pd.Timestamp("2025-06-30")].copy()
    sweep_df = df[(rd > pd.Timestamp("2025-06-30")) & (rd <= pd.Timestamp("2025-12-31"))].copy()
    test_df = df[(rd > pd.Timestamp("2025-12-31")) & (rd <= pd.Timestamp("2026-06-30"))].copy()
    X_ts = pd.concat([train_df, sweep_df])[features_B]
    y_ts = pd.concat([train_df, sweep_df])["pead_pass"].astype(int).values
    X_te = test_df[features_B]
    y_te = test_df["pead_pass"].astype(int).values
    clf_B = fit_clf(X_ts, y_ts, X_te, y_te, hp)
    imp = clf_B.feature_importances_
    imp_df = pd.DataFrame({"feature": features_B, "importance": imp}).sort_values("importance", ascending=False)
    print(f"  {'Feature':<35} {'Importance':>10}")
    for _, r in imp_df.iterrows():
        marker = " <-- MACRO" if r["feature"].startswith("macro_") else ""
        print(f"  {r['feature']:<35} {r['importance']:>10.4f}{marker}")

    print(f"\n{'='*105}")


if __name__ == "__main__":
    main()

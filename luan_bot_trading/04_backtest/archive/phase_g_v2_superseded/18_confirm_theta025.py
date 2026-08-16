"""Confirmation tests for theta=0.25:
  1. Random baseline: select same n_picks randomly, measure PEAD precision
  2. Bootstrap CIs on precision, recall, F1, PnL
  3. Per-fold stability check
"""
import sys, io, importlib.util, json
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

DB = tm.DB_FILE
SUNDAY_SAFE = pg.SUNDAY_SAFE_FEATURES
THETA = 0.25

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


def main():
    print("=" * 80)
    print(f"CONFIRMATION TESTS — theta={THETA}")
    print("=" * 80)

    # Load + prime + gates + paths
    print("\n[1] Loading + priming + gates + paths ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)
    print(f"    rows: {len(df)}, pead: {int(df['pead_pass'].sum())}")

    # Pre-train classifiers (gamma=3, same as nested CV selected)
    print("\n[2] Pre-training 4 fold classifiers (gamma=3) ...")
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
        print(f"    Fold {fi}: picks={len(picks)}, pead={int(picks['pead_pass'].sum())}")

    # --- TEST 1: Random baseline comparison ---
    print(f"\n{'='*80}")
    print("[3] RANDOM BASELINE COMPARISON")
    print(f"{'='*80}")
    print(f"  For each fold, select same n_picks randomly (1000 trials)")
    print(f"  and measure PEAD precision. Compare to model precision.\n")

    rng = np.random.default_rng(42)
    model_precisions = []
    random_precisions = []
    all_model_picks_pead = []
    all_random_picks_pead = []

    for fi in range(1, 5):
        test_df = fold_data[fi]["test_df"]
        picks = fold_data[fi]["picks"]
        n_picks = len(picks)
        n_pead_picks = int(picks["pead_pass"].sum())
        n_total = len(test_df)
        n_total_pead = int(test_df["pead_pass"].sum())
        model_prec = n_pead_picks / n_picks if n_picks > 0 else 0
        model_precisions.append(model_prec)

        # Random baseline: select n_picks randomly, count PEAD
        trial_precs = []
        for _ in range(1000):
            idx = rng.choice(n_total, size=min(n_picks, n_total), replace=False)
            random_pead = int(test_df.iloc[idx]["pead_pass"].sum())
            trial_precs.append(random_pead / n_picks if n_picks > 0 else 0)
        random_prec = np.mean(trial_precs)
        random_std = np.std(trial_precs)
        random_p95 = np.percentile(trial_precs, 95)
        random_precisions.append(random_prec)

        # Count how many random trials beat the model
        model_beats_random = np.mean(np.array(trial_precs) >= model_prec) * 100

        all_model_picks_pead.append(n_pead_picks)
        all_random_picks_pead.append(random_prec * n_picks)

        print(f"  Fold {fi}: model prec={model_prec*100:.1f}%  "
              f"random prec={random_prec*100:.1f}% (std={random_std*100:.1f}%, "
              f"p95={random_p95*100:.1f}%)  "
              f"model beats {100-model_beats_random:.1f}% of random")

    model_avg = np.mean(model_precisions)
    random_avg = np.mean(random_precisions)
    print(f"\n  AGGREGATE: model prec={model_avg*100:.1f}%  random prec={random_avg*100:.1f}%  "
          f"lift={((model_avg/random_avg)-1)*100:.0f}%")

    # --- TEST 2: Bootstrap CIs ---
    print(f"\n{'='*80}")
    print("[4] BOOTSTRAP CONFIDENCE INTERVALS (2000 resamples)")
    print(f"{'='*80}")

    # Collect all picks across folds
    all_picks = pd.concat([fold_data[fi]["picks"] for fi in range(1, 5)])
    all_test = pd.concat([fold_data[fi]["test_df"] for fi in range(1, 5)])
    n_all_picks = len(all_picks)
    n_all_pead = int(all_picks["pead_pass"].sum())
    n_all_total_pead = int(all_test["pead_pass"].sum())

    actual_prec = n_all_pead / n_all_picks
    actual_rec = n_all_pead / n_all_total_pead
    actual_f1 = 2 * actual_prec * actual_rec / (actual_prec + actual_rec) if (actual_prec + actual_rec) > 0 else 0

    # Bootstrap: resample picks with replacement
    rng_boot = np.random.default_rng(123)
    n_boot = 2000
    prec_boot = []
    rec_boot = []
    f1_boot = []
    pnl_all_boot = []
    pnl_pead_boot = []

    pick_indices = np.arange(n_all_picks)
    for _ in range(n_boot):
        idx = rng_boot.choice(pick_indices, size=n_all_picks, replace=True)
        sample = all_picks.iloc[idx]
        n_s = len(sample)
        n_pead_s = int(sample["pead_pass"].sum())
        prec_s = n_pead_s / n_s if n_s > 0 else 0
        rec_s = n_pead_s / n_all_total_pead
        f1_s = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0
        prec_boot.append(prec_s)
        rec_boot.append(rec_s)
        f1_boot.append(f1_s)
        # PnL
        arith = np.expm1(sample["path_pnl_t11_pct"])
        pnl_all_boot.append(float(arith.mean()) * 100)
        pead_sample = sample[sample["pead_pass"] == 1]
        if len(pead_sample) > 0:
            pead_arith = np.expm1(pead_sample["path_pnl_t11_pct"])
            pnl_pead_boot.append(float(pead_arith.mean()) * 100)
        else:
            pnl_pead_boot.append(0)

    def ci(arr, label, actual):
        arr = np.array(arr)
        lo = np.percentile(arr, 2.5)
        hi = np.percentile(arr, 97.5)
        print(f"  {label:30s}  actual={actual:.4f}  CI=[{lo:.4f}, {hi:.4f}]  "
              f"width={hi-lo:.4f}")
        return lo, hi

    print(f"\n  Total picks: {n_all_picks}, PEAD picks: {n_all_pead}, Total PEAD: {n_all_total_pead}")
    print()
    ci(prec_boot, "Precision", actual_prec)
    ci(rec_boot, "Recall", actual_rec)
    ci(f1_boot, "F1", actual_f1)
    ci(pnl_all_boot, "PnL per pick (all)", float(np.mean(pnl_all_boot)))
    ci(pnl_pead_boot, "PnL per pick (PEAD)", float(np.mean(pnl_pead_boot)))

    # Check if precision CI excludes base rate
    base_rate = n_all_total_pead / len(all_test)
    prec_lo = np.percentile(prec_boot, 2.5)
    prec_hi = np.percentile(prec_boot, 97.5)
    print(f"\n  Base rate (random precision): {base_rate:.4f}")
    print(f"  Precision CI: [{prec_lo:.4f}, {prec_hi:.4f}]")
    if prec_lo > base_rate:
        print(f"  *** CI EXCLUDES base rate — precision is statistically significant ***")
    else:
        print(f"  *** CI overlaps base rate — precision is NOT clearly significant ***")

    # Check if PnL CI excludes 0
    pnl_lo = np.percentile(pnl_all_boot, 2.5)
    pnl_hi = np.percentile(pnl_all_boot, 97.5)
    print(f"\n  PnL (all picks) CI: [{pnl_lo:.4f}, {pnl_hi:.4f}]")
    if pnl_lo > 0:
        print(f"  *** CI EXCLUDES 0 — PnL is positive at 95% confidence ***")
    else:
        print(f"  *** CI overlaps 0 — PnL is NOT clearly positive ***")

    pnl_pead_lo = np.percentile(pnl_pead_boot, 2.5)
    pnl_pead_hi = np.percentile(pnl_pead_boot, 97.5)
    print(f"  PnL (PEAD picks) CI: [{pnl_pead_lo:.4f}, {pnl_pead_hi:.4f}]")
    if pnl_pead_lo > 0:
        print(f"  *** CI EXCLUDES 0 — PEAD PnL is positive at 95% confidence ***")

    # --- TEST 3: Per-fold stability ---
    print(f"\n{'='*80}")
    print("[5] PER-FOLD STABILITY")
    print(f"{'='*80}")
    print(f"\n  {'Fold':>4s}  {'Picks':>5s} {'PEAD':>5s} {'Prec':>6s} {'Rand':>6s} {'Lift':>6s} "
          f"{'PnL_all':>8s} {'PnL_pead':>9s}")
    print("  " + "-" * 60)
    for fi in range(1, 5):
        picks = fold_data[fi]["picks"]
        test_df = fold_data[fi]["test_df"]
        n_p = len(picks)
        n_pead = int(picks["pead_pass"].sum())
        prec = n_pead / n_p if n_p > 0 else 0
        rand = int(test_df["pead_pass"].sum()) / len(test_df)
        arith = np.expm1(picks["path_pnl_t11_pct"]) if n_p > 0 else pd.Series()
        pnl_all = float(arith.mean()) * 100 if len(arith) > 0 else 0
        pead_picks = picks[picks["pead_pass"] == 1]
        pead_arith = np.expm1(pead_picks["path_pnl_t11_pct"]) if len(pead_picks) > 0 else pd.Series()
        pnl_pead = float(pead_arith.mean()) * 100 if len(pead_arith) > 0 else 0
        lift = ((prec / rand) - 1) * 100 if rand > 0 else 0
        print(f"  {fi:>4d}  {n_p:>5d} {n_pead:>5d} {prec*100:>5.1f}% {rand*100:>5.1f}% "
              f"{lift:>5.0f}% {pnl_all:>+7.2f}% {pnl_pead:>+8.2f}%")

    print(f"\n{'='*80}")
    print("CONFIRMATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

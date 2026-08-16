#!/usr/bin/env python3
"""
PEAD Capture Diagnostic — how many of the model's picks are TRUE PEAD?

This directly addresses the "objective mismatch" problem (features.md §6):
the old backtest optimized for PnL, which conflated PEAD drift with gap
mean-reversion. This script measures PEAD precision/recall alongside PnL.

For each nested-CV fold:
  1. Train classifier on TRAIN+SWEEP (same HP selection as 06/15 scripts)
  2. Select trades at NEG_only theta=0.20, gap [-15%, -2%]
  3. Check each selected trade against pead_pass (the 3-gate PEAD label)
  4. Report:
     - n_picks: total trades selected
     - n_pead_picks: trades that are TRUE PEAD (pead_pass=1)
     - pead_precision: n_pead_picks / n_picks (what % of picks are real PEAD)
     - n_total_pead: total PEAD events in TEST set
     - pead_recall: n_pead_picks / n_total_pead (what % of PEAD did we catch)
     - n_pead_missed: PEAD events the model missed
"""
import sys, io, importlib.util, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
# Import train_matrix helpers
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
THETA = 0.20
NEG_LO, NEG_HI = -0.15, -0.02

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]

SWEEP_GRID = [
    {"gamma": g, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
    for g in [3, 5, 10, 20]
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
    print("PEAD CAPTURE DIAGNOSTIC — NEG_only theta=0.20, 24 features")
    print("=" * 80)
    print(f"Features: {len(SUNDAY_SAFE)} Sunday-safe (is_bmo removed, ordinal momentum added)")
    print(f"Rule: P(PEAD)>={THETA} AND gap in [{NEG_LO},{NEG_HI}]")
    print()

    # Load + prime + gates + trade paths
    print("[1] Loading train_matrix + priming + PEAD gates ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)
    print(f"    rows: {len(df)}, pead_pass: {int(df['pead_pass'].sum())} ({df['pead_pass'].mean()*100:.1f}%)")

    fold_results = []
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        print(f"\n{'='*70}")
        print(f"  FOLD {fi}: SWEEP {te}->{sve} | TEST {sve}->{tse}")
        print(f"{'='*70}")

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

        # HP sweep
        best_hp = SWEEP_GRID[1]  # default gamma=5
        best_pnl = -999
        for hp in SWEEP_GRID:
            clf = fit_clf(X_tr, y_tr, X_sv, y_sv, hp)
            proba = clf.predict_proba(X_sv)[:, 1]
            d = sweep_df.copy()
            d["p"] = proba
            m = (d["p"] >= THETA) & (d["opening_gap_t1"] >= NEG_LO) & \
                (d["opening_gap_t1"] <= NEG_HI) & (d["path_pnl_t11_pct"].notna())
            picks = d[m]
            if len(picks) >= 10:
                pnl = np.expm1(picks["path_pnl_t11_pct"]).mean() * 100
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_hp = hp

        # Retrain on TRAIN+SWEEP
        X_ts = pd.concat([X_tr, X_sv])
        y_ts = np.concatenate([y_tr, y_sv])
        clf_final = fit_clf(X_ts, y_ts, X_te, y_te, best_hp)

        # Select trades on TEST
        proba_test = clf_final.predict_proba(X_te)[:, 1]
        test_df = test_df.copy()
        test_df["p"] = proba_test
        mask = (test_df["p"] >= THETA) & \
               (test_df["opening_gap_t1"] >= NEG_LO) & \
               (test_df["opening_gap_t1"] <= NEG_HI) & \
               (test_df["path_pnl_t11_pct"].notna())
        picks = test_df[mask]

        # PEAD stats
        n_picks = len(picks)
        n_pead_picks = int(picks["pead_pass"].sum()) if n_picks > 0 else 0
        n_total_pead = int(test_df["pead_pass"].sum())
        n_total_events = len(test_df)
        pead_precision = n_pead_picks / n_picks * 100 if n_picks > 0 else 0
        pead_recall = n_pead_picks / n_total_pead * 100 if n_total_pead > 0 else 0
        n_pead_missed = n_total_pead - n_pead_picks

        # PnL stats
        if n_picks > 0:
            arith = np.expm1(picks["path_pnl_t11_pct"])
            avg_pnl = arith.mean() * 100
            hit = (arith > 0).mean() * 100
        else:
            avg_pnl = 0
            hit = 0

        print(f"  gamma={best_hp['gamma']}")
        print(f"  TEST events: {n_total_events}, PEAD events: {n_total_pead} ({n_total_pead/n_total_events*100:.1f}%)")
        print()
        print(f"  {'Metric':<30s} {'Value':>10s}")
        print(f"  {'-'*30} {'-'*10}")
        print(f"  {'Trades selected (picks)':<30s} {n_picks:>10d}")
        print(f"  {'TRUE PEAD in picks':<30s} {n_pead_picks:>10d}")
        print(f"  {'PEAD precision (picks that are PEAD)':<30s} {pead_precision:>9.1f}%")
        print(f"  {'Total PEAD in TEST':<30s} {n_total_pead:>10d}")
        print(f"  {'PEAD recall (PEAD caught)':<30s} {pead_recall:>9.1f}%")
        print(f"  {'PEAD MISSED':<30s} {n_pead_missed:>10d}")
        print()
        print(f"  {'Avg PnL per pick':<30s} {avg_pnl:>+9.2f}%")
        print(f"  {'Hit rate':<30s} {hit:>9.1f}%")

        # Also show: what are the non-PEAD picks?
        non_pead_picks = n_picks - n_pead_picks
        print(f"\n  Breakdown of {n_picks} picks:")
        print(f"    TRUE PEAD:     {n_pead_picks:3d} ({pead_precision:.1f}%) -- real post-earnings drift")
        print(f"    NON-PEAD:      {non_pead_picks:3d} ({100-pead_precision:.1f}%) -- mean-reversion / noise")
        print(f"  Of {n_total_pead} PEAD events in TEST:")
        print(f"    CAUGHT:        {n_pead_picks:3d} ({pead_recall:.1f}%)")
        print(f"    MISSED:        {n_pead_missed:3d} ({100-pead_recall:.1f}%)")

        fold_results.append({
            "fold": fi, "gamma": best_hp["gamma"],
            "n_picks": n_picks, "n_pead_picks": n_pead_picks,
            "pead_precision": pead_precision,
            "n_total_pead": n_total_pead,
            "pead_recall": pead_recall,
            "n_pead_missed": n_pead_missed,
            "avg_pnl": avg_pnl, "hit": hit,
        })

    # Aggregate
    print(f"\n{'='*80}")
    print("AGGREGATE PEAD CAPTURE")
    print(f"{'='*80}")
    print(f"\n{'Fold':>4s}  {'Picks':>5s} {'PEAD picks':>10s} {'Precision':>10s} "
          f"{'Total PEAD':>10s} {'Recall':>8s} {'Missed':>7s} {'AvgPnL':>8s}")
    print("-" * 75)
    for r in fold_results:
        print(f"  {r['fold']:>2d}  {r['n_picks']:>5d} {r['n_pead_picks']:>10d} "
              f"{r['pead_precision']:>9.1f}% {r['n_total_pead']:>10d} "
              f"{r['pead_recall']:>7.1f}% {r['n_pead_missed']:>7d} "
              f"{r['avg_pnl']:>+7.2f}%")
    tot_picks = sum(r["n_picks"] for r in fold_results)
    tot_pead_picks = sum(r["n_pead_picks"] for r in fold_results)
    tot_pead = sum(r["n_total_pead"] for r in fold_results)
    tot_missed = sum(r["n_pead_missed"] for r in fold_results)
    avg_prec = tot_pead_picks / tot_picks * 100 if tot_picks > 0 else 0
    avg_recall = tot_pead_picks / tot_pead * 100 if tot_pead > 0 else 0
    print("-" * 75)
    print(f"  {'ALL':>4s}  {tot_picks:>5d} {tot_pead_picks:>10d} "
          f"{avg_prec:>9.1f}% {tot_pead:>10d} "
          f"{avg_recall:>7.1f}% {tot_missed:>7d}")

    print(f"\n  SUMMARY:")
    print(f"    Total picks across 4 folds:        {tot_picks}")
    print(f"    TRUE PEAD in picks:                {tot_pead_picks} ({avg_prec:.1f}% precision)")
    print(f"    Mean-reversion / noise in picks:   {tot_picks - tot_pead_picks} ({100-avg_prec:.1f}%)")
    print(f"    Total PEAD events in all TEST:      {tot_pead}")
    print(f"    PEAD caught:                        {tot_pead_picks} ({avg_recall:.1f}% recall)")
    print(f"    PEAD missed:                        {tot_missed} ({100-avg_recall:.1f}%)")
    print(f"\n  INTERPRETATION:")
    if avg_prec < 20:
        print(f"    PRECISION {avg_prec:.1f}% < 20%: the model is NOT a PEAD detector.")
        print(f"    {100-avg_prec:.1f}% of picks are mean-reversion, not PEAD drift.")
    else:
        print(f"    PRECISION {avg_prec:.1f}%: the model catches some real PEAD.")
    if avg_recall < 10:
        print(f"    RECALL {avg_recall:.1f}% < 10%: the model misses most PEAD events.")
        print(f"    {100-avg_recall:.1f}% of true PEAD events are NOT captured.")
    else:
        print(f"    RECALL {avg_recall:.1f}%: the model catches a meaningful fraction of PEAD.")

    print(f"\n{'='*80}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

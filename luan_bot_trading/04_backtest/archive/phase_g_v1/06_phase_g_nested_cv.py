#!/usr/bin/env python3
"""
Phase G v2 — Nested CV with PEAD capture as the PRIMARY objective.

REWRITE 2026-07-23: The old POS/NEG gap split is DELETED. The objective
is now PEAD capture (precision + recall + F1), not PnL. PnL is secondary
and reported only on PEAD-confirmed picks.

Design:
  - Trade selection: P(PEAD) >= theta ONLY. No gap filter.
  - HP sweep: select gamma by max PEAD F1 on SWEEP_VAL (not PnL).
  - TEST evaluation: PEAD precision/recall/F1 as primary metrics.
    PnL on all picks + PnL on PEAD-confirmed picks as secondary.
  - Random baseline: expected PEAD precision = base rate (~10%).

CLI:
    python luan_bot_trading/04_backtest/06_phase_g_nested_cv.py
    python luan_bot_trading/04_backtest/06_phase_g_nested_cv.py --theta 0.15

NO DB WRITES.
"""
from __future__ import annotations
import sys, importlib.util, json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

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


def select_trades(df, clf, theta):
    """Select trades at P(PEAD) >= theta. NO gap filter."""
    proba = clf.predict_proba(df[SUNDAY_SAFE])[:, 1]
    d = df.copy()
    d["pead_proba"] = proba
    mask = (d["pead_proba"] >= theta) & (d["path_pnl_t11_pct"].notna())
    return d[mask].copy().reset_index(drop=True)


def compute_pead_metrics(picks, test_df):
    """Compute PEAD capture metrics (primary objective)."""
    n_picks = len(picks)
    n_pead_picks = int(picks["pead_pass"].sum()) if n_picks > 0 else 0
    n_total_pead = int(test_df["pead_pass"].sum())
    precision = n_pead_picks / n_picks if n_picks > 0 else 0
    recall = n_pead_picks / n_total_pead if n_total_pead > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "n_picks": n_picks,
        "n_pead_picks": n_pead_picks,
        "n_total_pead": n_total_pead,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pead_missed": n_total_pead - n_pead_picks,
    }


def compute_pnl_metrics(picks):
    """Compute PnL metrics (secondary, for context only)."""
    if len(picks) == 0:
        return {"avg_pnl_all": 0, "hit_all": 0, "avg_pnl_pead": 0, "hit_pead": 0}
    arith = np.expm1(picks["path_pnl_t11_pct"])
    pead_picks = picks[picks["pead_pass"] == 1]
    pead_arith = np.expm1(pead_picks["path_pnl_t11_pct"]) if len(pead_picks) > 0 else pd.Series(dtype=float)
    return {
        "avg_pnl_all": float(arith.mean()) * 100 if len(arith) > 0 else 0,
        "hit_all": float((arith > 0).mean()) * 100 if len(arith) > 0 else 0,
        "avg_pnl_pead": float(pead_arith.mean()) * 100 if len(pead_arith) > 0 else 0,
        "hit_pead": float((pead_arith > 0).mean()) * 100 if len(pead_arith) > 0 else 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, default=0.20)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--n-slots", type=int, default=4)
    parser.add_argument("--n-rng-trials", type=int, default=100)
    args = parser.parse_args(argv)

    folds = DEFAULT_FOLDS[:args.n_folds]
    theta = args.theta

    print("=" * 80)
    print("PHASE G v2 — NESTED CV (PEAD capture objective, no gap filter)")
    print("=" * 80)
    print(f"Theta     : {theta}")
    print(f"Folds     : {len(folds)}")
    print(f"Features  : {len(SUNDAY_SAFE)} Sunday-safe")
    print(f"HP sweep  : gamma in {{3,5,10,20}}, select by max PEAD F1")
    print(f"Objective : PEAD precision + recall + F1 (PRIMARY)")
    print(f"Secondary : PnL on all picks + PnL on PEAD-confirmed picks")
    print("=" * 80)

    # --- Shared data load ---
    print("\n[1] Loading train_matrix + priming + gates + paths ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)
    print(f"    rows: {len(df)}, pead_pass: {int(df['pead_pass'].sum())} "
          f"({df['pead_pass'].mean()*100:.1f}%)")

    # Pre-cache trading calendar
    print("\n[2] Caching trading calendar ...")
    with pd.HDFStore(DB, mode="r") as sstore:
        all_dates = set()
        for key in sstore.keys():
            if not key.startswith("/sp400/"):
                continue
            d = sstore[key]
            all_dates.update(pd.to_datetime(d["Date"]).tolist())
    calendar = sorted(all_dates)
    print(f"    calendar: {len(calendar)} trading days")

    fold_results = []
    print(f"\n[3] Running {len(folds)} folds ...")

    for fi, (te, sve, tse) in enumerate(folds, 1):
        print(f"\n{'='*70}")
        print(f"  FOLD {fi}: TRAIN<={te} | SWEEP {te}->{sve} | TEST {sve}->{tse}")
        print(f"{'='*70}")

        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        print(f"  TRAIN={len(train_df)} SWEEP={len(sweep_df)} TEST={len(test_df)} "
              f"pead: TRAIN={int(train_df['pead_pass'].sum())} "
              f"SWEEP={int(sweep_df['pead_pass'].sum())} "
              f"TEST={int(test_df['pead_pass'].sum())}")

        X_tr = train_df[SUNDAY_SAFE]
        y_tr = train_df["pead_pass"].astype(int).values
        X_sv = sweep_df[SUNDAY_SAFE]
        y_sv = sweep_df["pead_pass"].astype(int).values
        X_te = test_df[SUNDAY_SAFE]
        y_te = test_df["pead_pass"].astype(int).values

        # --- HP sweep by PEAD F1 (NOT PnL) ---
        print(f"\n  Sweeping {len(SWEEP_GRID)} gamma configs (select by PEAD F1) ...")
        sweep_results = []
        for hp in SWEEP_GRID:
            t0 = time.time()
            clf = fit_clf(X_tr, y_tr, X_sv, y_sv, hp)
            picks_sv = select_trades(sweep_df, clf, theta)
            pead_m = compute_pead_metrics(picks_sv, sweep_df)
            sweep_results.append({"hp": hp, "f1": pead_m["f1"],
                                   "precision": pead_m["precision"],
                                   "recall": pead_m["recall"],
                                   "n_picks": pead_m["n_picks"]})
            print(f"    gamma={hp['gamma']:>3d}  F1={pead_m['f1']:.4f}  "
                  f"prec={pead_m['precision']*100:.1f}%  rec={pead_m['recall']*100:.1f}%  "
                  f"n={pead_m['n_picks']}  t={time.time()-t0:.1f}s")

        # Select best HP by F1 (fallback to gamma=5 if no valid)
        valid = [r for r in sweep_results if r["n_picks"] >= 5 and r["f1"] > 0]
        if valid:
            best = max(valid, key=lambda r: r["f1"])
            sel_hp = best["hp"]
            sel_reason = f"best F1={best['f1']:.4f} (prec={best['precision']*100:.1f}%, rec={best['recall']*100:.1f}%)"
        else:
            sel_hp = {"gamma": 5, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
            sel_reason = "fallback gamma=5 (no config reached F1>0 with n>=5)"
        print(f"\n  Selected HP: gamma={sel_hp['gamma']}  ({sel_reason})")

        # --- Retrain on TRAIN+SWEEP ---
        X_ts = pd.concat([X_tr, X_sv])
        y_ts = np.concatenate([y_tr, y_sv])
        clf_final = fit_clf(X_ts, y_ts, X_te, y_te, sel_hp)

        from sklearn.metrics import roc_auc_score
        auc_test = roc_auc_score(y_te, clf_final.predict_proba(X_te)[:, 1])
        print(f"  TEST AUC: {auc_test:.4f}")

        # --- Evaluate on TEST (PEAD capture primary, PnL secondary) ---
        picks = select_trades(test_df, clf_final, theta)
        pead_m = compute_pead_metrics(picks, test_df)
        pnl_m = compute_pnl_metrics(picks)

        print(f"\n  --- FOLD {fi} TEST RESULTS ---")
        print(f"  {'PEAD CAPTURE (PRIMARY)':^50s}")
        print(f"  {'Metric':<30s} {'Value':>10s}")
        print(f"  {'-'*30} {'-'*10}")
        print(f"  {'Picks':<30s} {pead_m['n_picks']:>10d}")
        print(f"  {'TRUE PEAD in picks':<30s} {pead_m['n_pead_picks']:>10d}")
        print(f"  {'Precision':<30s} {pead_m['precision']*100:>9.1f}%")
        print(f"  {'Total PEAD in TEST':<30s} {pead_m['n_total_pead']:>10d}")
        print(f"  {'Recall':<30s} {pead_m['recall']*100:>9.1f}%")
        print(f"  {'F1':<30s} {pead_m['f1']*100:>9.1f}%")
        print(f"  {'PEAD missed':<30s} {pead_m['n_pead_missed']:>10d}")
        print(f"  {'-'*30} {'-'*10}")
        print(f"  {'PNL (SECONDARY)':^50s}")
        print(f"  {'Avg PnL (all picks)':<30s} {pnl_m['avg_pnl_all']:>+9.2f}%")
        print(f"  {'Hit rate (all picks)':<30s} {pnl_m['hit_all']:>9.1f}%")
        print(f"  {'Avg PnL (PEAD picks only)':<30s} {pnl_m['avg_pnl_pead']:>+9.2f}%")
        print(f"  {'Hit rate (PEAD picks only)':<30s} {pnl_m['hit_pead']:>9.1f}%")

        # Random baseline for PEAD precision
        n_total = len(test_df)
        base_rate = test_df["pead_pass"].mean()
        print(f"\n  Random baseline: expected precision = {base_rate*100:.1f}% (base rate)")

        fold_results.append({
            "fold": fi, "gamma": sel_hp["gamma"], "auc_test": float(auc_test),
            **pead_m, **pnl_m,
            "base_rate": float(base_rate),
        })

    # --- Aggregate ---
    print(f"\n{'='*80}")
    print("AGGREGATE: PEAD CAPTURE (PRIMARY OBJECTIVE)")
    print(f"{'='*80}")

    print(f"\n{'Fold':>4s}  {'gamma':>5s}  {'Picks':>5s} {'PEAD':>5s} {'Prec':>6s} "
          f"{'Total':>5s} {'Recall':>7s} {'F1':>6s} {'Missed':>7s}  "
          f"{'PnL_all':>8s} {'PnL_pead':>9s}")
    print("-" * 85)
    for r in fold_results:
        print(f"  {r['fold']:>2d}  gamma={r['gamma']:>2d}  "
              f"{r['n_picks']:>5d} {r['n_pead_picks']:>5d} "
              f"{r['precision']*100:>5.1f}% "
              f"{r['n_total_pead']:>5d} {r['recall']*100:>6.1f}% "
              f"{r['f1']*100:>5.1f}% {r['n_pead_missed']:>7d}  "
              f"{r['avg_pnl_all']:>+7.2f}% {r['avg_pnl_pead']:>+8.2f}%")

    # Averages
    avg_prec = np.mean([r["precision"] for r in fold_results])
    avg_rec = np.mean([r["recall"] for r in fold_results])
    avg_f1 = np.mean([r["f1"] for r in fold_results])
    avg_pnl_all = np.mean([r["avg_pnl_all"] for r in fold_results])
    avg_pnl_pead = np.mean([r["avg_pnl_pead"] for r in fold_results])
    tot_picks = sum(r["n_picks"] for r in fold_results)
    tot_pead_picks = sum(r["n_pead_picks"] for r in fold_results)
    tot_pead = sum(r["n_total_pead"] for r in fold_results)

    print("-" * 85)
    print(f"  {'AVG':>4s}  {'':>7s}  "
          f"{int(np.mean([r['n_picks'] for r in fold_results])):>5.0f} "
          f"{int(np.mean([r['n_pead_picks'] for r in fold_results])):>5.0f} "
          f"{avg_prec*100:>5.1f}% "
          f"{int(np.mean([r['n_total_pead'] for r in fold_results])):>5.0f} "
          f"{avg_rec*100:>6.1f}% {avg_f1*100:>5.1f}% "
          f"{int(np.mean([r['n_pead_missed'] for r in fold_results])):>7.0f}  "
          f"{avg_pnl_all:>+7.2f}% {avg_pnl_pead:>+8.2f}%")

    print(f"\n  PEAD CAPTURE SUMMARY:")
    print(f"    Total picks (4 folds):        {tot_picks}")
    print(f"    TRUE PEAD in picks:            {tot_pead_picks} ({avg_prec*100:.1f}% precision)")
    print(f"    Total PEAD events in TEST:     {tot_pead}")
    print(f"    PEAD recall:                   {avg_rec*100:.1f}%")
    print(f"    PEAD F1:                       {avg_f1*100:.1f}%")
    print(f"    PEAD missed:                   {tot_pead - tot_pead_picks}")
    print(f"    Base rate (random precision): {np.mean([r['base_rate'] for r in fold_results])*100:.1f}%")
    print(f"    Lift over random:              {(avg_prec - np.mean([r['base_rate'] for r in fold_results]))*100:.1f}pp")
    print(f"\n  PnL (SECONDARY):")
    print(f"    Avg PnL per pick (all):         {avg_pnl_all:+.2f}%")
    print(f"    Avg PnL per pick (PEAD only):   {avg_pnl_pead:+.2f}%")

    # Save
    out_dir = HERE / f"phase_g_v2_pead_capture_n{args.n_folds}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_results).to_csv(out_dir / "fold_results.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "theta": theta, "n_folds": len(fold_results),
            "features": SUNDAY_SAFE, "sweep_grid": SWEEP_GRID,
            "folds": folds, "fold_results": fold_results,
            "aggregate": {
                "mean_precision": float(avg_prec),
                "mean_recall": float(avg_rec),
                "mean_f1": float(avg_f1),
                "total_picks": tot_picks, "total_pead_picks": tot_pead_picks,
                "total_pead": tot_pead,
                "mean_pnl_all": float(avg_pnl_all),
                "mean_pnl_pead": float(avg_pnl_pead),
            },
            "created_at": pd.Timestamp.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

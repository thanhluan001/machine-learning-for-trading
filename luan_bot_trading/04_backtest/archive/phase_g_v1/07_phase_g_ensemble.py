"""
Phase G v1.1 -- Multi-rule ensemble (positive-gap + negative-gap blend).

Per phase_g_findings.md §D.5.1 path (2) (multi-rule ensemble):
  explore whether blending the positive-gap rule (POS) with the
  negative-gap rule (NEG) smooths the cross-fold variance observed
  in Appendix D's nested CV.

This script reuses the Appendix D nested-CV infrastructure:
  - Same 4 anchored walk-forward folds (see DEFAULT_FOLDS).
  - Same selected HP per fold (read from
    `phase_g_v1_1_nested_cv_n4/fold_results.csv` -- no re-sweep, to
    keep this comparable to Appendix D's strategy selection).
  - Same n_slots=4 portfolio simulator + 100 random-trial baseline.

For each fold, we evaluate 4 alternative operating points on the
same TEST slice, using the same trained classifier:
  POS_only        : gap in [+2%, +15%],  theta_screen = 0.20
  NEG_only        : gap in [-15%, -2%],  theta_screen = 0.15
  UNION_equal     : union of POS@0.20 and NEG@0.20  (equal thresholds)
  UNION_split     : union of POS@0.20 and NEG@0.15  (NEG more permissive)

Generated trades are then fed into the portfolio simulator with
n_slots=4 (§B.2 derived optimum).

CLI:
  python luan_bot_trading/04_backtest/07_phase_g_ensemble.py
  python luan_bot_trading/04_backtest/07_phase_g_ensemble.py --n-folds 4 --n-slots 4

NO DB WRITES.
"""
from __future__ import annotations
import sys, importlib.util, json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "tm", HERE.parent / "03_model" / "01_train_model.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
pg_spec = importlib.util.spec_from_file_location(
    "pg", HERE.parent / "03_model" / "02_phase_g_sunday_classifier.py")
pg = importlib.util.module_from_spec(pg_spec); pg_spec.loader.exec_module(pg)
ps_spec = importlib.util.spec_from_file_location(
    "ps", HERE / "04_phase_g_portfolio.py")
ps = importlib.util.module_from_spec(ps_spec); ps_spec.loader.exec_module(ps)
rb_spec = importlib.util.spec_from_file_location(
    "rb", HERE / "_phase_g_random_baseline.py")
rb = importlib.util.module_from_spec(rb_spec); rb_spec.loader.exec_module(rb)

DB = tm.DB_FILE
SUNDAY_SAFE_FEATURES = pg.SUNDAY_SAFE_FEATURES

# Load Appendix D's per-fold selected HP (no re-sweep -- stay comparable)
APPD_RESULTS_CSV = HERE / "archive" / "experiments" / "phase_g_v1_1_nested_cv_n4" / "fold_results.csv"

# Operating points explored
RULES = {
    "POS_only": {
        "pos": {"theta": 0.20, "gap_lo": 0.02, "gap_hi": 0.15},
        "neg": None,
    },
    "NEG_only": {
        "pos": None,
        "neg": {"theta": 0.15, "gap_lo": -0.15, "gap_hi": -0.02},
    },
    "UNION_equal": {
        "pos": {"theta": 0.20, "gap_lo": 0.02, "gap_hi": 0.15},
        "neg": {"theta": 0.20, "gap_lo": -0.15, "gap_hi": -0.02},
    },
    "UNION_split": {
        "pos": {"theta": 0.20, "gap_lo": 0.02, "gap_hi": 0.15},
        "neg": {"theta": 0.15, "gap_lo": -0.15, "gap_hi": -0.02},
    },
}

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),   # Fold 1
    ("2024-06-30", "2024-12-31", "2025-06-30"),   # Fold 2
    ("2024-12-31", "2025-06-30", "2025-12-31"),   # Fold 3
    ("2025-06-30", "2025-12-31", "2026-06-30"),   # Fold 4
]


def fit_classifier(X_train, y_train, X_val, y_val, hp):
    import xgboost as xgb
    params = dict(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        n_estimators=hp["n_estimators"],
        learning_rate=0.05,
        max_depth=hp["max_depth"],
        min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"],
        reg_lambda=1.0,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
    )
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False)
    return clf


def select_trades_unified(df, clf, rule_def):
    """Unified trade selection: union of POS and NEG rule branches."""
    proba = clf.predict_proba(df[SUNDAY_SAFE_FEATURES])[:, 1]
    df2 = df.copy()
    df2["pead_proba"] = proba
    base_mask = df2["path_pnl_t11_pct"].notna()
    pick_idx = []
    for branch in ("pos", "neg"):
        cfg = rule_def[branch]
        if cfg is None:
            continue
        m = base_mask & \
            (df2["pead_proba"] >= cfg["theta"]) & \
            (df2["opening_gap_t1"] >= cfg["gap_lo"]) & \
            (df2["opening_gap_t1"] <= cfg["gap_hi"])
        pick_idx.append(df2.index[m].values)
    if not pick_idx:
        return df2.iloc[0:0].copy().reset_index(drop=True)
    all_idx = np.unique(np.concatenate(pick_idx))
    return df2.loc[all_idx].reset_index(drop=True)


def select_random_trades(val_df, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for week, g in val_df.groupby("calendar_week_group", sort=True):
        g_ok = g.dropna(subset=["path_pnl_t11_pct"])
        if g_ok.empty:
            continue
        idx = rng.integers(len(g_ok))
        rows.append(g_ok.iloc[idx])
    return pd.DataFrame(rows).reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--n-slots", type=int, default=4)
    parser.add_argument("--initial-nav", type=float, default=100_000.0)
    parser.add_argument("--n-rng-trials", type=int, default=100)
    args = parser.parse_args(argv)

    folds = DEFAULT_FOLDS[:args.n_folds]

    print("=" * 78)
    print("PHASE G v1.1 -- MULTI-RULE ENSEMBLE (POS + NEG blend)")
    print("=" * 78)
    print(f"Folds      : {len(folds)}")
    print(f"N_SLOTS    : {args.n_slots}")
    print(f"N_RNG      : {args.n_rng_trials}")
    print(f"Rules evaluated: {list(RULES.keys())}")
    for name, rdef in RULES.items():
        desc_parts = []
        for b in ("pos", "neg"):
            cfg = rdef[b]
            if cfg is None:
                desc_parts.append(f"{b}=OFF")
            else:
                desc_parts.append(
                    f"{b}=P>={cfg['theta']:.2f},gap[{cfg['gap_lo']:+.2f},{cfg['gap_hi']:+.2f}]")
        print(f"  {name:>14s}: " + " | ".join(desc_parts))
    print("=" * 78)

    # Read Appendix D per-fold selected HP to reproduce exactly
    print("\n[1] Loading Appendix D fold_results.csv for selected HP ...")
    appd = pd.read_csv(APPD_RESULTS_CSV)
    print(f"    Loaded {len(appd)} fold rows")
    print(f"    Columns: {list(appd.columns)}")
    # Use sel_gamma, sel_mcw, sel_md, sel_n_est as the per-fold HP
    fold_hp = []
    for fold_idx, row in appd.iterrows():
        fold_hp.append({
            "gamma": int(row["sel_gamma"]),
            "min_child_weight": int(row["sel_mcw"]),
            "max_depth": int(row["sel_md"]),
            "n_estimators": int(row["sel_n_est"]),
        })
        print(f"  Fold {int(row['fold'])}: gamma={int(row['sel_gamma'])}, "
              f"mcw={int(row['sel_mcw'])}, md={int(row['sel_md'])}, "
              f"n_est={int(row['sel_n_est'])}")

    # --- Shared data load + gates + entry-PnL + paths + calendar ---
    print("\n[2] Loading train_matrix + §12 cutoff + gate computation ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    print(f"    rows after §12 cut: {len(df)}")
    df = pg.v3.compute_pead_gates_full(df)
    print(f"    pead_pass positives: {int(df['pead_pass'].sum())} "
          f"({df['pead_pass'].mean()*100:.2f}%)")

    print("\n[3] Computing entry-PnL + 12-snap trade paths on ALL rows ...")
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)
    print(f"    coverage: {int(df['path_pnl_t11_pct'].notna().sum())}/{len(df)}")

    print("\n[4] Pre-caching trading calendar for portfolio sim speedup ...")
    with pd.HDFStore(DB, mode="r") as sstore:
        all_dates = set()
        for key in sstore.keys():
            if not key.startswith("/sp400/"):
                continue
            d = sstore[key]
            all_dates.update(pd.to_datetime(d["Date"]).tolist())
    calendar = sorted(all_dates)
    print(f"    calendar: {len(calendar)} trading days")

    # --- Per-fold ensemble evaluation ---
    fold_results = []
    print(f"\n[5] Running {len(folds)} folds x {len(RULES)} rules ...")
    for fold_idx, (train_end, sweep_end, test_end) in enumerate(folds, 1):
        print("\n" + "=" * 60)
        print(f"  FOLD {fold_idx}/{len(folds)}: "
              f"TEST {sweep_end}->{test_end}")
        print("=" * 60)

        train_ts = pd.Timestamp(train_end)
        sweep_ts = pd.Timestamp(sweep_end)
        test_ts = pd.Timestamp(test_end)
        rd = pd.to_datetime(df["report_date"])
        train_mask = rd <= train_ts
        sweep_mask = (rd > train_ts) & (rd <= sweep_ts)
        test_mask = (rd > sweep_ts) & (rd <= test_ts)
        train_df = df[train_mask].copy().reset_index(drop=True)
        sweep_df = df[sweep_mask].copy().reset_index(drop=True)
        test_df = df[test_mask].copy().reset_index(drop=True)
        print(f"  TRAIN={len(train_df)}  SWEEP={len(sweep_df)}  TEST={len(test_df)}")

        # Combine TRAIN+SWEEP_VAL for final-fit (matches Appendix D §4d)
        X_tr = train_df[SUNDAY_SAFE_FEATURES].copy()
        y_tr = train_df["pead_pass"].astype(int).values
        X_sv = sweep_df[SUNDAY_SAFE_FEATURES].copy()
        y_sv = sweep_df["pead_pass"].astype(int).values
        X_ts = pd.concat([X_tr, X_sv], axis=0).reset_index(drop=True)
        y_ts = np.concatenate([y_tr, y_sv])

        # Refit on TRAIN+SWEEP_VAL with Appendix-D selected HP
        hp = fold_hp[fold_idx - 1]
        X_te = test_df[SUNDAY_SAFE_FEATURES].copy()
        y_te = test_df["pead_pass"].astype(int).values
        print(f"  Training final classifier (gamma={hp['gamma']}) on "
              f"{len(y_ts)} TRAIN+SWEEP rows ...")
        clf = fit_classifier(X_ts, y_ts, X_te, y_te, hp)
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_test = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        ap_test = average_precision_score(y_te, clf.predict_proba(X_te)[:, 1])
        print(f"  TEST AUC: {auc_test:.4f}  AP: {ap_test:.4f}")

        # --- Apply each rule ---
        print(f"\n  Rule evaluation per rule on TEST slice ...")
        per_rule = {}
        for rule_name, rule_def in RULES.items():
            trades = select_trades_unified(test_df, clf, rule_def)
            n_picks = len(trades)
            # Per-trade arithmetic mean (quick metric, doesn't account for slot collisions)
            if n_picks >= 1:
                mean_pnl_pct = float(
                    np.expm1(trades["path_pnl_t11_pct"]).mean() * 100)
                hit_pct = float(
                    (np.expm1(trades["path_pnl_t11_pct"]) > 0).mean() * 100)
            else:
                mean_pnl_pct = float("nan")
                hit_pct = float("nan")
            # Portfolio sim
            result = rb._simulate_with_cached_calendar(
                trades, args.n_slots, args.initial_nav, calendar)
            s = result.get("summary", {})
            print(f"    {rule_name:>14s}: n_picks={n_picks:>3d}  "
                  f"mean_arith_pnl={mean_pnl_pct:+.3f}%  hit={hit_pct:>5.1f}%  "
                  f"trades_exec={s.get('n_trades_executed', 0):>3d}  "
                  f"IRR={s.get('irr_pct', float('nan')):+.2f}%  "
                  f"Sharpe={s.get('sharpe_liq_annualized', float('nan')):+.2f}  "
                  f"MaxDD={s.get('max_drawdown_pct', float('nan')):.2f}%")
            per_rule[rule_name] = {
                "n_picks": n_picks,
                "mean_arith_pnl_pct": mean_pnl_pct,
                "hit_pct": hit_pct,
                **{k: v for k, v in s.items()},
            }

        # --- Random baseline 100 trials per fold ---
        print(f"\n  Random baseline ({args.n_rng_trials} trials) ...")
        rand_rows = []
        for trial in range(args.n_rng_trials):
            seed = trial * 7 + 100
            rt = select_random_trades(test_df, seed)
            result_r = rb._simulate_with_cached_calendar(
                rt, args.n_slots, args.initial_nav, calendar)
            s_r = result_r.get("summary", {})
            if not s_r:
                continue
            rand_rows.append({
                "trial": trial,
                "irr": s_r.get("irr_pct", float("nan")),
                "sharpe": s_r.get("sharpe_liq_annualized", float("nan")),
                "max_dd": s_r.get("max_drawdown_pct", float("nan")),
            })
        rand_df = pd.DataFrame(rand_rows)
        rand_mean_sharpe = float(rand_df["sharpe"].mean()) if len(rand_df) else float("nan")
        rand_mean_irr    = float(rand_df["irr"].mean())    if len(rand_df) else float("nan")
        print(f"  Random: Sharpe mean={rand_mean_sharpe:+.2f}  "
              f"IRR mean={rand_mean_irr:+.2f}%  n={len(rand_df)}")

        # exceedance fractions per rule
        rule_exceedance = {}
        for rule_name in RULES:
            rule_s = per_rule[rule_name]
            rule_sharpe = rule_s.get("sharpe_liq_annualized", float("nan"))
            rule_irr = rule_s.get("irr_pct", float("nan"))
            if len(rand_df) and not np.isnan(rule_sharpe):
                frac_sh = float((rand_df["sharpe"] > rule_sharpe).mean())
                # invert: fraction exceeded = 1 - frac_greater
                frac_ex_sh = 1.0 - frac_sh
            else:
                frac_ex_sh = float("nan")
            if len(rand_df) and not np.isnan(rule_irr) and not rand_df['irr'].isna().all():
                frac_ir = float((rand_df["irr"] > rule_irr).mean())
                frac_ex_ir = 1.0 - frac_ir
            else:
                frac_ex_ir = float("nan")
            rule_exceedance[rule_name] = {
                "frac_random_exceeded_sharpe": frac_ex_sh,
                "frac_random_exceeded_irr": frac_ex_ir,
            }
            print(f"    {rule_name:>14s}: exceeds {frac_ex_sh*100:>5.1f}% "
                  f"random Sharpe / {frac_ex_ir*100:>5.1f}% random IRR")

        fold_results.append({
            "fold": fold_idx,
            "test_slice": f"{sweep_end}->{test_end}",
            "selected_hp": hp,
            "test_auc": float(auc_test),
            "test_ap": float(ap_test),
            "test_n_rows": len(test_df),
            "test_n_pos": int(test_df["pead_pass"].sum()),
            "random_mean_sharpe": rand_mean_sharpe,
            "random_mean_irr": rand_mean_irr,
            "random_n": len(rand_df),
            "rules": {
                rule_name: {
                    **per_rule[rule_name],
                    **rule_exceedance[rule_name],
                }
                for rule_name in RULES
            },
        })

    # --- Aggregate across folds ---
    print("\n" + "=" * 78)
    print("MULTI-RULE ENSEMBLE -- CROSS-FOLD COMPARISON")
    print("=" * 78)

    # Build a wide table for visual comparison
    rule_names = list(RULES.keys())
    metrics = ["n_trades_executed", "irr_pct", "sharpe_liq_annualized",
               "max_drawdown_pct", "hit_rate_pct", "avg_trade_pnl_pct",
               "frac_random_exceeded_sharpe", "frac_random_exceeded_irr",
               "mean_arith_pnl_pct"]

    # Per-fold per-rule table
    rows = []
    for fr in fold_results:
        for rn in rule_names:
            r = fr["rules"][rn]
            row = {"fold": fr["fold"], "rule": rn,
                   "test_slice": fr["test_slice"]}
            for m in metrics:
                row[m] = r.get(m, float("nan"))
            rows.append(row)
    wide = pd.DataFrame(rows)

    # Print per-fold table per rule
    for rn in rule_names:
        print(f"\n--- Rule: {rn} ---")
        sub = wide[wide["rule"] == rn].copy()
        print(f"{'Fold':>4}  {'slice':<24}  {'n_tr':>4}  {'IRR%':>7}  "
              f"{'Sharpe':>7}  {'MaxDD%':>7}  {'hit%':>5}  {'avgPnL%':>8}  "
              f"{'%rShEx':>7}  {'%rIREx':>7}")
        print("-" * 100)
        for _, r in sub.iterrows():
            print(f"  {int(r['fold']):>2d}  {r['test_slice']:>22s}  "
                  f"{int(r['n_trades_executed']):>3d}  "
                  f"{r['irr_pct']:>+6.2f}%  {r['sharpe_liq_annualized']:>+6.2f}  "
                  f"{r['max_drawdown_pct']:>+6.2f}%  "
                  f"{r['hit_rate_pct']:>4.1f}%  "
                  f"{r['avg_trade_pnl_pct']:>+7.3f}%  "
                  f"{r['frac_random_exceeded_sharpe']*100:>5.1f}%  "
                  f"{r['frac_random_exceeded_irr']*100:>5.1f}%")
        # means
        print(f"  {'AVG':>2s}")
        print(f"     n_tr={sub['n_trades_executed'].mean():>5.1f}  "
              f"IRR={sub['irr_pct'].mean():>+6.2f}%  "
              f"Sharpe={sub['sharpe_liq_annualized'].mean():>+6.2f}  "
              f"MaxDD={sub['max_drawdown_pct'].mean():>+6.2f}%  "
              f"hit={sub['hit_rate_pct'].mean():>4.1f}%  "
              f"avgPnL={sub['avg_trade_pnl_pct'].mean():>+7.3f}%  "
              f"%rShEx={sub['frac_random_exceeded_sharpe'].mean()*100:>5.1f}%  "
              f"%rIREx={sub['frac_random_exceeded_irr'].mean()*100:>5.1f}%")

    # Print per-fold row=WULE rule=METRICS compact comparison
    print(f"\n--- Compact cross-rule comparison (mean across 4 folds) ---")
    print(f"{'Rule':>14}  {'IRR%':>7}  {'Sharpe':>7}  {'MaxDD%':>7}  "
          f"{'hit%':>5}  {'avgPnL%':>8}  {'%rShEx':>7}  {'%rIREx':>7}  "
          f"{'n_trades':>9}")
    for rn in rule_names:
        sub = wide[wide["rule"] == rn]
        print(f"  {rn:>12s}  "
              f"{sub['irr_pct'].mean():>+6.2f}%  "
              f"{sub['sharpe_liq_annualized'].mean():>+6.2f}  "
              f"{sub['max_drawdown_pct'].mean():>+6.2f}%  "
              f"{sub['hit_rate_pct'].mean():>4.1f}%  "
              f"{sub['avg_trade_pnl_pct'].mean():>+7.3f}%  "
              f"{sub['frac_random_exceeded_sharpe'].mean()*100:>5.1f}%  "
              f"{sub['frac_random_exceeded_irr'].mean()*100:>5.1f}%  "
              f"{sub['n_trades_executed'].mean():>5.1f}")

    # Persist
    out_dir = HERE / f"phase_g_v1_1_ensemble_n{args.n_folds}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_dir / "ensemble_wide.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_folds": len(fold_results),
            "n_slots": args.n_slots,
            "n_rng_trials": args.n_rng_trials,
            "rules": RULES,
            "folds": folds,
            "fold_results": fold_results,
            "aggregate_per_rule": {
                rn: {
                    "mean_irr_pct": float(wide[wide["rule"]==rn]["irr_pct"].mean()),
                    "mean_sharpe": float(wide[wide["rule"]==rn]["sharpe_liq_annualized"].mean()),
                    "mean_max_dd_pct": float(wide[wide["rule"]==rn]["max_drawdown_pct"].mean()),
                    "mean_hit_pct": float(wide[wide["rule"]==rn]["hit_rate_pct"].mean()),
                    "mean_avg_pnl_pct": float(wide[wide["rule"]==rn]["avg_trade_pnl_pct"].mean()),
                    "mean_n_trades": float(wide[wide["rule"]==rn]["n_trades_executed"].mean()),
                    "mean_frac_random_exceeded_sharpe": float(wide[wide["rule"]==rn]["frac_random_exceeded_sharpe"].mean()),
                    "mean_frac_random_exceeded_irr": float(wide[wide["rule"]==rn]["frac_random_exceeded_irr"].mean()),
                }
                for rn in rule_names
            },
            "created_at": pd.Timestamp.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"\nSaved artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

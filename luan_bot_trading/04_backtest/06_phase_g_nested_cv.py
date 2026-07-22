"""
Phase G v1.1 -- Anchored walk-forward nested cross-validation.

Per phase_g_findings.md §C.5.1 (REVISED TOP PRIORITY): decouple
hyperparameter selection from VAL-test comparison via nested CV.

Design (anchored walk-forward):
  For each fold k, define three non-overlapping date windows:
    TRAIN     : 2015-01-01 -> split_train_end_k      (anchored, growing)
    SWEEP_VAL : split_train_end_k -> split_sweep_end_k  (HP selection)
    TEST      : split_sweep_end_k -> split_test_end_k   (OOS evaluation)

  TRAIN is used to fit the classifier at a given hyperparameter set;
  SWEEP_VAL is used to measure PnL at the deployable operating point
  and select the best HP set;
  TEST is the genuinely held-out OOS slice -- not seen at training
  AND not seen at HP-selection time.

  For each fold we:
    1. Sweep over a focused HP grid (gamma in {3, 5, 10, 20} with
       fixed min_child_weight=50, max_depth=3, n_estimators=300) --
       a tractable 4-config subset of the §A.1 72-config grid,
       specifically spanning the gamma dimension that mattered.
    2. Select the HP set that maximizes SWEEP_VAL PnL at
       P(PEAD)>=0.20 AND gap[+2%,+15%] with N_trades>=20.
    3. Retrain the classifier ON [TRAIN + SWEEP_VAL] with the
       selected HP set -- taking advantage of more data after
       hyperparameter selection.
    4. Evaluate the resulting rule on TEST (truly OOS).

  Aggregate the TEST-fold results: average IRR/Sharpe/MaxDD/per-event
  PnL across folds. This is the realistic estimate of OOS strategy
  performance that survives the §C.2 critique.

CLI:
  python luan_bot_trading/04_backtest/06_phase_g_nested_cv.py
  python luan_bot_trading/04_backtest/06_phase_g_nested_cv.py --n-folds 3
  python luan_bot_trading/04_backtest/06_phase_g_nested_cv.py --n-folds 4 --n-rng-trials 50

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

# Operating point (deployable rule per §B.6 / §C.5)
THETA_SCREEN = 0.20
GAP_LO = 0.02
GAP_HI = 0.15

# Focused sweep grid (4 configs) -- just vary gamma
SWEEP_GRID = [
    {"gamma": g, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
    for g in [3, 5, 10, 20]
]

# Fold definitions: (train_end, sweep_end, test_end)
DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),   # Fold 1
    ("2024-06-30", "2024-12-31", "2025-06-30"),   # Fold 2
    ("2024-12-31", "2025-06-30", "2025-12-31"),   # Fold 3
    ("2025-06-30", "2025-12-31", "2026-06-30"),   # Fold 4
]


def fit_classifier(X_train, y_train, X_val, y_val, hp):
    """Fit an XGBClassifier with the given hp dict, return the model."""
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


def select_v11_trades(val_df, clf):
    """Apply P(PEAD)>=theta AND gap in [lo,hi] filter."""
    proba = clf.predict_proba(val_df[SUNDAY_SAFE_FEATURES])[:, 1]
    df = val_df.copy()
    df["pead_proba"] = proba
    mask = (
        (df["pead_proba"] >= THETA_SCREEN) &
        (df["opening_gap_t1"] >= GAP_LO) &
        (df["opening_gap_t1"] <= GAP_HI) &
        (df["path_pnl_t11_pct"].notna())
    )
    return df[mask].copy().reset_index(drop=True)


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


def filter_by_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return rows where start < report_date <= end."""
    s = pd.Timestamp(start) if isinstance(start, str) else start
    e = pd.Timestamp(end) if isinstance(end, str) else end
    mask = (pd.to_datetime(df["report_date"]) > s) & \
           (pd.to_datetime(df["report_date"]) <= e)
    return df[mask].reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=4,
                        help="Number of folds to run (defaults to all 4)")
    parser.add_argument("--n-slots", type=int, default=4)
    parser.add_argument("--initial-nav", type=float, default=100_000.0)
    parser.add_argument("--n-rng-trials", type=int, default=100,
                        help="Random trials per fold for null distribution")
    args = parser.parse_args(argv)

    folds = DEFAULT_FOLDS[:args.n_folds]

    print("=" * 78)
    print("PHASE G v1.1 -- NESTED CROSS-VALIDATION (anchored walk-forward)")
    print("=" * 78)
    print(f"Folds      : {len(folds)}")
    for i, (te, sve, tse) in enumerate(folds, 1):
        print(f"  Fold {i}: TRAIN 2015-01 -> {te} | "
              f"SWEEP_VAL {te} -> {sve} | TEST {sve} -> {tse}")
    print(f"N_SLOTS    : {args.n_slots}")
    print(f"INITIAL_NAV: ${args.initial_nav:,.0f}")
    print(f"N_RNG      : {args.n_rng_trials}")
    print(f"OPERATING POINT: P(PEAD)>={THETA_SCREEN} AND gap in [{GAP_LO},{GAP_HI}]")
    print(f"SWEEP GRID: gamma in {{3,5,10,20}} ({len(SWEEP_GRID)} configs/fold)")
    print("=" * 78)

    # --- Shared data load ---
    print("\n[1] Loading train_matrix + §12 cutoff + gate computation ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    print(f"    rows after §12 cut: {len(df)}")
    df = pg.v3.compute_pead_gates_full(df)
    print(f"    pead_pass positives: {int(df['pead_pass'].sum())} "
          f"({df['pead_pass'].mean()*100:.2f}%)")

    # Compute entry-PnL + trade paths on ALL rows once (permaTicker keyed)
    print("\n[2] Computing entry-PnL + 12-snap trade paths on ALL rows ...")
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)
    # add a daily-bar repair util for the universe
    print(f"    coverage: {int(df['path_pnl_t11_pct'].notna().sum())}/{len(df)}")

    # Pre-cache trading calendar for fast portfolio simulation
    print("\n[3] Pre-caching trading calendar for portfolio sim speedup ...")
    with pd.HDFStore(DB, mode="r") as sstore:
        all_dates = set()
        for key in sstore.keys():
            if not key.startswith("/sp400/"):
                continue
            d = sstore[key]
            all_dates.update(pd.to_datetime(d["Date"]).tolist())
    calendar = sorted(all_dates)
    print(f"    calendar: {len(calendar)} trading days")

    # --- Per-fold nested CV eval ---
    fold_results = []
    print(f"\n[4] Running {len(folds)} folds ...")
    for fold_idx, (train_end, sweep_end, test_end) in enumerate(folds, 1):
        print("\n" + "=" * 60)
        print(f"  FOLD {fold_idx}/{len(folds)}: "
              f"TRAIN <= {train_end} | SWEEP {train_end}->{sweep_end} | "
              f"TEST {sweep_end}->{test_end}")
        print("=" * 60)

        # 4a. Split TRAIN, SWEEP_VAL, TEST by date
        # ALL data is df; TRAIN = report_date <= train_end (anchored),
        # SWEEP_VAL = train_end < report_date <= sweep_end,
        # TEST = sweep_end < report_date <= test_end
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
        print(f"  TRAIN    rows: {len(train_df)}  pead_pos: "
              f"{int(train_df['pead_pass'].sum())} "
              f"({train_df['pead_pass'].mean()*100:.2f}%)")
        print(f"  SWEEP_VALrows: {len(sweep_df)}  pead_pos: "
              f"{int(sweep_df['pead_pass'].sum())} "
              f"({sweep_df['pead_pass'].mean()*100:.2f}%)")
        print(f"  TEST     rows: {len(test_df)}   pead_pos: "
              f"{int(test_df['pead_pass'].sum())} "
              f"({test_df['pead_pass'].mean()*100:.2f}%)")

        # 4b. Sweep hyperparameters on TRAIN -> SWEEP_VAL
        print(f"\n  Sweeping {len(SWEEP_GRID)} configs on TRAIN->SWEEP_VAL ...")
        X_tr = train_df[SUNDAY_SAFE_FEATURES].copy()
        y_tr = train_df["pead_pass"].astype(int).values
        X_sv = sweep_df[SUNDAY_SAFE_FEATURES].copy()
        y_sv = sweep_df["pead_pass"].astype(int).values
        sweep_results = []
        for hp_idx, hp in enumerate(SWEEP_GRID):
            t0 = time.time()
            clf = fit_classifier(X_tr, y_tr, X_sv, y_sv, hp)
            # Evaluate SWEEP_VAL PnL at deployable op
            proba_sv = clf.predict_proba(X_sv)[:, 1]
            sweep_df2 = sweep_df.copy()
            sweep_df2["pead_proba"] = proba_sv
            mask = (
                (sweep_df2["pead_proba"] >= THETA_SCREEN) &
                (sweep_df2["opening_gap_t1"] >= GAP_LO) &
                (sweep_df2["opening_gap_t1"] <= GAP_HI) &
                (sweep_df2["path_pnl_t11_pct"].notna())
            )
            picks = sweep_df2[mask]
            if len(picks) >= 1:
                arith = np.expm1(picks["path_pnl_t11_pct"])
                sweep_pnl = float(arith.mean()) * 100
                sweep_hit = float((arith > 0).mean()) * 100
                sweep_n = len(picks)
            else:
                sweep_pnl = float("nan")
                sweep_hit = float("nan")
                sweep_n = 0
            # Also cohort VAL AUC + AP inline for completeness
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc_t = roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1])
            auc_s = roc_auc_score(y_sv, proba_sv)
            sweep_results.append({
                "hp_idx": hp_idx, "hp": hp, "sweep_val_n": sweep_n,
                "sweep_val_pnl_pct": sweep_pnl,
                "sweep_val_hit_pct": sweep_hit,
                "auc_train": auc_t, "auc_sweep_val": auc_s,
            })
            print(f"    gamma={hp['gamma']:>3d}  n={sweep_n:>3d}  "
                  f"pnl={sweep_pnl:>+6.3f}%  hit={sweep_hit:>5.1f}%  "
                  f"auc_sv={auc_s:.4f}  t={time.time()-t0:.1f}s")

        # 4c. Select best HP: max sweep_val_pnl_pct subject to n>=20
        sweep_df_out = pd.DataFrame([
            {**{k: v for k, v in r.items() if k != 'hp'}, **r['hp']}
            for r in sweep_results
        ])
        # If no config satisfies n>=20, fall back to gamma=5 (v1 baseline)
        valid = [r for r in sweep_results
                 if r["sweep_val_n"] >= 20 and not np.isnan(r["sweep_val_pnl_pct"])]
        if valid:
            best = max(valid, key=lambda r: r["sweep_val_pnl_pct"])
            sel_hp = best["hp"]
            sel_reason = f"best sweep_val_pnl (n={best['sweep_val_n']}, pnl={best['sweep_val_pnl_pct']:+.3f}%)"
        else:
            sel_hp = {"gamma": 5, "min_child_weight": 50,
                      "max_depth": 3, "n_estimators": 300}
            sel_reason = "fallback to gamma=5 (no config reached n>=20)"
        print(f"\n  Selected HP: gamma={sel_hp['gamma']}, mcw={sel_hp['min_child_weight']}, "
              f"md={sel_hp['max_depth']}, n_est={sel_hp['n_estimators']}")
        print(f"  Reason: {sel_reason}")

        # 4d. Retrain on TRAIN+SWEEP_VAL combined with selected HP, eval on TEST
        X_ts = pd.concat([X_tr, X_sv], axis=0).reset_index(drop=True)
        y_ts = np.concatenate([y_tr, y_sv])
        print(f"\n  Retraining on TRAIN+SWEEP_VAL ({len(y_ts)} rows) ...")
        X_te = test_df[SUNDAY_SAFE_FEATURES].copy()
        y_te = test_df["pead_pass"].astype(int).values
        clf_final = fit_classifier(X_ts, y_ts, X_te, y_te, sel_hp)
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_test = roc_auc_score(y_te, clf_final.predict_proba(X_te)[:, 1])
        ap_test = average_precision_score(y_te, clf_final.predict_proba(X_te)[:, 1])
        print(f"  TEST AUC: {auc_test:.4f}  AP: {ap_test:.4f}")

        # 4e. Run v1.1 portfolio sim on TEST
        print(f"\n  Running v1.1 portfolio sim on TEST (n_slots={args.n_slots}) ...")
        trades = select_v11_trades(test_df, clf_final)
        print(f"  v1.1 trades selected: {len(trades)}")
        result = rb._simulate_with_cached_calendar(
            trades, args.n_slots, args.initial_nav, calendar)
        s_v11 = result.get("summary", {})
        if not s_v11:
            print("  [!] v1.1 strategy produced no trades on this fold.")
            continue

        print(f"  v1.1 on TEST fold {fold_idx}:")
        print(f"    Trades:    {s_v11['n_trades_executed']}")
        print(f"    IRR:       {s_v11['irr_pct']:+.2f}%")
        print(f"    Sharpe:    {s_v11['sharpe_liq_annualized']:+.2f}")
        print(f"    MaxDD:     {s_v11['max_drawdown_pct']:.2f}%")
        print(f"    Hit%:      {s_v11['hit_rate_pct']:.1f}%")
        print(f"    AvgPnL%:   {s_v11['avg_trade_pnl_pct']:+.3f}%")

        # 4f. Random baseline N_RNG_TRIALS on TEST
        print(f"\n  Running {args.n_rng_trials} random trials on TEST ...")
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
                "hit_pct": s_r.get("hit_rate_pct", float("nan")),
                "avg_pnl_pct": s_r.get("avg_trade_pnl_pct", float("nan")),
            })
        rand_df = pd.DataFrame(rand_rows)
        if len(rand_df) >= 1:
            frac_above_sharpe = (
                rand_df["sharpe"] > s_v11["sharpe_liq_annualized"]
            ).mean()
            frac_above_irr = (
                (rand_df["irr"] > s_v11["irr_pct"])
                .mean() if not rand_df["irr"].isna().all() else float("nan")
            )
        else:
            frac_above_sharpe = float("nan")
            frac_above_irr = float("nan")
        print(f"  Random baseline mean (test fold {fold_idx}): "
              f"IRR={rand_df['irr'].mean():+.2f}%  "
              f"Sharpe={rand_df['sharpe'].mean():+.2f}  "
              f"MaxDD={rand_df['max_dd'].mean():+.2f}%")
        print(f"  v1.1 TEST Sharpe exceeds {frac_above_sharpe*100:.1f}% of "
              f"random trials; IRR exceeds {frac_above_irr*100:.1f}%.")

        fold_results.append({
            "fold": fold_idx,
            "train_end": train_end, "sweep_end": sweep_end,
            "test_end": test_end,
            "selected_hp": sel_hp,
            "selected_hp_reason": sel_reason,
            "test_n_trades": int(s_v11["n_trades_executed"]),
            "test_irr_pct": s_v11["irr_pct"],
            "test_sharpe": s_v11["sharpe_liq_annualized"],
            "test_max_dd_pct": s_v11["max_drawdown_pct"],
            "test_hit_pct": s_v11["hit_rate_pct"],
            "test_avg_pnl_pct": s_v11["avg_trade_pnl_pct"],
            "test_auc": float(auc_test),
            "test_ap": float(ap_test),
            "random_mean_irr": float(rand_df["irr"].mean()) if len(rand_df) else float("nan"),
            "random_mean_sharpe": float(rand_df["sharpe"].mean()) if len(rand_df) else float("nan"),
            "random_mean_max_dd": float(rand_df["max_dd"].mean()) if len(rand_df) else float("nan"),
            "frac_random_exceeded_sharpe": float(frac_above_sharpe),
            "frac_random_exceeded_irr": float(frac_above_irr),
            "n_random_trials": int(len(rand_df)),
        })

    # --- Aggregate across folds ---
    print("\n" + "=" * 78)
    print("NESTED CV AGGREGATE RESULTS")
    print("=" * 78)
    # Flatten selected_hp into individual columns for printing
    for r in fold_results:
        r["sel_gamma"] = r["selected_hp"]["gamma"]
        r["sel_mcw"] = r["selected_hp"]["min_child_weight"]
        r["sel_md"] = r["selected_hp"]["max_depth"]
        r["sel_n_est"] = r["selected_hp"]["n_estimators"]

    print(f"\n{'Fold':>4s}  {'TEST slice':<25s}  "
          f"{'sel_gamma':>9s}  {'n_t':>4s}  "
          f"{'IRR%':>7s}  {'Sharpe':>7s}  {'MaxDD%':>7s}  "
          f"{'hit%':>5s}  {'avgPnL%':>8s}  {'fracShEx':>8s}  {'fracIREx':>8s}")
    print("-" * 110)
    for r in fold_results:
        test_range = r['sweep_end'][:7] + '->' + r['test_end'][:7]
        print(f"  {r['fold']:>2d}  {test_range:>22s}  "
              f"gamma={r['sel_gamma']:>2d}    {r['test_n_trades']:>3d}  "
              f"{r['test_irr_pct']:>+6.2f}%  {r['test_sharpe']:>+6.2f}  "
              f"{r['test_max_dd_pct']:>+6.2f}%  {r['test_hit_pct']:>4.1f}%  "
              f"{r['test_avg_pnl_pct']:>+7.3f}%  "
              f"{r['frac_random_exceeded_sharpe']*100:>5.1f}%  "
              f"{r['frac_random_exceeded_irr']*100:>5.1f}%")

    # Average across folds
    print(f"\n{'AVG':>4s}  {'':>22s}  {'':>8s}  "
          f"{int(np.mean([r['test_n_trades'] for r in fold_results])):>3d}  "
          f"{np.mean([r['test_irr_pct'] for r in fold_results]):>+6.2f}%  "
          f"{np.mean([r['test_sharpe'] for r in fold_results]):>+6.2f}  "
          f"{np.mean([r['test_max_dd_pct'] for r in fold_results]):>+6.2f}%  "
          f"{np.mean([r['test_hit_pct'] for r in fold_results]):>4.1f}%  "
          f"{np.mean([r['test_avg_pnl_pct'] for r in fold_results]):>+7.3f}%  "
          f"{np.mean([r['frac_random_exceeded_sharpe'] for r in fold_results])*100:>5.1f}%  "
          f"{np.mean([r['frac_random_exceeded_irr'] for r in fold_results])*100:>5.1f}%")

    # Persist
    out_dir = HERE / f"phase_g_v1_1_nested_cv_n{args.n_folds}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "selected_hp"}
                  for r in fold_results]).to_csv(out_dir / "fold_results.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_folds": len(fold_results),
            "n_slots": args.n_slots,
            "n_rng_trials": args.n_rng_trials,
            "operating_point": {
                "theta_screen": THETA_SCREEN,
                "gap_lo": GAP_LO,
                "gap_hi": GAP_HI,
            },
            "sweep_grid": SWEEP_GRID,
            "folds": folds,
            "fold_results": fold_results,
            "aggregate": {
                "mean_test_irr_pct": float(np.mean([r["test_irr_pct"] for r in fold_results])),
                "mean_test_sharpe": float(np.mean([r["test_sharpe"] for r in fold_results])),
                "mean_test_max_dd_pct": float(np.mean([r["test_max_dd_pct"] for r in fold_results])),
                "mean_test_hit_pct": float(np.mean([r["test_hit_pct"] for r in fold_results])),
                "mean_test_avg_pnl_pct": float(np.mean([r["test_avg_pnl_pct"] for r in fold_results])),
                "mean_frac_random_exceeded_sharpe": float(np.mean([r["frac_random_exceeded_sharpe"] for r in fold_results])),
                "mean_frac_random_exceeded_irr": float(np.mean([r["frac_random_exceeded_irr"] for r in fold_results])),
            },
            "created_at": pd.Timestamp.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"\nSaved artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

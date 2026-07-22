"""
Phase G v1.1 -- NEG_only theta sweep (POS-tuned HP, vary theta only).

Per `phase_g_neg_tuned_findings.md` §6 item (2). The NEG-tuned
retrain showed that re-sweeping HP for the NEG target over-tunes
on tiny SWEEP_VAL samples. We now keep the Appendix D POS-tuned HP
FIXED per fold, and sweep ONLY the NEG_only operating-point
threshold theta in {0.10, 0.15, 0.20, 0.25}.

For each fold we:
  1. Take the Appendix D POS-tuned HP for that fold (gamma=10/5/3/3
     for folds 1-4 respectively, all with mcw=50, md=3, n_est=300).
  2. Retrain final classifier on TRAIN+SWEEP_VAL with the
     POS-tuned HP.
  3. For each candidate theta in {0.10, 0.15, 0.20, 0.25}:
     - Pick trades with P(PEAD) >= theta AND gap in [-15%, -2%]
       AND valid path_pnl_t11_pct.
     - Run n_slots=4 portfolio sim -> IRR / Sharpe / MaxDD etc.
  4. Per-fold random baseline: 100 trials (seed = trial*7+100,
     same as Appendix D/E). The random baseline distribution is
     INDEPENDENT of theta, so we compute it once per fold and
     reuse it for all 4 theta evaluations.

  5. Per-fold per-theta: compute fraction of 100 random trials
     exceeded on Sharpe + IRR.

Cross-fold ANNUAL aggregate: best theta = max mean Sharpe across
the 4 folds (with mean n_trades >= some floor so we don't pick
an over-thin slice that's just 2 outliers).

CLI:
  python luan_bot_trading/04_backtest/09_phase_g_neg_theta_sweep.py
  python luan_bot_trading/04_backtest/09_phase_g_neg_theta_sweep.py --n-folds 4 --n-slots 4

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

# POS-tuned HP from Appendix D's fold_results.csv
APPD_RESULTS_CSV = HERE / "archive" / "experiments" / "phase_g_v1_1_nested_cv_n4" / "fold_results.csv"

# NEG_only gap range (fixed at §4.4 anomaly range; gap range SEPARATE sweep)
NEG_GAP_LO = -0.15
NEG_GAP_HI = -0.02

THETAS = [0.10, 0.15, 0.20, 0.25]

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
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def select_neg_picks(df, proba, theta):
    df2 = df.copy()
    df2["pead_proba"] = proba
    mask = (
        (df2["pead_proba"] >= theta) &
        (df2["opening_gap_t1"] >= NEG_GAP_LO) &
        (df2["opening_gap_t1"] <= NEG_GAP_HI) &
        (df2["path_pnl_t11_pct"].notna())
    )
    return df2[mask].copy().reset_index(drop=True)


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
    parser.add_argument("--min-n-trades", type=int, default=5,
                        help="Floor on mean n_trades for selecting best theta")
    args = parser.parse_args(argv)

    folds = DEFAULT_FOLDS[:args.n_folds]

    print("=" * 78)
    print("PHASE G v1.1 -- NEG_only THETA SWEEP (POS-tuned HP fixed)")
    print("=" * 78)
    print(f"Folds      : {len(folds)}")
    for i, (te, sve, tse) in enumerate(folds, 1):
        print(f"  Fold {i}: TEST {sve}->{tse}")
    print(f"N_SLOTS    : {args.n_slots}")
    print(f"N_RNG      : {args.n_rng_trials}")
    print(f"NEG gap    : [{NEG_GAP_LO:+.2f}, {NEG_GAP_HI:+.2f}] (fixed)")
    print(f"Thetas     : {THETAS}")
    print(f"HP source  : App D fold_results.csv POS-tuned per-fold HP "
          f"(gamma=10/5/3/3 for folds 1-4)")
    print(f"Floor for theta selection: mean n_trades >= {args.min_n_trades}")
    print("=" * 78)

    print("\n[1] Loading Appendix D fold_results.csv for POS-tuned HP ...")
    appd = pd.read_csv(APPD_RESULTS_CSV)
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

    print("\n[3] Computing entry-PnL + 12-snap trade paths ...")
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)
    print(f"    coverage: {int(df['path_pnl_t11_pct'].notna().sum())}/{len(df)}")

    print("\n[4] Pre-caching trading calendar ...")
    with pd.HDFStore(DB, mode="r") as sstore:
        all_dates = set()
        for key in sstore.keys():
            if not key.startswith("/sp400/"):
                continue
            d = sstore[key]
            all_dates.update(pd.to_datetime(d["Date"]).tolist())
    calendar = sorted(all_dates)
    print(f"    calendar: {len(calendar)} trading days")

    # --- Per-fold theta sweep ---
    fold_results = []
    per_fold_random = {}  # fold_idx -> list of 100 random Sharpe values
    print(f"\n[5] Running {len(folds)} folds ...")
    for fold_idx, (train_end, sweep_end, test_end) in enumerate(folds, 1):
        print("\n" + "=" * 60)
        print(f"  FOLD {fold_idx}/{len(folds)}: TEST {sweep_end}->{test_end}")
        print("=" * 60)

        train_ts = pd.Timestamp(train_end)
        sweep_ts = pd.Timestamp(sweep_end)
        test_ts = pd.Timestamp(test_end)
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= train_ts].copy().reset_index(drop=True)
        sweep_df = df[(rd > train_ts) & (rd <= sweep_ts)].copy().reset_index(drop=True)
        test_df  = df[(rd > sweep_ts) & (rd <= test_ts)].copy().reset_index(drop=True)
        print(f"  TRAIN={len(train_df)}  SWEEP={len(sweep_df)}  "
              f"TEST={len(test_df)}")

        X_tr = train_df[SUNDAY_SAFE_FEATURES].copy()
        y_tr = train_df["pead_pass"].astype(int).values
        X_sv = sweep_df[SUNDAY_SAFE_FEATURES].copy()
        y_sv = sweep_df["pead_pass"].astype(int).values
        X_ts = pd.concat([X_tr, X_sv], axis=0).reset_index(drop=True)
        y_ts = np.concatenate([y_tr, y_sv])
        X_te = test_df[SUNDAY_SAFE_FEATURES].copy()
        y_te = test_df["pead_pass"].astype(int).values

        hp = fold_hp[fold_idx - 1]
        print(f"  Training classifier (gamma={hp['gamma']}, POS-tuned) on "
              f"{len(y_ts)} TRAIN+SWEEP rows ...")
        clf = fit_classifier(X_ts, y_ts, X_te, y_te, hp)
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_test = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        ap_test  = average_precision_score(
            y_te, clf.predict_proba(X_te)[:, 1])
        print(f"  TEST AUC: {auc_test:.4f}  AP: {ap_test:.4f}")

        proba_te = clf.predict_proba(X_te)[:, 1]

        # --- Random baseline ONCE per fold ---
        print(f"\n  Random baseline ({args.n_rng_trials} trials) ...")
        rand_sharpes = []
        rand_irrs = []
        for trial in range(args.n_rng_trials):
            seed = trial * 7 + 100
            rt = select_random_trades(test_df, seed)
            result_r = rb._simulate_with_cached_calendar(
                rt, args.n_slots, args.initial_nav, calendar)
            s_r = result_r.get("summary", {})
            if not s_r:
                continue
            rand_sharpes.append(s_r.get("sharpe_liq_annualized", float("nan")))
            rand_irrs.append(s_r.get("irr_pct", float("nan")))
        rand_sharpes = np.array(rand_sharpes)
        rand_irrs = np.array(rand_irrs)
        per_fold_random[fold_idx] = {"sharpes": rand_sharpes.tolist(),
                                     "irrs": rand_irrs.tolist()}
        print(f"  rand_n={len(rand_sharpes)}  "
              f"rand Sharpe mean={np.nanmean(rand_sharpes):+.2f}  "
              f"rand IRR mean={np.nanmean(rand_irrs):+.2f}%")

        # --- For each theta, evaluate ---
        print(f"\n  Theta sweep on TEST slice ...")
        per_theta = {}
        for theta in THETAS:
            picks = select_neg_picks(test_df, proba_te, theta)
            n_picks = len(picks)
            if n_picks >= 1:
                arith = np.expm1(picks["path_pnl_t11_pct"])
                mean_arith = float(arith.mean()) * 100
                hit_pct = float((arith > 0).mean()) * 100
            else:
                mean_arith = float("nan")
                hit_pct = float("nan")
            result = rb._simulate_with_cached_calendar(
                picks, args.n_slots, args.initial_nav, calendar)
            s = result.get("summary", {})
            sharpe = s.get("sharpe_liq_annualized", float("nan"))
            irr = s.get("irr_pct", float("nan"))
            # Fraction of random trials exceeded:
            if len(rand_sharpes) and not np.isnan(sharpe):
                frac_ex_sh = 1.0 - float(
                    np.nanmean(rand_sharpes > sharpe))
            else:
                frac_ex_sh = float("nan")
            if len(rand_irrs) and not np.isnan(irr) and not np.all(np.isnan(rand_irrs)):
                frac_ex_ir = 1.0 - float(
                    np.nanmean(rand_irrs > irr))
            else:
                frac_ex_ir = float("nan")
            per_theta[theta] = {
                "n_picks": n_picks,
                "n_trades_executed": s.get("n_trades_executed", 0),
                "irr_pct": irr,
                "sharpe": sharpe,
                "max_drawdown_pct": s.get("max_drawdown_pct", float("nan")),
                "hit_rate_pct": s.get("hit_rate_pct", float("nan")),
                "avg_trade_pnl_pct": s.get("avg_trade_pnl_pct", float("nan")),
                "mean_arith_pnl_pct": mean_arith,
                "hit_pct_per_pick": hit_pct,
                "frac_random_exceeded_sharpe": frac_ex_sh,
                "frac_random_exceeded_irr": frac_ex_ir,
                "auc_test": float(auc_test),
                "ap_test": float(ap_test),
                "selected_gamma": int(hp["gamma"]),
            }
            print(f"    theta={theta:.2f}  n_picks={n_picks:>3d}  "
                  f"n_exec={s.get('n_trades_executed',0):>3d}  "
                  f"IRR={irr:>+7.2f}%  Sharpe={sharpe:>+5.2f}  "
                  f"MaxDD={s.get('max_drawdown_pct',float('nan')):>+6.2f}%  "
                  f"hit={s.get('hit_rate_pct',float('nan')):>4.1f}%  "
                  f"avgPnL={s.get('avg_trade_pnl_pct',float('nan')):>+6.3f}%  "
                  f"%rShEx={frac_ex_sh*100:>4.0f}%  "
                  f"%rIREx={frac_ex_ir*100:>4.0f}%")

        fold_results.append({
            "fold": fold_idx,
            "train_end": train_end, "sweep_end": sweep_end,
            "test_end": test_end,
            "test_slice": f"{sweep_end}->{test_end}",
            "pos_tuned_hp": hp,
            "test_n_rows": len(test_df),
            "test_n_pos": int(test_df["pead_pass"].sum()),
            "auc_test": float(auc_test),
            "ap_test": float(ap_test),
            "random_mean_sharpe": float(np.nanmean(rand_sharpes)),
            "random_mean_irr": float(np.nanmean(rand_irrs)),
            "random_n": int(len(rand_sharpes)),
            "per_theta": {str(k): v for k, v in per_theta.items()},
        })

    # --- Aggregate across folds: per-theta cross-fold means ---
    print("\n" + "=" * 78)
    print("NEG_only THETA SWEEP -- CROSS-FOLD AGGREGATES")
    print("=" * 78)

    print(f"\n{'theta':>6s}  {'mean IRR%':>9s}  {'mean Sharpe':>11s}  "
          f"{'mean MaxDD%':>11s}  {'mean hit%':>9s}  "
          f"{'mean avgPnL%':>12s}  {'mean %rShEx':>11s}  "
          f"{'mean %rIREx':>11s}  {'mean n_tr':>9s}  {'mean n_pick':>11s}")
    print("-" * 120)

    aggregate_per_theta = {}
    for theta in THETAS:
        irr_vals = [fr["per_theta"][str(theta)]["irr_pct"]
                    for fr in fold_results
                    if str(theta) in fr["per_theta"]]
        sh_vals  = [fr["per_theta"][str(theta)]["sharpe"]
                    for fr in fold_results
                    if str(theta) in fr["per_theta"]]
        dd_vals  = [fr["per_theta"][str(theta)]["max_drawdown_pct"]
                    for fr in fold_results
                    if str(theta) in fr["per_theta"]]
        hit_vals = [fr["per_theta"][str(theta)]["hit_rate_pct"]
                    for fr in fold_results
                    if str(theta) in fr["per_theta"]]
        avg_vals = [fr["per_theta"][str(theta)]["avg_trade_pnl_pct"]
                    for fr in fold_results
                    if str(theta) in fr["per_theta"]]
        fr_ex_sh = [fr["per_theta"][str(theta)]["frac_random_exceeded_sharpe"]
                    for fr in fold_results
                    if str(theta) in fr["per_theta"]]
        fr_ex_ir = [fr["per_theta"][str(theta)]["frac_random_exceeded_irr"]
                    for fr in fold_results
                    if str(theta) in fr["per_theta"]]
        n_tr  = [fr["per_theta"][str(theta)]["n_trades_executed"]
                 for fr in fold_results
                 if str(theta) in fr["per_theta"]]
        n_pick = [fr["per_theta"][str(theta)]["n_picks"]
                  for fr in fold_results
                  if str(theta) in fr["per_theta"]]
        agg = {
            "mean_irr_pct": float(np.nanmean(irr_vals)),
            "mean_sharpe": float(np.nanmean(sh_vals)),
            "mean_maxdd_pct": float(np.nanmean(dd_vals)),
            "mean_hit_pct": float(np.nanmean(hit_vals)),
            "mean_avg_pnl_pct": float(np.nanmean(avg_vals)),
            "mean_frac_rand_exceed_sh": float(np.nanmean(fr_ex_sh)),
            "mean_frac_rand_exceed_ir": float(np.nanmean(fr_ex_ir)),
            "mean_n_trades": float(np.nanmean(n_tr)),
            "mean_n_picks": float(np.nanmean(n_pick)),
        }
        aggregate_per_theta[theta] = agg
        print(f"{theta:>6.2f}  {agg['mean_irr_pct']:>+8.2f}%  "
              f"{agg['mean_sharpe']:>+10.2f}  "
              f"{agg['mean_maxdd_pct']:>+10.2f}%  "
              f"{agg['mean_hit_pct']:>8.1f}%  "
              f"{agg['mean_avg_pnl_pct']:>+11.3f}%  "
              f"{agg['mean_frac_rand_exceed_sh']*100:>10.1f}%  "
              f"{agg['mean_frac_rand_exceed_ir']*100:>10.1f}%  "
              f"{agg['mean_n_trades']:>8.1f}  {agg['mean_n_picks']:>10.1f}")

    # Best theta = max mean Sharpe subject to mean_n_trades >= floor
    valid_thetas = [t for t in THETAS
                    if aggregate_per_theta[t]["mean_n_trades"] >= args.min_n_trades]
    if valid_thetas:
        best_theta = max(valid_thetas, key=lambda t:
                         aggregate_per_theta[t]["mean_sharpe"])
        print(f"\n  Best theta (mean Sharpe, floor n_tr>={args.min_n_trades}): "
              f"{best_theta:.2f}")
        print(f"    -> mean Sharpe = {aggregate_per_theta[best_theta]['mean_sharpe']:+.2f}")
        print(f"    -> mean IRR = {aggregate_per_theta[best_theta]['mean_irr_pct']:+.2f}%")
        print(f"    -> mean MaxDD = {aggregate_per_theta[best_theta]['mean_maxdd_pct']:+.2f}%")
        print(f"    -> mean % rShEx = "
              f"{aggregate_per_theta[best_theta]['mean_frac_rand_exceed_sh']*100:.1f}%")
    else:
        best_theta = None
        print(f"\n  No theta crossed floor n_trades={args.min_n_trades}. "
              f"Review raw numbers.")

    # --- Persist ---
    out_dir = HERE / f"phase_g_v1_1_neg_theta_sweep_n{args.n_folds}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {**{"fold": fr["fold"], "test_slice": fr["test_slice"]},
         **fr["per_theta"][str(theta)],
         "theta": theta}
        for fr in fold_results
        for theta in THETAS
    ]).to_csv(out_dir / "neg_theta_sweep_wide.csv", index=False)
    summary = {
        "n_folds": len(fold_results),
        "n_slots": args.n_slots,
        "n_rng_trials": args.n_rng_trials,
        "neg_gap_lo": NEG_GAP_LO,
        "neg_gap_hi": NEG_GAP_HI,
        "thetas": THETAS,
        "pos_tuned_hp_source": "Appendix D fold_results.csv",
        "fold_pos_tuned_hps": fold_hp,
        "folds": folds,
        "fold_results": fold_results,
        "per_fold_random_baseline": per_fold_random,
        "aggregate_per_theta": {str(k): v for k, v in aggregate_per_theta.items()},
        "best_theta": best_theta,
        "selection_floor_min_n_trades": args.min_n_trades,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

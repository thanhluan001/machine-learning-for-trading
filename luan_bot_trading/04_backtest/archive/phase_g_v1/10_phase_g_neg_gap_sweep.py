"""
Phase G v1.1 -- NEG_only gap range sweep at theta=0.20.

Per `phase_g_neg_theta_sweep_findings.md` §F.6 item (3). At the new
theta=0.20 operating point, sweep the NEG_gap range alternative:
  (a) [-15%, -2%]  -- App F recommended baseline
  (b) [-20%, -2%]  -- extended downside (large drops)
  (c) [-12%, -2%]  -- mild downside only (small capacity)
  (d) [-15%, -3%]  -- skipping tiny gaps (tick-noise threshold)
  (e) [-10%, -2%]  -- moderate 10% cap
  (f) [-10%, -3%]  -- moderate cap + skip tiny

Same 4 nested CV scaffolding as App D / E / F:
  - 4 anchored walk-forward folds
  - Per-fold POS-tuned HP fixed from App D's fold_results.csv
  - Random baseline computed ONCE per fold, reused across all gap configs

CLI:
  python luan_bot_trading/04_backtest/10_phase_g_neg_gap_sweep.py

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

APPD_RESULTS_CSV = HERE / "archive" / "experiments" / "phase_g_v1_1_nested_cv_n4" / "fold_results.csv"

# Theta fixed from App F (Doc F §F.5)
NEG_THETA = 0.20

# Gap range candidates -- 6 alternatives
GAPS = [
    ("[-15,-2]",  -0.15, -0.02),  # App F baseline
    ("[-20,-2]",  -0.20, -0.02),  # extended downside
    ("[-12,-2]",  -0.12, -0.02),  # mild downside only
    ("[-15,-3]",  -0.15, -0.03),  # skip tiny gaps (just < -3%)
    ("[-10,-2]",  -0.10, -0.02),  # moderate 10% cap
    ("[-10,-3]",  -0.10, -0.03),  # moderate cap + skip < -3%
]

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
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


def select_neg_picks(df, proba, gap_lo, gap_hi):
    df2 = df.copy()
    df2["pead_proba"] = proba
    mask = (
        (df2["pead_proba"] >= NEG_THETA) &
        (df2["opening_gap_t1"] >= gap_lo) &
        (df2["opening_gap_t1"] <= gap_hi) &
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
                        help="Floor on mean n_trades for selecting best gap")
    args = parser.parse_args(argv)

    folds = DEFAULT_FOLDS[:args.n_folds]

    print("=" * 78)
    print("PHASE G v1.1 -- NEG_only GAP RANGE SWEEP (theta=0.20 fixed)")
    print("=" * 78)
    print(f"Folds      : {len(folds)}")
    for i, (te, sve, tse) in enumerate(folds, 1):
        print(f"  Fold {i}: TEST {sve}->{tse}")
    print(f"N_SLOTS    : {args.n_slots}")
    print(f"N_RNG      : {args.n_rng_trials}")
    print(f"NEG theta  : {NEG_THETA:.2f}")
    print(f"Gap ranges : {[c[0] for c in GAPS]}")
    print(f"HP source  : App D fold_results.csv POS-tuned per-fold HP "
          f"(gamma=10/5/3/3 for folds 1-4)")
    print(f"Floor for gap selection: mean n_trades >= {args.min_n_trades}")
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

    # --- Shared data load ---
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

    # --- Per-fold gap sweep ---
    fold_results = []
    print(f"\n[5] Running {len(folds)} folds x {len(GAPS)} gap configs ...")
    for fold_idx, (train_end, sweep_end, test_end) in enumerate(folds, 1):
        print("\n" + "=" * 60)
        print(f"  FOLD {fold_idx}/{len(folds)}: TEST {sweep_end}->{test_end}")
        print("=" * 60)

        train_ts = pd.Timestamp(train_end)
        sweep_ts = pd.Timestamp(sweep_end)
        test_ts  = pd.Timestamp(test_end)
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
        print(f"\n  Random baseline ({args.n_rng_trials} trials) -- "
              f"random NOT gap-dependent")
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
        print(f"  rand_n={len(rand_sharpes)}  "
              f"rand Sharpe mean={np.nanmean(rand_sharpes):+.2f}  "
              f"rand IRR mean={np.nanmean(rand_irrs):+.2f}%")

        # --- For each gap range ---
        print(f"\n  Gap config sweep on TEST slice ...")
        per_gap = {}
        for gap_name, gap_lo, gap_hi in GAPS:
            picks = select_neg_picks(test_df, proba_te, gap_lo, gap_hi)
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
                frac_ex_sh = 1.0 - float(np.nanmean(rand_sharpes > sharpe))
            else:
                frac_ex_sh = float("nan")
            if len(rand_irrs) and not np.isnan(irr) and not np.all(np.isnan(rand_irrs)):
                frac_ex_ir = 1.0 - float(np.nanmean(rand_irrs > irr))
            else:
                frac_ex_ir = float("nan")
            per_gap[gap_name] = {
                "gap_lo": gap_lo, "gap_hi": gap_hi,
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
            print(f"    {gap_name:>10s}  n_picks={n_picks:>3d}  "
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
            "per_gap": per_gap,
        })

    # --- Aggregate across folds: per-gap cross-fold means ---
    print("\n" + "=" * 78)
    print("NEG_only GAP SWEEP -- CROSS-FOLD AGGREGATES")
    print("=" * 78)

    print(f"\n{'gap range':>12s}  {'mean IRR%':>9s}  {'mean Sharpe':>11s}  "
          f"{'mean MaxDD%':>11s}  {'mean hit%':>9s}  "
          f"{'mean avgPnL%':>12s}  {'mean %rShEx':>11s}  "
          f"{'mean %rIREx':>11s}  {'mean n_tr':>9s}  {'mean n_pick':>11s}")
    print("-" * 130)

    aggregate_per_gap = {}
    for gap_name, gap_lo, gap_hi in GAPS:
        irr_vals = [fr["per_gap"][gap_name]["irr_pct"]
                    for fr in fold_results if gap_name in fr["per_gap"]]
        sh_vals  = [fr["per_gap"][gap_name]["sharpe"]
                    for fr in fold_results if gap_name in fr["per_gap"]]
        dd_vals  = [fr["per_gap"][gap_name]["max_drawdown_pct"]
                    for fr in fold_results if gap_name in fr["per_gap"]]
        hit_vals = [fr["per_gap"][gap_name]["hit_rate_pct"]
                    for fr in fold_results if gap_name in fr["per_gap"]]
        avg_vals = [fr["per_gap"][gap_name]["avg_trade_pnl_pct"]
                    for fr in fold_results if gap_name in fr["per_gap"]]
        fr_ex_sh = [fr["per_gap"][gap_name]["frac_random_exceeded_sharpe"]
                    for fr in fold_results if gap_name in fr["per_gap"]]
        fr_ex_ir = [fr["per_gap"][gap_name]["frac_random_exceeded_irr"]
                    for fr in fold_results if gap_name in fr["per_gap"]]
        n_tr  = [fr["per_gap"][gap_name]["n_trades_executed"]
                 for fr in fold_results if gap_name in fr["per_gap"]]
        n_pick = [fr["per_gap"][gap_name]["n_picks"]
                  for fr in fold_results if gap_name in fr["per_gap"]]
        agg = {
            "gap_lo": gap_lo, "gap_hi": gap_hi,
            "mean_irr_pct": float(np.nanmean(irr_vals)),
            "mean_sharpe": float(np.nanmean(sh_vals)),
            "std_sharpe": float(np.nanstd(sh_vals, ddof=1))
                if sum(not np.isnan(v) for v in sh_vals) >= 2 else float("nan"),
            "ci95_sharpe_lo": float(np.nanmean(sh_vals) - 2*np.nanstd(sh_vals, ddof=1)/np.sqrt(sum(not np.isnan(v) for v in sh_vals)))
                if sum(not np.isnan(v) for v in sh_vals) >= 2 else float("nan"),
            "ci95_sharpe_hi": float(np.nanmean(sh_vals) + 2*np.nanstd(sh_vals, ddof=1)/np.sqrt(sum(not np.isnan(v) for v in sh_vals)))
                if sum(not np.isnan(v) for v in sh_vals) >= 2 else float("nan"),
            "mean_maxdd_pct": float(np.nanmean(dd_vals)),
            "mean_hit_pct": float(np.nanmean(hit_vals)),
            "mean_avg_pnl_pct": float(np.nanmean(avg_vals)),
            "mean_frac_rand_exceed_sh": float(np.nanmean(fr_ex_sh)),
            "mean_frac_rand_exceed_ir": float(np.nanmean(fr_ex_ir)),
            "mean_n_trades": float(np.nanmean(n_tr)),
            "mean_n_picks": float(np.nanmean(n_pick)),
            "per_fold_sharpes": sh_vals,
        }
        aggregate_per_gap[gap_name] = agg
        print(f"{gap_name:>12s}  {agg['mean_irr_pct']:>+8.2f}%  "
              f"{agg['mean_sharpe']:>+10.2f}  "
              f"{agg['mean_maxdd_pct']:>+10.2f}%  "
              f"{agg['mean_hit_pct']:>8.1f}%  "
              f"{agg['mean_avg_pnl_pct']:>+11.3f}%  "
              f"{agg['mean_frac_rand_exceed_sh']*100:>10.1f}%  "
              f"{agg['mean_frac_rand_exceed_ir']*100:>10.1f}%  "
              f"{agg['mean_n_trades']:>8.1f}  {agg['mean_n_picks']:>10.1f}")

    # Best gap = max mean Sharpe subject to mean_n_trades >= floor
    valid_gaps = [g for g, a in aggregate_per_gap.items()
                  if a["mean_n_trades"] >= args.min_n_trades]
    if valid_gaps:
        best_gap = max(valid_gaps, key=lambda g:
                       aggregate_per_gap[g]["mean_sharpe"])
        a = aggregate_per_gap[best_gap]
        print(f"\n  Best gap (mean Sharpe, floor n_tr>={args.min_n_trades}): "
              f"{best_gap} (gap range [{a['gap_lo']:+.2f}, {a['gap_hi']:+.2f}])")
        print(f"    -> mean Sharpe = {a['mean_sharpe']:+.2f}")
        print(f"    -> std Sharpe  = {a['std_sharpe']:.2f}")
        print(f"    -> 95% CI on mean Sharpe = "
              f"[{a['ci95_sharpe_lo']:+.2f}, {a['ci95_sharpe_hi']:+.2f}]")
        print(f"    -> mean IRR = {a['mean_irr_pct']:+.2f}%")
        print(f"    -> mean MaxDD = {a['mean_maxdd_pct']:+.2f}%")
        print(f"    -> mean % rShEx = "
              f"{a['mean_frac_rand_exceed_sh']*100:.1f}%")
    else:
        best_gap = None
        print(f"\n  No gap config crossed floor n_trades={args.min_n_trades}.")

    # --- Persist ---
    out_dir = HERE / f"phase_g_v1_1_neg_gap_sweep_n{args.n_folds}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {**{"fold": fr["fold"], "test_slice": fr["test_slice"]},
         **fr["per_gap"][gap_name], "gap_name": gap_name}
        for fr in fold_results
        for gap_name, _, _ in GAPS
    ]).to_csv(out_dir / "neg_gap_sweep_wide.csv", index=False)
    summary = {
        "n_folds": len(fold_results),
        "n_slots": args.n_slots,
        "n_rng_trials": args.n_rng_trials,
        "neg_theta": NEG_THETA,
        "gaps": [{"name": n, "lo": l, "hi": h} for n, l, h in GAPS],
        "pos_tuned_hp_source": "Appendix D fold_results.csv",
        "fold_pos_tuned_hps": fold_hp,
        "folds": folds,
        "fold_results": fold_results,
        "aggregate_per_gap": aggregate_per_gap,
        "best_gap": best_gap,
        "selection_floor_min_n_trades": args.min_n_trades,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

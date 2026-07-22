"""
Phase G v1.1 -- Forward-shifted out-of-sample validation.

Per phase_g_findings.md §B.5 item 3:
  Re-train the v1.1 Sunday classifier on TRAIN = 2015-01-01 to
  SPLIT_DATE (forward-shifted) and evaluate it on VAL = report_date
  > SPLIT_DATE.

Default SPLIT_DATE = 2024-12-31 -- this holds out the most recent 18
months (2025-01 -> 2026-07) for a true forward-shifted test. The
original Phase G v1.1 split was 2024-01-01, so the original VAL
window (2024-01 -> 2026-07) OVERLAPPED with this new TRAIN slice
(2015 -> 2024-12). To avoid that, we re-train on the shifted TRAIN.

For honesty:
  * Hyperparameters are FIXED at the v1.1 sweep winner (gamma=10,
    min_child_weight=50, max_depth=3, n_estimators=300) -- NOT
    re-swept. Re-sweeping on the new training set would normalize
    the hyperparameter-selection step within the new TRAIN, but
    would also leak the VAL period's information if we used VAL
    AUC/PnL as the sweep selection criterion. We commit to the
    v1.1 hyperparameters and only re-fit on the new TRAIN.
  * Random baseline (100 trials) is recomputed for the new VAL
    window to provide a null-distribution-tail test.
  * n_slots = 4 (deployable rule per §B.6).

Outputs:
  04_backtest/phase_g_v1_1_oos_<split_date_clean>/
    equity_curve.csv, trades.csv, summary.json,
    random_baseline_dist.csv
"""
from __future__ import annotations
import sys, importlib.util, json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

HERE = Path(__file__).resolve().parent
# Reuse the train module + Phase G helpers
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

# Fixed v1.1 hyperparameters (no re-sweep)
V1_1_PARAMS = dict(
    objective="binary:logistic",
    eval_metric=["logloss", "auc"],
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=50,
    gamma=10.0,    # <-- v1.1's winning value
    reg_lambda=1.0,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1,
)

# Operating-point parameters (unchanged from §B.6 deployable rule)
THETA_SCREEN = 0.20
GAP_LO = 0.02
GAP_HI = 0.15

# Benchmark original VAL numbers (from Appendix B §B.2 n=4)
ORIG_VAL_BENCHMARK = {
    "split_date": "2024-01-01",
    "n_trades": 56,
    "irr_pct": 21.29,
    "sharpe": 1.92,
    "max_dd_pct": -6.07,
    "hit_rate_pct": 66.1,
    "avg_pnl_pct": 3.41,
}


def select_v11_trades(val_df, clf):
    """Apply the v1.1_two_stage selection rule to val_df."""
    proba = clf.predict_proba(val_df[SUNDAY_SAFE_FEATURES])[:, 1]
    val_df = val_df.copy()
    val_df["pead_proba"] = proba
    mask = (
        (val_df["pead_proba"] >= THETA_SCREEN) &
        (val_df["opening_gap_t1"] >= GAP_LO) &
        (val_df["opening_gap_t1"] <= GAP_HI) &
        (val_df["path_pnl_t11_pct"].notna())
    )
    return val_df[mask].copy().reset_index(drop=True)


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
    parser.add_argument("--split-date", default="2024-12-31",
                        help="End of TRAIN period (inclusive). "
                             "VAL = report_date > split_date.")
    parser.add_argument("--n-slots", type=int, default=4)
    parser.add_argument("--initial-nav", type=float, default=100_000.0)
    parser.add_argument("--n-rng-trials", type=int, default=100)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("PHASE G v1.1 -- FORWARD-SHIFTED OUT-OF-SAMPLE VALIDATION")
    print("=" * 78)
    print(f"SPLIT_DATE      : {args.split_date}")
    print(f"N_SLOTS         : {args.n_slots}")
    print(f"INITIAL_NAV     : ${args.initial_nav:,.0f}")
    print(f"N_RNG_TRIALS    : {args.n_rng_trials}")
    print(f"OPERATING POINT : P(PEAD) >= {THETA_SCREEN} AND "
          f"opening_gap_t1 in [{GAP_LO}, {GAP_HI}]")
    print(f"V1.1 PARAMS     : gamma=10, mcw=50, md=3, n_est=300")
    print(f"                      (FIXED -- no re-sweep)")
    print("=" * 78)

    # --- Load + cut + gates ---
    print("\n[1] Loading train_matrix + §12 cutoff + gate computation ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    print(f"    rows after §12 cut: {len(df)}")
    df = pg.v3.compute_pead_gates_full(df)

    # Forward-shifted split
    train_df, val_df = tm.split_walk_forward(df, args.split_date)
    train_df, _ = tm.drop_sparse_weeks(train_df, tm.DEFAULT_MIN_GROUP_SIZE)
    val_df, _ = tm.drop_sparse_weeks(val_df, tm.DEFAULT_MIN_GROUP_SIZE)
    train_df = train_df.sort_values(
        ["calendar_week_group", "permaTicker", "report_date"]
    ).reset_index(drop=True)
    val_df = val_df.sort_values(
        ["calendar_week_group", "permaTicker", "report_date"]
    ).reset_index(drop=True)
    print(f"    TRAIN rows      : {len(train_df)}  "
          f"(report_date <= {args.split_date})  "
          f"pead_pos: {int(train_df['pead_pass'].sum())} "
          f"({train_df['pead_pass'].mean()*100:.2f}%)")
    print(f"    VAL rows        : {len(val_df)}  "
          f"(report_date > {args.split_date})  "
          f"pead_pos: {int(val_df['pead_pass'].sum())} "
          f"({val_df['pead_pass'].mean()*100:.2f}%)")
    val_date_min = pd.to_datetime(val_df["report_date"]).min()
    val_date_max = pd.to_datetime(val_df["report_date"]).max()
    print(f"    VAL date range  : {val_date_min.date()} -> {val_date_max.date()}")

    # --- Train v1.1 classifier on the new TRAIN ---
    print("\n[2] Training v1.1 Sunday classifier on NEW TRAIN ...")
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, average_precision_score
    X_train = train_df[SUNDAY_SAFE_FEATURES].copy()
    y_train = train_df["pead_pass"].astype(int).values
    X_val = val_df[SUNDAY_SAFE_FEATURES].copy()
    y_val = val_df["pead_pass"].astype(int).values
    t0 = time.time()
    clf = xgb.XGBClassifier(**V1_1_PARAMS)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"    Trained in {time.time() - t0:.1f}s")
    auc_train = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    auc_val = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    ap_train = average_precision_score(y_train, clf.predict_proba(X_train)[:, 1])
    ap_val = average_precision_score(y_val, clf.predict_proba(X_val)[:, 1])
    print(f"    TRAIN AUC: {auc_train:.4f}  AP: {ap_train:.4f}")
    print(f"    VAL   AUC: {auc_val:.4f}  AP: {ap_val:.4f}")

    # --- Compute trade paths on VAL ---
    print("\n[3] Computing realized entry trade paths on VAL ...")
    # pg.compute_entry_pnl adds `ret_open_t1_close_t11`; ps.compute_trade_paths
    # adds the 12 daily snap columns (path_pnl_t0..t11_pct).
    val_df = pg.compute_entry_pnl(val_df)
    val_df = ps.compute_trade_paths(val_df)

    # --- v1.1 portfolio sim on VAL ---
    print("\n[4] Selecting v1.1 trades on VAL + running portfolio sim ...")
    trades_v11 = select_v11_trades(val_df, clf)
    print(f"    v1.1 trades selected: {len(trades_v11)}")
    result_v11 = ps.simulate_portfolio(trades_v11, n_slots=args.n_slots,
                                       initial_nav=args.initial_nav)
    s_v11 = result_v11["summary"]
    if not s_v11:
        print("    [!] No trades executed. Cannot compute v1.1 strategy metrics.")
    else:
        print(f"    v1.1 strategy on OOS VAL:")
        print(f"      Trades executed   : {s_v11['n_trades_executed']}")
        print(f"      Slots-full skips  : {s_v11['n_slots_full_skips']}")
        print(f"      Initial NAV       : ${s_v11['initial_nav']:,.2f}")
        print(f"      Final NAV         : ${s_v11['final_nav']:,.2f}")
        print(f"      Trading days      : {s_v11['n_trading_days']}")
        print(f"      Years             : {s_v11['n_years']:.2f}")
        print(f"      IRR (annualized)  : {s_v11['irr_pct']:+.2f}%")
        print(f"      Sharpe (liq ann.) : {s_v11['sharpe_liq_annualized']:+.2f}")
        print(f"      Max drawdown      : {s_v11['max_drawdown_pct']:.2f}%")
        print(f"      Per-trade hit rate: {s_v11['hit_rate_pct']:.1f}%")
        print(f"      Avg trade pnl     : {s_v11['avg_trade_pnl_pct']:+.3f}%")

    # --- Random baseline (100 trials) for this OOS window ---
    print(f"\n[5] Running N_RNG_TRIALS={args.n_rng_trials} random trials "
          f"on same OOS VAL ...")
    # Pre-cache calendar once to speed up
    with pd.HDFStore(DB, mode="r") as sstore:
        all_dates = set()
        for key in sstore.keys():
            if not key.startswith("/sp400/"):
                continue
            d = sstore[key]
            all_dates.update(pd.to_datetime(d["Date"]).tolist())
    calendar = sorted(all_dates)
    print(f"    [cached calendar] {len(calendar)} trading days")

    rows = []
    for trial in range(args.n_rng_trials):
        seed = trial * 7 + 100
        trades = select_random_trades(val_df, seed)
        # Use the cached-calendar fast simulator
        result = rb._simulate_with_cached_calendar(
            trades, args.n_slots, args.initial_nav, calendar)
        s = result.get("summary", {})
        if not s:
            continue
        rows.append({
            "trial": trial, "seed": seed,
            "n_trades": s.get("n_trades_executed", 0),
            "irr": s.get("irr_pct", float("nan")),
            "sharpe": s.get("sharpe_liq_annualized", float("nan")),
            "max_dd": s.get("max_drawdown_pct", float("nan")),
            "hit_pct": s.get("hit_rate_pct", float("nan")),
            "avg_pnl_pct": s.get("avg_trade_pnl_pct", float("nan")),
        })
    rand_df = pd.DataFrame(rows)
    if rand_df.empty:
        print("    [!] No random trials completed (VAL too sparse).")
    else:
        print(f"\n    {len(rand_df)}-trial random baseline on OOS VAL:")
        print(f"      IRR    mean={rand_df['irr'].mean():+.2f}%  "
              f"median={rand_df['irr'].median():+.2f}%  std={rand_df['irr'].std():.2f}  "
              f"5%-95% CI=[{rand_df['irr'].quantile(0.05):+.2f}%, "
              f"{rand_df['irr'].quantile(0.95):+.2f}%]")
        print(f"      Sharpe mean={rand_df['sharpe'].mean():+.2f}  "
              f"median={rand_df['sharpe'].median():+.2f}  std={rand_df['sharpe'].std():.2f}  "
              f"5%-95% CI=[{rand_df['sharpe'].quantile(0.05):+.2f}, "
              f"{rand_df['sharpe'].quantile(0.95):+.2f}]")
        print(f"      MaxDD  mean={rand_df['max_dd'].mean():+.2f}%  "
              f"median={rand_df['max_dd'].median():+.2f}%  std={rand_df['max_dd'].std():.2f}")

    # --- Comparison ---
    print(f"\n{'='*78}")
    print("OOS VALIDATION RESULT -- v1.1 vs OOS random baseline vs original VAL benchmark")
    print(f"{'='*78}")
    if s_v11:
        print(f"  {'Metric':<20s} {'OOS v1.1':>14s} {'OOS random mean':>18s} "
              f"{'OOS random best':>16s} {'Orig VAL v1.1':>16s}")
        print(f"  {'-'*86}")
        rand_best_irr = rand_df["irr"].max() if not rand_df.empty else float("nan")
        rand_best_sharpe = rand_df["sharpe"].max() if not rand_df.empty else float("nan")
        rand_best_maxdd = rand_df["max_dd"].min() if not rand_df.empty else float("nan")
        print(f"  {'IRR %':<20s} {s_v11['irr_pct']:>+13.2f}% "
              f"{rand_df['irr'].mean():>+17.2f}% "
              f"{rand_best_irr:>+15.2f}% "
              f"{ORIG_VAL_BENCHMARK['irr_pct']:>+15.2f}%")
        print(f"  {'Sharpe':<20s} {s_v11['sharpe_liq_annualized']:>+13.2f}  "
              f"{rand_df['sharpe'].mean():>+17.2f}  "
              f"{rand_best_sharpe:>+15.2f}  "
              f"{ORIG_VAL_BENCHMARK['sharpe']:>+15.2f}")
        print(f"  {'MaxDD %':<20s} {s_v11['max_drawdown_pct']:>+13.2f}% "
              f"{rand_df['max_dd'].mean():>+17.2f}% "
              f"{rand_best_maxdd:>+15.2f}% "
              f"{ORIG_VAL_BENCHMARK['max_dd_pct']:>+15.2f}%")
        print(f"  {'Hit rate %':<20s} {s_v11['hit_rate_pct']:>13.1f}% "
              f"{rand_df['hit_pct'].mean():>17.1f}% "
              f"{'--':>15s} "
              f"{ORIG_VAL_BENCHMARK['hit_rate_pct']:>15.1f}%")
        print(f"  {'Avg PnL/event %':<20s} {s_v11['avg_trade_pnl_pct']:>+13.3f}% "
              f"{rand_df['avg_pnl_pct'].mean():>+17.3f}% "
              f"{'--':>15s} "
              f"{ORIG_VAL_BENCHMARK['avg_pnl_pct']:>+15.3f}%")
        print(f"  {'Trades':<20s} {s_v11['n_trades_executed']:>13d} "
              f"{int(rand_df['n_trades'].mean()):>17d} "
              f"{'--':>15s} "
              f"{ORIG_VAL_BENCHMARK['n_trades']:>15d}")
        if not rand_df.empty:
            frac_above_sharpe = (rand_df["sharpe"] > s_v11["sharpe_liq_annualized"]).mean()
            frac_above_irr = (rand_df["irr"] > s_v11["irr_pct"]).mean()
            print(f"\n  v1.1 OOS Sharpe ({s_v11['sharpe_liq_annualized']:+.2f}) "
                  f"exceeds {frac_above_sharpe*100:.1f}% of OOS random trials.")
            print(f"  v1.1 OOS IRR    ({s_v11['irr_pct']:+.2f}%) "
                  f"exceeds {frac_above_irr*100:.1f}% of OOS random trials.")
    print(f"{'='*78}")

    # --- Persist artifacts ---
    split_clean = args.split_date.replace("-", "")
    out_dir = HERE / f"phase_g_v1_1_oos_{split_clean}_n{args.n_slots}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not result_v11["equity_curve"].empty:
        result_v11["equity_curve"].to_csv(out_dir / "equity_curve.csv", index=False)
    if not result_v11["trades_done"].empty:
        result_v11["trades_done"].to_csv(out_dir / "trades.csv", index=False)
    if s_v11:
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(s_v11, f, indent=2, default=str)
    if not rand_df.empty:
        rand_df.to_csv(out_dir / "random_baseline_dist.csv", index=False)
    # Save the retrained v1.1 classifier too for traceability
    clf.save_model(str(out_dir / "classifier.json"))
    print(f"\nSaved artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

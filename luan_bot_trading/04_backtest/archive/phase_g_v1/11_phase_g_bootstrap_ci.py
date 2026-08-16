"""
Phase G v1.1 -- Bootstrap CI on the recommended operating point.

Per `phase_g_neg_gap_sweep_findings.md` §G.6 item (4). The App F /
App G result is a cross-fold mean Sharpe of +1.31 across only 4
folds -- too few points to give a reliable CI from sample std alone
(the 95% effective CI = [+1.14, +1.48] is based on a 4-fold std
that has its own noise band).

This script computes three bootstrap distributions per OOS fold,
then aggregates them across the 4 folds:

  1. IID bootstrap on daily log returns of the equity curve
     - Captures day-level noise (the inner-band sampling variance of
       a single Sharpe observation).
     - Treats days as independent, which is somewhat violated by the
       10-day holding periods (several consecutive days share the
       same underlying trade).

  2. Block bootstrap on daily log returns with block length L=10
     (matching the 10-day hold period) -- more conservative CI
     - Preserves within-trade-day-cluster correlation.

  3. Trade-level IID bootstrap on realized_arith_pct of trades_done
     - Computes a per-bootstrap-trial synthetic mean PnL.
     - Captures trade-level variance (no autocorrelation).
     - For Sharpe: mimics by treating each bootstrap-resampled trade
       list as if executed back-to-back across the same observed
       time window, using the entry/exit dates from the original.

After computing per-fold Sharpe distributions, we aggregate:

  A. Parametric cross-fold mean CI -- use the 4 fold Sharpes (the
     best estimator for each fold) and a Student-t distribution
     with 3 degrees of freedom: 95% CI = mean ± t*std/sqrt(N).

  B. Non-parametric cross-fold mean CI -- bootstrap-resample the 4
     per-fold Sharpe estimates with replacement, N_BOOT=1000 trials,
     take 2.5% and 97.5% percentiles of the resampled mean.

Both A and B test "the cross-fold mean Sharpe" (population
parameter of the fold-level distribution). The per-fold CI from
methods 1-3 tests "the single-fold Sharpe" (uncertainty in a single
fold's Sharpe estimate).

The recommended operating point:
  theta = 0.20, gap [-0.15, -0.02]
  (per App F winner and App G confirmation)

CLI:
  python luan_bot_trading/04_backtest/11_phase_g_bootstrap_ci.py
  python luan_bot_trading/04_backtest/11_phase_g_bootstrap_ci.py --n-boot 5000
  python luan_bot_trading/04_backtest/11_phase_g_bootstrap_ci.py --block-len 5

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

# Recommended operating point (App F / App G)
NEG_THETA = 0.20
NEG_GAP_LO = -0.15
NEG_GAP_HI = -0.02

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


def select_neg_picks(df, proba):
    df2 = df.copy()
    df2["pead_proba"] = proba
    mask = (
        (df2["pead_proba"] >= NEG_THETA) &
        (df2["opening_gap_t1"] >= NEG_GAP_LO) &
        (df2["opening_gap_t1"] <= NEG_GAP_HI) &
        (df2["path_pnl_t11_pct"].notna())
    )
    return df2[mask].copy().reset_index(drop=True)


def sharpe_from_log_returns(log_rets: np.ndarray, ann_factor=252.0) -> float:
    """Sharpe on a log-return series, annualized."""
    if len(log_rets) < 2:
        return float("nan")
    m = float(np.nanmean(log_rets))
    s = float(np.nanstd(log_rets, ddof=1))
    if s < 1e-12:
        return 0.0
    return (m / s) * np.sqrt(ann_factor)


def bootstrap_iid_daily(log_rets: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """IID bootstrap of daily log returns -> Sharpe distribution."""
    n = len(log_rets)
    if n == 0:
        return np.array([])
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = sharpe_from_log_returns(log_rets[idx])
    return out


def bootstrap_block_daily(log_rets: np.ndarray, n_boot: int,
                          block_len: int, rng) -> np.ndarray:
    """Block bootstrap on daily log returns -> Sharpe distribution.

    Resample CONTIGUOUS blocks of length block_len (with wraparound via
    i.i.d. selection of block start indices). Concatenate to length n.
    """
    n = len(log_rets)
    if n == 0 or block_len < 1:
        return np.array([])
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=int(np.ceil(n / block_len)) + 1)
        samples = []
        for s in starts:
            samples.append(log_rets[s:s + block_len])
        sample = np.concatenate(samples)[:n]
        out[b] = sharpe_from_log_returns(sample)
    return out


def bootstrap_trade_pnl(trade_arith_pcts: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """IID bootstrap on per-trade arithmetic PnL -> dist of mean PnL (%).

    Returns array of mean_arith_pnl_pct values per bootstrap trial.
    """
    n = len(trade_arith_pcts)
    if n == 0:
        return np.array([])
    return rng.choice(trade_arith_pcts, size=(n_boot, n), replace=True).mean(axis=1) * 100.0


def t_student_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Parametric Student-t CI on the mean of values."""
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    m = float(np.nanmean(values))
    s = float(np.nanstd(values, ddof=1))
    from scipy.stats import t
    t_crit = t.ppf(1 - alpha / 2, df=n - 1)
    se = s / np.sqrt(n)
    return (m - t_crit * se, m + t_crit * se)


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Non-parametric CI via percentile method."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(values, 100 * alpha / 2))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return (lo, hi)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--n-slots", type=int, default=4)
    parser.add_argument("--initial-nav", type=float, default=100_000.0)
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Number of bootstrap trials per fold")
    parser.add_argument("--block-len", type=int, default=10,
                        help="Block length for block bootstrap (default "
                             "10 = hold period)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    folds = DEFAULT_FOLDS[:args.n_folds]
    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print("PHASE G v1.1 -- BOOTSTRAP CI ON RECOMMENDED OPERATING POINT")
    print("=" * 78)
    print(f"Folds      : {len(folds)}")
    print(f"N_SLOTS    : {args.n_slots}")
    print(f"N_BOOT     : {args.n_boot} trials/bootstrap")
    print(f"BLOCK_LEN  : {args.block_len} (for block bootstrap)")
    print(f"Operating pt: theta={NEG_THETA}, gap [{NEG_GAP_LO:+.2f}, {NEG_GAP_HI:+.2f}]")
    print(f"HP source  : App D fold_results.csv POS-tuned per-fold HP "
          f"(gamma=10/5/3/3 for folds 1-4)")
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
              f"mcw={int(row['sel_mcw'])}")

    # --- Load data ---
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

    # --- Per-fold: train, run rule, capture equity_curve + trades_done ---
    print(f"\n[5] Running {len(folds)} folds to capture equity curves "
          f"w/ recommended op pt ...")
    fold_data = []
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

        X_tr = train_df[SUNDAY_SAFE_FEATURES]; y_tr = train_df["pead_pass"].astype(int).values
        X_sv = sweep_df[SUNDAY_SAFE_FEATURES]; y_sv = sweep_df["pead_pass"].astype(int).values
        X_ts = pd.concat([X_tr, X_sv], axis=0).reset_index(drop=True)
        y_ts = np.concatenate([y_tr, y_sv])
        X_te = test_df[SUNDAY_SAFE_FEATURES]; y_te = test_df["pead_pass"].astype(int).values
        hp = fold_hp[fold_idx - 1]
        print(f"  Training classifier (gamma={hp['gamma']}) on "
              f"{len(y_ts)} TRAIN+SWEEP rows ...")
        clf = fit_classifier(X_ts, y_ts, X_te, y_te, hp)
        proba_te = clf.predict_proba(X_te)[:, 1]
        picks = select_neg_picks(test_df, proba_te)
        print(f"  Picks: n={len(picks)} (theta={NEG_THETA}, gap[{NEG_GAP_LO:+.2f}, {NEG_GAP_HI:+.2f}])")
        result = rb._simulate_with_cached_calendar(
            picks, args.n_slots, args.initial_nav, calendar)
        eq = result["equity_curve"]   # columns: date, day_idx, cash, n_open_positions, realized_dollars_so_far, nav
        td = result["trades_done"]    # columns incl realized_arith_pct, entry_date, exit_date
        sumS = result["summary"]
        if eq.empty:
            print("  [!] fold produced empty equity curve")
            fold_data.append(None)
            continue
        # Compute log returns from the equity curve
        eq_log = pd.to_numeric(eq["nav"], errors="coerce")
        log_rets = np.log(eq_log / eq_log.shift(1)).fillna(0.0).to_numpy()
        # Drop the first 0.0 (start artifact)
        if len(log_rets) > 1:
            log_rets = log_rets[1:]
        # Per-trade arith pnl
        if td is not None and len(td):
            trade_pnls = pd.to_numeric(
                td["realized_arith_pct"], errors="coerce").dropna().to_numpy()
        else:
            trade_pnls = np.array([])
        fold_data.append({
            "fold": fold_idx,
            "test_slice": f"{sweep_end}->{test_end}",
            "pos_tuned_hp": hp,
            "equity_curve": eq,
            "trades_done": td,
            "summary": sumS,
            "log_rets": log_rets,
            "trade_pnls": trade_pnls,
            "empirical_sharpe": float(sumS.get("sharpe_liq_annualized", float("nan"))),
            "empirical_irr_pct": float(sumS.get("irr_pct", float("nan"))),
            "empirical_n_trades": int(sumS.get("n_trades_executed", 0)),
        })
        print(f"  Empirical: n_trades={sumS.get('n_trades_executed', 0)}, "
              f"Sharpe={sumS.get('sharpe_liq_annualized', float('nan')):+.2f}, "
              f"IRR={sumS.get('irr_pct', float('nan')):+.2f}%, "
              f"MaxDD={sumS.get('max_drawdown_pct', float('nan')):+.2f}%, "
              f"n_days_equity={len(eq)}")

    # --- Bootstrap each fold ---
    print(f"\n[6] Bootstrapping {args.n_boot} trials per fold x "
          f"3 methods (IID-day, block-day, trade) ...")
    fold_boot_results = []
    for fd in fold_data:
        if fd is None:
            fold_boot_results.append(None)
            continue
        log_rets = fd["log_rets"]
        trade_pnls = fd["trade_pnls"]
        boot_iid = bootstrap_iid_daily(log_rets, args.n_boot, rng)
        boot_block = bootstrap_block_daily(log_rets, args.n_boot,
                                           args.block_len, rng)
        boot_trade = bootstrap_trade_pnl(trade_pnls, args.n_boot, rng)
        fold_boot_results.append({
            "fold": fd["fold"],
            "test_slice": fd["test_slice"],
            "empirical_sharpe": fd["empirical_sharpe"],
            "empirical_n_trades": fd["empirical_n_trades"],
            "n_days": len(log_rets),
            "boot_sharpe_iid": boot_iid,
            "boot_sharpe_block": boot_block,
            "boot_trade_mean_pct": boot_trade,
        })
        print(f"\n  Fold {fd['fold']} ({fd['test_slice']}, "
              f"n_days={len(log_rets)}, n_trades={fd['empirical_n_trades']})")
        print(f"    Empirical Sharpe = {fd['empirical_sharpe']:+.4f}")
        for name, dist in [("IID-day", boot_iid),
                           ("block-day", boot_block)]:
            lo, hi = percentile_ci(dist)
            sh_mean = float(np.nanmean(dist))
            sh_med = float(np.nanmedian(dist))
            print(f"    Bootstrap {name:>10s}: Sharpe mean={sh_mean:+.3f} "
                  f"median={sh_med:+.3f}  "
                  f"95% CI = [{lo:+.3f}, {hi:+.3f}]")
        if len(boot_trade):
            tlo, thi = percentile_ci(boot_trade)
            print(f"    Bootstrap trade-mean (mean PnL%): mean="
                  f"{float(np.nanmean(boot_trade)):+.3f}%  "
                  f"95% CI = [{tlo:+.3f}%, {thi:+.3f}%]")

    # --- Cross-fold aggregation ---
    print("\n" + "=" * 78)
    print("CROSS-FOLD AGGREGATION")
    print("=" * 78)

    # The aggregate Sharpe: 4-fold mean Sharpe
    empirical_sharpes = np.array([fd["empirical_sharpe"]
                                  for fd in fold_data if fd is not None])
    mean_empirical_sharpe = float(np.mean(empirical_sharpes))
    std_empirical_sharpe = float(np.std(empirical_sharpes, ddof=1))

    print(f"\n  Empirical cross-fold Sharpes: {[float(x) for x in empirical_sharpes]}")
    print(f"  Mean = {mean_empirical_sharpe:+.4f}")
    print(f"  Sample std (ddof=1) = {std_empirical_sharpe:.4f}")

    # A. parametric Student-t CI
    t_lo, t_hi = t_student_ci(empirical_sharpes)
    print(f"\n  A. Parametric Student-t CI on cross-fold mean Sharpe:")
    print(f"     [{t_lo:+.3f}, {t_hi:+.3f}]  (n={len(empirical_sharpes)}, df={len(empirical_sharpes)-1})")

    # B. Non-parametric bootstrap of the 4 fold Sharpes
    boot_fold_means = []
    for b in range(args.n_boot * 10):
        idx = rng.integers(0, len(empirical_sharpes), size=len(empirical_sharpes))
        boot_fold_means.append(float(np.mean(empirical_sharpes[idx])))
    boot_fold_means = np.array(boot_fold_means)
    b_lo, b_hi = percentile_ci(boot_fold_means)
    print(f"\n  B. Non-parametric bootstrap of cross-fold mean Sharpe "
          f"({args.n_boot * 10} trials):")
    print(f"     [{b_lo:+.3f}, {b_hi:+.3f}]  (median="
          f"{np.median(boot_fold_means):+.3f}, "
          f"std={np.std(boot_fold_means, ddof=1):.3f})")

    # C. Pool-method: per-fold Sharpe CIs PER method, then average
    print(f"\n  C. Per-fold Sharpe CIs (percentile, from "
          f"{args.n_boot} bootstrap trials):")
    method_names = ["IID-day", "Block-day"]
    method_keys = ["boot_sharpe_iid", "boot_sharpe_block"]
    for mname, mkey in zip(method_names, method_keys):
        fold_means = []
        for fbr in fold_boot_results:
            if fbr is None:
                continue
            dist = fbr[mkey]
            if len(dist) == 0:
                continue
            lo, hi = percentile_ci(dist)
            print(f"    [{mname}] Fold {fbr['fold']}: "
                  f"empirical={fbr['empirical_sharpe']:+.3f}  "
                  f"boot mean={np.nanmean(dist):+.3f}  "
                  f"95% CI=[{lo:+.3f}, {hi:+.3f}]")
            fold_means.append(float(np.nanmean(dist)))
        if fold_means:
            print(f"    [{mname}] Cross-fold mean-of-means = "
                  f"{np.mean(fold_means):+.3f}")

    # Trade-level cross-fold mean PnL CI
    print(f"\n  D. Per-fold trade-mean PnL CI:")
    trade_boot_means = []
    for fbr in fold_boot_results:
        if fbr is None or len(fbr["boot_trade_mean_pct"]) == 0:
            continue
        boot_trade = fbr["boot_trade_mean_pct"]
        lo, hi = percentile_ci(boot_trade)
        empirical_mean = float(np.mean(fbr["boot_trade_mean_pct"]))
        print(f"    Fold {fbr['fold']}: boot mean PnL = "
              f"{np.nanmean(boot_trade):+.3f}%  "
              f"95% CI=[{lo:+.3f}%, {hi:+.3f}%]")
        trade_boot_means.append(empirical_mean)
    if trade_boot_means:
        tlo, thi = percentile_ci(np.array(trade_boot_means))
        print(f"    Cross-fold mean-trade-PnL mean = "
              f"{np.mean(trade_boot_means):+.3f}%  "
              f"95% CI=[{tlo:+.3f}%, {thi:+.3f}%]")

    # --- Persist ---
    out_dir = HERE / f"phase_g_v1_1_bootstrap_ci_n{args.n_folds}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save key distributions + summaries JSON
    summary = {
        "n_folds": len(fold_boot_results),
        "n_boot": args.n_boot,
        "block_len": args.block_len,
        "n_slots": args.n_slots,
        "operating_point": {"theta": NEG_THETA, "gap_lo": NEG_GAP_LO,
                            "gap_hi": NEG_GAP_HI},
        "empirical_cross_fold_sharpes": [float(x) for x in empirical_sharpes],
        "empirical_cross_fold_mean_sharpe": mean_empirical_sharpe,
        "empirical_cross_fold_std_sharpe": std_empirical_sharpe,
        "parametric_student_t_ci": {"lo": t_lo, "hi": t_hi,
                                    "df": len(empirical_sharpes) - 1},
        "bootstrap_cross_fold_mean_sharpe_ci": {"lo": b_lo, "hi": b_hi,
            "median": float(np.median(boot_fold_means)),
            "std": float(np.std(boot_fold_means, ddof=1))},
        "fold_boot_results": [
            {
                "fold": fbr["fold"],
                "test_slice": fbr["test_slice"],
                "empirical_sharpe": fbr["empirical_sharpe"],
                "empirical_n_trades": fbr["empirical_n_trades"],
                "n_days": fbr["n_days"],
                "boot_sharpe_iid_dist_summary": {
                    "mean": float(np.nanmean(fbr["boot_sharpe_iid"])),
                    "median": float(np.nanmedian(fbr["boot_sharpe_iid"])),
                    "ci_lo_95": float(np.percentile(fbr["boot_sharpe_iid"], 2.5)),
                    "ci_hi_95": float(np.percentile(fbr["boot_sharpe_iid"], 97.5)),
                },
                "boot_sharpe_block_dist_summary": {
                    "mean": float(np.nanmean(fbr["boot_sharpe_block"])),
                    "median": float(np.nanmedian(fbr["boot_sharpe_block"])),
                    "ci_lo_95": float(np.percentile(fbr["boot_sharpe_block"], 2.5)),
                    "ci_hi_95": float(np.percentile(fbr["boot_sharpe_block"], 97.5)),
                },
                "boot_trade_mean_pct_dist_summary": {
                    "mean": float(np.nanmean(fbr["boot_trade_mean_pct"])),
                    "median": float(np.nanmedian(fbr["boot_trade_mean_pct"])),
                    "ci_lo_95": float(np.percentile(fbr["boot_trade_mean_pct"], 2.5)),
                    "ci_hi_95": float(np.percentile(fbr["boot_trade_mean_pct"], 97.5)),
                },
            }
            for fbr in fold_boot_results if fbr is not None
        ],
        "created_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save raw distribution arrays as npz for later re-analysis
    fold_boot_dict = {}
    for fbr in fold_boot_results:
        if fbr is None:
            continue
        fold_boot_dict[f"fold{fbr['fold']}_sharpe_iid"] = fbr["boot_sharpe_iid"]
        fold_boot_dict[f"fold{fbr['fold']}_sharpe_block"] = fbr["boot_sharpe_block"]
        fold_boot_dict[f"fold{fbr['fold']}_trade_mean_pct"] = fbr["boot_trade_mean_pct"]
    if fold_boot_dict:
        np.savez(out_dir / "bootstrap_distributions.npz", **fold_boot_dict)

    print(f"\nSaved artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

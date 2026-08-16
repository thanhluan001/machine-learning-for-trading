"""
Phase G v1.1 -- Dead-zone skip rule test.

Per `phase_g_bootstrap_ci_findings.md` §H.9.2 item (6). The Doc G
gap-bucket diagnostic (§G.4) showed the (-10%, -5%] bucket is
ANTI-ALPHA (-2.88% avg PnL/trade, 33% hit, n=6). The recommended
rule keeps [-15%, -2%] which includes this dead zone.

This script tests 5 gap selection rules, all at the SAME operating
point (theta=0.20, n_slots=4) and SAME per-fold POS-tuned HP from
App D. The ONLY intervention is the gap selection rule. This
isolates the dead-zone skip as the sole independent variable.

Rules tested (in bucket-notation; (a, b] = a < x <= b):

  baseline      : [-15%, -2%]                    = [(-0.15, -0.02)]
                  (control, current recommended rule)

  skip_deadzone : [-15%, -10%] U (-5%, -2%]     = two sub-ranges
                  (the proposed dead-zone skip;
                   excludes (-10%, -5%])

  placebo_skip  : [-15%, -5%] U (-3%, -2%]      = bogus skip of
                  (-5%, -3%] (HEALTHY bucket,
                  +2.28% avg). Should HURT or be neutral.

  no_deep       : [-10%, -2%]                   = exclude buckets
                  (-15%, -10%] AND we don't filter dead zone;
                  combines (-10%, -5%] dead zone + (-5%, -2%]
                  alpha engine. (Test if dropping the small-n=2
                  deep bucket (-15%, -10%] hurts.)

  tight_only    : [-5%, -2%]                    = (-5%, -2%] only;
                  drops BOTH dead zone AND the deep (-15%, -10%]
                  and keeps the alpha engine buckets (-5%, -3%]
                  + (-3%, -2%].

  engine_only   : (-3%, -2%]                    = ONLY the core
                  alpha engine bucket. Super tight, smallest
                  expected sample.

The 'skip_deadzone' rule is the hypothesis test. If hypothesis
true: skip_deadzone Sharpe > baseline Sharpe.

Per-fold empirical-projection (from Doc G bucket arithmetic
means: (-15,-10]: +7.14% n=2; (-10,-5]: -2.88% n=6;
(-5,-3]: +2.28% n=12; (-3,-2]: +3.53% n=12):

  baseline (32)    :       ~ +2.09%
  skip_deadzone (26):      ~ +3.23%  (expected lift +1.14% per trade)
  placebo_skip (20) :      ~ +1.97%  (expected slight drop)
  no_deep (30)     :       ~ +1.74%  (lose +14% contribution from n=2)
  tight_only (24)  :       ~ +2.89%  (drop dead + unreliables)
  engine_only (12) :       ~ +3.53%  (highest per-trade, smallest n)

CLI:
  python luan_bot_trading/04_backtest/12_phase_g_deadzone_skip.py
  python luan_bot_trading/04_backtest/12_phase_g_deadzone_skip.py --n-boot 1000

NO DB WRITES. NO classifier retraining changes (uses same HP per fold).
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
N_SLOTS = 4

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]


# Rule definitions: each rule = (label, list of (lo, hi, lo_inclusive, hi_inclusive))
# Default bucket convention is right-inclusive (a, b] -- so a is
# exclusive, b is inclusive. We allow rules to specify boundary
# inclusiveness for excel-level rigor.
def baseline_rule():
    # [-15%, -2%] fully inclusive -- matches App F/G selection exactly
    return [(-0.15, -0.02, True, True)]


def skip_deadzone_rule():
    # [-15%, -10%] ∪ (-5%, -2%]
    # First range inclusive both ends; second uses lo> -0.05 strict
    # (so excludes gap == -0.05 which is in (-10, -5] dead zone).
    return [
        (-0.15, -0.10, True, True),   # [-15%, -10%]
        (-0.05, -0.02, False, True),  # (-5%, -2%]
    ]


def placebo_skip_rule():
    # [-15%, -5%] ∪ (-3%, -2%]
    # Intentionally bogus: drops the (-5%, -3%] HEALTHY bucket.
    return [
        (-0.15, -0.05, True, True),   # [-15%, -5%]
        (-0.03, -0.02, False, True),  # (-3%, -2%]
    ]


def no_deep_rule():
    # [-10%, -2%] -- exclude the (-15%, -10%] deep unreliables only.
    return [(-0.10, -0.02, True, True)]


def tight_only_rule():
    # [-5%, -2%] -- only the alpha engine buckets.
    return [(-0.05, -0.02, True, True)]


def engine_only_rule():
    # (-3%, -2%] -- only the core alpha engine.
    return [(-0.03, -0.02, False, True)]


RULES = [
    ("baseline",       baseline_rule,       "[-15%, -2%] control"),
    ("skip_deadzone",  skip_deadzone_rule,  "[-15%, -10%] U (-5%, -2%]  proposed dead-zone skip"),
    ("placebo_skip",   placebo_skip_rule,    "[-15%, -5%] U (-3%, -2%]  bogus -- skips HEALTHY (-5%, -3%]"),
    ("no_deep",        no_deep_rule,         "[-10%, -2%]                drops (-15%, -10%] deep (n=2 unstable)"),
    ("tight_only",     tight_only_rule,      "[-5%, -2%]                 drops dead zone AND deep"),
    ("engine_only",    engine_only_rule,     "(-3%, -2%]                 super-tight core alpha engine only"),
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


def select_neg_picks_ranges(df, proba, gap_ranges):
    """df + per-row proba -> picks matching a list of (lo, hi, lo_inc, hi_inc).

    A row's opening_gap_t1 is selected iff:
      proba >= NEG_THETA AND path_pnl_t11_pct notna AND
      (gap matches ANY tuple in gap_ranges).
    """
    df2 = df.copy()
    df2["pead_proba"] = proba
    base = (
        (df2["pead_proba"] >= NEG_THETA) &
        (df2["path_pnl_t11_pct"].notna())
    )
    gap_mask = pd.Series(False, index=df2.index)
    gap_col = df2["opening_gap_t1"]
    for lo, hi, lo_inc, hi_inc in gap_ranges:
        if lo_inc and hi_inc:
            m = (gap_col >= lo) & (gap_col <= hi)
        elif lo_inc and not hi_inc:
            m = (gap_col >= lo) & (gap_col < hi)
        elif not lo_inc and hi_inc:
            m = (gap_col > lo) & (gap_col <= hi)
        else:
            m = (gap_col > lo) & (gap_col < hi)
        gap_mask = gap_mask | m
    return df2[base & gap_mask].copy().reset_index(drop=True)


def sharpe_from_log_returns(log_rets, ann_factor=252.0):
    if len(log_rets) < 2:
        return float("nan")
    m = float(np.nanmean(log_rets))
    s = float(np.nanstd(log_rets, ddof=1))
    if s < 1e-12:
        return 0.0
    return (m / s) * np.sqrt(ann_factor)


def bootstrap_iid_daily(log_rets, n_boot, rng):
    n = len(log_rets)
    if n == 0:
        return np.array([])
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = sharpe_from_log_returns(log_rets[idx])
    return out


def bootstrap_trade_pnl(trade_pcts, n_boot, rng):
    n = len(trade_pcts)
    if n == 0:
        return np.array([])
    return rng.choice(trade_pcts, size=(n_boot, n), replace=True).mean(axis=1) * 100.0


def percentile_ci(values, alpha=0.05):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2))))


def t_student_ci(values, alpha=0.05):
    from scipy.stats import t
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    m = float(np.nanmean(values))
    s = float(np.nanstd(values, ddof=1))
    t_crit = t.ppf(1 - alpha / 2, df=n - 1)
    se = s / np.sqrt(n)
    return (m - t_crit * se, m + t_crit * se)


def classify_gap_bucket(g):
    """Map a gap value to a bucket label. Returns '(-15,-10]', etc."""
    if pd.isna(g):
        return "n/a"
    elif g <= -0.10:
        return "(-15,-10] (n=2 deep)"
    elif g <= -0.05:
        return "(-10,-5] (DEAD ZONE)"
    elif g <= -0.03:
        return "(-5,-3]"
    elif g <= -0.02:
        return "(-3,-2] (alpha engine)"
    else:
        return "(outside range)"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--n-slots", type=int, default=N_SLOTS)
    parser.add_argument("--initial-nav", type=float, default=100_000.0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    folds = DEFAULT_FOLDS[:args.n_folds]
    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print("PHASE G v1.1 -- DEAD-ZONE SKIP RULE TEST")
    print("=" * 78)
    print(f"Folds : {len(folds)}")
    print(f"N_SLOTS: {args.n_slots}")
    print(f"N_BOOT : {args.n_boot}")
    print(f"theta  : {NEG_THETA}")
    print("HP source: App D fold_results.csv POS-tuned per-fold HP "
          "(gamma=10/5/3/3, mcw=50, md=3, n_est=300)")
    print(f"Rules  : {len(RULES)}")
    for label, _, desc in RULES:
        print(f"  - {label:14s} : {desc}")
    print("=" * 78)

    print("\n[1] Loading App D fold_results.csv for POS-tuned HP ...")
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

    print("\n[2] Loading train_matrix + computing pead_pass + paths ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = pg.v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)
    print(f"    rows: {len(df)}, pead_pass: {int(df['pead_pass'].sum())}, "
          f"path coverage: {int(df['path_pnl_t11_pct'].notna().sum())}")

    print("\n[3] Pre-caching trading calendar ...")
    with pd.HDFStore(DB, mode="r") as sstore:
        all_dates = set()
        for key in sstore.keys():
            if not key.startswith("/sp400/"):
                continue
            d = sstore[key]
            all_dates.update(pd.to_datetime(d["Date"]).tolist())
    calendar = sorted(all_dates)
    print(f"    calendar: {len(calendar)} trading days")

    # --- Per-fold: train ONE classifier per fold (cached), then evaluate all
    # rules against the SAME predicted proba on the SAME test_df. This
    # isolates the gap rule as the sole independent variable.
    print(f"\n[4] Training+predicting per fold once, evaluating {len(RULES)} "
          f"gap rules per fold ...")
    fold_model_outputs = []  # list of (fold_idx, test_df, proba_te)
    for fold_idx, (train_end, sweep_end, test_end) in enumerate(folds, 1):
        train_ts = pd.Timestamp(train_end)
        sweep_ts = pd.Timestamp(sweep_end)
        test_ts  = pd.Timestamp(test_end)
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= train_ts].copy().reset_index(drop=True)
        sweep_df = df[(rd > train_ts) & (rd <= sweep_ts)].copy().reset_index(drop=True)
        test_df  = df[(rd > sweep_ts) & (rd <= test_ts)].copy().reset_index(drop=True)
        X_tr = train_df[SUNDAY_SAFE_FEATURES]; y_tr = train_df["pead_pass"].astype(int).values
        X_sv = sweep_df[SUNDAY_SAFE_FEATURES]; y_sv = sweep_df["pead_pass"].astype(int).values
        X_ts = pd.concat([X_tr, X_sv], axis=0).reset_index(drop=True)
        y_ts = np.concatenate([y_tr, y_sv])
        X_te = test_df[SUNDAY_SAFE_FEATURES]; y_te = test_df["pead_pass"].astype(int).values
        hp = fold_hp[fold_idx - 1]
        print(f"\n  Fold {fold_idx}: TEST {sweep_end}->{test_end}, "
              f"TRAIN+SWEEP={len(y_ts)}, TEST={len(test_df)}")
        print(f"    Training classifier (gamma={hp['gamma']}) ...")
        clf = fit_classifier(X_ts, y_ts, X_te, y_te, hp)
        proba_te = clf.predict_proba(X_te)[:, 1]
        print(f"    proba summary: min={proba_te.min():.3f}, "
              f"median={np.median(proba_te):.3f}, "
              f"max={proba_te.max():.3f}, "
              f"n>=theta={int((proba_te>=NEG_THETA).sum())}")
        fold_model_outputs.append((fold_idx, test_df, proba_te))

    # --- Evaluate each rule per fold ---
    print(f"\n[5] Running {len(RULES)} gap rules x {len(folds)} folds ...")
    # rule_fold_results[(rule_idx, fold_idx)] = {...}
    rule_fold_results = {}
    for rule_idx, (rule_name, rule_fn, rule_desc) in enumerate(RULES):
        gap_ranges = rule_fn()
        print(f"\n  RULE: {rule_name}  --  {rule_desc}")
        print(f"    gap ranges: {gap_ranges}")
        for fold_idx, test_df, proba_te in fold_model_outputs:
            picks = select_neg_picks_ranges(test_df, proba_te, gap_ranges)
            result = rb._simulate_with_cached_calendar(
                picks, args.n_slots, args.initial_nav, calendar)
            eq = result["equity_curve"]
            td = result["trades_done"]
            sumS = result["summary"] if result["summary"] else {}
            if eq.empty:
                print(f"    Fold {fold_idx}: EMPTY (n_picks={len(picks)})")
                rule_fold_results[(rule_idx, fold_idx)] = {
                    "n_picks": int(len(picks)), "n_trades": 0,
                    "sharpe": float("nan"), "irr_pct": float("nan"),
                    "max_dd_pct": float("nan"), "hit_rate_pct": float("nan"),
                    "avg_pnl_pct": float("nan"), "log_rets": np.array([]),
                    "trade_pnls": np.array([]),
                }
                continue
            # log returns from nav series
            eq_nav = pd.to_numeric(eq["nav"], errors="coerce")
            log_rets = np.log(eq_nav / eq_nav.shift(1)).fillna(0.0).to_numpy()
            if len(log_rets) > 1:
                log_rets = log_rets[1:]
            if td is not None and len(td):
                trade_pnls = pd.to_numeric(
                    td["realized_arith_pct"], errors="coerce").dropna().to_numpy()
            else:
                trade_pnls = np.array([])
            rule_fold_results[(rule_idx, fold_idx)] = {
                "n_picks": int(len(picks)),
                "n_trades": int(sumS.get("n_trades_executed", 0)),
                "sharpe": float(sumS.get("sharpe_liq_annualized", float("nan"))),
                "irr_pct": float(sumS.get("irr_pct", float("nan"))),
                "max_dd_pct": float(sumS.get("max_drawdown_pct", float("nan"))),
                "hit_rate_pct": float(sumS.get("hit_rate_pct", float("nan"))),
                "avg_pnl_pct": float(sumS.get("avg_trade_pnl_pct", float("nan"))),
                "log_rets": log_rets,
                "trade_pnls": trade_pnls,
                "picks": picks,
            }
            print(f"    Fold {fold_idx}: n_picks={len(picks):2d}, "
                  f"n_trades={sumS.get('n_trades_executed',0):2d}, "
                  f"Sharpe={sumS.get('sharpe_liq_annualized', float('nan')):+.3f}, "
                  f"IRR={sumS.get('irr_pct', float('nan')):+.2f}%, "
                  f"MaxDD={sumS.get('max_drawdown_pct', float('nan')):+.2f}%, "
                  f"hit={sumS.get('hit_rate_pct', float('nan')):.1f}%, "
                  f"avg_pnl={sumS.get('avg_trade_pnl_pct', float('nan')):+.2f}%")

    # --- Pick-bucket histogram per rule per fold ---
    print(f"\n[6] Gap-bucket distribution of picks per rule per fold ...")
    print()
    header = "rule/fold".ljust(16) + " ".join(
        f"F{i+1}".ljust(15) for i in range(len(folds)))
    print(header)
    bucket_names = [
        "(-15,-10] deep",
        "(-10,-5] DEAD",
        "(-5,-3]",
        "(-3,-2] alpha",
    ]
    for rule_idx, (rule_name, _, _) in enumerate(RULES):
        row = f"{rule_name}".ljust(16)
        for fold_idx in range(1, len(folds) + 1):
            rfr = rule_fold_results.get((rule_idx, fold_idx))
            if rfr is None or "picks" not in rfr:
                row += "n/a".ljust(15)
                continue
            picks = rfr["picks"]
            if not len(picks):
                row += "0".ljust(15)
                continue
            gaps = pd.to_numeric(picks["opening_gap_t1"], errors="coerce")
            counts = []
            for lo, hi in [(-0.15, -0.10), (-0.10, -0.05),
                           (-0.05, -0.03), (-0.03, -0.02)]:
                counts.append(int(((gaps > lo) & (gaps <= hi)).sum()))
                # Note: gap == exactly -0.15 handled here as not counted;
                # effectively zero in our data.
            row += "/".join(str(c) for c in counts)
            row += f" (n={len(picks)})"
            row = row.ljust(15) if len(row) < (16 + 15 * fold_idx) else row + " "
        print(row)
    print()
    print("    Format per fold cell: deep / DEAD / (-5,-3] / (-3,-2]  (n_total)")

    # --- Cross-fold aggregate per rule ---
    print("\n" + "=" * 78)
    print("CROSS-FOLD AGGREGATE PER RULE")
    print("=" * 78)
    print()
    hdr = (f"{'rule':14s}  {'mean Sharpe':>12s}  {'std':>6s}  "
           f"{'mean IRR%':>10s}  {'mean MaxDD%':>11s}  {'mean hit':>8s}  "
           f"{'mean avg_pnl%':>13s}  {'n_trades_tot':>13s}")
    print(hdr)
    print("-" * len(hdr))
    cross_fold_summaries = []
    for rule_idx, (rule_name, _, _) in enumerate(RULES):
        sharpes = []
        irrs = []
        maxdds = []
        hits = []
        avgpnls = []
        ntrades_total = 0
        for fold_idx in range(1, len(folds) + 1):
            rfr = rule_fold_results.get((rule_idx, fold_idx))
            if rfr is None:
                continue
            if not np.isnan(rfr["sharpe"]):
                sharpes.append(rfr["sharpe"])
            if not np.isnan(rfr["irr_pct"]):
                irrs.append(rfr["irr_pct"])
            if not np.isnan(rfr["max_dd_pct"]):
                maxdds.append(rfr["max_dd_pct"])
            if not np.isnan(rfr["hit_rate_pct"]):
                hits.append(rfr["hit_rate_pct"])
            if not np.isnan(rfr["avg_pnl_pct"]):
                avgpnls.append(rfr["avg_pnl_pct"])
            ntrades_total += rfr.get("n_trades", 0)
        if not sharpes:
            print(f"  {rule_name:14s}  (no data)")
            continue
        mean_s = float(np.mean(sharpes))
        std_s = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
        mean_irr = float(np.mean(irrs)) if irrs else float("nan")
        mean_dd  = float(np.mean(maxdds)) if maxdds else float("nan")
        mean_hit = float(np.mean(hits)) if hits else float("nan")
        mean_apn  = float(np.mean(avgpnls)) if avgpnls else float("nan")
        print(f"  {rule_name:14s}  "
              f"{mean_s:+12.3f}  {std_s:6.3f}  "
              f"{mean_irr:+10.2f}  {mean_dd:+11.2f}  "
              f"{mean_hit:8.1f}  {mean_apn:+13.3f}  "
              f"{ntrades_total:13d}")
        cross_fold_summaries.append({
            "rule": rule_name,
            "sharpes": sharpes,
            "mean_sharpe": mean_s,
            "std_sharpe": std_s,
            "mean_irr": mean_irr,
            "mean_max_dd": mean_dd,
            "mean_hit": mean_hit,
            "mean_avg_pnl": mean_apn,
            "n_trades_total": ntrades_total,
        })

    # --- Bootstrap CIs per rule on cross-fold Sharpes ---
    print("\n" + "=" * 78)
    print("BOOTSTRAP CIs ON CROSS-FOLD MEAN SHARPE PER RULE")
    print("=" * 78)
    print()
    print(f"{'rule':14s}  {'mean':>7s}  {'t-ci':>22s}  {'boot-ci':>22s}")
    print("-" * 70)
    for cfs in cross_fold_summaries:
        sharpes_arr = np.array(cfs["sharpes"], dtype=float)
        if len(sharpes_arr) < 2:
            print(f"  {cfs['rule']:14s}  insufficient data")
            continue
        t_lo, t_hi = t_student_ci(sharpes_arr)
        # Bootstrap of cross-fold means
        boot_means = []
        for b in range(args.n_boot * 10):
            idx = rng.integers(0, len(sharpes_arr), size=len(sharpes_arr))
            boot_means.append(float(np.mean(sharpes_arr[idx])))
        boot_means = np.array(boot_means)
        b_lo, b_hi = percentile_ci(boot_means)
        print(f"  {cfs['rule']:14s}  "
              f"{cfs['mean_sharpe']:+7.3f}  "
              f"[{t_lo:+6.3f}, {t_hi:+6.3f}]   "
              f"[{b_lo:+6.3f}, {b_hi:+6.3f}]")
        cfs["t_ci_lo"] = t_lo
        cfs["t_ci_hi"] = t_hi
        cfs["boot_ci_lo"] = b_lo
        cfs["boot_ci_hi"] = b_hi

    # --- Trade-level bootstrap per rule ---
    print("\n" + "=" * 78)
    print("PER-TRADE PnL BOOTSTRAP CI PER RULE (cross-fold pooled)")
    print("=" * 78)
    print()
    print(f"{'rule':14s}  {'mean':>8s}  {'n_trades_tot':>13s}  "
          f"{'boot 95% CI':>22s}")
    print("-" * 62)
    for rule_idx, (rule_name, _, _) in enumerate(RULES):
        all_trades = []
        for fold_idx in range(1, len(folds) + 1):
            rfr = rule_fold_results.get((rule_idx, fold_idx))
            if rfr is None: continue
            all_trades.append(rfr["trade_pnls"])
        if not all_trades:
            print(f"  {rule_name:14s}  no trades")
            continue
        all_trades_arr = np.concatenate(all_trades)
        if len(all_trades_arr) == 0:
            print(f"  {rule_name:14s}  no trades")
            continue
        boot_trade = bootstrap_trade_pnl(all_trades_arr, args.n_boot, rng)
        lo, hi = percentile_ci(boot_trade)
        mean_pct = float(np.mean(all_trades_arr) * 100.0)
        print(f"  {rule_name:14s}  "
              f"{mean_pct:+8.3f}%  "
              f"{len(all_trades_arr):13d}   "
              f"[{lo:+7.3f}%, {hi:+7.3f}%]")
        # find this rule's cross_fold_summary entry and save
        for cfs in cross_fold_summaries:
            if cfs["rule"] == rule_name:
                cfs["trade_boot_mean_pct"] = mean_pct
                cfs["trade_boot_ci_lo"] = lo
                cfs["trade_boot_ci_hi"] = hi
                cfs["n_trades_pooled"] = int(len(all_trades_arr))
                cfs["trade_pnls_pooled"] = [float(x) for x in all_trades_arr]

    # --- Compare skip_deadzone vs baseline ---
    print("\n" + "=" * 78)
    print("DEAD-ZONE SKIP HYPOTHESIS TEST (skip_deadzone vs baseline)")
    print("=" * 78)
    base = next((c for c in cross_fold_summaries if c["rule"] == "baseline"), None)
    skip = next((c for c in cross_fold_summaries if c["rule"] == "skip_deadzone"), None)
    plcb = next((c for c in cross_fold_summaries if c["rule"] == "placebo_skip"), None)
    if base and skip:
        delta_sharpe = skip["mean_sharpe"] - base["mean_sharpe"]
        delta_irr    = skip["mean_irr"] - base["mean_irr"]
        delta_apn    = skip["mean_avg_pnl"] - base["mean_avg_pnl"]
        print(f"\n  Baseline [-15,-2]       : Sharpe={base['mean_sharpe']:+.4f}  "
              f"IRR={base['mean_irr']:+.2f}%  "
              f"avg_pnl={base['mean_avg_pnl']:+.3f}%  "
              f"n_trades={base['n_trades_total']}")
        print(f"  skip_deadzone           : Sharpe={skip['mean_sharpe']:+.4f}  "
              f"IRR={skip['mean_irr']:+.2f}%  "
              f"avg_pnl={skip['mean_avg_pnl']:+.3f}%  "
              f"n_trades={skip['n_trades_total']}")
        print()
        print(f"  Delta Sharpe  = {delta_sharpe:+.4f}   "
              f"({'LIFT' if delta_sharpe > 0 else 'drop'})")
        print(f"  Delta IRR     = {delta_irr:+.2f}%")
        print(f"  Delta avg_pnl = {delta_apn:+.3f}% per trade")
        # Per-fold pair comparison
        print()
        print(f"  Per-fold Sharpe pair:")
        for fold_idx in range(1, len(folds) + 1):
            base_rfr = rule_fold_results.get((0, fold_idx))
            skip_rfr = rule_fold_results.get((1, fold_idx))
            if base_rfr and skip_rfr:
                print(f"    Fold {fold_idx}: "
                      f"baseline={base_rfr['sharpe']:+.3f} (n={base_rfr['n_trades']}), "
                      f"skip={skip_rfr['sharpe']:+.3f} (n={skip_rfr['n_trades']}), "
                      f"delta={skip_rfr['sharpe']-base_rfr['sharpe']:+.3f}")
    if plcb:
        print()
        print(f"  PLACEBO CHECK (drop (-5%, -3%] HEALTHY bucket):")
        print(f"    placebo mean Sharpe = {plcb['mean_sharpe']:+.4f}  "
              f"(baseline={base['mean_sharpe']:+.4f})")
        if plcb["mean_sharpe"] < base["mean_sharpe"]:
            print(f"    -> PLACEBO HURT as expected (good sanity check).")
        else:
            print(f"    -> PLACEBO did NOT hurt (sanity check failed).")

    # --- Persist ---
    out_dir = HERE / f"phase_g_v1_1_deadzone_skip_n{args.n_folds}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_folds": len(folds),
        "n_boot": args.n_boot,
        "n_slots": args.n_slots,
        "theta": NEG_THETA,
        "fold_hp": fold_hp,
        "rules": [
            {"name": rule_name, "desc": rule_desc,
             "gap_ranges": rule_fn()}
            for rule_name, rule_fn, rule_desc in RULES
        ],
        "cross_fold_summaries": [
            {k: v for k, v in cfs.items()
             if k not in ("trade_pnls_pooled", "sharpes")}
            for cfs in cross_fold_summaries
        ],
        "fold_results": [
            {
                "rule": RULES[rule_idx][0],
                "fold": fold_idx,
                "n_picks": rule_fold_results.get((rule_idx, fold_idx), {}).get("n_picks", 0),
                "n_trades": rule_fold_results.get((rule_idx, fold_idx), {}).get("n_trades", 0),
                "sharpe": rule_fold_results.get((rule_idx, fold_idx), {}).get("sharpe", float("nan")),
                "irr_pct": rule_fold_results.get((rule_idx, fold_idx), {}).get("irr_pct", float("nan")),
                "max_dd_pct": rule_fold_results.get((rule_idx, fold_idx), {}).get("max_dd_pct", float("nan")),
                "hit_rate_pct": rule_fold_results.get((rule_idx, fold_idx), {}).get("hit_rate_pct", float("nan")),
                "avg_pnl_pct": rule_fold_results.get((rule_idx, fold_idx), {}).get("avg_pnl_pct", float("nan")),
            }
            for rule_idx in range(len(RULES))
            for fold_idx in range(1, len(folds) + 1)
        ],
        "created_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save trade-level PnL pools per rule for re-bootstrapping later
    pools_dict = {}
    for cfs in cross_fold_summaries:
        if "trade_pnls_pooled" in cfs:
            pools_dict[f"{cfs['rule']}_trade_pnls_pct"] = np.array(
                cfs["trade_pnls_pooled"]) * 100.0
    if pools_dict:
        np.savez(out_dir / "trade_pnls_pools.npz", **pools_dict)

    print(f"\nSaved artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

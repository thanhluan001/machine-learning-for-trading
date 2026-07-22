"""
Phase G v1.1 -- DEAD-ZONE SKIP NESTED CV (proper OOS test).

Per `phase_g_deadzone_skip_findings.md` §I.11.7. The user critique:
Doc I's (-10%, -5%) dead-zone boundary came from Doc G's pd.cut
discretization of the SAME OOS picks Doc I evaluated -- in-sample
rule selection. The user said: "the next deadzone can be [-7, -10]".

This script runs the proper TEST-of-DEAD-ZONE under nested CV, where
for each fold a dead-zone rule is SELECTED on SWEEP_VAL and then
evaluated on a separate TEST slice.

Per-fold procedure (matches App D nested CV scaffold):
1. TRAIN: data <= train_end_k. Fit classifier with App D's
   per-fold-selected HP (gamma=10/5/3/3 from phase_g_v1_1_nested_cv_n4
   fold_results.csv). These HPs already survived the user's critique
   since App D's nested CV proper structure SWEEP-(separate from)-TEST.
2. SWEEP_VAL: (train_end_k, sweep_end_k].
   Apply each of 10 candidate gap rules to the predicted P(PEAD) on
   SWEEP_VAL. Compute SWEEP_VAL per-trade mean arith PnL for each rule.
   Select the rule with max PnL (with n_trades >= 3 to be considered).
3. Refit classifier on TRAIN + SWEEP_VAL with same selected HP.
4. TEST: (sweep_end_k, test_end_k].
   Apply the SELECTED rule to the predicted P(PEAD) on TEST. Evaluate
   the resulting trades via the portfolio simulator -> TEST Sharpe.

Aggregate TEST-fold Sharpe / IRR / per-trade PnL across the 4 folds.
This is the OOS-defensible estimate of the dead-zone-skip strategy
under proper nested CV.

The H_B test of the user's critique: examine the PER-FOLD SELECTED
RULES. If they CONVERGE around the same boundary {-10, -5}),
the dead zone is structural (H_B=true) and Doc I's rule is OOS.
If they DIFFER ridiculously {-10, -5, -11, -6, -7, -4, -9, -4},
the dead zone is in-sample-fit noise (H_B=false) and the user is
right.

Candidate gap rules (10 total):

  no_skip    : gap in [-15%, -2%]                       baseline
  dz_-10_-5  : EXCLUDE gap in (-10%, -5%]              Doc G/I observed
  dz_-9_-5   : EXCLUDE gap in (-9%, -5%]
  dz_-8_-5   : EXCLUDE gap in (-8%, -5%]
  dz_-7_-4   : EXCLUDE gap in (-7%, -4%]
  dz_-7_-3   : EXCLUDE gap in (-7%, -3%]
  dz_-10_-6  : EXCLUDE gap in (-10%, -6%]
  dz_-11_-6  : EXCLUDE gap in (-11%, -6%]
  dz_-9_-4   : EXCLUDE gap in (-9%, -4%]
  dz_-12_-7  : EXCLUDE gap in (-12%, -7%]             user's literal example

CLI:
  python luan_bot_trading/04_backtest/13_phase_g_deadzone_nested_cv.py
  python luan_bot_trading/04_backtest/13_phase_g_deadzone_nested_cv.py --n-rng-trials 100

NO DB WRITES. Same classifier HP per fold from App D -- no HP sweeping here.
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

# Operating point constants (from App F/G -- the carrier operating
# point the App D nested CV used. NOT re-swept here.)
NEG_THETA = 0.20
N_SLOTS = 4
INITIAL_NAV = 100_000.0

APPD_RESULTS_CSV = HERE / "archive" / "experiments" / "phase_g_v1_1_nested_cv_n4" / "fold_results.csv"

# Fold definitions: (train_end, sweep_end, test_end) -- same as App D
DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]


# ------------------- Candidate dead-zone rules -------------------
# Each rule = (label, exclude_range) where exclude_range = (lo, hi)
# or None for "no_skip".
# The rule's KEPT set: gap in [-15%, -2%] AND NOT (gap > lo AND gap <= hi).
# (lo, hi] convention (right-inclusive in pd.cut sense).
GAP_RULES = [
    ("no_skip",     None),
    ("dz_-10_-5",   (-0.10, -0.05)),
    ("dz_-9_-5",    (-0.09, -0.05)),
    ("dz_-8_-5",    (-0.08, -0.05)),
    ("dz_-7_-4",    (-0.07, -0.04)),
    ("dz_-7_-3",    (-0.07, -0.03)),
    ("dz_-10_-6",   (-0.10, -0.06)),
    ("dz_-11_-6",   (-0.11, -0.06)),
    ("dz_-9_-4",    (-0.09, -0.04)),
    ("dz_-12_-7",   (-0.12, -0.07)),
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


def select_picks(df, proba, exclude_range):
    """Apply NEG_only selection with optional dead-zone exclusion.

    KEPT set:
      proba >= NEG_THETA AND path_pnl_t11_pct notna AND
      gap >= -0.15 AND gap <= -0.02 AND
      NOT (exclude_range[0] < gap <= exclude_range[1])
    """
    df2 = df.copy()
    df2["pead_proba"] = proba
    gap_col = df2["opening_gap_t1"]
    base = (
        (df2["pead_proba"] >= NEG_THETA) &
        (gap_col >= -0.15) &
        (gap_col <= -0.02) &
        (df2["path_pnl_t11_pct"].notna())
    )
    if exclude_range is not None:
        lo, hi = exclude_range
        dead = (gap_col > lo) & (gap_col <= hi)
        base = base & ~dead
    return df2[base].copy().reset_index(drop=True)


def select_random_trades(val_df, seed):
    """Random 1-per-week selection with explicit seed."""
    rng = np.random.default_rng(seed)
    rows = []
    for week, g in val_df.groupby("calendar_week_group", sort=True):
        g_ok = g.dropna(subset=["path_pnl_t11_pct"])
        gap = pd.to_numeric(g_ok["opening_gap_t1"], errors="coerce")
        neg_only = g_ok[(gap >= -0.15) & (gap <= -0.02)]
        if neg_only.empty:
            continue
        idx = rng.integers(len(neg_only))
        rows.append(neg_only.iloc[idx])
    return pd.DataFrame(rows).reset_index(drop=True)


def sharpe_from_log_returns(log_rets, ann_factor=252.0):
    if len(log_rets) < 2:
        return float("nan")
    m = float(np.nanmean(log_rets))
    s = float(np.nanstd(log_rets, ddof=1))
    if s < 1e-12:
        return 0.0
    return (m / s) * np.sqrt(ann_factor)


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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--n-slots", type=int, default=N_SLOTS)
    parser.add_argument("--initial-nav", type=float, default=INITIAL_NAV)
    parser.add_argument("--n-rng-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sweep-n", type=int, default=3,
                        help="Minimum SWEEP_VAL n_trades for a rule "
                             "candidate to be selectable. Below this, "
                             "the rule's SWEEP_VAL PnL estimate is too "
                             "noisy.")
    args = parser.parse_args(argv)

    folds = DEFAULT_FOLDS[:args.n_folds]
    rng_seed = args.seed

    print("=" * 78)
    print("PHASE G v1.1 -- DEAD-ZONE SKIP NESTED CV (PROPER OOS TEST)")
    print("=" * 78)
    print(f"Folds      : {len(folds)}")
    print(f"N_SLOTS    : {args.n_slots}")
    print(f"N_RNG_TRIALS: {args.n_rng_trials}")
    print(f"NEG_THETA  : {NEG_THETA}")
    print(f"MIN_SWEEP_N: {args.min_sweep_n}")
    print(f"HP source  : App D phase_g_v1_1_nested_cv_n4/fold_results.csv (gamma=10/5/3/3)")
    print(f"# candidate rules: {len(GAP_RULES)}")
    for label, excl in GAP_RULES:
        if excl is None:
            print(f"  - {label:11s} : no skip (baseline)")
        else:
            print(f"  - {label:11s} : exclude gap in ({excl[0]*100:.0f}%, {excl[1]*100:.0f}%]")
    print("=" * 78)

    print("\n[1] Loading App D fold_results.csv for POS-tuned HP per fold ...")
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

    # ------------------- Per-fold loop -------------------
    print(f"\n[4] Running {len(folds)} folds (per-fold: fit TRAIN classifier, "
          f"sweep gap rule on SWEEP_VAL, select, refit TRAIN+SWEEP_VAL, "
          f"eval TEST) ...")
    fold_results = []

    for fold_idx, (train_end, sweep_end, test_end) in enumerate(folds, 1):
        print("\n" + "=" * 78)
        print(f"  FOLD {fold_idx}/{len(folds)}: "
              f"TRAIN <= {train_end} | SWEEP_VAL {train_end}->{sweep_end} | "
              f"TEST {sweep_end}->{test_end}")
        print("=" * 78)
        train_ts = pd.Timestamp(train_end)
        sweep_ts = pd.Timestamp(sweep_end)
        test_ts  = pd.Timestamp(test_end)
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= train_ts].copy().reset_index(drop=True)
        sweep_df = df[(rd > train_ts) & (rd <= sweep_ts)].copy().reset_index(drop=True)
        test_df  = df[(rd > sweep_ts) & (rd <= test_ts)].copy().reset_index(drop=True)
        hp = fold_hp[fold_idx - 1]
        print(f"  TRAIN={len(train_df)}  SWEEP_VAL={len(sweep_df)}  "
              f"TEST={len(test_df)}  (HP: gamma={hp['gamma']})")

        # ----- 4a. Fit classifier on TRAIN only (for SWEEP_VAL eval) -----
        print(f"\n  Step 4a: Fit classifier on TRAIN only "
              f"({len(train_df)} rows, gamma={hp['gamma']}) ...")
        X_tr = train_df[SUNDAY_SAFE_FEATURES].copy()
        y_tr = train_df["pead_pass"].astype(int).values
        X_sv = sweep_df[SUNDAY_SAFE_FEATURES].copy()
        y_sv = sweep_df["pead_pass"].astype(int).values
        X_te = test_df[SUNDAY_SAFE_FEATURES].copy()
        y_te = test_df["pead_pass"].astype(int).values
        clf_train = fit_classifier(X_tr, y_tr, X_sv, y_sv, hp)
        proba_sv_4a = clf_train.predict_proba(X_sv)[:, 1]

        # ----- 4b. Sweep gap rules on SWEEP_VAL -----
        print(f"\n  Step 4b: Sweep {len(GAP_RULES)} gap rules on SWEEP_VAL ...")
        sweep_eval_rows = []
        for label, excl in GAP_RULES:
            picks = select_picks(sweep_df, proba_sv_4a, excl)
            n_pick = len(picks)
            if n_pick == 0:
                sweep_eval_rows.append({
                    "rule": label, "exclude_range": excl,
                    "n_picks": 0, "mean_arith_pct": float("nan"),
                    "median_arith_pct": float("nan"),
                    "hit_pct": float("nan"),
                    "sharpe_picks": float("nan"),
                })
                continue
            arith = np.expm1(pd.to_numeric(picks["path_pnl_t11_pct"],
                                          errors="coerce").dropna().values)
            sweep_eval_rows.append({
                "rule": label, "exclude_range": excl,
                "n_picks": int(n_pick),
                "mean_arith_pct": float(arith.mean() * 100),
                "median_arith_pct": float(np.median(arith) * 100),
                "hit_pct": float((arith > 0).mean() * 100),
                "sharpe_picks": sharpe_from_log_returns(
                    np.log1p(pd.to_numeric(picks["path_pnl_t11_pct"],
                                           errors="coerce").dropna().values)),
            })
            print(f"    {label:11s} (n={n_pick:>3d}): "
                  f"mean={arith.mean()*100:+.3f}%  "
                  f"hit={((arith>0).mean())*100:>5.1f}%  "
                  f"med={np.median(arith)*100:.3f}%")
        sweep_eval_df = pd.DataFrame(sweep_eval_rows)

        # ----- 4c. Select gap rule on SWEEP_VAL -----
        # Selection criterion: MAX SWEEP_VAL mean_arith_pct with n_picks >= min_sweep_n
        valid_mask = (sweep_eval_df["n_picks"] >= args.min_sweep_n) & \
                     (~sweep_eval_df["mean_arith_pct"].isna())
        valid_df = sweep_eval_df[valid_mask].copy()
        if len(valid_df) >= 1:
            sel_row = valid_df.loc[
                valid_df["mean_arith_pct"].idxmax()].to_dict()
            sel_rule = sel_row["rule"]
            sel_excl = sel_row["exclude_range"]
            sel_reason = (f"SWEEP_VAL max mean_arith_pct "
                          f"(n={sel_row['n_picks']}, "
                          f"pnl={sel_row['mean_arith_pct']:+.3f}%)")
        else:
            # Fallback: no_skip
            sel_rule = "no_skip"
            sel_excl = None
            sel_reason = (f"fallback to no_skip (no rule reached "
                          f"n_picks >= {args.min_sweep_n} on SWEEP_VAL)")
        print(f"\n  Step 4c: SELECTED rule = {sel_rule}  ({sel_reason})")

        # ----- 4d. Refit classifier on TRAIN+SWEEP_VAL with same HP -----
        print(f"\n  Step 4d: Refit classifier on TRAIN+SWEEP_VAL "
              f"({len(train_df)+len(sweep_df)} rows, gamma={hp['gamma']}) ...")
        X_ts = pd.concat([X_tr, X_sv], axis=0).reset_index(drop=True)
        y_ts = np.concatenate([y_tr, y_sv])
        clf_final = fit_classifier(X_ts, y_ts, X_te, y_te, hp)
        proba_te = clf_final.predict_proba(X_te)[:, 1]
        # Also predicted proba on SWEEP_VAL with refit classifier (for reference)
        proba_sv_final = clf_final.predict_proba(X_sv)[:, 1]

        # ----- 4e. Apply selected rule on TEST, eval portfolio -----
        print(f"\n  Step 4e: Apply SELECTED rule '{sel_rule}' to TEST ...")
        sel_picks = select_picks(test_df, proba_te, sel_excl)
        print(f"    TEST picks (n={len(sel_picks)}, rule={sel_rule})")
        result = rb._simulate_with_cached_calendar(
            sel_picks, args.n_slots, args.initial_nav, calendar)
        s = result.get("summary", {})
        if not s:
            print("  [!] TEST produced no trades")
            test_sharpe = float("nan"); test_irr = float("nan")
            test_max_dd = float("nan"); test_hit = float("nan")
            test_avg_pnl = float("nan"); test_n_trades = 0
        else:
            test_sharpe = float(s["sharpe_liq_annualized"])
            test_irr = float(s["irr_pct"])
            test_max_dd = float(s["max_drawdown_pct"])
            test_hit = float(s["hit_rate_pct"])
            test_avg_pnl = float(s["avg_trade_pnl_pct"])
            test_n_trades = int(s["n_trades_executed"])
            print(f"    TEST fold {fold_idx}: n_trades={test_n_trades}, "
                  f"Sharpe={test_sharpe:+.3f}, IRR={test_irr:+.2f}%, "
                  f"MaxDD={test_max_dd:+.2f}%, hit={test_hit:.1f}%, "
                  f"avg_pnl={test_avg_pnl:+.3f}%")

        # ----- 4f. TEST: also evaluate ALL rules to see which would
        # win if we had perfect knowledge (oracle test) -----
        print(f"\n  Step 4f: Oracle evaluation -- all rules on TEST ...")
        oracle_rows = []
        for label, excl in GAP_RULES:
            picks = select_picks(test_df, proba_te, excl)
            n_pick = len(picks)
            if n_pick == 0:
                oracle_rows.append({
                    "rule": label, "n_picks_test": 0,
                    "test_mean_arith_pct": float("nan"),
                    "test_sharpe": float("nan"),
                })
                continue
            res_r = rb._simulate_with_cached_calendar(
                picks, args.n_slots, args.initial_nav, calendar)
            s_r = res_r.get("summary", {})
            if not s_r:
                oracle_rows.append({
                    "rule": label, "n_picks_test": n_pick,
                    "test_mean_arith_pct": float("nan"),
                    "test_sharpe": float("nan"),
                })
                continue
            arith = np.expm1(pd.to_numeric(picks["path_pnl_t11_pct"],
                                          errors="coerce").dropna().values)
            oracle_rows.append({
                "rule": label, "n_picks_test": n_pick,
                "test_mean_arith_pct": float(arith.mean() * 100),
                "test_sharpe": float(s_r["sharpe_liq_annualized"]),
                "test_irr": float(s_r["irr_pct"]),
            })
            print(f"    {label:11s} (n={n_pick:>3d}): "
                  f"Sharpe={s_r['sharpe_liq_annualized']:+.3f}, "
                  f"IRR={s_r['irr_pct']:+.2f}%, "
                  f"mean_arith={arith.mean()*100:+.3f}%")
        oracle_df = pd.DataFrame(oracle_rows)

        # ----- 4g. Random baseline trials on TEST (NEG-only universe) -----
        print(f"\n  Step 4g: {args.n_rng_trials} random-baseline trials "
              f"on TEST (NEG_only universe) ...")
        rand_rows = []
        for trial in range(args.n_rng_trials):
            seed = trial * 7 + 100 + fold_idx
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
        if len(rand_df) >= 1 and not np.isnan(test_sharpe):
            frac_above_sharpe = float((rand_df["sharpe"] > test_sharpe).mean())
        else:
            frac_above_sharpe = float("nan")
        if len(rand_df):
            print(f"    Random baseline mean (fold {fold_idx}): "
                  f"Sharpe={rand_df['sharpe'].mean():+.2f}  "
                  f"IRR={rand_df['irr'].mean():+.2f}%  "
                  f"n_trials={len(rand_df)}")
        print(f"    Selected-rule TEST Sharpe beats {frac_above_sharpe * 100 if not np.isnan(frac_above_sharpe) else 0:.1f}% of random trials")

        fold_results.append({
            "fold": fold_idx,
            "train_end": train_end, "sweep_end": sweep_end,
            "test_end": test_end,
            "selected_rule": sel_rule,
            "selected_exclude_range": (
                [float(sel_excl[0]), float(sel_excl[1])]
                if sel_excl is not None else None),
            "selected_rule_reason": sel_reason,
            "fold_hp": hp,
            "sweep_eval": sweep_eval_df.to_dict(orient="records"),
            "oracle_test_eval": oracle_df.to_dict(orient="records"),
            "test_n_trades": test_n_trades,
            "test_sharpe": test_sharpe,
            "test_irr_pct": test_irr,
            "test_max_dd_pct": test_max_dd,
            "test_hit_pct": test_hit,
            "test_avg_pnl_pct": test_avg_pnl,
            "random_mean_sharpe": float(rand_df["sharpe"].mean()) if len(rand_df) else float("nan"),
            "random_mean_irr":    float(rand_df["irr"].mean())    if len(rand_df) else float("nan"),
            "frac_random_exceeded_sharpe": frac_above_sharpe,
        })

    # ------------------- Aggregate across folds -------------------
    print("\n" + "=" * 78)
    print("AGGREGATE: DEAD-ZONE SKIP NESTED CV RESULTS")
    print("=" * 78)
    print(f"\n{'Fold':>4s}  {'SWEEP slice':<25s}  "
          f"{'selected_rule':<13s}  {'n_t':>4s}  "
          f"{'TEST Sharpe':>11s}  {'TEST IRR%':>9s}  "
          f"{'TEST MaxDD%':>11s}  {'TEST hit%':>9s}  "
          f"{'avg_pnl%':>8s}  {'%rand_sh':>8s}")
    print("-" * 105)
    for r in fold_results:
        test_range = r['sweep_end'][:7] + '->' + r['test_end'][:7]
        excl = r["selected_exclude_range"]
        rule_str = r["selected_rule"]
        if excl:
            rule_str = f"{rule_str} {excl[0]:.2f},{excl[1]:.2f}"
        print(f"  {r['fold']:>2d}  {test_range:>22s}  {rule_str:<20s}  "
              f"{r['test_n_trades']:>3d}  "
              f"{r['test_sharpe']:>+10.3f}  {r['test_irr_pct']:>+8.2f}%  "
              f"{r['test_max_dd_pct']:>+10.2f}%  "
              f"{r['test_hit_pct']:>8.1f}%  "
              f"{r['test_avg_pnl_pct']:>+7.3f}%  "
              f"{r['frac_random_exceeded_sharpe']*100 if not np.isnan(r['frac_random_exceeded_sharpe']) else 0:>5.1f}%")

    # Per-fold selected rules -- the H_B test
    print("\n  H_B test -- per-fold SELECTED rule shapes:")
    for r in fold_results:
        excl = r["selected_exclude_range"]
        if excl:
            print(f"    Fold {r['fold']}: dz_({excl[0]*100:.0f}, {excl[1]*100:.0f}]")
        else:
            print(f"    Fold {r['fold']}: no_skip  (baseline returned)")

    # Cross-fold mean + CIs
    sharpes = np.array([r["test_sharpe"] for r in fold_results
                        if not np.isnan(r["test_sharpe"])])
    irrs = np.array([r["test_irr_pct"] for r in fold_results
                    if not np.isnan(r["test_irr_pct"])])
    max_dds = np.array([r["test_max_dd_pct"] for r in fold_results
                        if not np.isnan(r["test_max_dd_pct"])])
    hits = np.array([r["test_hit_pct"] for r in fold_results
                    if not np.isnan(r["test_hit_pct"])])
    avgpnls = np.array([r["test_avg_pnl_pct"] for r in fold_results
                        if not np.isnan(r["test_avg_pnl_pct"])])
    print(f"\n  Cross-fold metrics:")
    print(f"    Mean Sharpe = {np.mean(sharpes):+.3f}  (std {np.std(sharpes, ddof=1):.3f}, n={len(sharpes)})")
    print(f"    Mean IRR    = {np.mean(irrs):+.2f}%")
    print(f"    Mean MaxDD  = {np.mean(max_dds):+.2f}%")
    print(f"    Mean hit    = {np.mean(hits):.1f}%")
    print(f"    Mean avg_pnl = {np.mean(avgpnls):+.3f}%")
    if len(sharpes) >= 2:
        t_lo, t_hi = t_student_ci(sharpes)
        print(f"    Parametric Student-t CI on mean Sharpe (n={len(sharpes)}, df={len(sharpes)-1}): "
              f"[{t_lo:+.3f}, {t_hi:+.3f}]")
        # Bootstrap CI
        boot_means = []
        rng = np.random.default_rng(rng_seed)
        for b in range(10000):
            idx = rng.integers(0, len(sharpes), size=len(sharpes))
            boot_means.append(float(np.mean(sharpes[idx])))
        boot_means = np.array(boot_means)
        b_lo, b_hi = percentile_ci(boot_means)
        print(f"    Bootstrap CI (10k trials): [{b_lo:+.3f}, {b_hi:+.3f}]")

    # ------------------- Compare against baseline-only nested CV -------------------
    # The Doc H baseline rule ([-15,-2], no skip) under proper nested CV with
    # the SAME App-D-selected HPs would NOT select any dead zone. That's
    # equivalent to "manually force no_skip on every fold".
    print("\n" + "=" * 78)
    print("VS-BASELINE-COMPARISON: does dead-zone selection actually help?")
    print("=" * 78)
    print()
    print("'no_skip' performance per fold (from oracle_test_eval):")
    for r in fold_results:
        oracle = pd.DataFrame(r["oracle_test_eval"])
        nsk = oracle[oracle["rule"] == "no_skip"]
        if len(nsk):
            nsk_row = nsk.iloc[0]
            print(f"  Fold {r['fold']}: no_skip TEST Sharpe = "
                  f"{nsk_row['test_sharpe']:+.3f}, "
                  f"mean_arith = {nsk_row['test_mean_arith_pct']:+.3f}%, "
                  f"n_picks = {nsk_row['n_picks_test']}")
    print()
    print(f"Selected-rule vs no_skip:")
    for r in fold_results:
        oracle = pd.DataFrame(r["oracle_test_eval"])
        nsk = oracle[oracle["rule"] == "no_skip"].iloc[0] if len(oracle[oracle["rule"] == "no_skip"]) else None
        sel_rule = r["selected_rule"]
        selected_oracle = oracle[oracle["rule"] == sel_rule]
        if len(selected_oracle):
            sel_oracle = selected_oracle.iloc[0]
            d_sharpe = r["test_sharpe"] - nsk["test_sharpe"] if (nsk is not None and not pd.isna(r["test_sharpe"]) and not pd.isna(nsk["test_sharpe"])) else float("nan")
            print(f"  Fold {r['fold']}: "
                  f"selected='{sel_rule:11s}'  "
                  f"selected Sharpe={r['test_sharpe']:+.3f}  "
                  f"vs no_skip={nsk['test_sharpe']:+.3f}  "
                  f"delta={d_sharpe:+.3f}")

    # ------------------- Test H_B: convergence -------------------
    print("\n" + "=" * 78)
    print("H_B: Is the dead-zone shape a STABLE structural feature?")
    print("=" * 78)
    selected_rules = [r["selected_rule"] for r in fold_results]
    selected_excl = [r["selected_exclude_range"] for r in fold_results]
    print(f"\n  Selected rules per fold: {selected_rules}")
    print(f"  Selected exclude ranges per fold: {selected_excl}")
    if all(s == "no_skip" for s in selected_rules):
        print("\n >> H_B = FALSE: NO fold selected a dead-zone rule --")
        print("    >> The user is RIGHT and the (-10,-5) rule of Doc I ")
        print("    >> is an in-sample-only artifact.")
    elif sum(s != "no_skip" for s in selected_rules) == 1:
        print("\n  >> H_B = FALSE: Only 1 fold selected a dead-zone rule --")
        print("    >> No convergence. User critique confirmed.")
    else:
        # Check if the boundary is consistent
        excl_arr = [e for e in selected_excl if e is not None]
        n_selected = len(excl_arr)
        if n_selected >= 2:
            lo_arr = np.array([e[0] for e in excl_arr])
            hi_arr = np.array([e[1] for e in excl_arr])
            std_lo = float(np.std(lo_arr, ddof=1)) if len(lo_arr) >= 2 else 0.0
            std_hi = float(np.std(hi_arr, ddof=1)) if len(hi_arr) >= 2 else 0.0
            print(f"\n  >> H_B PARTIAL: {n_selected}/{len(folds)} folds selected a dead-zone rule.")
            print(f"    >> Selected lo boundaries: {[float(x) for x in lo_arr]} (std = {std_lo:.3f})")
            print(f"    >> Selected hi boundaries: {[float(x) for x in hi_arr]} (std = {std_hi:.3f})")
            # Convergence test: how many pairs of folds selected EXACTLY the same rule?
            from collections import Counter
            rule_counts = Counter(selected_rules)
            same_rule_pairs = sum(c * (c - 1) / 2 for c in rule_counts.values() if c >= 2)
            total_pairs = len(folds) * (len(folds) - 1) / 2
            print(f"    >> Same-rule pairs: {int(same_rule_pairs)}/{int(total_pairs)} possible.")

    # ------------------- Persist -------------------
    out_dir = HERE / f"phase_g_v1_1_deadzone_nested_cv_n{args.n_folds}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_folds": len(folds),
        "n_slots": args.n_slots,
        "n_rng_trials": args.n_rng_trials,
        "theta": NEG_THETA,
        "min_sweep_n": args.min_sweep_n,
        "gap_rules": [{"label": lbl, "exclude_range":
                       ([float(e[0]), float(e[1])] if e is not None else None)}
                      for lbl, e in GAP_RULES],
        "fold_hp": fold_hp,
        "fold_results": [
            {k: v for k, v in r.items()
             if k not in ("fold",)}
            | {"fold": r["fold"]}
            for r in fold_results
        ],
        "cross_fold_mean_sharpe": float(np.mean(sharpes)) if len(sharpes) else float("nan"),
        "cross_fold_std_sharpe": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else float("nan"),
        "cross_fold_mean_irr":  float(np.mean(irrs))    if len(irrs)    else float("nan"),
        "cross_fold_mean_max_dd": float(np.mean(max_dds)) if len(max_dds) else float("nan"),
        "cross_fold_mean_hit":   float(np.mean(hits))   if len(hits)   else float("nan"),
        "cross_fold_mean_avg_pnl": float(np.mean(avgpnls)) if len(avgpnls) else float("nan"),
        "selected_rules": selected_rules,
        "selected_exclude_ranges": selected_excl,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save the sweep_eval and oracle_test_eval per fold as CSV
    sweep_rows = []
    oracle_rows = []
    for r in fold_results:
        for s in r["sweep_eval"]:
            sr = {"fold": r["fold"]}; sr.update(s)
            sweep_rows.append(sr)
        for o in r["oracle_test_eval"]:
            orow = {"fold": r["fold"]}; orow.update(o)
            oracle_rows.append(orow)
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(
            out_dir / "sweep_eval_per_fold.csv", index=False)
    if oracle_rows:
        pd.DataFrame(oracle_rows).to_csv(
            out_dir / "oracle_test_eval_per_fold.csv", index=False)

    print(f"\nSaved artifacts to {out_dir}")
    print(f"  - summary.json   (cross-fold aggregates + selected rules)")
    print(f"  - sweep_eval_per_fold.csv ({len(sweep_rows)} rows)")
    print(f"  - oracle_test_eval_per_fold.csv ({len(oracle_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

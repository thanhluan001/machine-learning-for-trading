"""
Phase G v1.1 -- NEG-tuned retrain (negative-gap-tuned hyperparameters).

Per phase_g_findings.md §E.5.4 item (5): re-sweep hyperparameters
where SWEEP_VAL evaluates NEG_only PnL (rather than POS_only PnL)
as the selection criterion.

This script parallels `06_phase_g_nested_cv.py` (Appendix D), with
three differences:

1. SWEEP_VAL selection metric is NEG_only per-event PnL at
   theta=0.15, gap[-15%, -2%], n_trades >= 10 floor.
2. The final classifier is retrained on TRAIN+SWEEP_VAL with the
   NEG-selected HP.
3. TEST evaluation runs BOTH POS_only AND NEG_only -- so we can
   compare the NEG-tuned model against POS_only AND NEG_only on
   the same TEST slice, against Appendix E's POS-tuned baseline.

This builds the full 2x2 (POS-tuned vs NEG-tuned) x (POS_only vs
NEG_only) matrix:

  Tune        Pos-only        Neg-only
  POS-tuned   App D = App E   App E (NEW)
  NEG-tuned   THIS RUN         THIS RUN (NEW)

CLI:
  python luan_bot_trading/04_backtest/08_phase_g_neg_tuned.py
  python luan_bot_trading/04_backtest/08_phase_g_neg_tuned.py --n-folds 4 --n-slots 4

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

# NEG_only operating point (§4.4)
NEG_THETA = 0.15
NEG_GAP_LO = -0.15
NEG_GAP_HI = -0.02

# For comparison, also evaluate POS_only
POS_THETA = 0.20
POS_GAP_LO = 0.02
POS_GAP_HI = 0.15

# Focused sweep grid (4 configs) -- vary gamma only (same as App D)
SWEEP_GRID = [
    {"gamma": g, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
    for g in [3, 5, 10, 20]
]

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


def select_picks(df, proba, theta, gap_lo, gap_hi):
    """Generic trade picker: P(PEAD) >= theta AND gap in [gap_lo, gap_hi]."""
    df2 = df.copy()
    df2["pead_proba"] = proba
    mask = (
        (df2["pead_proba"] >= theta) &
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
    args = parser.parse_args(argv)

    folds = DEFAULT_FOLDS[:args.n_folds]

    print("=" * 78)
    print("PHASE G v1.1 -- NEG-TUNED RETRAIN (HP selection by NEG_only PnL)")
    print("=" * 78)
    print(f"Folds      : {len(folds)}")
    for i, (te, sve, tse) in enumerate(folds, 1):
        print(f"  Fold {i}: TRAIN 2015-01 -> {te} | SWEEP {te}->{sve} | "
              f"TEST {sve}->{tse}")
    print(f"N_SLOTS    : {args.n_slots}")
    print(f"N_RNG      : {args.n_rng_trials}")
    print(f"NEG selection criterion: P(PEAD)>={NEG_THETA}, "
          f"gap [{NEG_GAP_LO:+.2f},{NEG_GAP_HI:+.2f}], "
          f"floor n>=10")
    print(f"Also-eval POS_only    : P(PEAD)>={POS_THETA}, "
          f"gap [{POS_GAP_LO:+.2f},{POS_GAP_HI:+.2f}]")
    print(f"SWEEP GRID: gamma in {{3,5,10,20}} ({len(SWEEP_GRID)} configs/fold)")
    print("=" * 78)

    # --- Shared data load + gates + entry-PnL + paths + calendar ---
    print("\n[1] Loading train_matrix + §12 cutoff + gate computation ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    print(f"    rows after §12 cut: {len(df)}")
    df = pg.v3.compute_pead_gates_full(df)
    print(f"    pead_pass positives: {int(df['pead_pass'].sum())} "
          f"({df['pead_pass'].mean()*100:.2f}%)")

    print("\n[2] Computing entry-PnL + 12-snap trade paths ...")
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)
    print(f"    coverage: {int(df['path_pnl_t11_pct'].notna().sum())}/{len(df)}")

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

    # --- Per-fold: sweep on SWEEP_VAL using NEG_only PnL ---
    fold_results = []
    print(f"\n[4] Running {len(folds)} folds ...")
    for fold_idx, (train_end, sweep_end, test_end) in enumerate(folds, 1):
        print("\n" + "=" * 60)
        print(f"  FOLD {fold_idx}/{len(folds)}: "
              f"TRAIN <= {train_end} | SWEEP {train_end}->{sweep_end} | "
              f"TEST {sweep_end}->{test_end}")
        print("=" * 60)

        train_ts = pd.Timestamp(train_end)
        sweep_ts = pd.Timestamp(sweep_end)
        test_ts = pd.Timestamp(test_end)
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= train_ts].copy().reset_index(drop=True)
        sweep_df = df[(rd > train_ts) & (rd <= sweep_ts)].copy().reset_index(drop=True)
        test_df  = df[(rd > sweep_ts) & (rd <= test_ts)].copy().reset_index(drop=True)
        print(f"  TRAIN={len(train_df)}  SWEEP={len(sweep_df)}  TEST={len(test_df)}")

        X_tr = train_df[SUNDAY_SAFE_FEATURES].copy()
        y_tr = train_df["pead_pass"].astype(int).values
        X_sv = sweep_df[SUNDAY_SAFE_FEATURES].copy()
        y_sv = sweep_df["pead_pass"].astype(int).values
        X_te = test_df[SUNDAY_SAFE_FEATURES].copy()
        y_te = test_df["pead_pass"].astype(int).values

        # --- Inner sweep: train on TRAIN, evaluate NEG_only PnL on SWEEP_VAL ---
        print(f"\n  Sweeping {len(SWEEP_GRID)} configs using NEG_only PnL on SWEEP_VAL ...")
        sweep_results = []
        for hp_idx, hp in enumerate(SWEEP_GRID):
            t0 = time.time()
            clf = fit_classifier(X_tr, y_tr, X_sv, y_sv, hp)
            proba_sv = clf.predict_proba(X_sv)[:, 1]
            neg_picks = select_picks(sweep_df, proba_sv,
                                     NEG_THETA, NEG_GAP_LO, NEG_GAP_HI)
            if len(neg_picks) >= 1:
                arith = np.expm1(neg_picks["path_pnl_t11_pct"])
                neg_sweep_pnl = float(arith.mean()) * 100
                neg_sweep_hit = float((arith > 0).mean()) * 100
                neg_sweep_n = len(neg_picks)
            else:
                neg_sweep_pnl = float("nan")
                neg_sweep_hit = float("nan")
                neg_sweep_n = 0
            # also compute POS_only PnL just for visibility/comparison
            pos_picks = select_picks(sweep_df, proba_sv,
                                     POS_THETA, POS_GAP_LO, POS_GAP_HI)
            if len(pos_picks) >= 1:
                arith_p = np.expm1(pos_picks["path_pnl_t11_pct"])
                pos_sweep_pnl = float(arith_p.mean()) * 100
                pos_sweep_n = len(pos_picks)
            else:
                pos_sweep_pnl = float("nan")
                pos_sweep_n = 0
            from sklearn.metrics import roc_auc_score
            auc_sv = roc_auc_score(y_sv, proba_sv)
            sweep_results.append({
                "hp_idx": hp_idx, "hp": hp,
                "neg_sweep_n": neg_sweep_n,
                "neg_sweep_pnl_pct": neg_sweep_pnl,
                "neg_sweep_hit_pct": neg_sweep_hit,
                "pos_sweep_n": pos_sweep_n,
                "pos_sweep_pnl_pct": pos_sweep_pnl,
                "auc_sweep_val": auc_sv,
            })
            print(f"    gamma={hp['gamma']:>3d}  "
                  f"NEG n={neg_sweep_n:>2d}  NEG pnl={neg_sweep_pnl:>+6.3f}%  "
                  f"POS n={pos_sweep_n:>3d}  POS pnl={pos_sweep_pnl:>+6.3f}%  "
                  f"auc_sv={auc_sv:.4f}  t={time.time()-t0:.1f}s")

        # --- Select best HP: max NEG sweep_val pnl, floor n>=10 ---
        # If no config satisfies n>=10, fall back to gamma=5
        valid = [r for r in sweep_results
                 if r["neg_sweep_n"] >= 10 and not np.isnan(r["neg_sweep_pnl_pct"])]
        if valid:
            best = max(valid, key=lambda r: r["neg_sweep_pnl_pct"])
            sel_hp = best["hp"]
            sel_reason = (f"best NEG sweep_val pnl "
                          f"(n={best['neg_sweep_n']}, "
                          f"pnl={best['neg_sweep_pnl_pct']:+.3f}%)")
        else:
            # If NO config has n>=10, drop the floor to n>=5, then n>=1
            looser = [r for r in sweep_results
                      if r["neg_sweep_n"] >= 5
                      and not np.isnan(r["neg_sweep_pnl_pct"])]
            if looser:
                best = max(looser, key=lambda r: r["neg_sweep_pnl_pct"])
                sel_hp = best["hp"]
                sel_reason = (f"floor lowered to n>=5; "
                              f"NEG sweep_val n={best['neg_sweep_n']}, "
                              f"pnl={best['neg_sweep_pnl_pct']:+.3f}%")
            else:
                any_pos = [r for r in sweep_results
                           if r["neg_sweep_n"] >= 1
                           and not np.isnan(r["neg_sweep_pnl_pct"])]
                if any_pos:
                    best = max(any_pos, key=lambda r: r["neg_sweep_pnl_pct"])
                    sel_hp = best["hp"]
                    sel_reason = (f"floor to n>=1; "
                                  f"NEG sweep n={best['neg_sweep_n']}, "
                                  f"pnl={best['neg_sweep_pnl_pct']:+.3f}%")
                else:
                    sel_hp = {"gamma": 5, "min_child_weight": 50,
                              "max_depth": 3, "n_estimators": 300}
                    sel_reason = "fallback gamma=5 (no NEG picks in SWEEP_VAL)"
        print(f"\n  Selected HP: gamma={sel_hp['gamma']}, "
              f"mcw={sel_hp['min_child_weight']}, "
              f"md={sel_hp['max_depth']}, n_est={sel_hp['n_estimators']}")
        print(f"  Reason: {sel_reason}")

        # --- Retrain on TRAIN+SWEEP_VAL with selected HP ---
        X_ts = pd.concat([X_tr, X_sv], axis=0).reset_index(drop=True)
        y_ts = np.concatenate([y_tr, y_sv])
        print(f"\n  Retraining on TRAIN+SWEEP_VAL ({len(y_ts)} rows) ...")
        clf_final = fit_classifier(X_ts, y_ts, X_te, y_te, sel_hp)
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_test = roc_auc_score(y_te, clf_final.predict_proba(X_te)[:, 1])
        ap_test = average_precision_score(y_te, clf_final.predict_proba(X_te)[:, 1])
        print(f"  TEST AUC: {auc_test:.4f}  AP: {ap_test:.4f}")

        # --- Evaluate both POS_only and NEG_only on TEST ---
        print(f"\n  Portfolio eval on TEST (n_slots={args.n_slots}) ...")
        proba_te = clf_final.predict_proba(X_te)[:, 1]

        rule_results = {}
        for rule_name, theta, glo, ghi in [
            ("POS_only", POS_THETA, POS_GAP_LO, POS_GAP_HI),
            ("NEG_only", NEG_THETA, NEG_GAP_LO, NEG_GAP_HI),
        ]:
            picks = select_picks(test_df, proba_te, theta, glo, ghi)
            n_picks = len(picks)
            if n_picks >= 1:
                mean_arith = float(np.expm1(picks["path_pnl_t11_pct"]).mean() * 100)
                hit_pct = float((np.expm1(picks["path_pnl_t11_pct"]) > 0).mean() * 100)
            else:
                mean_arith = float("nan")
                hit_pct = float("nan")
            result = rb._simulate_with_cached_calendar(
                picks, args.n_slots, args.initial_nav, calendar)
            s = result.get("summary", {})
            print(f"    {rule_name:>10s}: n_picks={n_picks:>3d}  "
                  f"mean_arith_pnl={mean_arith:+.3f}%  hit={hit_pct:>5.1f}%  "
                  f"trades_exec={s.get('n_trades_executed', 0):>3d}  "
                  f"IRR={s.get('irr_pct', float('nan')):+.2f}%  "
                  f"Sharpe={s.get('sharpe_liq_annualized', float('nan')):+.2f}  "
                  f"MaxDD={s.get('max_drawdown_pct', float('nan')):.2f}%")
            rule_results[rule_name] = {
                "n_picks": n_picks,
                "mean_arith_pnl_pct": mean_arith,
                "hit_pct": hit_pct,
                **{k: v for k, v in s.items()},
            }

        # --- Random baseline ---
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
        print(f"  Random mean: Sharpe={rand_mean_sharpe:+.2f}  "
              f"IRR={rand_mean_irr:+.2f}%  n={len(rand_df)}")

        # Exceedance fractions per rule
        rule_exceed = {}
        for rule_name in ("POS_only", "NEG_only"):
            rr = rule_results[rule_name]
            rsh = rr.get("sharpe_liq_annualized", float("nan"))
            rir = rr.get("irr_pct", float("nan"))
            if len(rand_df) and not np.isnan(rsh):
                frac_ex_sh = 1.0 - float((rand_df["sharpe"] > rsh).mean())
            else:
                frac_ex_sh = float("nan")
            if len(rand_df) and not np.isnan(rir) and not rand_df['irr'].isna().all():
                frac_ex_ir = 1.0 - float((rand_df["irr"] > rir).mean())
            else:
                frac_ex_ir = float("nan")
            rule_exceed[rule_name] = {
                "frac_random_exceeded_sharpe": frac_ex_sh,
                "frac_random_exceeded_irr": frac_ex_ir,
            }
            print(f"    {rule_name:>10s}: exceeds {frac_ex_sh*100:>5.1f}% "
                  f"random Sharpe / {frac_ex_ir*100:>5.1f}% random IRR")

        fold_results.append({
            "fold": fold_idx,
            "train_end": train_end,
            "sweep_end": sweep_end,
            "test_end": test_end,
            "test_slice": f"{sweep_end}->{test_end}",
            "selected_hp": sel_hp,
            "selected_hp_reason": sel_reason,
            "sweep_results": sweep_results,
            "test_auc": float(auc_test),
            "test_ap": float(ap_test),
            "test_n_rows": len(test_df),
            "test_n_pos": int(test_df["pead_pass"].sum()),
            "random_mean_sharpe": rand_mean_sharpe,
            "random_mean_irr": rand_mean_irr,
            "random_n": len(rand_df),
            "rules": {
                rn: {**rule_results[rn], **rule_exceed[rn]}
                for rn in ("POS_only", "NEG_only")
            },
        })

    # --- Aggregate ---
    print("\n" + "=" * 78)
    print("NEG-TUNED RETRAIN -- CROSS-FOLD RESULTS")
    print("=" * 78)

    rows = []
    for fr in fold_results:
        for rn in ("POS_only", "NEG_only"):
            r = fr["rules"][rn]
            row = {"fold": fr["fold"], "rule": rn,
                   "test_slice": fr["test_slice"],
                   "sel_gamma": fr["selected_hp"]["gamma"]}
            for m in ["n_trades_executed", "irr_pct", "sharpe_liq_annualized",
                      "max_drawdown_pct", "hit_rate_pct", "avg_trade_pnl_pct",
                      "mean_arith_pnl_pct",
                      "frac_random_exceeded_sharpe", "frac_random_exceeded_irr"]:
                row[m] = r.get(m, float("nan"))
            rows.append(row)
    wide = pd.DataFrame(rows)

    # Print per-rule per-fold
    for rn in ("POS_only", "NEG_only"):
        print(f"\n--- Rule: {rn} ---")
        sub = wide[wide["rule"] == rn].copy()
        print(f"{'Fold':>4}  {'slice':<24}  {'gamma':>5}  {'n_tr':>4}  "
              f"{'IRR%':>7}  {'Sharpe':>7}  {'MaxDD%':>7}  {'hit%':>5}  "
              f"{'avgPnL%':>8}  {'%rShEx':>7}  {'%rIREx':>7}")
        print("-" * 110)
        for _, r in sub.iterrows():
            print(f"  {int(r['fold']):>2d}  {r['test_slice']:>22s}  "
                  f"{int(r['sel_gamma']):>3d}    "
                  f"{int(r['n_trades_executed']):>3d}  "
                  f"{r['irr_pct']:>+6.2f}%  "
                  f"{r['sharpe_liq_annualized']:>+6.2f}  "
                  f"{r['max_drawdown_pct']:>+6.2f}%  "
                  f"{r['hit_rate_pct']:>4.1f}%  "
                  f"{r['avg_trade_pnl_pct']:>+7.3f}%  "
                  f"{r['frac_random_exceeded_sharpe']*100:>5.1f}%  "
                  f"{r['frac_random_exceeded_irr']*100:>5.1f}%")
        print(f"  {'AVG':>4s}  gamma={sub['sel_gamma'].mean():>4.1f}  "
              f"n_tr={sub['n_trades_executed'].mean():>5.1f}  "
              f"IRR={sub['irr_pct'].mean():>+6.2f}%  "
              f"Sharpe={sub['sharpe_liq_annualized'].mean():>+6.2f}  "
              f"MaxDD={sub['max_drawdown_pct'].mean():>+6.2f}%  "
              f"hit={sub['hit_rate_pct'].mean():>4.1f}%  "
              f"avgPnL={sub['avg_trade_pnl_pct'].mean():>+7.3f}%  "
              f"%rShEx={sub['frac_random_exceeded_sharpe'].mean()*100:>5.1f}%  "
              f"%rIREx={sub['frac_random_exceeded_irr'].mean()*100:>5.1f}%")

    # Compact 2x2 vs Appendix E (POS-tuned)
    print("\n--- Compact 2x2 (Tune x Rule) mean across 4 folds ---")
    print(f"{'Tune':<10}  {'Rule':<10}  {'IRR%':>7}  {'Sharpe':>7}  "
          f"{'MaxDD%':>7}  {'hit%':>5}  {'avgPnL%':>8}  {'%rShEx':>7}  "
          f"{'%rIREx':>7}  {'n_tr':>5}")
    # Print POS-tuned values (from Appendix E -- recompute via the POS_only
    # row already produced by THIS run, since the classifier is identical to
    # Appendix E's when gamma happens to match the POS-tuned per-fold selection)
    pos_tuned_neg = [
        # (mean IRR, mean Sharpe, mean MaxDD, mean hit, mean avgPnL, mean %rShEx, mean %rIREx, n_tr)
        # From Appendix E (run 07_phase_g_ensemble.py): NEG_only means
        (+14.36, +1.01, -6.94, 58.1, +1.70, 80.3, 74.2, 13.0),
    ]
    pos_tuned_pos = [
        (+13.06, +0.86, -10.37, 54.9, +1.14, 58.2, 57.2, 15.2),
    ]
    print(f"  {'POS-tuned':<10}  {'POS_only':<10}  "
          f"{pos_tuned_pos[0][0]:>+6.2f}  {pos_tuned_pos[0][1]:>+6.2f}  "
          f"{pos_tuned_pos[0][2]:>+6.2f}  {pos_tuned_pos[0][3]:>4.1f}  "
          f"{pos_tuned_pos[0][4]:>+7.2f}  {pos_tuned_pos[0][5]:>5.1f}  "
          f"{pos_tuned_pos[0][6]:>5.1f}  {pos_tuned_pos[0][7]:>5.1f}  (from App D)")
    print(f"  {'POS-tuned':<10}  {'NEG_only':<10}  "
          f"{pos_tuned_neg[0][0]:>+6.2f}  {pos_tuned_neg[0][1]:>+6.2f}  "
          f"{pos_tuned_neg[0][2]:>+6.2f}  {pos_tuned_neg[0][3]:>4.1f}  "
          f"{pos_tuned_neg[0][4]:>+7.2f}  {pos_tuned_neg[0][5]:>5.1f}  "
          f"{pos_tuned_neg[0][6]:>5.1f}  {pos_tuned_neg[0][7]:>5.1f}  (from App E)")

    for rn in ("POS_only", "NEG_only"):
        sub = wide[wide["rule"] == rn]
        print(f"  {'NEG-tuned':<10}  {rn:<10}  "
              f"{sub['irr_pct'].mean():>+6.2f}  "
              f"{sub['sharpe_liq_annualized'].mean():>+6.2f}  "
              f"{sub['max_drawdown_pct'].mean():>+6.2f}  "
              f"{sub['hit_rate_pct'].mean():>4.1f}  "
              f"{sub['avg_trade_pnl_pct'].mean():>+7.2f}  "
              f"{sub['frac_random_exceeded_sharpe'].mean()*100:>5.1f}  "
              f"{sub['frac_random_exceeded_irr'].mean()*100:>5.1f}  "
              f"{sub['n_trades_executed'].mean():>5.1f}  (this run)")

    # Persist
    out_dir = HERE / f"phase_g_v1_1_neg_tuned_n{args.n_folds}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_dir / "neg_tuned_wide.csv", index=False)
    summary = {
        "n_folds": len(fold_results),
        "n_slots": args.n_slots,
        "n_rng_trials": args.n_rng_trials,
        "sweep_grid": SWEEP_GRID,
        "neg_op_point": {"theta": NEG_THETA,
                         "gap_lo": NEG_GAP_LO, "gap_hi": NEG_GAP_HI},
        "pos_op_point": {"theta": POS_THETA,
                         "gap_lo": POS_GAP_LO, "gap_hi": POS_GAP_HI},
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
            for rn in ("POS_only", "NEG_only")
        },
        "comparison_to_appendix_e_pos_tuned": {
            "POS_only (App D)": pos_tuned_pos[0],
            "NEG_only (App E)": pos_tuned_neg[0],
        },
        "created_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

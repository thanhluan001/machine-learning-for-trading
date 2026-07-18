#!/usr/bin/env python3
"""
Stage 4: Backtest harness for Phase F v2 models  [VAL period]
====================================================================

PURPOSE
-------
Walk the VAL-period calendar_week_groups chronologically. For each week,
score all earnings events with the trained ranker, select top-N long
positions (equal-weighted), and compute realized per-event PnL from
`car_10d` (the 10-trading-day abnormal log return vs the IJH benchmark
already stored in /features/train_matrix).

PORTFOLIO CONVENTIONS (Phase F v2 baseline; user-confirmed)
-----------------------------------------------------------
  * Universe         : /features/train_matrix rows with report_date > split_date.
  * Rebalancing      : every calendar_week_group in chronological order.
  * Selection        : top-N long positions by predicted raw rank score.
  * Sizing           : equal-weight (f = 1/N).
  * Holding period   : 10 trading days (implied by `car_10d` window).
  * Per-event PnL     : arithmetic form of stored car_10d, i.e. `np.expm1(car_10d)`.
                       For NaN car_10d rows, treated as 0 PnL (no contribution).
  * Baseline SPLIT_DATE: 2024-01-01 (same as Stage 3 walk-forward).
  * No transaction costs; no slippage. Pure stat-arb simulation.

PERFORMANCE METRICS
--------------------
  * Cumulative log PnL and arithmetic cumulative PnL
  * Mean weekly log return, std, Sharpe (weekly / annualized)
  * Win rate (weeks with positive return)
  * Hit rate of selected events (selected events with positive car_10d)
  * Max drawdown over the VAL period
  * Average per-event contribution

CLI
---
    python luan_bot_trading/04_backtest/01_val_backtest.py
    python luan_bot_trading/04_backtest/01_val_backtest.py --model-dir ...\\phase_f_v2_baseline_ndcg
    python luan_bot_trading/04_backtest/01_val_backtest.py --top-n 5
    python luan_bot_trading/04_backtest/01_backtest.py --model-dir foo --baseline-dd RAND-COMPARE

This is a BASELINE v2 backtest. Tuning, transaction costs, and Kelly sizing
are deferred (computes baseline envelope only). See STOP_DOING_EXTRA_SHIT.md
sec "do exactly what is asked".

HDF5 WRITE SAFETY
-----------------
NONE — this script is READ-ONLY on db.h5. All output is printed to stdout.
No persistence (user decided no DB writes; tuning later).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# Allow utf-8 stdout on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


HERE = Path(__file__).resolve().parent
TRAIN_SCRIPT = HERE.parent / "03_model" / "01_train_model.py"

# Load the stage-3 module (re-use its data path + helpers)
spec = importlib.util.spec_from_file_location("train_model", TRAIN_SCRIPT)
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)


def load_model(model_dir: Path) -> tuple:
    """Load (ranker, calibrator, meta) from a Stage-3 model directory.

    ranker: xgboost.XGBRanker, restored from ranker.json.
    calibrator: sklearn.isotonic.IsotonicRegression restored from calibrator.pkl.
    meta: dict from meta.json.
    """
    import xgboost as xgb  # local for faster startup
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    ranker_path = model_dir / "ranker.json"
    calib_path = model_dir / "calibrator.pkl"
    meta_path = model_dir / "meta.json"
    if not all(p.exists() for p in (ranker_path, calib_path, meta_path)):
        raise FileNotFoundError(
            f"Missing one of ranker.json / calibrator.pkl / meta.json in {model_dir}"
        )
    ranker = xgb.XGBRanker()
    ranker.load_model(str(ranker_path))
    with open(calib_path, "rb") as f:
        calib = pickle.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return ranker, calib, meta


def build_val_table(split_date: str) -> pd.DataFrame:
    """Load /features/train_matrix, apply §12 priming cutoff (Stage-3 design),
    drop sparse weeks (min_group_size=3), restrict VAL to report_date > split_date.

    Returns the VAL rows sorted by [calendar_week_group, permaTicker, report_date].
    """
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    train_df, val_df = tm.split_walk_forward(df, split_date)
    train_df, _ = tm.drop_sparse_weeks(train_df, tm.DEFAULT_MIN_GROUP_SIZE)
    val_df, _ = tm.drop_sparse_weeks(val_df, tm.DEFAULT_MIN_GROUP_SIZE)
    val_df = val_df.sort_values(
        [tm.GROUP_COLUMN, "permaTicker", "report_date"]
    ).reset_index(drop=True)
    return val_df


def backtest_top_n(
    ranker,
    val_df: pd.DataFrame,
    top_n: int,
) -> dict:
    """Walk VAL weeks chronologically. Each week, score all events with the
    ranker's predict(). Take top-N by predicted score (= highest expected
    post-event CAR). Equal-weight. Compute per-event realized CAR via
    expm1(car_10d). Aggregate weekly portfolio arithmetic returns.

    Returns dict with per-week portfolio returns, total cum_pnl (arithmetic),
    log PnL, hit rates, etc.
    """
    feature_cols = tm.FEATURE_COLUMNS
    label_col = "car_10d"
    group_col = "calendar_week_group"

    per_week_rows = []  # list of dicts, one per week
    per_event_pnls = []  # per selected event

    grouped = val_df.groupby(group_col, sort=True)
    for week, g in grouped:
        if len(g) < 1:
            continue
        X = g[feature_cols].copy()
        raw_scores = ranker.predict(X)
        # Sort by predicted score desc; pick top_n
        order_idx = np.argsort(-raw_scores)
        # Cap n at the actual group size
        n_pick = min(top_n, len(g))
        top_idx = order_idx[:n_pick]
        # Subset selected events
        sel_rows = g.iloc[top_idx]
        sel_scores = raw_scores[top_idx]
        # Per-event realized arithmetic CAR: expm1(car_10d)
        sel_labels = pd.to_numeric(sel_rows[label_col], errors="coerce").to_numpy()
        # NaN labels -> 0 contribution (no position taken that day)
        sel_arith = np.expm1(np.nan_to_num(sel_labels, nan=0.0))
        # Equal-weight arithmetic PnL for the week
        # NB: meaning is "average return over the 10-day window per week"
        week_pnl_arith = float(np.mean(sel_arith))
        # Log-form aggregate via mean of log CARs (more comparable across weeks)
        sel_log = np.nan_to_num(sel_labels, nan=0.0)
        week_pnl_log = float(np.mean(sel_log))
        # Hit rate (fraction of selected events with positive arithmetic CAR)
        n_hits = int(np.sum(sel_arith > 0))
        n_valid = int(np.sum(~np.isnan(sel_labels)))
        per_week_rows.append({
            "calendar_week_group": week,
            "n_events": int(len(g)),
            "n_pick": int(n_pick),
            "n_hits": n_hits,
            "n_valid_labels": n_valid,
            "pnl_arith_mean": week_pnl_arith,
            "pnl_log_mean": week_pnl_log,
            "mean_raw_score": float(np.mean(sel_scores)),
            # Track individual selected-event pnls for diagnostics
        })
        for label, score in zip(sel_labels, sel_scores):
            per_event_pnls.append({"label_log": label, "raw_score": float(score)})

    weeks_df = pd.DataFrame(per_week_rows)
    events_df = pd.DataFrame(per_event_pnls)
    return {"weeks": weeks_df, "events": events_df}


def aggregate_metrics(bt_result: dict, top_n: int) -> dict:
    """Compute summary metrics from backtest output."""
    w = bt_result["weeks"]
    e = bt_result["events"]
    if w.empty:
        return {"error": "no weeks"}

    n_weeks = int(len(w))
    n_events_avg = float(w["n_events"].mean())

    # Aggregate log returns (sum across weeks -- naive compounding). 
    # DOES NOT MODEL REAL CAPITAL ALLOCATION ACROSS OVERLAPPING 10-DAY HOLDS.
    # These cumulative figures are informational only; true PnL requires a 
    # multi-period-position portfolio simulator (out of baseline scope).
    sum_log = float(w["pnl_log_mean"].sum())
    cum_log = sum_log  # log PnL accumulator (informational only)
    cum_arith = float(np.expm1(sum_log))  # geometric cumulative arithmetic (informational)

    # Mean / std of weekly arithmetic returns -- raw signal-strength measure
    mean_arith = float(w["pnl_arith_mean"].mean())
    std_arith = float(w["pnl_arith_mean"].std(ddof=1))
    # Sharpe (weekly -> annualized by sqrt(52)). NOTE: this is the PER-WEEK 
    # portfolio arithmetic edge, NOT a true portfolio Sharpe with overlapping
    # holds. Use it as a relative signal-strength metric vs the random trial
    # distribution with the same caveat.
    sharpe_weekly = (mean_arith / std_arith) * np.sqrt(52.0) if std_arith > 0 else float("nan")

    # Win rate: weeks with positive arith pnl
    n_win_weeks = int((w["pnl_arith_mean"] > 0).sum())
    win_rate_week = n_win_weeks / n_weeks if n_weeks else float("nan")

    # Per-event hit rate -- most reliable edge metric on this baseline.
    if e.empty:
        per_event_hit_rate = float("nan")
        n_total_events = 0
    else:
        labels = e["label_log"].dropna()
        n_total_events = int(len(labels))
        n_hit_events = int((labels > 0).sum())
        per_event_hit_rate = n_hit_events / n_total_events if n_total_events else float("nan")
    # Max drawdown on cumulative log PnL stream (informational only)
    log_stream = np.cumsum(w["pnl_log_mean"].to_numpy())
    running_max = np.maximum.accumulate(log_stream)
    drawdown = running_max - log_stream
    max_drawdown = float(np.max(drawdown)) if len(drawdown) else float("nan")
    first_week = w["calendar_week_group"].iloc[0]
    last_week = w["calendar_week_group"].iloc[-1]
    # NEW: mean per-event realized arith PnL (the un-scaled per-event signal)
    if not e.empty:
        ev_arith = np.expm1(e['label_log'].fillna(0))
        mean_event_arith = float(ev_arith.mean())
        median_event_arith = float(ev_arith.median())
    else:
        mean_event_arith = float("nan")
        median_event_arith = float("nan")
    return {
        "top_n": top_n,
        "n_weeks": n_weeks,
        "n_events_avg": n_events_avg,
        "n_total_events": n_total_events,
        "date_range": f"{first_week} -> {last_week}",
        "mean_event_arith": mean_event_arith,
        "median_event_arith": median_event_arith,
        "mean_arith_week": mean_arith,
        "std_arith_week": std_arith,
        "cum_log_pnl": cum_log,
        "cum_arith_pnl": cum_arith,
        "sharpe_weekly_annualized": sharpe_weekly,
        "win_rate_week": win_rate_week,
        "per_event_hit_rate": per_event_hit_rate,
        "max_drawdown_log": max_drawdown,
        "max_drawdown_arith": float(np.expm1(-max_drawdown) - 1) if not np.isnan(max_drawdown) else float("nan"),
    }


def backtest_random_baseline(val_df: pd.DataFrame, top_n: int,
                             n_trials: int = 100, seed: int = 42) -> dict:
    """Compute random-selection baseline: at each week, pick top_n events
    uniformly at random; report mean sharpe and PnL across n_trials.

    This gives a null distribution to compare the model against, since the
    VAL-period overfit gap suggests weak signal that may be hard to distinguish
    from random.
    """
    rng = np.random.default_rng(seed)
    per_trial_aggs = []
    for t in range(n_trials):
        # Simulate random selection per week; use a fixed seed for reproducibility
        # REUSE build_val_table: we use the SAME val_df shuffled per week, no
        # ranker score, just rng.choice.
        per_week_rows = []
        grouped = val_df.groupby("calendar_week_group", sort=True)
        for week, g in grouped:
            if len(g) < 1:
                continue
            n_pick = min(top_n, len(g))
            top_idx = rng.choice(len(g), size=n_pick, replace=False)
            sel_rows = g.iloc[top_idx]
            sel_labels = pd.to_numeric(sel_rows["car_10d"], errors="coerce").to_numpy()
            sel_arith = np.expm1(np.nan_to_num(sel_labels, nan=0.0))
            week_pnl_arith = float(np.mean(sel_arith))
            week_pnl_log = float(np.mean(np.nan_to_num(sel_labels, nan=0.0)))
            per_week_rows.append({
                "calendar_week_group": week,
                "pnl_arith_mean": week_pnl_arith,
                "pnl_log_mean": week_pnl_log,
            })
        w = pd.DataFrame(per_week_rows)
        if w.empty:
            continue
        sum_log = float(w["pnl_log_mean"].sum())
        mean_a = float(w["pnl_arith_mean"].mean())
        std_a = float(w["pnl_arith_mean"].std(ddof=1))
        sharpe_ann = (mean_a / std_a) * np.sqrt(52.0) if std_a > 0 else float("nan")
        per_trial_aggs.append({"sum_log": sum_log, "sharpe": sharpe_ann,
                               "mean_arith": mean_a, "std_arith": std_a,
                               "win_rate": float((w["pnl_arith_mean"] > 0).mean())})
    if not per_trial_aggs:
        return {"error": "no trials"}
    pa = pd.DataFrame(per_trial_aggs)
    return {
        "n_trials": int(len(pa)),
        "top_n": top_n,
        "mean_sharpe_annualized": float(pa["sharpe"].mean()),
        "median_sharpe_annualized": float(pa["sharpe"].median()),
        "std_sharpe_annualized": float(pa["sharpe"].std(ddof=1)),
        "p_sharpe_gt_zero": float((pa["sharpe"] > 0).mean()),
        "mean_cum_log": float(pa["sum_log"].mean()),
        "median_cum_log": float(pa["sum_log"].median()),
        "mean_win_rate": float(pa["win_rate"].mean()),
    }


def run_for_model(model_dir: Path, top_n: int, split_date: str,
                  include_random: bool = True) -> dict:
    """Run the backtest for a single model directory. Print summary to stdout."""
    ranker, calib, meta = load_model(model_dir)
    val_df = build_val_table(split_date)
    obj = meta["xgb_params"]["objective"]
    print(f"\n{'=' * 70}")
    print(f"BACKTEST: {model_dir.name}")
    print(f"  model objective: {obj}  max_depth={meta['xgb_params'].get('max_depth')}")
    print(f"  top_n (long positions per week): {top_n}")
    print(f"  VAL rows: {len(val_df)}  weeks: {val_df['calendar_week_group'].nunique()}")
    print(f"  VAL date range: {val_df['report_date'].min()} -> {val_df['report_date'].max()}")
    print(f"{'=' * 70}")

    bt = backtest_top_n(ranker, val_df, top_n)
    agg = aggregate_metrics(bt, top_n)
    print("MODEL performance:")
    print(f"  Weeks traded                  : {agg['n_weeks']}")
    print(f"  Avg events per week           : {agg['n_events_avg']:.2f}")
    print(f"  Total selected events         : {agg['n_total_events']}")
    print(f"  Date range                    : {agg['date_range']}")
    print()
    print("  --- Core edge metrics (primary truth) ---")
    print(f"  Per-event hit rate             : {agg['per_event_hit_rate']*100:.1f}%")
    print(f"  Mean per-event arith PnL       : {agg['mean_event_arith']*100:+.3f}%")
    print(f"  Median per-event arith PnL     : {agg['median_event_arith']*100:+.3f}%")
    print(f"  Week win rate                 : {agg['win_rate_week']*100:.1f}%")
    print()
    print("  --- Weekly signal-strength (NOT a true portfolio Sharpe) ---")
    print(f"  Mean weekly arith PnL         : {agg['mean_arith_week']*100:+.5f}%")
    print(f"  Std  weekly arith PnL         : {agg['std_arith_week']*100:+.5f}%")
    print(f"  Sharpe (weekly, sqrt(52)-annualized): {agg['sharpe_weekly_annualized']:+.4f}")
    print()
    print("  --- Informational naive compounding (informational only) ---")
    print(f"  Cumulative log PnL (naive)    : {agg['cum_log_pnl']*100:+.2f}% (log units)")
    print(f"  Cumulative arith PnL (naive)  : {agg['cum_arith_pnl']*100:+.2f}%")
    print(f"  Max drawdown (log)            : {agg['max_drawdown_log']:+.5f}")
    print(f"  Max drawdown (arith, naive)   : {agg['max_drawdown_arith']*100:+.2f}%")

    if include_random:
        print("\nComputing random baseline (100 trials, identical setup)...")
        rand = backtest_random_baseline(val_df, top_n, n_trials=100)
        print("RANDOM baseline (100 trials):")
        print(f"  Mean Sharpe annualized        : {rand['mean_sharpe_annualized']:+.4f} (std over trials: {rand['std_sharpe_annualized']:.4f})")
        print(f"  Median Sharpe                 : {rand['median_sharpe_annualized']:+.4f}")
        print(f"  P(trial Sharpe > 0)           : {rand['p_sharpe_gt_zero']*100:.1f}%")
        print(f"  Mean cumulative log PnL       : {rand['mean_cum_log']:+.5f}")
        print(f"  Median cumulative log PnL     : {rand['median_cum_log']:+.5f}")
        print(f"  Mean week win rate            : {rand['mean_win_rate']*100:.1f}%")
        # Model vs random quantile
        model_sharpe = agg["sharpe_weekly_annualized"]
        # Re-derive trial list to compare
        # Just print rank against approx random distribution
        # Recompute with fresh rng for fair comparison
        print("\n  Model Sharpe vs random trial distribution:")
        # Compute p-value (one-sided): how many trials >= model Sharpe?
        rng_seed = 42
        rng = np.random.default_rng(rng_seed)
        trial_sharpes = []
        for _ in range(100):
            per_week_rows = []
            grouped = val_df.groupby("calendar_week_group", sort=True)
            for week, g in grouped:
                if len(g) < 1: continue
                n_pick = min(top_n, len(g))
                idx = rng.choice(len(g), size=n_pick, replace=False)
                sel = g.iloc[idx]
                labels = pd.to_numeric(sel["car_10d"], errors="coerce").to_numpy()
                arith = np.expm1(np.nan_to_num(labels, nan=0.0))
                wpa = float(np.mean(arith))
                wpl = float(np.mean(np.nan_to_num(labels, nan=0.0)))
                per_week_rows.append({"pnl_arith_mean": wpa, "pnl_log_mean": wpl})
            w = pd.DataFrame(per_week_rows)
            if w.empty: continue
            m = float(w["pnl_arith_mean"].mean())
            s = float(w["pnl_arith_mean"].std(ddof=1))
            trial_sharpes.append((m / s) * np.sqrt(52) if s > 0 else float("nan"))
        trial_sharpes = np.array(trial_sharpes)
        if not np.isnan(model_sharpe):
            p_val = float((trial_sharpes >= model_sharpe).mean())
            print(f"    P(random trial Sharpe >= model Sharpe)  =  {p_val:.2f}")
            # Percentile of model sharpe vs random dist
            pct = float((trial_sharpes < model_sharpe).mean() * 100)
            print(f"    Model Sharpe {model_sharpe:+.4f} is at the {pct:.0f}th percentile of random trials")

    return {"model_name": model_dir.name, "metrics": agg}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 4 baseline backtest: VAL period top-N long only "
                    "(equal-weight) on Phase F v2 trained ranks."
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to model dir (containing ranker.json, calibrator.pkl, meta.json). "
             "If omitted, BOTH phase_f_v2_baseline_ndcg and phase_f_v2_baseline_pairwise "
             "are tested in sequence.",
    )
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top-ranked events to long each week (default 5).")
    parser.add_argument("--split", type=str, default=tm.DEFAULT_SPLIT_DATE,
                        help=f"VAL starts after this date (inclusive val: > split). "
                             f"Default {tm.DEFAULT_SPLIT_DATE}.")
    parser.add_argument("--no-random", action="store_true",
                        help="Skip random-baseline 100-trial comparison.")
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip top-N sensitivity sweep (n=1,3,5,10,20).")
    args = parser.parse_args(argv)

    print(f"VAL split = report_date > {args.split}")
    print(f"Top-N long  = {args.top_n}")

    if args.model_dir is None:
        models_root = HERE.parent / "03_model" / "models"
        dirs = [models_root / "phase_f_v2_baseline_ndcg",
                models_root / "phase_f_v2_baseline_pairwise"]
    else:
        dirs = [Path(args.model_dir)]

    results = [run_for_model(d, args.top_n, args.split,
                            include_random=(not args.no_random)) for d in dirs]

    if len(results) > 1:
        print("\n" + "=" * 78)
        print("COMPARISON SUMMARY (val period) -- top-N long only")
        print("=" * 78)
        h = f"{'model':35s} {'Sharpe':>7s} {'Hit%':>6s} {'Avg event':>10s} {'WinWeek%':>9s}"
        print(h)
        print("-" * 78)
        for r in results:
            m = r["metrics"]
            print(f"{r['model_name']:35s} "
                  f"{m['sharpe_weekly_annualized']:>7.3f} "
                  f"{m['per_event_hit_rate']*100:>5.1f}% "
                  f"{m['mean_event_arith']*100:>9.3f}% "
                  f"{m['win_rate_week']*100:>8.1f}%")

    # Optional: top-N sensitivity sweep
    if not args.no_sweep:
        print("\n" + "=" * 78)
        print("TOP-N SENSITIVITY SWEEP")
        print("=" * 78)
        sweep_summary = []
        # Compute symbol grid only once per model
        for d in dirs:
            try:
                ranker, _calib, _meta = load_model(d)
            except Exception as e:
                print(f"skip {d.name} sweep: {e}")
                continue
            val_df = build_val_table(args.split)
            row = {"model": d.name}
            for n in (1, 3, 5, 10, 20):
                bt = backtest_top_n(ranker, val_df, n)
                agg = aggregate_metrics(bt, n)
                row[f"hit_n{n}"] = agg.get("per_event_hit_rate", float("nan"))
                row[f"sharpe_n{n}"] = agg.get("sharpe_weekly_annualized", float("nan"))
                row[f"avg_n{n}"] = agg.get("mean_event_arith", float("nan"))
            sweep_summary.append(row)
        # Print sweep table compact
        print("\n" + "-" * 78)
        flag_hdr = f"  {'':48s}  " + \
                   "  ".join(f"{f'n={n}':>20s}" for n in (1, 3, 5, 10, 20))
        print(flag_hdr)
        for row in sweep_summary:
            print(f"\n{row['model']:48s}")
            for metric in ("hit", "avg", "sharpe"):
                cells = []
                for n in (1, 3, 5, 10, 20):
                    v = row.get(f"{metric}_n{n}", float("nan"))
                    if np.isnan(v):
                        s = "   -   "
                    elif metric == "hit":
                        s = f"{v*100:5.1f}%"
                    elif metric == "avg":
                        s = f"{v*100:+5.2f}%"
                    else:
                        s = f"{v:+5.2f}"
                    cells.append(s)
                line = f"  {metric.upper():46s}  " + \
                       "  ".join(f"{c:>20s}" for c in cells)
                print(line)
            
    return 0


if __name__ == "__main__":
    sys.exit(main())

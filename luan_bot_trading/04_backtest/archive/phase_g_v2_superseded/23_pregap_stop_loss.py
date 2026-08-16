#!/usr/bin/env python3
"""
Pre-gap entry + 3% stop-loss backtest.

Strategy:
  BMO: enter Close[T-1], stop at -3%, exit Close[T+4] or stop
  AMC: enter Close[T], stop at -3%, exit Close[T+5] or stop

Stop-loss logic (daily OHLC):
  For each day k from entry to exit:
    - If Open[k] <= entry * 0.97: gap-down below stop -> exit at Open[k]
    - Elif Low[k] <= entry * 0.97: intraday stop hit -> exit at entry*0.97
    - Else: continue holding
  If no stop hit by exit day: exit at Close[exit_day]

Compares 3 scenarios:
  A. Current: Open[T+1] entry, Close[T+5] exit (no stop)
  B. Pre-gap no stop: Close[T-1]/Close[T] entry, Close[T+5] exit
  C. Pre-gap + 3% stop: Close[T-1]/Close[T] entry, stop or Close[T+5] exit
"""
import sys, io, importlib.util
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tm", HERE.parent / "03_model" / "01_train_model.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
pg_spec = importlib.util.spec_from_file_location("pg", HERE.parent / "03_model" / "02_phase_g_sunday_classifier.py")
pg = importlib.util.module_from_spec(pg_spec); pg_spec.loader.exec_module(pg)
v3_spec = importlib.util.spec_from_file_location("v3", HERE / "_pead_target_retrain.py")
v3 = importlib.util.module_from_spec(v3_spec); v3_spec.loader.exec_module(v3)
ps_spec = importlib.util.spec_from_file_location("ps", HERE / "04_phase_g_portfolio.py")
ps = importlib.util.module_from_spec(ps_spec); ps_spec.loader.exec_module(ps)
rb_spec = importlib.util.spec_from_file_location("rb", HERE / "_phase_g_random_baseline.py")
rb = importlib.util.module_from_spec(rb_spec); rb_spec.loader.exec_module(rb)

DB = tm.DB_FILE
SUNDAY_SAFE = pg.SUNDAY_SAFE_FEATURES
THETA = 0.25
N_SLOTS = 4
EXIT_SNAP = 5
STOP_LOSS = 0.03  # -3% stop from entry

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]


def select_weekly_top_n(picks, n_slots=4):
    if picks.empty:
        return picks
    pk = picks.copy()
    pk["entry_date"] = pd.to_datetime(pk["entry_date"])
    pk["exit_date"] = pd.to_datetime(pk["exit_date"])
    iso = pk["entry_date"].dt.isocalendar()
    pk["_week_key"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    selected_rows = []
    active_positions = []
    for week_key, week_df in pk.groupby("_week_key", sort=True):
        week_start = week_df["entry_date"].min()
        active_positions = [ex for ex in active_positions if ex >= week_start]
        free_slots = n_slots - len(active_positions)
        if free_slots <= 0:
            continue
        week_sorted = week_df.sort_values("p", ascending=False)
        n_take = min(free_slots, len(week_sorted))
        taken = week_sorted.head(n_take)
        selected_rows.append(taken)
        for _, row in taken.iterrows():
            active_positions.append(row["exit_date"])
    if selected_rows:
        result = pd.concat(selected_rows).sort_values("entry_date").reset_index(drop=True)
    else:
        result = pd.DataFrame(columns=picks.columns)
    return result


def fit_clf(X_tr, y_tr, X_val, y_val, hp):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=hp["n_estimators"], learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def compute_pregap_stop_returns(trades_df, db_path, stop_loss=0.03, hold_days=5):
    """For each trade, compute returns under 3 scenarios:
    A. Current: Open[T+1] entry, Close[T+5] exit
    B. Pre-gap no stop: Close[T-1]/Close[T] entry, Close[T+5] exit
    C. Pre-gap + stop: Close[T-1]/Close[T] entry, stop or Close[T+5] exit

    BMO: entry = Close[T-1], first check day = T (announcement before Open[T])
    AMC: entry = Close[T], first check day = T+1 (announcement after Close[T])
    """
    results = []
    with pd.HDFStore(db_path, mode="r") as s:
        for _, trade in trades_df.iterrows():
            pt = trade["permaTicker"]
            key = f"/sp400/{pt}"
            if key not in s:
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_open = p["Adj_Open"].values
            p_high = p["Adj_High"].values
            p_low = p["Adj_Low"].values
            p_close = p["Adj_Close"].values

            rdate = pd.to_datetime(trade["report_date"]).to_datetime64()
            t_mask = p_index >= rdate
            if not t_mask.any():
                continue
            t_idx = int(np.argmax(t_mask))
            is_bmo = bool(trade.get("is_bmo", False))

            # Determine entry index and exit index
            if is_bmo:
                entry_idx = t_idx - 1  # Close[T-1]
                first_check = t_idx     # Open[T] (announcement day)
                exit_idx = t_idx + hold_days  # Close[T+5] (same exit as current for comparison)
            else:  # AMC
                entry_idx = t_idx       # Close[T]
                first_check = t_idx + 1  # Open[T+1] (reaction day)
                exit_idx = t_idx + hold_days  # Close[T+5]

            # Bounds check
            if entry_idx < 0 or exit_idx >= len(p_close):
                continue

            entry_price = p_close[entry_idx]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            # --- Scenario A: Current (Open[T+1] entry, Close[T+5] exit) ---
            o_t1 = p_open[t_idx + 1] if t_idx + 1 < len(p_open) else np.nan
            c_t5 = p_close[exit_idx]
            if pd.isna(o_t1) or pd.isna(c_t5) or o_t1 <= 0:
                ret_a = np.nan
            else:
                ret_a = float(c_t5 / o_t1 - 1.0)

            # --- Scenario B: Pre-gap no stop ---
            c_exit = p_close[exit_idx]
            if pd.isna(c_exit):
                ret_b = np.nan
            else:
                ret_b = float(c_exit / entry_price - 1.0)

            # --- Scenario C: Pre-gap + stop-loss ---
            stop_price = entry_price * (1.0 - stop_loss)
            ret_c = np.nan
            stop_hit_day = None
            stop_hit_price = None

            for k in range(first_check, exit_idx + 1):
                if k >= len(p_open):
                    break
                o_k = p_open[k]
                lo_k = p_low[k]
                hi_k = p_high[k]

                # Check if stop was hit
                if pd.notna(o_k) and o_k <= stop_price:
                    # Gap-down below stop: exit at open
                    ret_c = float(o_k / entry_price - 1.0)
                    stop_hit_day = k - first_check
                    stop_hit_price = o_k
                    break
                elif pd.notna(lo_k) and lo_k <= stop_price:
                    # Intraday stop hit: exit at stop price
                    ret_c = float(stop_price / entry_price - 1.0)
                    stop_hit_day = k - first_check
                    stop_hit_price = stop_price
                    break

            if ret_c is np.nan or pd.isna(ret_c):
                # No stop hit: exit at Close[exit_idx]
                if pd.notna(c_exit):
                    ret_c = float(c_exit / entry_price - 1.0)
                else:
                    ret_c = np.nan

            # Also compute the opening gap (for reference)
            if t_idx + 1 < len(p_open) and pd.notna(p_open[t_idx + 1]):
                gap = float(p_open[t_idx + 1] / p_close[t_idx] - 1.0)
            else:
                gap = np.nan

            results.append({
                "permaTicker": pt,
                "ticker": trade.get("canonical_ticker", pt),
                "report_date": trade["report_date"],
                "is_bmo": is_bmo,
                "pead_pass": trade.get("pead_pass", np.nan),
                "p": trade.get("p", np.nan),
                "fold": trade.get("fold", np.nan),
                "ret_current": ret_a,
                "ret_pregap_nostop": ret_b,
                "ret_pregap_stop": ret_c,
                "opening_gap": gap,
                "stop_hit_day": stop_hit_day,
                "stop_hit": stop_hit_day is not None,
            })

    return pd.DataFrame(results)


def print_stats(df, col, label):
    pnls = df[col].dropna()
    if len(pnls) == 0:
        print(f"  {label:<25} N=0")
        return
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / len(pnls) * 100
    avg_w = wins.mean() * 100 if len(wins) > 0 else 0
    avg_l = losses.mean() * 100 if len(losses) > 0 else 0
    payoff = avg_w / abs(avg_l) if avg_l != 0 else float('inf')
    print(f"  {label:<25} N={len(pnls):>3}  Win={wr:>5.1f}%  "
          f"Avg={pnls.mean()*100:>+6.2f}%  Win={avg_w:>+6.2f}%  "
          f"Loss={avg_l:>+6.2f}%  Payoff={payoff:>5.2f}  "
          f"Std={pnls.std()*100:>5.2f}%")


def main():
    print("=" * 90)
    print(f"PRE-GAP ENTRY + {STOP_LOSS*100:.0f}% STOP-LOSS BACKTEST")
    print(f"  theta={THETA}, {N_SLOTS} slots, {EXIT_SNAP}-day hold")
    print("=" * 90)

    print("\n[1] Loading + priming + gates + paths ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)

    with pd.HDFStore(DB, mode="r") as sstore:
        all_dates = set()
        for key in sstore.keys():
            if not key.startswith("/sp400/"):
                continue
            d = sstore[key]
            all_dates.update(pd.to_datetime(d["Date"]).tolist())
    calendar = sorted(all_dates)
    cal_idx = {d: i for i, d in enumerate(calendar)}
    df["_entry_idx"] = pd.to_datetime(df["entry_date"]).map(cal_idx)
    df["_exit_idx_new"] = df["_entry_idx"] + (EXIT_SNAP - 1)
    valid = df["_exit_idx_new"].notna() & (df["_exit_idx_new"] < len(calendar))
    df.loc[valid, "exit_date"] = df.loc[valid, "_exit_idx_new"].astype(int).map(lambda i: calendar[i])
    df["path_pnl_t11_pct"] = df[f"path_pnl_t{EXIT_SNAP}_pct"]
    df = df.drop(columns=["_entry_idx", "_exit_idx_new"])

    print("\n[2] Pre-training 4 fold classifiers ...")
    fold_data = {}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_tr = train_df[SUNDAY_SAFE]
        y_tr = train_df["pead_pass"].astype(int).values
        X_sv = sweep_df[SUNDAY_SAFE]
        y_sv = sweep_df["pead_pass"].astype(int).values
        X_te = test_df[SUNDAY_SAFE]
        y_te = test_df["pead_pass"].astype(int).values
        X_ts = pd.concat([X_tr, X_sv])
        y_ts = np.concatenate([y_tr, y_sv])
        hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
        clf = fit_clf(X_ts, y_ts, X_te, y_te, hp)
        proba = clf.predict_proba(test_df[SUNDAY_SAFE])[:, 1]
        test_df = test_df.copy()
        test_df["p"] = proba
        mask = (test_df["p"] >= THETA) & (test_df["path_pnl_t11_pct"].notna())
        picks = test_df[mask].copy()
        fold_data[fi] = {"picks": picks}

    print(f"\n[3] Running {N_SLOTS}-slot portfolio simulation + collecting trades ...")
    all_trades = []
    for fi in range(1, 5):
        picks = fold_data[fi]["picks"]
        selected = select_weekly_top_n(picks, N_SLOTS)
        result = rb._simulate_with_cached_calendar(selected, N_SLOTS, 100_000.0, calendar)
        executed = result.get("trades_done", pd.DataFrame())
        if len(executed) == 0:
            continue
        lookup = selected[["permaTicker", "entry_date", "pead_pass", "p",
                           "report_date", "canonical_ticker", "is_bmo"]].copy()
        lookup["entry_date"] = pd.to_datetime(lookup["entry_date"])
        executed["entry_date"] = pd.to_datetime(executed["entry_date"])
        executed = executed.merge(lookup, on=["permaTicker", "entry_date"], how="left")
        executed["fold"] = fi
        all_trades.append(executed)

    all_df = pd.concat(all_trades).reset_index(drop=True)
    print(f"  Total executed trades: {len(all_df)}")

    print(f"\n[4] Computing pre-gap + stop-loss returns ...")
    results = compute_pregap_stop_returns(all_df, DB, STOP_LOSS, EXIT_SNAP)
    print(f"  Computed returns for {len(results)} trades")

    # ===== ANALYSIS =====
    print(f"\n{'='*90}")
    print("RESULTS: 3 SCENARIOS COMPARED")
    print(f"{'='*90}")

    print(f"\n1. ALL TRADES ({len(results)})")
    print(f"   {'Scenario':<25} {'N':>4}  {'Win%':>6}  {'Avg':>7}  {'AvgWin':>7}  {'AvgLoss':>7}  {'Payoff':>6}  {'Std':>6}")
    print_stats(results, "ret_current", "A. Current (Open[T+1])")
    print_stats(results, "ret_pregap_nostop", "B. Pre-gap (no stop)")
    print_stats(results, "ret_pregap_stop", f"C. Pre-gap + {STOP_LOSS*100:.0f}% stop")

    print(f"\n2. BMO TRADES ({int(results['is_bmo'].sum())})")
    bmo_df = results[results["is_bmo"]]
    print_stats(bmo_df, "ret_current", "A. Current (Open[T+1])")
    print_stats(bmo_df, "ret_pregap_nostop", "B. Pre-gap (no stop)")
    print_stats(bmo_df, "ret_pregap_stop", f"C. Pre-gap + {STOP_LOSS*100:.0f}% stop")

    print(f"\n3. AMC TRADES ({int((~results['is_bmo']).sum())})")
    amc_df = results[~results["is_bmo"]]
    print_stats(amc_df, "ret_current", "A. Current (Open[T+1])")
    print_stats(amc_df, "ret_pregap_nostop", "B. Pre-gap (no stop)")
    print_stats(amc_df, "ret_pregap_stop", f"C. Pre-gap + {STOP_LOSS*100:.0f}% stop")

    print(f"\n4. PEAD TRADES ({int(results['pead_pass'].sum())})")
    pead_df = results[results["pead_pass"] == 1]
    print_stats(pead_df, "ret_current", "A. Current (Open[T+1])")
    print_stats(pead_df, "ret_pregap_nostop", "B. Pre-gap (no stop)")
    print_stats(pead_df, "ret_pregap_stop", f"C. Pre-gap + {STOP_LOSS*100:.0f}% stop")

    print(f"\n5. PEAD + BMO ({int(pead_df['is_bmo'].sum())})")
    pead_bmo = pead_df[pead_df["is_bmo"]]
    print_stats(pead_bmo, "ret_current", "A. Current (Open[T+1])")
    print_stats(pead_bmo, "ret_pregap_nostop", "B. Pre-gap (no stop)")
    print_stats(pead_bmo, "ret_pregap_stop", f"C. Pre-gap + {STOP_LOSS*100:.0f}% stop")

    print(f"\n6. PEAD + AMC ({int((~pead_df['is_bmo']).sum())})")
    pead_amc = pead_df[~pead_df["is_bmo"]]
    print_stats(pead_amc, "ret_current", "A. Current (Open[T+1])")
    print_stats(pead_amc, "ret_pregap_nostop", "B. Pre-gap (no stop)")
    print_stats(pead_amc, "ret_pregap_stop", f"C. Pre-gap + {STOP_LOSS*100:.0f}% stop")

    # Stop-loss effectiveness
    print(f"\n7. STOP-LOSS EFFECTIVENESS (Pre-gap + {STOP_LOSS*100:.0f}% stop)")
    stop_df = results[results["stop_hit"]]
    nostop_df = results[~results["stop_hit"]]
    print(f"  Stop hit:    {len(stop_df)}/{len(results)} trades ({len(stop_df)/len(results)*100:.1f}%)")
    print(f"  Stop NOT hit: {len(nostop_df)}/{len(results)} trades ({len(nostop_df)/len(results)*100:.1f}%)")
    if len(stop_df) > 0:
        print(f"\n  When stop hit ({len(stop_df)} trades):")
        print(f"    Avg ret (with stop):   {stop_df['ret_pregap_stop'].mean()*100:+.2f}%")
        print(f"    Avg ret (without stop): {stop_df['ret_pregap_nostop'].mean()*100:+.2f}%")
        print(f"    Stop saved:            {stop_df['ret_pregap_stop'].mean() - stop_df['ret_pregap_nostop'].mean():+.4f} per trade")
        print(f"    Stop hit day:          mean={stop_df['stop_hit_day'].mean():.1f}, med={stop_df['stop_hit_day'].median():.0f}")
    if len(nostop_df) > 0:
        print(f"\n  When stop NOT hit ({len(nostop_df)} trades):")
        print(f"    Avg ret (with stop):   {nostop_df['ret_pregap_stop'].mean()*100:+.2f}%")
        print(f"    Avg ret (without stop): {nostop_df['ret_pregap_nostop'].mean()*100:+.2f}%")
        print(f"    (should be identical - no stop triggered)")

    # Worst-case comparison
    print(f"\n8. WORST-CASE COMPARISON (bottom 5 trades by pre-gap no-stop)")
    worst = results.nsmallest(5, "ret_pregap_nostop")[
        ["ticker", "is_bmo", "pead_pass", "ret_current", "ret_pregap_nostop",
         "ret_pregap_stop", "opening_gap", "stop_hit"]]
    print(worst.to_string(index=False, float_format=lambda x: f"{x*100:+.2f}%"))

    print(f"\n{'='*90}")


if __name__ == "__main__":
    main()

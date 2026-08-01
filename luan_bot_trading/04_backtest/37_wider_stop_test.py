#!/usr/bin/env python3
"""
Test wider delayed stop-loss: -10%, -12%, -14% (skip gap day).

The user's concern: the worst losses (-14% to -37%) are hard to stomach.
A tight 3% stop hurt (cut winners). But a WIDER stop at -10% to -14%
has different dynamics:
  - Winners dip only -3% avg (never triggers)
  - Only catches catastrophic losers
  - Question: do catastrophic losers recover or keep falling after day 1?

Delayed stop logic (skip gap day, check from day 1 onward):
  - Day 0 (gap day): no check (let the gap settle)
  - Days 1..N: if Low[k] <= entry*(1-stop), exit at stop price
  - If Open[k] <= entry*(1-stop): gap-down below stop, exit at Open

Tests stops at: 10%, 12%, 14% (plus no-stop baseline)
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

DB = tm.DB_FILE
SUNDAY_SAFE = pg.SUNDAY_SAFE_FEATURES
N_SLOTS = 4
EXIT_SNAP = 5
THETA = 0.20

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]

STOPS = [None, 0.10, 0.12, 0.14]


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


def select_weekly_top_n(picks, n_slots=4, sort_col="p"):
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
        week_sorted = week_df.sort_values(sort_col, ascending=False)
        taken = week_sorted.head(min(free_slots, len(week_sorted)))
        selected_rows.append(taken)
        for _, row in taken.iterrows():
            active_positions.append(row["exit_date"])
    if selected_rows:
        return pd.concat(selected_rows).sort_values("entry_date").reset_index(drop=True)
    return pd.DataFrame(columns=picks.columns)


def compute_returns_with_stop(trades_df, db_path, stop_loss, hold_days=5):
    """Compute returns with a delayed stop (skip gap day).
    Returns: (return, stop_hit_day, stop_hit_price)"""
    results = []
    with pd.HDFStore(db_path, mode="r") as s:
        for _, trade in trades_df.iterrows():
            pt = trade["permaTicker"]
            key = f"/sp400/{pt}"
            if key not in s:
                results.append({"ret": np.nan, "stop_hit": False, "stop_day": None})
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_open = p["Adj_Open"].values
            p_low = p["Adj_Low"].values
            p_close = p["Adj_Close"].values
            rdate = pd.to_datetime(trade["report_date"]).to_datetime64()
            t_mask = p_index >= rdate
            if not t_mask.any():
                results.append({"ret": np.nan, "stop_hit": False, "stop_day": None})
                continue
            t_idx = int(np.argmax(t_mask))
            is_bmo = bool(trade.get("is_bmo", False))
            entry_t = t_idx - 1 if is_bmo else t_idx
            gap_day = t_idx if is_bmo else t_idx + 1
            exit_t = t_idx + hold_days
            if entry_t < 0 or exit_t >= len(p_close):
                results.append({"ret": np.nan, "stop_hit": False, "stop_day": None})
                continue
            entry_price = p_close[entry_t]
            exit_price = p_close[exit_t]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                results.append({"ret": np.nan, "stop_hit": False, "stop_day": None})
                continue

            ret_nostop = float(exit_price / entry_price - 1.0)

            if stop_loss is None:
                results.append({"ret": ret_nostop, "stop_hit": False, "stop_day": None,
                                "ret_nostop": ret_nostop})
                continue

            stop_price = entry_price * (1.0 - stop_loss)
            ret_stop = ret_nostop  # default: no stop hit
            stop_hit = False
            stop_day = None

            # Check from day AFTER gap day (gap_day + 1) to exit
            for k in range(gap_day + 1, exit_t + 1):
                if k >= len(p_open):
                    break
                o_k = p_open[k]
                lo_k = p_low[k]
                # Gap-down below stop at open
                if pd.notna(o_k) and o_k <= stop_price:
                    ret_stop = float(o_k / entry_price - 1.0)
                    stop_hit = True
                    stop_day = k - gap_day
                    break
                # Intraday stop hit
                elif pd.notna(lo_k) and lo_k <= stop_price:
                    ret_stop = float(stop_price / entry_price - 1.0)
                    stop_hit = True
                    stop_day = k - gap_day
                    break

            results.append({"ret": ret_stop, "stop_hit": stop_hit, "stop_day": stop_day,
                            "ret_nostop": ret_nostop})
    return pd.DataFrame(results)


def main():
    print("=" * 95)
    print(f"WIDER DELAYED STOP TEST: -10%, -12%, -14% (skip gap day)")
    print(f"  Binary P(PEAD)>={THETA}, pre-gap entry, {EXIT_SNAP}-day hold")
    print("=" * 95)

    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)

    # Pre-gap returns + entry/exit dates
    df["pregap_return"] = np.nan
    df["pregap_entry_date"] = pd.NaT
    df["pregap_exit_date"] = pd.NaT
    with pd.HDFStore(DB, mode="r") as s:
        for idx, row in df.iterrows():
            pt = row["permaTicker"]
            key = f"/sp400/{pt}"
            if key not in s:
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_close = p["Adj_Close"].values
            rdate = pd.to_datetime(row["report_date"]).to_datetime64()
            t_mask = p_index >= rdate
            if not t_mask.any():
                continue
            t_idx = int(np.argmax(t_mask))
            is_bmo = bool(row.get("is_bmo", False))
            entry_t = t_idx - 1 if is_bmo else t_idx
            exit_t = t_idx + EXIT_SNAP
            if entry_t < 0 or exit_t >= len(p_close):
                continue
            entry_price = p_close[entry_t]
            exit_price = p_close[exit_t]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue
            df.at[idx, "pregap_return"] = float(exit_price / entry_price - 1.0)
            df.at[idx, "pregap_entry_date"] = pd.Timestamp(p_index[entry_t])
            df.at[idx, "pregap_exit_date"] = pd.Timestamp(p_index[exit_t])

    # Train + select
    fold_data = {}
    hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_ts = pd.concat([train_df[SUNDAY_SAFE], sweep_df[SUNDAY_SAFE]])
        y_ts = pd.concat([train_df, sweep_df])["pead_pass"].astype(int).values
        y_te = test_df["pead_pass"].astype(int).values
        clf = fit_clf(X_ts, y_ts, test_df[SUNDAY_SAFE], y_te, hp)
        test_df = test_df.copy()
        test_df["p"] = clf.predict_proba(test_df[SUNDAY_SAFE])[:, 1]
        fold_data[fi] = {"test_df": test_df}

    all_exec = []
    for fi in range(1, 5):
        td = fold_data[fi]["test_df"]
        mask = (td["p"] >= THETA) & (td["pregap_return"].notna())
        picks = td[mask].copy()
        if len(picks) == 0:
            continue
        picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
        picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
        picks["fold"] = fi
        sel = select_weekly_top_n(picks, N_SLOTS, sort_col="p")
        if len(sel) > 0:
            all_exec.append(sel)
    exec_df = pd.concat(all_exec).reset_index(drop=True)
    print(f"\n  Executed trades: {len(exec_df)}")

    # ===== FIRST: Do catastrophic losers recover or worsen? =====
    print(f"\n{'='*95}")
    print("1. DO CATASTROPHIC LOSERS RECOVER OR WORSEN AFTER DAY 1?")
    print(f"{'='*95}")

    # Get the worst trades (no-stop return < -10%)
    worst = exec_df[exec_df["pregap_return"] < -0.10].sort_values("pregap_return").copy()
    print(f"\n  Trades with no-stop return < -10%: {len(worst)}")

    # For each, check the path: what was the return at day 1, day 2, ... exit?
    print(f"\n  {'Ticker':<10} {'Date':<12} {'NoStop':>8} {'Day1':>8} {'Day2':>8} {'Day3':>8} {'Day4':>8} {'Day5':>8} {'Recovered?':>12}")
    with pd.HDFStore(DB, mode="r") as s:
        for _, trade in worst.iterrows():
            pt = trade["permaTicker"]
            key = f"/sp400/{pt}"
            if key not in s:
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_close = p["Adj_Close"].values
            rdate = pd.to_datetime(trade["report_date"]).to_datetime64()
            t_mask = p_index >= rdate
            t_idx = int(np.argmax(t_mask))
            is_bmo = bool(trade.get("is_bmo", False))
            entry_t = t_idx - 1 if is_bmo else t_idx
            gap_day = t_idx if is_bmo else t_idx + 1
            exit_t = t_idx + EXIT_SNAP
            if entry_t < 0 or exit_t >= len(p_close):
                continue
            entry_price = p_close[entry_t]
            ticker = trade.get("canonical_ticker", pt)
            rd = str(pd.Timestamp(trade["report_date"]).date())
            nostop = trade["pregap_return"] * 100
            # Daily closes from gap_day to exit
            daily = []
            for k in range(gap_day, exit_t + 1):
                if k < len(p_close) and pd.notna(p_close[k]):
                    daily.append(f"{(p_close[k]/entry_price-1)*100:+.1f}%")
                else:
                    daily.append("?")
            # Did it recover from day 1 low?
            d1_ret = (p_close[gap_day] / entry_price - 1) if gap_day < len(p_close) else 0
            exit_ret = (p_close[exit_t] / entry_price - 1)
            recovered = "YES" if exit_ret > d1_ret + 0.02 else ("NO" if exit_ret < d1_ret - 0.02 else "FLAT")
            print(f"  {ticker:<10} {rd:<12} {nostop:>+7.1f}% {daily[0]:>8} {daily[1] if len(daily)>1 else '?':>8} "
                  f"{daily[2] if len(daily)>2 else '?':>8} {daily[3] if len(daily)>3 else '?':>8} "
                  f"{daily[4] if len(daily)>4 else '?':>8} {recovered:>12}")

    # ===== STOP-LOSS COMPARISON =====
    print(f"\n{'='*95}")
    print("2. STOP-LOSS COMPARISON (delayed, skip gap day)")
    print(f"{'='*95}")

    print(f"\n  {'Stop':>6} {'N':>4} {'Win%':>6} {'Avg':>8} {'Total':>8} {'Payoff':>7} {'AvgWin':>8} {'AvgLoss':>8} {'Worst':>8} {'Stops':>6}")
    print("  " + "-" * 90)

    for stop in STOPS:
        stop_label = f"-{stop*100:.0f}%" if stop else "None"
        stop_results = compute_returns_with_stop(exec_df, DB, stop, EXIT_SNAP)
        pnls = stop_results["ret"].dropna()
        n = len(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        wr = len(wins) / n * 100
        avg = pnls.mean() * 100
        aw = wins.mean() * 100 if len(wins) > 0 else 0
        al = losses.mean() * 100 if len(losses) > 0 else 0
        payoff = aw / abs(al) if al != 0 else float('inf')
        total = pnls.sum() * 100
        worst = pnls.min() * 100
        n_stops = int(stop_results["stop_hit"].sum())
        print(f"  {stop_label:>6} {n:>4} {wr:>5.1f}% {avg:>+7.2f}% {total:>+7.1f}% {payoff:>6.2f} "
              f"{aw:>+7.2f}% {al:>+7.2f}% {worst:>+7.1f}% {n_stops:>6}")

    # ===== STOP IMPACT DETAIL =====
    print(f"\n{'='*95}")
    print("3. STOP IMPACT DETAIL (which trades got stopped?)")
    print(f"{'='*95}")

    for stop in [0.10, 0.12, 0.14]:
        stop_results = compute_returns_with_stop(exec_df, DB, stop, EXIT_SNAP)
        stopped = stop_results[stop_results["stop_hit"]]
        not_stopped = stop_results[~stop_results["stop_hit"]]

        print(f"\n  Stop at -{stop*100:.0f}%:")
        print(f"    Stopped:     {len(stopped)} trades")
        if len(stopped) > 0:
            print(f"      Avg return WITH stop:    {stopped['ret'].mean()*100:+.2f}%")
            print(f"      Avg return WITHOUT stop: {stopped['ret_nostop'].mean()*100:+.2f}%")
            print(f"      Stop SAVED:              {(stopped['ret'] - stopped['ret_nostop']).mean()*100:+.2f}%/trade")
            print(f"      Stop day:                mean={stopped['stop_day'].mean():.1f}, med={stopped['stop_day'].median():.0f}")
            # Were the stopped trades winners or losers without the stop?
            would_win = (stopped["ret_nostop"] > 0).sum()
            would_lose = (stopped["ret_nostop"] <= 0).sum()
            print(f"      Would have been: {would_win} wins, {would_lose} losses (without stop)")
        print(f"    Not stopped: {len(not_stopped)} trades (identical to no-stop)")

    # ===== WORST TRADES: with vs without stop =====
    print(f"\n{'='*95}")
    print("4. WORST 10 TRADES: no-stop vs -10% stop vs -14% stop")
    print(f"{'='*95}")

    nostop = compute_returns_with_stop(exec_df, DB, None, EXIT_SNAP)
    s10 = compute_returns_with_stop(exec_df, DB, 0.10, EXIT_SNAP)
    s14 = compute_returns_with_stop(exec_df, DB, 0.14, EXIT_SNAP)

    combined = pd.DataFrame({
        "ticker": exec_df["canonical_ticker"].values,
        "date": exec_df["report_date"].values,
        "nostop": nostop["ret"].values * 100,
        "s10": s10["ret"].values * 100,
        "s14": s14["ret"].values * 100,
        "s10_hit": s10["stop_hit"].values,
        "s14_hit": s14["stop_hit"].values,
    })
    worst10 = combined.nsmallest(10, "nostop")
    print(f"\n  {'Ticker':<10} {'Date':<12} {'NoStop':>8} {'-10%':>8} {'-14%':>8} {'-10%hit':>8} {'-14%hit':>8}")
    for _, r in worst10.iterrows():
        print(f"  {r['ticker']:<10} {str(pd.Timestamp(r['date']).date()):<12} "
              f"{r['nostop']:>+7.1f}% {r['s10']:>+7.1f}% {r['s14']:>+7.1f}% "
              f"{'YES' if r['s10_hit'] else '':>8} {'YES' if r['s14_hit'] else '':>8}")

    print(f"\n{'='*95}")


if __name__ == "__main__":
    main()

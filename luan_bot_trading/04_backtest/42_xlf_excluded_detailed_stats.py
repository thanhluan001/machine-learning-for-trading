#!/usr/bin/env python3
"""
Detailed trade statistics for the XLF-excluded binary model.

Produces the same detailed stats as 36_binary_detailed_stats.py but with
XLF exclusion applied at inference. This is now the FINAL deployable
operating point.
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
EXCLUDE_SECTORS = ["XLF"]

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]


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


def compute_pregap_with_path(df, db_path, hold_days=5, stop_loss=0.10):
    """Compute pre-gap PnL, intra-hold max drawdown, and stop-loss exit."""
    df = df.copy()
    df["pregap_return"] = np.nan
    df["pregap_entry_date"] = pd.NaT
    df["pregap_exit_date"] = pd.NaT
    df["pregap_max_drawdown"] = np.nan
    df["pregap_max_favorable"] = np.nan
    df["pregap_stop_exit"] = 0
    df["pregap_hold_return_no_stop"] = np.nan
    with pd.HDFStore(db_path, mode="r") as s:
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
            exit_t = t_idx + hold_days
            if entry_t < 0 or exit_t >= len(p_close):
                continue
            entry_price = p_close[entry_t]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            # Intra-hold path (gap day + hold days)
            path_start = t_idx  # gap day = day after entry
            path_end = exit_t + 1  # inclusive
            path_prices = p_close[path_start:path_end]
            path_valid = path_prices[~np.isnan(path_prices)]
            if len(path_valid) == 0:
                continue

            # Max drawdown / favorable from entry
            running_max = np.maximum.accumulate(path_valid)
            running_min = np.minimum.accumulate(path_valid)
            mdd = np.nanmin(path_valid / entry_price - 1.0)
            mfe = np.nanmax(path_valid / entry_price - 1.0)

            # Full 5-day hold return (no stop)
            exit_price = p_close[exit_t]
            if pd.isna(exit_price):
                continue
            hold_ret = float(exit_price / entry_price - 1.0)

            # Stop-loss check (delayed: skip gap day, check days 1+)
            stop_price = entry_price * (1.0 - stop_loss)
            # Days after gap: path_valid[1:] (gap day = path_valid[0])
            stop_days = path_valid[1:]
            stop_exit = 0
            final_ret = hold_ret
            for sp in stop_days:
                if not np.isnan(sp) and sp <= stop_price:
                    final_ret = float(sp / entry_price - 1.0)
                    stop_exit = 1
                    break

            df.at[idx, "pregap_return"] = final_ret
            df.at[idx, "pregap_entry_date"] = pd.Timestamp(p_index[entry_t])
            df.at[idx, "pregap_exit_date"] = pd.Timestamp(p_index[exit_t])
            df.at[idx, "pregap_max_drawdown"] = float(mdd)
            df.at[idx, "pregap_max_favorable"] = float(mfe)
            df.at[idx, "pregap_stop_exit"] = stop_exit
            df.at[idx, "pregap_hold_return_no_stop"] = hold_ret
    return df


def main():
    print("=" * 100)
    print("XLF-EXCLUDED BINARY MODEL — DETAILED TRADE STATISTICS")
    print("(Binary theta=0.20, pre-gap entry, 5-day hold, -10% delayed stop, exclude XLF)")
    print("=" * 100)

    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = compute_pregap_with_path(df, DB, EXIT_SNAP, stop_loss=0.10)

    # Join sector
    with pd.HDFStore(DB, mode="r") as s:
        pt_meta = s["/metadata/sp400_permatickers"]
    sector_lookup = pt_meta[["permaTicker", "index_ref"]].drop_duplicates("permaTicker")
    df = df.merge(sector_lookup, on="permaTicker", how="left")
    df["sector"] = df["index_ref"]

    # Train + predict
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

    # Execute with XLF exclusion
    all_exec = []
    for fi in range(1, 5):
        td = fold_data[fi]["test_df"]
        mask = (td["p"] >= THETA) & (td["pregap_return"].notna())
        mask = mask & (~td["sector"].isin(EXCLUDE_SECTORS))
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

    pnls = exec_df["pregap_return"].dropna()
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n_pead = int(exec_df["pead_pass"].sum())

    # ===== SECTION 1: HEADLINE STATS =====
    print(f"\n{'='*100}")
    print("1. HEADLINE STATISTICS")
    print(f"{'='*100}")

    print(f"\n  {'Metric':<45} {'Value':>15}")
    print("  " + "-" * 60)
    print(f"  {'Total executed trades':<45} {n:>15}")
    print(f"  {'True PEAD events (label=1)':<45} {n_pead:>15}")
    print(f"  {'False positives (label=0)':<45} {n - n_pead:>15}")
    print(f"  {'PEAD precision':<45} {n_pead/n*100:>14.1f}%")
    print(f"  {'Win rate (return > 0)':<45} {len(wins)/n*100:>14.1f}%")
    print(f"  {'Loss rate (return <= 0)':<45} {len(losses)/n*100:>14.1f}%")
    print(f"  {'Number of wins':<45} {len(wins):>15}")
    print(f"  {'Number of losses':<45} {len(losses):>15}")
    print(f"  {'Number of breakeven':<45} {len(pnls[pnls==0]):>15}")
    print(f"  {'Expectancy per trade':<45} {pnls.mean()*100:>+14.2f}%")
    print(f"  {'Median return per trade':<45} {pnls.median()*100:>+14.2f}%")
    print(f"  {'Std dev per trade':<45} {pnls.std()*100:>14.2f}%")
    print(f"  {'Min return':<45} {pnls.min()*100:>+14.2f}%")
    print(f"  {'Max return':<45} {pnls.max()*100:>+14.2f}%")
    print(f"  {'Skewness':<45} {pnls.skew():>15.2f}")
    print(f"  {'Kurtosis (excess)':<45} {pnls.kurtosis():>15.2f}")

    # ===== SECTION 2: WIN/LOSS BREAKDOWN =====
    print(f"\n{'='*100}")
    print("2. WIN / LOSS BREAKDOWN")
    print(f"{'='*100}")

    print(f"\n  {'Metric':<45} {'Value':>15}")
    print("  " + "-" * 60)
    print(f"  {'Avg win':<45} {wins.mean()*100:>+14.2f}%")
    print(f"  {'Avg loss':<45} {losses.mean()*100:>+14.2f}%")
    print(f"  {'Median win':<45} {wins.median()*100:>+14.2f}%")
    print(f"  {'Median loss':<45} {losses.median()*100:>+14.2f}%")
    print(f"  {'Max win':<45} {wins.max()*100:>+14.2f}%")
    print(f"  {'Max loss (worst)':<45} {losses.min()*100:>+14.2f}%")
    print(f"  {'Payoff ratio (avg win / |avg loss|)':<45} {wins.mean()/abs(losses.mean()):>15.2f}")
    print(f"  {'Profit factor (sum wins / |sum losses|)':<45} {wins.sum()/abs(losses.sum()):>15.2f}")

    # Win/loss percentile distribution
    print(f"\n  Win percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"    p{p}: {np.percentile(wins, p)*100:>+6.2f}%")
    print(f"\n  Loss percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"    p{p}: {np.percentile(losses, p)*100:>+6.2f}%")

    # ===== SECTION 3: RETURN DISTRIBUTION BUCKETS =====
    print(f"\n{'='*100}")
    print("3. RETURN DISTRIBUTION BUCKETS")
    print(f"{'='*100}")

    buckets = [
        ("<= -20%", pnls <= -0.20),
        ("-20% to -10%", (pnls > -0.20) & (pnls <= -0.10)),
        ("-10% to -5%", (pnls > -0.10) & (pnls <= -0.05)),
        ("-5% to 0%", (pnls > -0.05) & (pnls <= 0)),
        ("0% to +5%", (pnls > 0) & (pnls <= 0.05)),
        ("+5% to +10%", (pnls > 0.05) & (pnls <= 0.10)),
        ("+10% to +20%", (pnls > 0.10) & (pnls <= 0.20)),
        ("+20% to +30%", (pnls > 0.20) & (pnls <= 0.30)),
        ("> +30%", pnls > 0.30),
    ]
    print(f"\n  {'Bucket':<20} {'Count':>6} {'%':>6} {'Cumulative':>12}")
    print("  " + "-" * 50)
    cum = 0
    for label, mask in buckets:
        cnt = int(mask.sum())
        pct = cnt / n * 100
        cum += cnt
        bar = "#" * int(pct)
        print(f"  {label:<20} {cnt:>6} {pct:>5.1f}% {bar}")

    # ===== SECTION 4: PATH ANALYSIS (MAX DRAWDOWN / FAVORABLE) =====
    print(f"\n{'='*100}")
    print("4. PATH ANALYSIS — Intra-hold max drawdown & favorable excursion")
    print(f"{'='*100}")

    mdd = exec_df["pregap_max_drawdown"].dropna() * 100
    mfe = exec_df["pregap_max_favorable"].dropna() * 100

    print(f"\n  Max Drawdown (worst point during hold, from entry):")
    print(f"    {'Metric':<35} {'Value':>10}")
    print(f"    {'-'*45}")
    print(f"    {'Mean MDD':<35} {mdd.mean():>+9.2f}%")
    print(f"    {'Median MDD':<35} {mdd.median():>+9.2f}%")
    print(f"    {'p10 MDD':<35} {np.percentile(mdd, 10):>+9.2f}%")
    print(f"    {'p25 MDD':<35} {np.percentile(mdd, 25):>+9.2f}%")
    print(f"    {'p75 MDD':<35} {np.percentile(mdd, 75):>+9.2f}%")
    print(f"    {'p90 MDD':<35} {np.percentile(mdd, 90):>+9.2f}%")
    print(f"    {'Worst MDD':<35} {mdd.min():>+9.2f}%")

    print(f"\n  Max Favorable Excursion (best point during hold):")
    print(f"    {'Metric':<35} {'Value':>10}")
    print(f"    {'-'*45}")
    print(f"    {'Mean MFE':<35} {mfe.mean():>+9.2f}%")
    print(f"    {'Median MFE':<35} {mfe.median():>+9.2f}%")
    print(f"    {'p25 MFE':<35} {np.percentile(mfe, 25):>+9.2f}%")
    print(f"    {'p75 MFE':<35} {np.percentile(mfe, 75):>+9.2f}%")
    print(f"    {'Best MFE':<35} {mfe.max():>+9.2f}%")

    # MDD by win/loss
    print(f"\n  Max drawdown by outcome:")
    win_mask = exec_df["pregap_return"] > 0
    loss_mask = exec_df["pregap_return"] <= 0
    for label, mask in [("Winners", win_mask), ("Losers", loss_mask)]:
        sub_mdd = exec_df.loc[mask, "pregap_max_drawdown"].dropna() * 100
        sub_mfe = exec_df.loc[mask, "pregap_max_favorable"].dropna() * 100
        print(f"    {label}: MDD mean={sub_mdd.mean():+.2f}%, median={sub_mdd.median():+.2f}%, "
              f"MFE mean={sub_mfe.mean():+.2f}%")

    # ===== SECTION 5: STOP-LOSS IMPACT =====
    print(f"\n{'='*100}")
    print("5. STOP-LOSS IMPACT (-10% delayed stop)")
    print(f"{'='*100}")

    n_stopped = int(exec_df["pregap_stop_exit"].sum())
    no_stop_pnl = exec_df["pregap_hold_return_no_stop"].dropna()
    print(f"\n  {'Metric':<45} {'Value':>15}")
    print("  " + "-" * 60)
    print(f"  {'Trades stopped out':<45} {n_stopped:>15}")
    print(f"  {'Stop rate':<45} {n_stopped/n*100:>14.1f}%")
    print(f"  {'Total PnL with stop':<45} {pnls.sum()*100:>+14.1f}%")
    print(f"  {'Total PnL without stop (pure 5-day hold)':<45} {no_stop_pnl.sum()*100:>+14.1f}%")
    print(f"  {'Avg per trade with stop':<45} {pnls.mean()*100:>+14.2f}%")
    print(f"  {'Avg per trade without stop':<45} {no_stop_pnl.mean()*100:>+14.2f}%")

    stopped = exec_df[exec_df["pregap_stop_exit"] == 1]
    if len(stopped) > 0:
        print(f"\n  Stopped trades detail ({len(stopped)} trades):")
        print(f"    {'Ticker':<10} {'Date':<12} {'Entry':<12} {'Stop ret':>9} {'No-stop ret':>11} {'Saved?':>7}")
        for _, t in stopped.sort_values("pregap_return").iterrows():
            ticker = t.get("canonical_ticker", t["permaTicker"])
            rd = str(pd.Timestamp(t["report_date"]).date())
            sr = t["pregap_return"] * 100
            nsr = t["pregap_hold_return_no_stop"] * 100
            saved = nsr - sr
            saved_flag = "YES" if saved > 0.5 else "no"
            print(f"    {ticker:<10} {rd:<12} {sr:>+8.1f}% {nsr:>+10.1f}% {saved_flag:>7}")

    # ===== SECTION 6: PEAD vs NON-PEAD BREAKDOWN =====
    print(f"\n{'='*100}")
    print("6. PEAD vs NON-PEAD TRADE BREAKDOWN")
    print(f"{'='*100}")

    pead = exec_df[exec_df["pead_pass"] == 1]
    fp = exec_df[exec_df["pead_pass"] == 0]

    print(f"\n  {'Group':<20} {'N':>4} {'Win%':>6} {'Avg':>8} {'Median':>8} {'Total':>10} {'Avg Win':>9} {'Avg Loss':>9}")
    print("  " + "-" * 80)
    for label, sub in [("True PEAD", pead), ("False positive", fp), ("ALL", exec_df)]:
        r = sub["pregap_return"].dropna()
        if len(r) == 0:
            continue
        w = r[r > 0]; l = r[r <= 0]
        wr = len(w)/len(r)*100
        avg = r.mean()*100
        med = r.median()*100
        total = r.sum()*100
        aw = w.mean()*100 if len(w) > 0 else 0
        al = l.mean()*100 if len(l) > 0 else 0
        print(f"  {label:<20} {len(r):>4} {wr:>5.0f}% {avg:>+7.2f}% {med:>+7.2f}% {total:>+9.1f}% {aw:>+8.2f}% {al:>+8.2f}%")

    # ===== SECTION 7: LARGE PEAD ANALYSIS =====
    print(f"\n{'='*100}")
    print("7. LARGE PEAD ANALYSIS (CAR >= 10%)")
    print(f"{'='*100}")

    exec_df["car_10d_pct"] = np.expm1(exec_df["car_10d"]) * 100
    large_pead = exec_df[(exec_df["pead_pass"] == 1) & (exec_df["car_10d_pct"] >= 10)]
    small_pead = exec_df[(exec_df["pead_pass"] == 1) & (exec_df["car_10d_pct"] < 10)]

    print(f"\n  {'Group':<25} {'N':>4} {'Win%':>6} {'Avg':>8} {'Total':>10} {'Contribution':>13}")
    print("  " + "-" * 70)
    for label, sub in [("Large PEAD (CAR>=10%)", large_pead),
                        ("Small PEAD (CAR<10%)", small_pead),
                        ("Non-PEAD (FP)", fp),
                        ("ALL", exec_df)]:
        r = sub["pregap_return"].dropna()
        if len(r) == 0:
            continue
        wr = (r > 0).mean()*100
        avg = r.mean()*100
        total = r.sum()*100
        contrib = total / pnls.sum() * 100
        print(f"  {label:<25} {len(r):>4} {wr:>5.0f}% {avg:>+7.2f}% {total:>+9.1f}% {contrib:>11.1f}%")

    # ===== SECTION 8: PER-FOLD BREAKDOWN =====
    print(f"\n{'='*100}")
    print("8. PER-FOLD BREAKDOWN")
    print(f"{'='*100}")

    print(f"\n  {'Fold':<8} {'Period':<18} {'N':>4} {'Win%':>6} {'Avg':>8} {'Med':>8} {'Total':>10} {'PEAD':>6} {'Prec':>6}")
    print("  " + "-" * 80)
    for fi in range(1, 5):
        sub = exec_df[exec_df["fold"] == fi]
        if len(sub) == 0:
            continue
        r = sub["pregap_return"].dropna()
        wr = (r > 0).mean()*100
        avg = r.mean()*100
        med = r.median()*100
        total = r.sum()*100
        n_p = int(sub["pead_pass"].sum())
        prec = n_p/len(r)*100
        period = DEFAULT_FOLDS[fi-1][1][:7] + " → " + DEFAULT_FOLDS[fi-1][2][:7]
        print(f"  {fi:<8} {period:<18} {len(r):>4} {wr:>5.0f}% {avg:>+7.2f}% {med:>+7.2f}% {total:>+9.1f}% {n_p:>5} {prec:>5.0f}%")

    # ===== SECTION 9: SECTOR BREAKDOWN =====
    print(f"\n{'='*100}")
    print("9. SECTOR BREAKDOWN")
    print(f"{'='*100}")

    print(f"\n  {'Sector':<8} {'N':>4} {'Win%':>6} {'Avg':>8} {'Total':>10} {'PEAD':>5} {'Prec':>6}")
    print("  " + "-" * 55)
    for sec in sorted(exec_df["sector"].dropna().unique()):
        sub = exec_df[exec_df["sector"] == sec]
        r = sub["pregap_return"].dropna()
        if len(r) == 0:
            continue
        wr = (r > 0).mean()*100
        avg = r.mean()*100
        total = r.sum()*100
        n_p = int(sub["pead_pass"].sum())
        prec = n_p/len(r)*100
        print(f"  {sec:<8} {len(r):>4} {wr:>5.0f}% {avg:>+7.2f}% {total:>+9.1f}% {n_p:>4} {prec:>5.0f}%")

    # ===== SECTION 10: TIME IN MARKET / PORTFOLIO EFFICIENCY =====
    print(f"\n{'='*100}")
    print("10. PORTFOLIO EFFICIENCY")
    print(f"{'='*100}")

    entry_dates = pd.to_datetime(exec_df["pregap_entry_date"])
    exit_dates = pd.to_datetime(exec_df["pregap_exit_date"])
    total_days = (exit_dates.max() - entry_dates.min()).days
    total_hold_days = (exit_dates - entry_dates).dt.days.sum()
    # 4 slots available
    slot_days = total_days * N_SLOTS
    utilization = total_hold_days / slot_days * 100 if slot_days > 0 else 0

    print(f"\n  {'Metric':<45} {'Value':>15}")
    print("  " + "-" * 60)
    print(f"  {'Backtest period':<45} {str(entry_dates.min().date()) + ' → ' + str(exit_dates.max().date()):>15}")
    print(f"  {'Total calendar days':<45} {total_days:>15}")
    print(f"  {'Total trade-hold days':<45} {int(total_hold_days):>15}")
    print(f"  {'Max slots':<45} {N_SLOTS:>15}")
    print(f"  {'Available slot-days':<45} {int(slot_days):>15}")
    print(f"  {'Slot utilization':<45} {utilization:>14.1f}%")
    print(f"  {'Trades per year':<45} {n / (total_days/365):>15.1f}")
    print(f"  {'Avg hold days per trade':<45} {(exit_dates - entry_dates).dt.days.mean():>15.1f}")

    # ===== SECTION 11: EQUITY CURVE STATISTICS =====
    print(f"\n{'='*100}")
    print("11. EQUITY CURVE STATISTICS (equal-weight 1/4 NAV per slot)")
    print(f"{'='*100}")

    # Simplified: treat as sum of returns (no compounding across slots)
    # For true equity curve, we'd simulate the 4-slot portfolio with compounding
    cumulative = pnls.cumsum() * 100
    # Max drawdown of cumulative sum
    running_max = cumulative.cummax()
    eq_dd = (cumulative - running_max)
    max_dd = eq_dd.min()
    max_dd_idx = eq_dd.idxmin()

    print(f"\n  {'Metric':<45} {'Value':>15}")
    print("  " + "-" * 60)
    print(f"  {'Final cumulative PnL':<45} {cumulative.iloc[-1]:>+14.1f}%")
    print(f"  {'Max drawdown (cumulative sum)':<45} {max_dd:>+14.1f}%")
    print(f"  {'Avg return per trade':<45} {pnls.mean()*100:>+14.2f}%")
    print(f"  {'Sharpe per trade (mean/std)':<45} {pnls.mean()/pnls.std():>15.3f}")
    # Annualized (approximate, ~50 trades/yr)
    trades_per_yr = n / (total_days/365)
    ann_sharpe = (pnls.mean()/pnls.std()) * np.sqrt(trades_per_yr)
    print(f"  {'Trades per year':<45} {trades_per_yr:>15.1f}")
    print(f"  {'Annualized Sharpe (approx)':<45} {ann_sharpe:>15.2f}")

    # ===== SECTION 12: ENTRY TIMING BMO vs AMC =====
    print(f"\n{'='*100}")
    print("12. BMO vs AMC ENTRY TIMING")
    print(f"{'='*100}")

    print(f"\n  {'Timing':<10} {'N':>4} {'Win%':>6} {'Avg':>8} {'Total':>10} {'PEAD':>6}")
    print("  " + "-" * 50)
    for label, mask in [("BMO", exec_df["is_bmo"] == 1), ("AMC", exec_df["is_bmo"] == 0)]:
        sub = exec_df[mask]
        r = sub["pregap_return"].dropna()
        if len(r) == 0:
            continue
        wr = (r > 0).mean()*100
        avg = r.mean()*100
        total = r.sum()*100
        n_p = int(sub["pead_pass"].sum())
        print(f"  {label:<10} {len(r):>4} {wr:>5.0f}% {avg:>+7.2f}% {total:>+9.1f}% {n_p:>5}")

    print(f"\n{'='*100}")
    print("DONE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()

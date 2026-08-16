#!/usr/bin/env python3
"""
Detailed trade-level statistics for 3-class P(any)>=0.20 operating point.

Reports:
  1. Core stats: N, win rate, avg win/loss, payoff, expectancy, std
  2. Return distribution: min, percentiles, max
  3. Max drawdown per trade (intra-hold, using Adj_Low)
  4. PEAD vs non-PEAD breakdown
  5. Per-fold breakdown
  6. Weekly trade frequency
  7. BMO vs AMC breakdown
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
CAR_LARGE_THRESH = 10.0
THETA = 0.20

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]


def fit_clf_3class(X_tr, y_tr, X_val, y_val, hp):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss",
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


def compute_pregap_and_paths(df, db_path, hold_days=5):
    """Compute pre-gap returns AND intra-hold daily path for max drawdown."""
    df = df.copy()
    df["pregap_return"] = np.nan
    df["pregap_entry_date"] = pd.NaT
    df["pregap_exit_date"] = pd.NaT
    df["max_dd_during_hold"] = np.nan
    df["min_path_during_hold"] = np.nan
    df["day1_return"] = np.nan  # return on the gap day (entry close -> next close)

    with pd.HDFStore(db_path, mode="r") as s:
        for idx, row in df.iterrows():
            pt = row["permaTicker"]
            key = f"/sp400/{pt}"
            if key not in s:
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_close = p["Adj_Close"].values
            p_low = p["Adj_Low"].values
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
            exit_price = p_close[exit_t]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue

            df.at[idx, "pregap_return"] = float(exit_price / entry_price - 1.0)
            df.at[idx, "pregap_entry_date"] = pd.Timestamp(p_index[entry_t])
            df.at[idx, "pregap_exit_date"] = pd.Timestamp(p_index[exit_t])

            # Day-1 return (entry close -> next trading day close)
            if entry_t + 1 < len(p_close):
                d1 = p_close[entry_t + 1]
                if pd.notna(d1):
                    df.at[idx, "day1_return"] = float(d1 / entry_price - 1.0)

            # Intra-hold path: daily closes from entry_t to exit_t
            path_closes = p_close[entry_t:exit_t + 1]
            path_lows = p_low[entry_t:exit_t + 1]
            if len(path_closes) >= 2:
                cum_ret = path_closes / entry_price - 1.0
                # Max drawdown from peak
                running_max = np.maximum.accumulate(cum_ret)
                drawdowns = cum_ret - running_max
                df.at[idx, "max_dd_during_hold"] = float(np.nanmin(drawdowns))
                df.at[idx, "min_path_during_hold"] = float(np.nanmin(cum_ret))

    return df


def main():
    print("=" * 90)
    print(f"DETAILED STATS: 3-class P(any)>={THETA}")
    print(f"  Pre-gap entry, {EXIT_SNAP}-day hold, {N_SLOTS} slots, weekly batch")
    print("=" * 90)

    print("\n[1] Loading + priming + gates ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)

    df["car_10d_pct"] = np.expm1(df["car_10d"]) * 100
    df["label_3class"] = 0
    df.loc[(df["pead_pass"] == 1) & (df["car_10d_pct"] < CAR_LARGE_THRESH), "label_3class"] = 1
    df.loc[(df["pead_pass"] == 1) & (df["car_10d_pct"] >= CAR_LARGE_THRESH), "label_3class"] = 2

    print("[2] Computing pre-gap returns + intra-hold paths ...")
    df = compute_pregap_and_paths(df, DB, EXIT_SNAP)

    print("[3] Training 3-class classifiers per fold ...")
    fold_data = {}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_tr = train_df[SUNDAY_SAFE]; X_sv = sweep_df[SUNDAY_SAFE]
        X_te = test_df[SUNDAY_SAFE]
        X_ts = pd.concat([X_tr, X_sv])
        y_tr_3 = train_df["label_3class"].values
        y_sv_3 = sweep_df["label_3class"].values
        y_te_3 = test_df["label_3class"].values
        y_ts_3 = np.concatenate([y_tr_3, y_sv_3])
        hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
        clf_3 = fit_clf_3class(X_ts, y_ts_3, X_te, y_te_3, hp)
        proba_3 = clf_3.predict_proba(X_te)
        test_df = test_df.copy()
        test_df["p_any_pead"] = proba_3[:, 1] + proba_3[:, 2]
        test_df["p_large"] = proba_3[:, 2]
        fold_data[fi] = {"test_df": test_df}

    print("[4] Running weekly batch selection + collecting trades ...")
    all_exec_list = []
    for fi in range(1, 5):
        test_df = fold_data[fi]["test_df"]
        mask = (test_df["p_any_pead"] >= THETA) & (test_df["pregap_return"].notna())
        picks = test_df[mask].copy()
        if len(picks) == 0:
            continue
        picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
        picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
        picks["fold"] = fi
        selected = select_weekly_top_n(picks, N_SLOTS, sort_col="p_any_pead")
        if len(selected) > 0:
            all_exec_list.append(selected)

    exec_df = pd.concat(all_exec_list).reset_index(drop=True)
    print(f"  Total executed trades: {len(exec_df)}")

    pnls = exec_df["pregap_return"].dropna()
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n = len(pnls)

    # ===== ANALYSIS =====
    print(f"\n{'='*90}")
    print(f"CORE TRADE-LEVEL STATS (3-class P(any)>={THETA})")
    print(f"{'='*90}")

    print(f"\n  1. CORE STATS")
    print(f"    N trades:              {n}")
    print(f"    Win rate:              {len(wins)/n*100:.1f}%")
    print(f"    Avg win:               {wins.mean()*100:+.2f}%")
    print(f"    Avg loss:              {losses.mean()*100:+.2f}%")
    print(f"    Payoff ratio:          {wins.mean()/abs(losses.mean()):.2f}")
    print(f"    Expectancy/trade:      {pnls.mean()*100:+.2f}%")
    print(f"    Median return:         {pnls.median()*100:+.2f}%")
    print(f"    Std per trade:          {pnls.std()*100:.2f}%")
    print(f"    Total PnL (sum):       {pnls.sum()*100:+.1f}%")

    print(f"\n  2. RETURN DISTRIBUTION")
    vals = pnls.values * 100
    print(f"    min={vals.min():+.2f}%, p10={np.percentile(vals,10):+.2f}%, "
          f"p25={np.percentile(vals,25):+.2f}%, p50={np.percentile(vals,50):+.2f}%, "
          f"p75={np.percentile(vals,75):+.2f}%, p90={np.percentile(vals,90):+.2f}%, "
          f"max={vals.max():+.2f}%")
    print(f"    Trades < -5%:   {(vals < -5).sum()}")
    print(f"    Trades < -10%:  {(vals < -10).sum()}")
    print(f"    Trades > +5%:   {(vals > 5).sum()}")
    print(f"    Trades > +10%:  {(vals > 10).sum()}")
    print(f"    Trades > +20%:  {(vals > 20).sum()}")

    # Win/loss detail
    print(f"\n  3. WIN/LOSS DETAIL")
    print(f"    Wins:   n={len(wins)}, avg={wins.mean()*100:+.2f}%, "
          f"med={wins.median()*100:+.2f}%, max={wins.max()*100:+.2f}%")
    print(f"    Losses: n={len(losses)}, avg={losses.mean()*100:+.2f}%, "
          f"med={losses.median()*100:+.2f}%, min={losses.min()*100:+.2f}%")

    # Max drawdown during hold
    print(f"\n  4. MAX DRAWDOWN DURING HOLD (intra-hold, using daily closes)")
    dd = exec_df["max_dd_during_hold"].dropna() * 100
    minp = exec_df["min_path_during_hold"].dropna() * 100
    print(f"    All trades:     mean={dd.mean():+.2f}%, med={dd.median():+.2f}%, "
          f"min={dd.min():+.2f}%, max={dd.max():+.2f}%")
    dd_wins = exec_df.loc[wins.index, "max_dd_during_hold"].dropna() * 100
    dd_losses = exec_df.loc[losses.index, "max_dd_during_hold"].dropna() * 100
    print(f"    Winners:        mean={dd_wins.mean():+.2f}%, med={dd_wins.median():+.2f}%")
    print(f"    Losers:         mean={dd_losses.mean():+.2f}%, med={dd_losses.median():+.2f}%")

    print(f"\n    Lowest point during hold (min path):")
    print(f"      All trades:     mean={minp.mean():+.2f}%, med={minp.median():+.2f}%")
    print(f"      Winners:        mean={exec_df.loc[wins.index, 'min_path_during_hold'].mean()*100:+.2f}%")
    print(f"      Losers:         mean={exec_df.loc[losses.index, 'min_path_during_hold'].mean()*100:+.2f}%")

    # Day-1 return (gap day)
    print(f"\n  5. DAY-1 RETURN (entry close -> next day close)")
    d1 = exec_df["day1_return"].dropna() * 100
    print(f"    All trades:     mean={d1.mean():+.2f}%, med={d1.median():+.2f}%")
    d1_wins = exec_df.loc[wins.index, "day1_return"].dropna() * 100
    d1_losses = exec_df.loc[losses.index, "day1_return"].dropna() * 100
    print(f"    Winners:        mean={d1_wins.mean():+.2f}%, med={d1_wins.median():+.2f}%")
    print(f"    Losers:         mean={d1_losses.mean():+.2f}%, med={d1_losses.median():+.2f}%")

    # PEAD breakdown
    print(f"\n  6. PEAD vs NON-PEAD BREAKDOWN")
    for label, mask in [("No PEAD", exec_df["label_3class"] == 0),
                        ("Small PEAD", exec_df["label_3class"] == 1),
                        ("Large PEAD", exec_df["label_3class"] == 2),
                        ("Any PEAD", exec_df["label_3class"] >= 1)]:
        sub = exec_df[mask]
        if len(sub) == 0:
            continue
        sp = sub["pregap_return"].dropna()
        sw = sp[sp > 0]
        sl = sp[sp <= 0]
        wr = len(sw) / len(sp) * 100 if len(sp) > 0 else 0
        aw = sw.mean() * 100 if len(sw) > 0 else 0
        al = sl.mean() * 100 if len(sl) > 0 else 0
        payoff = aw / abs(al) if al != 0 else float('inf')
        sdd = sub["max_dd_during_hold"].dropna().mean() * 100
        print(f"    {label:<12} N={len(sp):>3} ({len(sp)/n*100:>4.1f}%)  "
              f"Win={wr:>5.1f}%  Avg={sp.mean()*100:>+6.2f}%  "
              f"Win={aw:>+6.2f}%  Loss={al:>+6.2f}%  Payoff={payoff:>5.2f}  "
              f"MaxDD={sdd:>+6.2f}%")

    # BMO vs AMC
    print(f"\n  7. BMO vs AMC BREAKDOWN")
    exec_df["is_bmo"] = exec_df["is_bmo"].fillna(0).astype(int) == 1
    for label, mask in [("BMO", exec_df["is_bmo"]),
                        ("AMC", ~exec_df["is_bmo"])]:
        sub = exec_df[mask]
        if len(sub) == 0:
            continue
        sp = sub["pregap_return"].dropna()
        sw = sp[sp > 0]
        sl = sp[sp <= 0]
        wr = len(sw) / len(sp) * 100 if len(sp) > 0 else 0
        aw = sw.mean() * 100 if len(sw) > 0 else 0
        al = sl.mean() * 100 if len(sl) > 0 else 0
        payoff = aw / abs(al) if al != 0 else float('inf')
        print(f"    {label:<5} N={len(sp):>3} ({len(sp)/n*100:>4.1f}%)  "
              f"Win={wr:>5.1f}%  Avg={sp.mean()*100:>+6.2f}%  "
              f"Win={aw:>+6.2f}%  Loss={al:>+6.2f}%  Payoff={payoff:>5.2f}")

    # Per-fold
    print(f"\n  8. PER-FOLD BREAKDOWN")
    print(f"    {'Fold':>4} {'N':>4} {'Win%':>6} {'Avg':>8} {'Total':>8} {'Std':>7} {'MaxDD':>8}")
    for fi in range(1, 5):
        sub = exec_df[exec_df["fold"] == fi]
        sp = sub["pregap_return"].dropna()
        if len(sp) == 0:
            continue
        wr = (sp > 0).mean() * 100
        avg = sp.mean() * 100
        total = sp.sum() * 100
        std = sp.std() * 100
        dd = sub["max_dd_during_hold"].dropna().mean() * 100
        print(f"    {fi:>4} {len(sp):>4} {wr:>5.1f}% {avg:>+7.2f}% {total:>+7.1f}% {std:>6.2f}% {dd:>+7.2f}%")

    # Weekly stats
    print(f"\n  9. WEEKLY TRADE FREQUENCY")
    exec_df["entry_date"] = pd.to_datetime(exec_df["pregap_entry_date"])
    exec_df["year_week"] = exec_df["entry_date"].dt.strftime("%G-W%V")
    weekly = exec_df.groupby("year_week").size()
    print(f"    Weeks with trades:   {len(weekly)}")
    print(f"    Trades per week:     mean={weekly.mean():.1f}, med={weekly.median():.0f}, max={weekly.max()}")
    print(f"    Weeks with 1 trade:  {(weekly==1).sum()}")
    print(f"    Weeks with 2 trades: {(weekly==2).sum()}")
    print(f"    Weeks with 3 trades: {(weekly==3).sum()}")
    print(f"    Weeks with 4 trades: {(weekly==4).sum()}")
    print(f"    Trades per year:     ~{n/4*2:.0f} (2 years of OOS)")

    # Summary comparison vs binary
    print(f"\n  10. COMPARISON vs BINARY theta=0.25 (from prior run)")
    print(f"    {'Metric':<25} {'Binary 0.25':>15} {'3-class 0.20':>15} {'Delta':>10}")
    print(f"    {'-'*65}")
    print(f"    {'N trades':<25} {'69':>15} {n:>15} {n-69:>+10}")
    print(f"    {'Win rate':<25} {'65.2%':>15} {len(wins)/n*100:>14.1f}% {(len(wins)/n-0.652)*100:>+9.1f}pp")
    print(f"    {'Avg win':<25} {'+14.55%':>15} {wins.mean()*100:>+14.2f}% {(wins.mean()-0.1455)*100:>+9.2f}pp")
    print(f"    {'Avg loss':<25} {'-5.41%':>15} {losses.mean()*100:>+14.2f}% {(losses.mean()+0.0541)*100:>+9.2f}pp")
    print(f"    {'Payoff':<25} {'2.69':>15} {wins.mean()/abs(losses.mean()):>15.2f}")
    print(f"    {'Expectancy/trade':<25} {'+7.61%':>15} {pnls.mean()*100:>+14.2f}%")
    print(f"    {'TOTAL PnL':<25} {'+525.2%':>15} {pnls.sum()*100:>+14.1f}%")

    print(f"\n{'='*90}")


if __name__ == "__main__":
    main()

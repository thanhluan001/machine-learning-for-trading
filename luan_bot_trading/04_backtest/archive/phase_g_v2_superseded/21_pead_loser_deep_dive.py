#!/usr/bin/env python3
"""
Deep dive: why do PEAD-labeled trades lose money at T+5 exit?

Two hypotheses:
  H1: PEAD = CAR (abnormal return). If market crashes, stock can have
      positive CAR (vs market) but negative raw return. We hold the
      stock (raw), so we lose money even on "correct" PEAD calls.
  H2: PEAD drift is T+1->T+10 (10 days). We exit at T+5. If drift
      concentrates in days 6-10, we miss it. PEAD losers at T+5 may
      recover by T+10.

This script checks both for the PEAD losers in the 5-day-hold backtest.
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
HOLD_DAYS = 5
EXIT_SNAP = 5

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


def compute_ijh_path(report_date, ijh_index, ijh_close, n_days=11):
    """Compute IJH (market benchmark) raw return path from T+1 to T+n_days.
    Returns array of cumulative returns (Close[T+1+t]/Close[T] - 1) for t=0..n_days.
    This is the MARKET return, to compare against stock's raw return."""
    rdate = pd.to_datetime(report_date).to_datetime64()
    mask = ijh_index >= rdate
    if not mask.any():
        return None
    t_idx = int(np.argmax(mask))
    if t_idx + n_days >= len(ijh_close):
        return None
    base = ijh_close[t_idx]  # Close[T] (report day)
    if pd.isna(base) or base <= 0:
        return None
    path = []
    for t in range(n_days + 1):
        c = ijh_close[t_idx + t]
        if pd.isna(c):
            path.append(np.nan)
        else:
            path.append(float(c / base - 1.0))
    return np.array(path)


def main():
    print("=" * 80)
    print(f"PEAD LOSER DEEP DIVE — theta={THETA}, {N_SLOTS} slots, {HOLD_DAYS}-day hold")
    print("=" * 80)

    # Load + prime + gates + paths
    print("\n[1] Loading + priming + gates + paths ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)
    print(f"    rows: {len(df)}, pead: {int(df['pead_pass'].sum())}")

    # Load IJH benchmark
    print("[2] Loading IJH market benchmark ...")
    with pd.HDFStore(DB, mode="r") as s:
        ijh_df = s["/macros/IJH"]
    ijh_index = pd.to_datetime(ijh_df["Date"]).values
    ijh_close = ijh_df["Close"].values
    print(f"    IJH: {len(ijh_df)} rows, {ijh_index[0]} -> {ijh_index[-1]}")

    # Cache calendar + remap exit to 5-day
    print("[3] Caching trading calendar + remapping to 5-day hold ...")
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

    # Pre-train classifiers
    print("\n[4] Pre-training 4 fold classifiers (gamma=3) ...")
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
        fold_data[fi] = {"test_df": test_df, "picks": picks}

    # Run simulation + collect executed trades
    print(f"\n[5] Running {N_SLOTS}-slot portfolio simulation per fold ...")
    all_trades = []
    for fi in range(1, 5):
        picks = fold_data[fi]["picks"]
        selected = select_weekly_top_n(picks, N_SLOTS)
        result = rb._simulate_with_cached_calendar(selected, N_SLOTS, 100_000.0, calendar)
        executed = result.get("trades_done", pd.DataFrame())
        if len(executed) == 0:
            continue
        pead_lookup = selected[["permaTicker", "entry_date", "pead_pass", "p"]].copy()
        pead_lookup["entry_date"] = pd.to_datetime(pead_lookup["entry_date"])
        executed["entry_date"] = pd.to_datetime(executed["entry_date"])
        executed = executed.merge(pead_lookup, on=["permaTicker", "entry_date"], how="left")
        # Join full path columns (t0 through t11)
        path_cols = [f"path_pnl_t{t}_pct" for t in range(12)]
        path_lookup = selected[["permaTicker", "entry_date"] + path_cols].copy()
        path_lookup["entry_date"] = pd.to_datetime(path_lookup["entry_date"])
        executed = executed.merge(path_lookup, on=["permaTicker", "entry_date"], how="left")
        # Also join report_date and other metadata
        meta_lookup = selected[["permaTicker", "entry_date", "report_date",
                                "canonical_ticker"]].copy()
        meta_lookup["entry_date"] = pd.to_datetime(meta_lookup["entry_date"])
        executed = executed.merge(meta_lookup, on=["permaTicker", "entry_date"], how="left")
        executed["fold"] = fi
        all_trades.append(executed)

    all_trades_df = pd.concat(all_trades).reset_index(drop=True)
    print(f"  Total executed trades: {len(all_trades_df)}")

    # Compute IJH market path for each trade
    print(f"\n[6] Computing IJH market benchmark path for each trade ...")
    ijh_paths = []
    ijh_t5 = []  # market return at T+5
    ijh_t10 = []  # market return at T+10
    for _, row in all_trades_df.iterrows():
        rdate = row["report_date"]
        mpath = compute_ijh_path(rdate, ijh_index, ijh_close, n_days=11)
        if mpath is not None:
            ijh_paths.append(mpath)
            ijh_t5.append(mpath[5])  # Close[T+5]/Close[T]-1
            ijh_t10.append(mpath[10])  # Close[T+10]/Close[T]-1
        else:
            ijh_paths.append(None)
            ijh_t5.append(np.nan)
            ijh_t10.append(np.nan)
    all_trades_df["ijh_ret_t5"] = ijh_t5
    all_trades_df["ijh_ret_t10"] = ijh_t10

    # Classification
    all_trades_df["final_return_t5"] = all_trades_df["realized_arith_pct"]  # T+5 exit
    all_trades_df["final_return_t10"] = all_trades_df["path_pnl_t10_pct"]   # T+10 (if held longer)
    all_trades_df["is_win_t5"] = all_trades_df["final_return_t5"] > 0
    all_trades_df["is_pead"] = all_trades_df["pead_pass"] == 1

    # ===== ANALYSIS =====
    print(f"\n{'='*80}")
    print("PEAD LOSER DEEP DIVE")
    print(f"{'='*80}")

    pead_trades = all_trades_df[all_trades_df["is_pead"]]
    pead_winners = pead_trades[pead_trades["is_win_t5"]]
    pead_losers = pead_trades[~pead_trades["is_win_t5"]]

    print(f"\n1. PEAD TRADES OVERVIEW ({len(pead_trades)} total)")
    print(f"   Winners (at T+5): {len(pead_winners)} ({len(pead_winners)/len(pead_trades)*100:.1f}%)")
    print(f"   Losers (at T+5):  {len(pead_losers)} ({len(pead_losers)/len(pead_trades)*100:.1f}%)")

    # H1: Market context
    print(f"\n2. HYPOTHESIS H1: Market downturn (CAR vs raw return)")
    print(f"   PEAD = CAR (abnormal return) >= 3%. If market drops, stock")
    print(f"   can have positive CAR but negative raw return.\n")
    print(f"   {'Group':<20} {'Stock T+5':>12} {'IJH T+5':>12} {'CAR T+5':>12} {'Stock T+10':>12} {'IJH T+10':>12} {'CAR T+10':>12}")
    for label, sub in [("PEAD winners", pead_winners),
                       ("PEAD losers", pead_losers),
                       ("PEAD all", pead_trades)]:
        s5 = sub["final_return_t5"].mean() * 100
        m5 = sub["ijh_ret_t5"].mean() * 100
        car5 = s5 - m5
        s10 = sub["final_return_t10"].mean() * 100
        m10 = sub["ijh_ret_t10"].mean() * 100
        car10 = s10 - m10
        print(f"   {label:<20} {s5:>+11.2f}% {m5:>+11.2f}% {car5:>+11.2f}% {s10:>+11.2f}% {m10:>+11.2f}% {car10:>+11.2f}%")

    print(f"\n   PEAD losers — was the market down?")
    mkt_down = (pead_losers["ijh_ret_t5"] < 0).sum()
    mkt_up = (pead_losers["ijh_ret_t5"] >= 0).sum()
    print(f"   Market DOWN at T+5: {mkt_down}/{len(pead_losers)} ({mkt_down/len(pead_losers)*100:.1f}%)")
    print(f"   Market UP at T+5:   {mkt_up}/{len(pead_losers)} ({mkt_up/len(pead_losers)*100:.1f}%)")
    print(f"   Market mean return at T+5: {pead_losers['ijh_ret_t5'].mean()*100:+.2f}%")

    # H2: T+5 vs T+10 recovery
    print(f"\n3. HYPOTHESIS H2: PEAD drift happens after T+5 exit")
    print(f"   PEAD is defined over T+1->T+10. We exit at T+5.")
    print(f"   Do PEAD losers at T+5 recover by T+10?\n")
    recovered = 0
    worsened = 0
    for _, row in pead_losers.iterrows():
        r5 = row["final_return_t5"]
        r10 = row["final_return_t10"]
        if pd.isna(r10):
            continue
        if r10 > r5 + 0.005:
            recovered += 1
        elif r10 < r5 - 0.005:
            worsened += 1
    n_valid = len(pead_losers) - pead_losers["final_return_t10"].isna().sum()
    print(f"   PEAD losers that RECOVERED by T+10: {recovered}/{n_valid} ({recovered/n_valid*100:.1f}%)")
    print(f"   PEAD losers that WORSENED by T+10: {worsened}/{n_valid} ({worsened/n_valid*100:.1f}%)")
    print(f"   PEAD losers T+5 mean: {pead_losers['final_return_t5'].mean()*100:+.2f}%")
    print(f"   PEAD losers T+10 mean: {pead_losers['final_return_t10'].mean()*100:+.2f}%")

    # Compute IJH path per trade (indexed by trade position)
    all_trades_df["_ijh_path"] = ijh_paths

    # Full path comparison
    print(f"\n4. AVERAGE PATH: PEAD losers vs PEAD winners (stock raw return)")
    print(f"   {'Day':<6} {'PEAD losers':>14} {'PEAD winners':>14} {'IJH (losers)':>14} {'IJH (winners)':>14}")
    for t in range(12):
        col = f"path_pnl_t{t}_pct"
        loser_v = pead_losers[col].mean() * 100 if col in pead_losers.columns else 0
        winner_v = pead_winners[col].mean() * 100 if col in pead_winners.columns else 0
        # IJH path for losers and winners
        loser_ijh = []
        winner_ijh = []
        for idx, row in all_trades_df.iterrows():
            p = row["_ijh_path"]
            if p is None or t >= len(p) or pd.isna(p[t]):
                continue
            if row["is_pead"] and not row["is_win_t5"]:
                loser_ijh.append(p[t] * 100)
            elif row["is_pead"] and row["is_win_t5"]:
                winner_ijh.append(p[t] * 100)
        ijh_l = np.mean(loser_ijh) if loser_ijh else 0
        ijh_w = np.mean(winner_ijh) if winner_ijh else 0
        print(f"   t{t:<5} {loser_v:>+13.2f}% {winner_v:>+13.2f}% {ijh_l:>+13.2f}% {ijh_w:>+13.2f}%")

    # Per-trade detail of PEAD losers
    print(f"\n5. PEAD LOSER DETAIL ({len(pead_losers)} trades)")
    print(f"   {'Ticker':<10} {'Fold':<5} {'Report Date':<12} {'P(PEAD)':>8} {'T+5 ret':>10} {'T+10 ret':>10} {'IJH T+5':>10} {'CAR T+5':>10} {'Recover?':>10}")
    for _, row in pead_losers.sort_values("final_return_t5").iterrows():
        ticker = row.get("canonical_ticker", row["permaTicker"])
        r5 = row["final_return_t5"] * 100
        r10 = row["final_return_t10"] * 100 if pd.notna(row["final_return_t10"]) else float('nan')
        m5 = row["ijh_ret_t5"] * 100 if pd.notna(row["ijh_ret_t5"]) else float('nan')
        car5 = r5 - m5 if pd.notna(m5) else float('nan')
        recover = "YES" if pd.notna(r10) and r10 > r5 + 0.5 else ("NO" if pd.notna(r10) else "?")
        print(f"   {ticker:<10} {int(row['fold']):<5} {str(row['report_date'])[:10]:<12} "
              f"{row['p']:>8.3f} {r5:>+9.2f}% {r10:>+9.2f}% {m5:>+9.2f}% {car5:>+9.2f}% {recover:>10}")

    # Aggregate: what if we held to T+10 instead of T+5?
    print(f"\n6. COUNTERFACTUAL: Hold PEAD trades to T+10 instead of T+5")
    for label, sub in [("PEAD all", pead_trades),
                       ("PEAD winners", pead_winners),
                       ("PEAD losers", pead_losers)]:
        r5 = sub["final_return_t5"].mean() * 100
        r10 = sub["final_return_t10"].mean() * 100
        print(f"   {label:<15} T+5: {r5:>+6.2f}%  ->  T+10: {r10:>+6.2f}%  (delta: {r10-r5:+.2f}%)")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

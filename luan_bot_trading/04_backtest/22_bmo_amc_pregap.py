#!/usr/bin/env python3
"""
Investigate BMO vs AMC timing for PEAD losers, and test Close[T-1]/Close[T]
entry (pre-gap) counterfactual.

TIMING LOGIC:
  BMO (before market open): announced before Open[T+1].
    - Gap happens at Open[T+1].
    - To capture gap: enter at Close[T-1] (prior day close, before BMO news).
    - Sunday predicts PEAD -> buy Monday close -> Tuesday BMO -> gap up.

  AMC (after market close): announced after Close[T].
    - Gap happens at Open[T+1] (next morning).
    - To capture gap: enter at Close[T] (same-day close, before AMC news).
    - Sunday predicts PEAD -> buy Tuesday close -> Tuesday evening AMC ->
      Wednesday gap up.

Both require the Sunday classifier to predict PEAD BEFORE the earnings
announcement. The current model enters at Open[T+1] (post-gap), which is
why the gap eats the drift.
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


def main():
    print("=" * 80)
    print("BMO vs AMC TIMING + PRE-GAP ENTRY COUNTERFACTUAL")
    print("=" * 80)

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

    # Pre-train classifiers
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

    # Simulate + collect executed trades
    all_trades = []
    for fi in range(1, 5):
        picks = fold_data[fi]["picks"]
        selected = select_weekly_top_n(picks, N_SLOTS)
        result = rb._simulate_with_cached_calendar(selected, N_SLOTS, 100_000.0, calendar)
        executed = result.get("trades_done", pd.DataFrame())
        if len(executed) == 0:
            continue
        lookup = selected[["permaTicker", "entry_date", "pead_pass", "p",
                           "report_date", "canonical_ticker",
                           "is_bmo"]].copy()
        lookup["entry_date"] = pd.to_datetime(lookup["entry_date"])
        executed["entry_date"] = pd.to_datetime(executed["entry_date"])
        executed = executed.merge(lookup, on=["permaTicker", "entry_date"], how="left")
        executed["fold"] = fi
        all_trades.append(executed)

    all_df = pd.concat(all_trades).reset_index(drop=True)

    # Check before_after_market coverage
    print(f"\n[2] BMO/AMC coverage on executed trades (is_bmo column)")
    print(f"  Total trades: {len(all_df)}")
    print(f"  BMO (is_bmo=1): {int(all_df['is_bmo'].sum())}")
    print(f"  AMC (is_bmo=0): {int((~all_df['is_bmo']).sum())}")

    # Split into BMO/AMC (is_bmo=1 means BMO, is_bmo=0 means AMC)
    all_df["is_bmo"] = all_df["is_bmo"].fillna(0).astype(int) == 1
    all_df["is_amc"] = all_df["is_bmo"] == False  # i.e. is_bmo=0

    # ===== ANALYSIS =====
    print(f"\n{'='*80}")
    print("BMO vs AMC ANALYSIS")
    print(f"{'='*80}")

    print(f"\n1. CURRENT ENTRY (Open[T+1], post-gap)")
    print(f"   {'Group':<20} {'N':>5} {'Win%':>7} {'Avg ret':>10} {'Avg win':>10} {'Avg loss':>10}")
    for label, mask in [("All", pd.Series([True]*len(all_df))),
                        ("BMO", all_df["is_bmo"]),
                        ("AMC", all_df["is_amc"]),
                        ("PEAD", all_df["pead_pass"] == 1),
                        ("PEAD + BMO", all_df["is_bmo"] & (all_df["pead_pass"] == 1)),
                        ("PEAD + AMC", all_df["is_amc"] & (all_df["pead_pass"] == 1))]:
        sub = all_df[mask]
        if len(sub) == 0:
            print(f"   {label:<20} {0:>5} (no trades)")
            continue
        pnls = sub["realized_arith_pct"]
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        wr = len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0
        avg_w = wins.mean() * 100 if len(wins) > 0 else 0
        avg_l = losses.mean() * 100 if len(losses) > 0 else 0
        print(f"   {label:<20} {len(sub):>5} {wr:>6.1f}% {pnls.mean()*100:>+9.2f}% "
              f"{avg_w:>+9.2f}% {avg_l:>+9.2f}%")

    # ===== PRE-GAP ENTRY COUNTERFACTUAL =====
    print(f"\n{'='*80}")
    print("2. PRE-GAP ENTRY COUNTERFACTUAL")
    print(f"{'='*80}")
    print(f"\n   Timing logic:")
    print(f"     BMO: announced before Open[T+1]")
    print(f"       -> Current:  enter Open[T+1]  (post-gap)")
    print(f"       -> Pre-gap:  enter Close[T-1]  (day before BMO)")
    print(f"     AMC: announced after Close[T]")
    print(f"       -> Current:  enter Open[T+1]  (post-gap)")
    print(f"       -> Pre-gap:  enter Close[T]    (same-day close, before AMC)")
    print(f"\n   Exit stays Close[T+5] for both (5-day hold from entry).")
    print(f"\n   NOTE: Pre-gap entry requires buying BEFORE knowing the earnings")
    print(f"   result. The Sunday classifier predicts PEAD, so we buy on")
    print(f"   conviction. This captures the gap instead of being eaten by it.")

    # Compute pre-gap returns
    print(f"\n   Computing pre-gap returns ...")
    pre_gap_returns = []
    with pd.HDFStore(DB, mode="r") as s:
        for idx, row in all_df.iterrows():
            pt = row["permaTicker"]
            key = f"/sp400/{pt}"
            if key not in s:
                pre_gap_returns.append(np.nan)
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_close = p["Adj_Close"].values

            rdate = pd.to_datetime(row["report_date"]).to_datetime64()
            t_mask = p_index >= rdate
            if not t_mask.any():
                pre_gap_returns.append(np.nan)
                continue
            t_idx = int(np.argmax(t_mask))

            is_bmo = bool(row.get("is_bmo", False))
            is_amc = not is_bmo

            if is_bmo:
                # Pre-gap entry: Close[T-1], exit Close[T+5] (from T-1 base, that's 6 trading days)
                if t_idx < 1 or t_idx + 5 >= len(p_close):
                    pre_gap_returns.append(np.nan)
                    continue
                entry_price = p_close[t_idx - 1]  # Close[T-1]
                exit_price = p_close[t_idx + 5]   # Close[T+5]
            elif is_amc:
                # Pre-gap entry: Close[T], exit Close[T+5] (from T base, that's 5 trading days)
                if t_idx + 5 >= len(p_close):
                    pre_gap_returns.append(np.nan)
                    continue
                entry_price = p_close[t_idx]       # Close[T]
                exit_price = p_close[t_idx + 5]    # Close[T+5]
            else:
                # Unknown timing — skip
                pre_gap_returns.append(np.nan)
                continue

            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                pre_gap_returns.append(np.nan)
                continue
            pre_gap_returns.append(float(exit_price / entry_price - 1.0))

    all_df["pregap_return"] = pre_gap_returns

    # Also compute the gap itself for reference
    gap_returns = []
    with pd.HDFStore(DB, mode="r") as s:
        for idx, row in all_df.iterrows():
            pt = row["permaTicker"]
            key = f"/sp400/{pt}"
            if key not in s:
                gap_returns.append(np.nan)
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_close = p["Adj_Close"].values
            p_open = p["Adj_Open"].values
            rdate = pd.to_datetime(row["report_date"]).to_datetime64()
            t_mask = p_index >= rdate
            if not t_mask.any():
                gap_returns.append(np.nan)
                continue
            t_idx = int(np.argmax(t_mask))
            if t_idx + 1 >= len(p_open):
                gap_returns.append(np.nan)
                continue
            c_t = p_close[t_idx]
            o_t1 = p_open[t_idx + 1]
            if pd.isna(c_t) or pd.isna(o_t1) or c_t <= 0:
                gap_returns.append(np.nan)
                continue
            gap_returns.append(float(o_t1 / c_t - 1.0))
    all_df["opening_gap"] = gap_returns

    # Compare current vs pre-gap
    print(f"\n3. CURRENT vs PRE-GAP ENTRY (all trades with known BMO/AMC)")
    print(f"   {'Group':<20} {'N':>5} {'Current':>10} {'Pre-gap':>10} {'Delta':>10} {'Avg gap':>10}")
    for label, mask in [("All", pd.Series([True]*len(all_df))),
                        ("BMO", all_df["is_bmo"]),
                        ("AMC", all_df["is_amc"]),
                        ("PEAD", all_df["pead_pass"] == 1),
                        ("PEAD + BMO", all_df["is_bmo"] & (all_df["pead_pass"] == 1)),
                        ("PEAD + AMC", all_df["is_amc"] & (all_df["pead_pass"] == 1)),
                        ("PEAD losers", (all_df["pead_pass"] == 1) & (all_df["realized_arith_pct"] <= 0))]:
        sub = all_df[mask & all_df["pregap_return"].notna()]
        if len(sub) == 0:
            print(f"   {label:<20} {0:>5} (no trades with pre-gap data)")
            continue
        cur = sub["realized_arith_pct"].mean() * 100
        pre = sub["pregap_return"].mean() * 100
        gap = sub["opening_gap"].mean() * 100
        delta = pre - cur
        print(f"   {label:<20} {len(sub):>5} {cur:>+9.2f}% {pre:>+9.2f}% {delta:>+9.2f}% {gap:>+9.2f}%")

    # Win/loss breakdown for pre-gap
    print(f"\n4. PRE-GAP ENTRY: WIN/LOSS BREAKDOWN")
    print(f"   {'Group':<20} {'N':>5} {'Win%':>7} {'Avg ret':>10} {'Avg win':>10} {'Avg loss':>10} {'Payoff':>8}")
    for label, mask in [("All", pd.Series([True]*len(all_df))),
                        ("BMO", all_df["is_bmo"]),
                        ("AMC", all_df["is_amc"]),
                        ("PEAD", all_df["pead_pass"] == 1),
                        ("PEAD + BMO", all_df["is_bmo"] & (all_df["pead_pass"] == 1)),
                        ("PEAD + AMC", all_df["is_amc"] & (all_df["pead_pass"] == 1))]:
        sub = all_df[mask & all_df["pregap_return"].notna()]
        if len(sub) == 0:
            print(f"   {label:<20} {0:>5} (no trades)")
            continue
        pnls = sub["pregap_return"]
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        wr = len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0
        avg_w = wins.mean() * 100 if len(wins) > 0 else 0
        avg_l = losses.mean() * 100 if len(losses) > 0 else 0
        payoff = avg_w / abs(avg_l) if avg_l != 0 else float('inf')
        print(f"   {label:<20} {len(sub):>5} {wr:>6.1f}% {pnls.mean()*100:>+9.2f}% "
              f"{avg_w:>+9.2f}% {avg_l:>+9.2f}% {payoff:>7.2f}")

    # Per-trade detail of PEAD losers with pre-gap
    print(f"\n5. PEAD LOSERS: CURRENT vs PRE-GAP (per trade)")
    pead_losers = all_df[(all_df["pead_pass"] == 1) & (all_df["realized_arith_pct"] <= 0)]
    print(f"   {'Ticker':<8} {'BMO/AMC':<8} {'Date':<12} {'Current':>10} {'Pre-gap':>10} {'Delta':>10} {'Gap':>10} {'CAR_10d':>10}")
    for _, row in pead_losers.sort_values("realized_arith_pct").iterrows():
        ticker = row.get("canonical_ticker", row["permaTicker"])
        bam = "BMO" if row.get("is_bmo", False) else "AMC"
        cur = row["realized_arith_pct"] * 100
        pre = row["pregap_return"] * 100 if pd.notna(row["pregap_return"]) else float('nan')
        gap = row["opening_gap"] * 100 if pd.notna(row["opening_gap"]) else float('nan')
        # car_10d is log, convert
        delta = pre - cur if pd.notna(pre) else float('nan')
        print(f"   {ticker:<8} {bam:<8} {str(row['report_date'])[:10]:<12} "
              f"{cur:>+9.2f}% {pre:>+9.2f}% {delta:>+9.2f}% {gap:>+9.2f}%")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

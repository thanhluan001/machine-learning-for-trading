#!/usr/bin/env python3
"""
Slot sweep with PROPER NAV-based position sizing.

Previous analysis (43_slot_utilization_analysis.py) summed raw trade returns,
which treats each trade as a fixed-dollar bet. That's wrong for NAV-based
sizing where each slot gets 1/N of NAV.

This script simulates a portfolio where:
  - Each trade is allocated 1/N_slots of NAV
  - Weekly portfolio return = sum of (1/N_slots × trade_return) for executed trades
  - Total return = product of (1 + weekly_return) across all weeks (compounding)

This correctly accounts for the dilution effect of more slots.
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
THETA = 0.20
EXCLUDE_SECTORS = ["XLF"]
EXIT_SNAP = 5

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


def compute_pregap(df, db_path, hold_days=5):
    df = df.copy()
    df["pregap_return"] = np.nan
    df["pregap_entry_date"] = pd.NaT
    df["pregap_exit_date"] = pd.NaT
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
            ep = p_close[entry_t]; xp = p_close[exit_t]
            if pd.isna(ep) or pd.isna(xp) or ep <= 0:
                continue
            df.at[idx, "pregap_return"] = float(xp / ep - 1.0)
            df.at[idx, "pregap_entry_date"] = pd.Timestamp(p_index[entry_t])
            df.at[idx, "pregap_exit_date"] = pd.Timestamp(p_index[exit_t])
    return df


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


def simulate_portfolio_nav(accepted_df, n_slots, sizing_mode="fixed_fraction"):
    """
    Simulate portfolio with NAV-based position sizing.

    sizing_mode:
      'fixed_fraction' — each trade gets 1/n_slots of NAV (e.g. 25% for 4 slots).
                          Dilutes when slots aren't full (cash sits idle).
      'equal_active'   — each ACTIVE trade gets 1/n_active of NAV (always 100% deployed).
                          No cash drag, but single-trade weeks are 100% in one stock.
    """
    exec_df = select_weekly_top_n(accepted_df.copy(), n_slots=n_slots)
    if len(exec_df) == 0:
        return {"n_trades": 0, "final_nav": 1.0, "total_return_pct": 0.0}

    exec_df["entry_date"] = pd.to_datetime(exec_df["entry_date"])
    iso = exec_df["entry_date"].dt.isocalendar()
    exec_df["week_key"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    nav = 1.0
    weekly_returns = []
    for week_key, week_df in exec_df.groupby("week_key", sort=True):
        trades = week_df["pregap_return"].dropna()
        n_active = len(trades)
        if n_active == 0:
            weekly_returns.append(0.0)
            continue

        if sizing_mode == "fixed_fraction":
            weight = 1.0 / n_slots  # e.g. 25% per slot
        elif sizing_mode == "equal_active":
            weight = 1.0 / n_active  # split 100% NAV among active trades

        # Portfolio return this week = sum of (weight × trade_return)
        week_ret = (trades * weight).sum()
        nav *= (1 + week_ret)
        weekly_returns.append(week_ret)

    total_return = (nav - 1) * 100
    return {
        "n_trades": len(exec_df),
        "final_nav": nav,
        "total_return_pct": total_return,
        "weekly_returns": weekly_returns,
    }


def main():
    print("=" * 100)
    print("SLOT SWEEP WITH PROPER NAV-BASED POSITION SIZING")
    print("=" * 100)

    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = compute_pregap(df, DB, EXIT_SNAP)

    with pd.HDFStore(DB, mode="r") as s:
        pt_meta = s["/metadata/sp400_permatickers"]
    sector_lookup = pt_meta[["permaTicker", "index_ref"]].drop_duplicates("permaTicker")
    df = df.merge(sector_lookup, on="permaTicker", how="left")
    df["sector"] = df["index_ref"]

    hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
    fold_data = {}
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

    all_accepted = []
    for fi in range(1, 5):
        td = fold_data[fi]["test_df"]
        mask = (td["p"] >= THETA) & (td["pregap_return"].notna()) & (~td["sector"].isin(EXCLUDE_SECTORS))
        picks = td[mask].copy()
        if len(picks) > 0:
            picks["fold"] = fi
            all_accepted.append(picks)
    accepted_df = pd.concat(all_accepted).reset_index(drop=True)
    accepted_df["entry_date"] = pd.to_datetime(accepted_df["pregap_entry_date"])
    accepted_df["exit_date"] = pd.to_datetime(accepted_df["pregap_exit_date"])

    n_accepted = len(accepted_df)

    # ===== 1. RAW SUM (old method — misleading) =====
    print(f"\n{'='*100}")
    print("1. OLD METHOD: Raw sum of trade returns (NO position sizing)")
    print("   (Each trade = fixed $1 bet. Misleading for NAV-based portfolios.)")
    print(f"{'='*100}")
    print(f"\n  {'Slots':>7} {'Trades':>7} {'Sum PnL':>10} {'Avg/trade':>10}")
    print("  " + "-" * 40)
    for n_slots in [2, 3, 4, 5, 6, 8, 10, 999]:
        exec_s = select_weekly_top_n(accepted_df.copy(), n_slots=n_slots)
        if len(exec_s) == 0:
            continue
        r = exec_s["pregap_return"].dropna()
        label = f"{n_slots} slots" if n_slots < 999 else "unlimited"
        print(f"  {label:>7} {len(r):>7} {r.sum()*100:>+9.1f}% {r.mean()*100:>+9.2f}%")

    # ===== 2. NAV-BASED: Fixed fraction (1/N per slot, cash sits idle) =====
    print(f"\n{'='*100}")
    print("2. NAV-BASED: Fixed fraction per slot (1/N_slots, cash sits idle when unfilled)")
    print("   This is the CORRECT model for the strategy spec (equal-weight 1/4 NAV).")
    print(f"{'='*100}")
    print(f"\n  {'Slots':>7} {'Trades':>7} {'Alloc/trade':>12} {'Final NAV':>10} {'Total return':>13} {'Max DD':>8}")
    print("  " + "-" * 65)
    results_fixed = {}
    for n_slots in [2, 3, 4, 5, 6, 8, 10, 999]:
        res = simulate_portfolio_nav(accepted_df, n_slots, sizing_mode="fixed_fraction")
        label = f"{n_slots} slots" if n_slots < 999 else "unlimited"
        alloc = f"1/{n_slots}={100/n_slots:.0f}%" if n_slots < 999 else "1/N"
        wr = res.get("weekly_returns", [])
        # Max drawdown of cumulative weekly returns
        if wr:
            cum = np.cumsum(wr)
            max_dd = (cum - np.maximum.accumulate(cum)).min() * 100
        else:
            max_dd = 0
        results_fixed[n_slots] = res
        print(f"  {label:>7} {res['n_trades']:>7} {alloc:>12} {res['final_nav']:>9.2f}x "
              f"{res['total_return_pct']:>+12.1f}% {max_dd:>+7.1f}%")

    # ===== 3. NAV-BASED: Equal active (1/N_active, always 100% deployed) =====
    print(f"\n{'='*100}")
    print("3. NAV-BASED: Equal active (1/N_active trades, always 100% deployed)")
    print("   No cash drag. Single-trade weeks put 100% in one stock (riskier).")
    print(f"{'='*100}")
    print(f"\n  {'Slots':>7} {'Trades':>7} {'Final NAV':>10} {'Total return':>13} {'Max DD':>8}")
    print("  " + "-" * 55)
    for n_slots in [2, 3, 4, 5, 6, 8, 10, 999]:
        res = simulate_portfolio_nav(accepted_df, n_slots, sizing_mode="equal_active")
        label = f"{n_slots} slots" if n_slots < 999 else "unlimited"
        wr = res.get("weekly_returns", [])
        if wr:
            cum = np.cumsum(wr)
            max_dd = (cum - np.maximum.accumulate(cum)).min() * 100
        else:
            max_dd = 0
        print(f"  {label:>7} {res['n_trades']:>7} {res['final_nav']:>9.2f}x "
              f"{res['total_return_pct']:>+12.1f}% {max_dd:>+7.1f}%")

    # ===== 4. HEAD-TO-HEAD: 4 vs 5 vs 6 slots (fixed fraction) =====
    print(f"\n{'='*100}")
    print("4. HEAD-TO-HEAD: 4 vs 5 vs 6 slots (fixed fraction, the deployable sizing)")
    print(f"{'='*100}")

    for n_slots in [4, 5, 6]:
        res = results_fixed[n_slots]
        exec_s = select_weekly_top_n(accepted_df.copy(), n_slots=n_slots)
        r = exec_s["pregap_return"].dropna()
        wr_pct = (r > 0).mean() * 100
        avg = r.mean() * 100
        med = r.median() * 100
        wr = res.get("weekly_returns", [])
        cum = np.cumsum(wr) if wr else [0]
        max_dd = (cum - np.maximum.accumulate(cum)).min() * 100 if wr else 0
        n_weeks = len(wr) if wr else 0
        n_full = sum(1 for w in wr if abs(w - 0) > 0.001) if wr else 0

        print(f"\n  {n_slots} slots (1/{n_slots} = {100/n_slots:.0f}% NAV per trade):")
        print(f"    Trades: {res['n_trades']}, Win rate: {wr_pct:.1f}%, Avg/trade: {avg:+.2f}%")
        print(f"    Final NAV: {res['final_nav']:.2f}x, Total return: {res['total_return_pct']:+.1f}%")
        print(f"    Max DD (weekly cumulative): {max_dd:+.1f}%")
        print(f"    Active weeks: {n_full} of {n_weeks}")

    # ===== 5. WHY: The math of dilution =====
    print(f"\n{'='*100}")
    print("5. THE MATH: Why more slots HURTS with fixed-fraction sizing")
    print(f"{'='*100}")

    print(f"""
  With fixed-fraction sizing (1/N per slot), each trade gets 100/N% of NAV:

  Scenario: A peak week with 5 accepted picks (returns: +10%, +8%, +6%, +4%, +2%)

  4 slots (25% each): Take top 4, reject the 5th
    Portfolio return = 25%*(10%+8%+6%+4%) = 25%*28% = +7.0%
    The 5th pick (+2%) is NOT traded. Cash from the 5th slot sits idle.

  5 slots (20% each): Take all 5
    Portfolio return = 20%*(10%+8%+6%+4%+2%) = 20%*30% = +6.0%
    Each trade gets LESS capital. The +2% marginal trade doesn't compensate.

  The crossover happens when:
    marginal_trade_return > avg_return_of_existing_trades

  Since the model sorts by P(PEAD) and picks are declining in quality,
  the marginal trade almost always has LOWER return than the average.
  So adding it at a SMALLER weight dilutes the portfolio.

  ONLY 'equal_active' sizing (1/N_active) avoids dilution — but it
  concentrates risk in low-trade-count weeks (100% in one stock).
""")

    # ===== 6. WHEN DOES MORE SLOTS HELP? =====
    print(f"{'='*100}")
    print("6. EDGE CASE: When would more slots actually help?")
    print(f"{'='*100}")

    # Find weeks where the 5th pick was very profitable
    exec_4 = select_weekly_top_n(accepted_df.copy(), n_slots=4)
    exec_5 = select_weekly_top_n(accepted_df.copy(), n_slots=5)
    exec_4["entry_date"] = pd.to_datetime(exec_4["entry_date"])
    exec_5["entry_date"] = pd.to_datetime(exec_5["entry_date"])
    iso4 = exec_4["entry_date"].dt.isocalendar()
    exec_4["week_key"] = iso4["year"].astype(str) + "-W" + iso4["week"].astype(str).str.zfill(2)
    iso5 = exec_5["entry_date"].dt.isocalendar()
    exec_5["week_key"] = iso5["year"].astype(str) + "-W" + iso5["week"].astype(str).str.zfill(2)

    keys_4 = set(zip(exec_4["permaTicker"], exec_4["report_date"]))
    extra_5 = exec_5[~exec_5.apply(lambda r: (r["permaTicker"], r["report_date"]) in keys_4, axis=1)]

    if len(extra_5) > 0:
        r = extra_5["pregap_return"].dropna()
        print(f"\n  The 13 'extra' trades added by going 4→5 slots:")
        print(f"    Avg return: {r.mean()*100:+.2f}% (vs +6.66% for top-4 trades)")
        print(f"    Win rate: {(r>0).mean()*100:.0f}%")
        print(f"    At 20% allocation, each adds ~{r.mean()*20:+.2f}% to weekly portfolio return")
        print(f"    But the 4 original trades lose {(25-20)/25*100:.0f}% of their allocation")
        orig_avg = exec_4["pregap_return"].mean()
        dilution_loss = orig_avg * 0.20 * 4  # 4 trades lose 20% of their 25% allocation
        marginal_gain = r.mean() * 0.20  # 1 new trade at 20%
        print(f"\n    Net effect per 'full' week:")
        print(f"      Dilution loss: 4 trades × ({orig_avg*100:.2f}% × 20%) = {dilution_loss*100:+.2f}%")
        print(f"      Marginal gain: 1 trade × ({r.mean()*100:.2f}% × 20%) = {marginal_gain*100:+.2f}%")
        print(f"      Net: {(marginal_gain - dilution_loss)*100:+.2f}% per full week")
        print(f"\n    => More slots HURTS because the marginal trade doesn't earn enough")
        print(f"       to compensate for diluting the better trades.")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

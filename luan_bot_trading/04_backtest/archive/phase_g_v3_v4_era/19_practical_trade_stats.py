#!/usr/bin/env python3
"""
Practical trade-level statistics with 4-slot portfolio constraint.

Answers:
  1. How many trades actually get EXECUTED (after 4-slot constraint)?
  2. Win rate, avg win, avg loss, expectancy per EXECUTED trade?
  3. How many trades per week?
  4. PEAD precision on EXECUTED trades (not raw picks)?
"""
import sys, io, importlib.util, json, time
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
HOLD_DAYS = 5   # exit at Close[T+5] (1 trading week) -- was 11
EXIT_SNAP = 5   # path_pnl_t5_pct = Close[T+5]/Open[T+1] - 1
# NOTE: PEAD label stays CAR(T+1 -> T+10) (unchanged). Only the
# holding period changes: exit at T+5 instead of T+11, freeing slots
# weekly so more picks can execute.


def select_weekly_top_n(picks: pd.DataFrame, n_slots: int = 4) -> pd.DataFrame:
    """Weekly-batch selection: for each ISO week, sort that week's picks
    by P(PEAD) descending and take top N where N = free slots.

    Free slots = n_slots - (positions still open from prior weeks).
    A position entered on entry_idx is open from entry_idx through
    entry_idx + HOLD_DAYS (exit_idx).

    This implements the user's per-week model:
      Week 1: 5 events -> sort -> take top 4
      Week 2: slots freed (5-day hold) -> sort -> take top 4
    A Friday entry still blocks into next week (T+5 = next Thursday).
    """
    if picks.empty:
        return picks

    pk = picks.copy()
    pk["entry_date"] = pd.to_datetime(pk["entry_date"])
    pk["exit_date"] = pd.to_datetime(pk["exit_date"])

    # ISO week key for grouping
    iso = pk["entry_date"].dt.isocalendar()
    pk["_week_key"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    selected_rows = []
    active_positions = []  # list of exit_date Timestamps

    for week_key, week_df in pk.groupby("_week_key", sort=True):
        # Free up slots: remove positions whose exit_date has passed
        # (exit_date < earliest entry_date in this week)
        week_start = week_df["entry_date"].min()
        active_positions = [ex for ex in active_positions if ex >= week_start]
        free_slots = n_slots - len(active_positions)

        if free_slots <= 0:
            # All slots busy this week
            continue

        # Sort this week's picks by P(PEAD) descending
        week_sorted = week_df.sort_values("p", ascending=False)
        n_take = min(free_slots, len(week_sorted))
        taken = week_sorted.head(n_take)
        selected_rows.append(taken)

        # Track new positions
        for _, row in taken.iterrows():
            active_positions.append(row["exit_date"])

    if selected_rows:
        result = pd.concat(selected_rows).sort_values("entry_date").reset_index(drop=True)
    else:
        result = pd.DataFrame(columns=picks.columns)
    return result

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


def main():
    print("=" * 80)
    print(f"PRACTICAL TRADE STATS \u2014 theta={THETA}, {N_SLOTS} slots, {HOLD_DAYS}-day hold, no gap filter")
    print("=" * 80)

    # Load + prime + gates + paths
    print("\n[1] Loading + priming + gates + paths ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)
    print(f"    rows: {len(df)}, pead: {int(df['pead_pass'].sum())}")

    # Cache trading calendar
    print("[2] Caching trading calendar ...")
    with pd.HDFStore(DB, mode="r") as sstore:
        all_dates = set()
        for key in sstore.keys():
            if not key.startswith("/sp400/"):
                continue
            d = sstore[key]
            all_dates.update(pd.to_datetime(d["Date"]).tolist())
    calendar = sorted(all_dates)
    print(f"    {len(calendar)} trading days")

    # [2b] Remap exit to 5-day hold (Close[T+5] instead of Close[T+11])
    # PEAD label stays CAR(T+1 -> T+10); only the holding period changes.
    print(f"[2b] Remapping exit to {HOLD_DAYS}-day hold (Close[T+{EXIT_SNAP}]) ...")
    cal_idx = {d: i for i, d in enumerate(calendar)}
    df["_entry_idx"] = pd.to_datetime(df["entry_date"]).map(cal_idx)
    # T+5 is (EXIT_SNAP - 1) trading days after T+1 (the entry day)
    df["_exit_idx_new"] = df["_entry_idx"] + (EXIT_SNAP - 1)
    valid = df["_exit_idx_new"].notna() & (df["_exit_idx_new"] < len(calendar))
    df.loc[valid, "exit_date"] = df.loc[valid, "_exit_idx_new"].astype(int).map(lambda i: calendar[i])
    # Override the t11 PnL column with the 5-day snap so the existing
    # simulator (which reads path_arr[11] at exit) uses 5-day PnL.
    df["path_pnl_t11_pct"] = df[f"path_pnl_t{EXIT_SNAP}_pct"]
    n_valid = int(valid.sum())
    print(f"    remapped {n_valid}/{len(df)} rows to {HOLD_DAYS}-day exit")
    df = df.drop(columns=["_entry_idx", "_exit_idx_new"])

    # Pre-train classifiers (gamma=3)
    print("\n[3] Pre-training 4 fold classifiers (gamma=3) ...")
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
        # Select picks above theta (NO global sort — sorting is per-week in step 4)
        mask = (test_df["p"] >= THETA) & (test_df["path_pnl_t11_pct"].notna())
        picks = test_df[mask].copy()

        fold_data[fi] = {"test_df": test_df, "picks": picks}
        print(f"    Fold {fi}: raw picks={len(picks)}, pead={int(picks['pead_pass'].sum())}")

    # Run portfolio simulation per fold
    print(f"\n[4] Running 4-slot portfolio simulation per fold ...")
    all_executed = []
    all_raw_picks = []

    for fi in range(1, 5):
        picks = fold_data[fi]["picks"]
        test_df = fold_data[fi]["test_df"]

        print(f"\n  --- FOLD {fi} ---")
        print(f"  Raw picks: {len(picks)} (PEAD: {int(picks['pead_pass'].sum())})")

        # Weekly batch selection: per-week sort by P(PEAD), take top N = free slots
        selected = select_weekly_top_n(picks, N_SLOTS)
        n_skipped = len(picks) - len(selected)
        print(f"  Weekly selected: {len(selected)} (skipped {n_skipped} due to full slots)")
        print(f"  PEAD in selected: {int(selected['pead_pass'].sum())} ({int(selected['pead_pass'].sum())/max(len(selected),1)*100:.1f}% precision)")

        # Run portfolio simulation on selected picks (no contention expected)
        result = rb._simulate_with_cached_calendar(selected, N_SLOTS, 100_000.0, calendar)
        s = result.get("summary", {})

        if not s:
            print("  [!] No trades executed.")
            continue

        n_exec = s["n_trades_executed"]
        n_skips = s.get("n_slots_full_skips", 0)
        n_raw = len(picks)
        n_sel = len(selected)

        print(f"  Executed: {n_exec} (sim skips: {n_skips})")
        print(f"  Slot utilization: {n_exec}/{n_raw} raw picks ({n_exec/n_raw*100:.1f}%)")

        # Get executed trades from the result
        executed_trades = result.get("trades_done", pd.DataFrame())
        if len(executed_trades) == 0:
            print("  [!] No trades executed.")
            continue

        # Join with pead_pass from the original picks (on permaTicker + entry_date)
        if "permaTicker" in executed_trades.columns and "entry_date" in executed_trades.columns:
            pead_lookup = picks[["permaTicker", "entry_date", "pead_pass", "p"]].copy()
            # Normalize entry_date to Timestamp on both sides for clean join
            pead_lookup["entry_date"] = pd.to_datetime(pead_lookup["entry_date"])
            executed_trades["entry_date"] = pd.to_datetime(executed_trades["entry_date"])
            executed_trades = executed_trades.merge(
                pead_lookup, on=["permaTicker", "entry_date"], how="left"
            )

        # Trade-level stats
        if "realized_arith_pct" in executed_trades.columns:
            pnls = executed_trades["realized_arith_pct"].dropna()
        elif "path_pnl_t11_pct" in executed_trades.columns:
            pnls = np.expm1(executed_trades["path_pnl_t11_pct"].dropna())
        else:
            pnls = pd.Series(dtype=float)

        if len(pnls) > 0:
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            n_w = len(wins)
            n_l = len(losses)
            wr = n_w / len(pnls) * 100 if len(pnls) > 0 else 0
            avg_win = wins.mean() * 100 if n_w > 0 else 0
            avg_loss = losses.mean() * 100 if n_l > 0 else 0
            expectancy = pnls.mean() * 100
            payoff = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')

            # PEAD stats on executed trades
            n_pead_exec = int(executed_trades["pead_pass"].sum()) if "pead_pass" in executed_trades.columns else 0
            prec_exec = n_pead_exec / n_exec * 100 if n_exec > 0 else 0

            print(f"\n  TRADE-LEVEL STATS (executed trades only):")
            print(f"    N executed:           {n_exec}")
            print(f"    Win rate:              {wr:.1f}%")
            print(f"    Avg win:              {avg_win:+.2f}%")
            print(f"    Avg loss:             {avg_loss:+.2f}%")
            print(f"    Payoff ratio:         {payoff:.2f}")
            print(f"    Expectancy/trade:     {expectancy:+.2f}%")
            print(f"    PEAD in executed:     {n_pead_exec}/{n_exec} ({prec_exec:.1f}% precision)")
            print(f"    IRR:                  {s.get('irr_pct', 0):+.2f}%")
            print(f"    Sharpe:               {s.get('sharpe_liq_annualized', 0):+.2f}")
            print(f"    MaxDD:                {s.get('max_drawdown_pct', 0):.2f}%")

            # Weekly stats
            if "entry_date" in executed_trades.columns:
                exec_copy = executed_trades.copy()
                exec_copy["week"] = pd.to_datetime(exec_copy["entry_date"]).dt.isocalendar().week
                exec_copy["year"] = pd.to_datetime(exec_copy["entry_date"]).dt.year
                exec_copy["year_week"] = exec_copy["year"].astype(str) + "-W" + exec_copy["week"].astype(str).str.zfill(2)
                weekly = exec_copy.groupby("year_week").size()
                print(f"\n  WEEKLY STATS:")
                print(f"    Weeks with trades:    {len(weekly)}")
                print(f"    Trades per week:      mean={weekly.mean():.1f}, med={weekly.median():.0f}, max={weekly.max()}")
                print(f"    Weeks with 0 trades:  {s.get('n_slots_full_skips', 0)} (all slots full or no picks)")

            all_executed.append(executed_trades)

        # For aggregate: raw picks = all theta-passing picks, selected = weekly batch
        all_raw_picks.append(picks)  # raw = before weekly selection

    # Aggregate across folds
    print(f"\n{'='*80}")
    print("AGGREGATE (4 folds, 4-slot constraint)")
    print(f"{'='*80}")

    all_exec_df = pd.concat(all_executed) if all_executed else pd.DataFrame()
    all_raw = pd.concat(all_raw_picks) if all_raw_picks else pd.DataFrame()

    if len(all_exec_df) > 0:
        if "realized_arith_pct" in all_exec_df.columns:
            pnls = all_exec_df["realized_arith_pct"].dropna()
        elif "path_pnl_t11_pct" in all_exec_df.columns:
            pnls = np.expm1(all_exec_df["path_pnl_t11_pct"].dropna())
        else:
            pnls = pd.Series(dtype=float)

        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        n_total_exec = len(pnls)
        n_raw_total = len(all_raw)
        n_pead_exec = int(all_exec_df["pead_pass"].sum()) if "pead_pass" in all_exec_df.columns else 0
        n_pead_raw = int(all_raw["pead_pass"].sum())

        print(f"\n  Raw picks (theta threshold):     {n_raw_total}")
        print(f"  Executed trades (4-slot):       {n_total_exec}")
        print(f"  Slot utilization:               {n_total_exec/n_raw_total*100:.1f}%")
        print(f"  Skipped (weekly batch):         {n_raw_total - n_total_exec}")

        print(f"\n  TRADE-LEVEL STATS (executed):")
        print(f"    Win rate:          {(len(wins)/n_total_exec)*100:.1f}%")
        print(f"    Avg win:           {wins.mean()*100:+.2f}%")
        print(f"    Avg loss:          {losses.mean()*100:+.2f}%")
        print(f"    Payoff ratio:      {wins.mean()/abs(losses.mean()):.2f}")
        print(f"    Expectancy/trade:  {pnls.mean()*100:+.2f}%")
        print(f"    Std per trade:     {pnls.std()*100:.2f}%")

        print(f"\n  PEAD CAPTURE (executed):")
        print(f"    PEAD in executed:  {n_pead_exec}/{n_total_exec} ({n_pead_exec/n_total_exec*100:.1f}% precision)")
        print(f"    PEAD in raw picks: {n_pead_raw}/{n_raw_total} ({n_pead_raw/n_raw_total*100:.1f}% precision)")
        print(f"    Precision lift:    {(n_pead_exec/n_total_exec - n_pead_raw/n_raw_total)*100:+.1f}pp (weekly batch effect)")

        # Per-week
        if "entry_date" in all_exec_df.columns:
            exec_copy = all_exec_df.copy()
            exec_copy["entry_date"] = pd.to_datetime(exec_copy["entry_date"])
            exec_copy["year_week"] = exec_copy["entry_date"].dt.strftime("%G-W%V")
            weekly = exec_copy.groupby("year_week").size()
            print(f"\n  WEEKLY STATS:")
            print(f"    Total weeks in TEST:   ~{sum(1 for _, g in all_raw.groupby(all_raw['report_date'].dt.strftime('%G-W%V')))}")
            print(f"    Weeks with trades:      {len(weekly)}")
            print(f"    Trades per week:        mean={weekly.mean():.1f}, med={weekly.median():.0f}, max={weekly.max()}")
            print(f"    Best week:              {weekly.max()} trades")
            print(f"    Weeks with 1 trade:     {(weekly==1).sum()}")
            print(f"    Weeks with 2 trades:    {(weekly==2).sum()}")
            print(f"    Weeks with 3 trades:    {(weekly==3).sum()}")
            print(f"    Weeks with 4 trades:    {(weekly==4).sum()}")

    # Build executed keys for matching
    if "permaTicker" in all_exec_df.columns and "entry_date" in all_exec_df.columns:
        all_exec_df["entry_date"] = pd.to_datetime(all_exec_df["entry_date"])
        exec_keys = set(zip(all_exec_df["permaTicker"], all_exec_df["entry_date"]))
    else:
        exec_keys = set()

    # Tag raw picks as executed or skipped
    all_raw = all_raw.copy()
    all_raw["entry_date"] = pd.to_datetime(all_raw["entry_date"])
    all_raw["_key"] = list(zip(all_raw["permaTicker"], all_raw["entry_date"]))
    all_raw["executed"] = all_raw["_key"].isin(exec_keys)
    all_raw["skipped"] = ~all_raw["executed"]

    print(f"\n  RAW vs EXECUTED vs SKIPPED (bias check):")
    raw_pnl = all_raw["path_pnl_t11_pct"].dropna() * 100
    exec_raw = all_raw[all_raw["executed"]]["path_pnl_t11_pct"].dropna() * 100
    skip_raw = all_raw[all_raw["skipped"]]["path_pnl_t11_pct"].dropna() * 100
    print(f"    Raw picks  PnL:      mean={raw_pnl.mean():+.2f}%, med={raw_pnl.median():+.2f}% (n={len(raw_pnl)})")
    print(f"    Executed   PnL:      mean={exec_raw.mean():+.2f}%, med={exec_raw.median():+.2f}% (n={len(exec_raw)})")
    print(f"    Skipped    PnL:      mean={skip_raw.mean():+.2f}%, med={skip_raw.median():+.2f}% (n={len(skip_raw)})")
    print(f"    Slot bias (exec-skip): {exec_raw.mean() - skip_raw.mean():+.2f}%")

    # Break down by PEAD vs non-PEAD
    print(f"\n  PEAD vs NON-PEAD breakdown (raw path_pnl_t11):")
    for label, mask in [("PEAD", all_raw["pead_pass"] == 1), ("non-PEAD", all_raw["pead_pass"] == 0)]:
        sub = all_raw[mask]
        r_pnl = sub["path_pnl_t11_pct"].dropna() * 100
        e_pnl = sub[sub["executed"]]["path_pnl_t11_pct"].dropna() * 100
        s_pnl = sub[sub["skipped"]]["path_pnl_t11_pct"].dropna() * 100
        print(f"    {label:10s} raw:      mean={r_pnl.mean():+.2f}% (n={len(r_pnl)})")
        print(f"    {label:10s} executed:  mean={e_pnl.mean():+.2f}% (n={len(e_pnl)})")
        print(f"    {label:10s} skipped:   mean={s_pnl.mean():+.2f}% (n={len(s_pnl)})")
        if len(e_pnl) > 0:
            e_wins = e_pnl[e_pnl > 0]
            e_losses = e_pnl[e_pnl <= 0]
            print(f"    {label:10s} exec wins:   n={len(e_wins)}, avg={e_wins.mean() if len(e_wins)>0 else 0:+.2f}%")
            print(f"    {label:10s} exec losses: n={len(e_losses)}, avg={e_losses.mean() if len(e_losses)>0 else 0:+.2f}%")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

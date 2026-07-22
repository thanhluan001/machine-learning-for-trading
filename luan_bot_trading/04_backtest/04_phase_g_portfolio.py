"""
Phase G -- Multi-period position simulator (true Sharpe with overlapping
10-day holds).

Per phase_g_findings.md §A.7 (item 3):
  Convert per-event alpha into strategy-level Sharpe / equity curve /
  IRR by simulating overlapping 10-day holds across the VAL period.

WHY THIS SCRIPT EXISTS
----------------------
The prior backtest outputs (`01_val_backtest.py`, `_pead_target_retrain.py`,
`_pead_classifier.py`, `_pead_gap_strategy.py`) all aggregate per-event
PnL by treating each week as an INDEPENDENT 1-week hold with full
capital. That is the "naive +540% annualized" caveat explicitly flagged
in `pead_target_findings.md` §11. In reality each event enters at
`Open[T+1]` and exits at `Close[T+11]` -- a ~10 trading-day hold that
overlaps with adjacent weeks' trades.

This script simulates the actual portfolio:

  * Universe                     : VAL rows from /features/train_matrix
                                   (report_date > 2024-01-01).
  * Selection rule (deployable) : Phase G v1.1 two-stage filter:
                                   P(PEAD) >= 0.20  (sunday classifier)
                                   AND opening_gap_t1 in [+2%, +15%]
                                   (T+1-morning confirmation).
  * Capital allocation          : equal-fraction per-trade slot.
                                   At each entry, allocate f = 1/N_max
                                   of CURRENT NAV to the new position,
                                   where N_max is the max simultaneous
                                   positions allowed.
  * Holding                     : from Open[T+1] to Close[T+11]
                                   (~10 trading days). Position is mark-
                                   to-market each trading day using
                                   intraday adj OHLC.
  * NAV updates                 : end-of-trading-day mark, summing all
                                   open positions' contributions.
                                   Capital is recycled after exit.
  * No transaction costs; no slippage. Pure stat-arb simulation.
  * No leverage assumption -- if maxed out on slots, skip the new trade
    (logged as `slots_full_skip`).

OUTPUTS
-------
Capital curves in `phase_g_v1_1_sweep_portfoliosim_<N>_positions/`:
  * `equity_curve.csv`  -- daily NAV + cash + open_positions count.
  * `trades.csv`        -- one row per executed trade with entry/exit/
                           realized_arith_return / pnl_dollar_contribution.
  * `summary.json`      -- headline metrics (IRR, Sharpe, maxDD, etc.).
No DB writes. Read-only on db.h5.

CLI
---
  python luan_bot_trading/04_backtest/04_phase_g_portfolio.py
  python luan_bot_trading/04_backtest/04_phase_g_portfolio.py --n-slots 4
  python luan_bot_trading/04_backtest/04_phase_g_portfolio.py --strategy v1
  python luan_bot_trading/04_backtest/04_phase_g_portfolio.py --strategy sunday_only
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
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

DB = tm.DB_FILE
SUNDAY_SAFE_FEATURES = pg.SUNDAY_SAFE_FEATURES

# ------------------ strategy registry ------------------
DEFAULT_STRATEGY = "v1_1_two_stage"

STRATEGIES = {
    # name  : (model_path,                   theta, gap_lo, gap_hi, label)
    "v1_1_two_stage": (
        HERE.parent / "03_model" / "models" / "phase_g_v1_1_sunday_sweep" / "classifier.json",
        0.20, 0.02, 0.15,
        "Phase G v1.1: P>=0.20 AND gap in [+2%,+15%]"),
    "v1_two_stage": (
        HERE.parent / "03_model" / "models" / "phase_g_v1_sunday_classifier" / "classifier.json",
        0.20, 0.02, 0.15,
        "Phase G v1: P>=0.20 AND gap in [+2%,+15%]"),
    "v1_1_sunday_passthru": (
        HERE.parent / "03_model" / "models" / "phase_g_v1_1_sunday_sweep" / "classifier.json",
        0.20, None, None,
        "Phase G v1.1: P>=0.20 (no gap filter)"),
    "random_universe": (
        None, 0.0, None, None,
        "Random baseline: enter 1 random event per week (no model)"),
}


# ------------------ data loading -----------------------
def load_universe() -> pd.DataFrame:
    """Load val_matrix + §12 cut + walk-forward VAL split + compute
    pead_pass labels + entry PnL (Open[T+1] -> Close[T+11]).
    """
    print("[1] Loading train_matrix + applying §12 cutoff + walk-forward ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    print(f"    rows after §12 cut: {len(df)}")
    print("[2] Computing 3 PEAD gates ...")
    df = pg.v3.compute_pead_gates_full(df)
    train_df, val_df = tm.split_walk_forward(df, tm.DEFAULT_SPLIT_DATE)
    val_df, _ = tm.drop_sparse_weeks(val_df, tm.DEFAULT_MIN_GROUP_SIZE)
    val_df = val_df.sort_values(
        ["calendar_week_group", "permaTicker", "report_date"]
    ).reset_index(drop=True)
    print(f"    VAL rows: {len(val_df)}  pead_pos: {int(val_df['pead_pass'].sum())}")
    print("[3] Computing entry-PnL helper ...")
    val_df = pg.compute_entry_pnl(val_df)
    valid = val_df["ret_open_t1_close_t11"].notna()
    print(f"    coverage: {int(valid.sum())}/{len(val_df)}")
    return val_df


# ------------------ per-event realized path computation ------
def compute_trade_paths(val_df: pd.DataFrame) -> pd.DataFrame:
    """For each event in val_df compute its DAILY adj-close path from T+1
    open through T+11 close. We need the FULL per-day path, not just the
    single log return, since we'll mark-to-market the position each day
    for NAV accounting.

    Adds two columns per event row:
      - `entry_date` : Timestamp of T+1 (entry day)
      - `exit_date`  : Timestamp of T+11 (exit close day)
      - `path_pnl_t_pct` : columns path_pnl_t0_pct, path_pnl_t1_pct, ...
                           path_pnl_t11_pct -- cumulative arithmetic
                           return of the position from entry open at
                           each of 0..11 trading days from T.
                           path_pnl_t0_pct = 0 (entry at open, no time
                            elapses), path_pnl_t1_pct = close[T+1] /
                            open[T+1] - 1, path_pnl_t11_pct = exit.
    """
    print("[4] Computing per-event realized trade paths (T+1 open -> T+11 close, 11 daily snaps) ...")
    val_df = val_df.copy()
    # Pre-allocate the 12 path columns (snap 0 = entry open, snap 11 = exit close)
    for t in range(12):
        val_df[f"path_pnl_t{t}_pct"] = np.nan
    val_df["entry_date"] = pd.NaT
    val_df["exit_date"] = pd.NaT
    with pd.HDFStore(DB, mode="r") as s:
        pts = val_df["permaTicker"].unique()
        n_done = 0
        for pt in pts:
            key = f"/sp400/{pt}"
            if key not in s:
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_open = p["Adj_Open"].values
            p_close = p["Adj_Close"].values
            sub = val_df[val_df["permaTicker"] == pt]
            for idx, row in sub.iterrows():
                rdate = pd.to_datetime(row["report_date"]).to_datetime64()
                t_mask = p_index >= rdate
                if not t_mask.any():
                    continue
                t_idx = int(np.argmax(t_mask))
                if t_idx + 11 >= len(p_close):
                    continue
                o_t1 = p_open[t_idx + 1]
                if pd.isna(o_t1) or o_t1 <= 0:
                    continue
                # Entry date = day T+1's date (column Date at index t_idx+1)
                entry_date = pd.Timestamp(p_index[t_idx + 1])
                exit_date = pd.Timestamp(p_index[t_idx + 11])
                # snap t = closing price T+1+t vs entry open, for t in 1..11
                # snap 0 = 0 (entry at open)
                val_df.loc[idx, f"path_pnl_t0_pct"] = 0.0
                val_df.loc[idx, "entry_date"] = entry_date
                val_df.loc[idx, "exit_date"] = exit_date
                for t in range(1, 12):
                    c = p_close[t_idx + 1 + (t - 1)]
                    if pd.isna(c):
                        continue
                    val_df.loc[idx, f"path_pnl_t{t}_pct"] = float(c / o_t1 - 1.0)
            n_done += 1
            if n_done % 100 == 0:
                print(f"    [paths] {n_done}/{len(pts)} permaTickers")
    # Note: convention here is path_pnl_tN_pct = cum PnL ADJUSTED so that
    #   t=0 = entry at open (=0 by definition)
    #   t=1 = close of T+1 -- the first end-of-day mark after entry
    #   ...
    #   t=10 = close of T+10
    #   t=11 = close of T+11 = exit
    valid = val_df["path_pnl_t11_pct"].notna()
    print(f"    coverage: {int(valid.sum())}/{len(val_df)}")
    return val_df


# ------------------- selection / portfolio sim ----------
def select_trades(val_df: pd.DataFrame, strategy: str,
                  rng_seed: int = 42) -> pd.DataFrame:
    """Apply the strategy's selection filter to val_df and return only
    the rows that should be entered as trades (with a permaTicker +
    entry_date + exit_date + path columns)."""
    model_path, theta, gap_lo, gap_hi, label = STRATEGIES[strategy]
    print(f"[5] Selecting trades via strategy: {strategy}  ({label})")

    if model_path is None:
        # Random baseline strategy -- pick 1 random event per week
        rng = np.random.default_rng(rng_seed)
        rows = []
        for week, g in val_df.groupby("calendar_week_group", sort=True):
            g_ok = g.dropna(subset=["path_pnl_t11_pct"])
            if g_ok.empty:
                continue
            idx = rng.integers(len(g_ok))
            rows.append(g_ok.iloc[idx])
        sel = pd.DataFrame(rows)
        print(f"    trades selected: {len(sel)} (random 1/week)")
        return sel.reset_index(drop=True)

    # Real model: load + predict + filter
    import xgboost as xgb
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    proba = clf.predict_proba(val_df[SUNDAY_SAFE_FEATURES])[:, 1]
    val_df = val_df.copy()
    val_df["pead_proba"] = proba
    mask = (val_df["pead_proba"] >= theta)
    if gap_lo is not None:
        mask = mask & (val_df["opening_gap_t1"] >= gap_lo) \
                     & (val_df["opening_gap_t1"] <= gap_hi)
    mask = mask & val_df["path_pnl_t11_pct"].notna()
    sel = val_df[mask].copy()
    print(f"    trades selected: {len(sel)}")
    return sel.reset_index(drop=True)


def simulate_portfolio(trades: pd.DataFrame, n_slots: int,
                       initial_nav: float = 100_000.0) -> dict:
    """Run the multi-period portfolio simulator.

    Each trade occupies one slot for the duration of its hold
    (entry_date -> exit_date, inclusive). At entry, allocate
    f = 1/n_slots of CURRENT NAV to the position. The position is
    mark-to-market each INTERMEDIATE trading day using its
    path_pnl_tN_pct column. At exit, the realized dollar return is
    added to cash and the slot is freed.

    If all n_slots are already occupied when a new trade's entry
    fires, the trade is SKIPPED (logged as slots_full_skip).

    The equity curve records NAV at each trading day where ANY
    position is open or any position exits.

    Returns: dict with equity_curve (DataFrame), trades_done
    (DataFrame), slots_full_skips (int), summary_metrics (dict).
    """
    print(f"[6] Simulating portfolio with n_slots={n_slots}, "
          f"initial_nav=${initial_nav:,.0f}")
    if trades.empty:
        print("    no trades to simulate.")
        return {"equity_curve": pd.DataFrame(),
                "trades_done": pd.DataFrame(),
                "slots_full_skips": 0,
                "summary": {}}

    # Build the master calendar of all trading days during the val window.
    # Use a union of all permaTicker Date series for safety.
    print("    [sim] building master trading calendar ...")
    with pd.HDFStore(DB, mode="r") as s:
        all_dates = set()
        for key in s.keys():
            if not key.startswith("/sp400/"):
                continue
            d = s[key]
            all_dates.update(pd.to_datetime(d["Date"]).tolist())
    calendar = sorted(all_dates)
    cal_idx = {d: i for i, d in enumerate(calendar)}
    print(f"    [sim] master calendar: {len(calendar)} trading days from "
          f"{calendar[0].date()} -> {calendar[-1].date()}")

    # Truncate we only care about days where any trade is open
    trades = trades.copy()
    trades["entry_idx"] = trades["entry_date"].map(cal_idx)
    trades["exit_idx"] = trades["exit_date"].map(cal_idx)
    # Drop trades where entry/exit not on the master calendar (rare)
    bad = trades["entry_idx"].isna() | trades["exit_idx"].isna()
    if bad.any():
        print(f"    [sim] dropping {int(bad.sum())} trades without "
              f"calendar match (entry/exit not on trading day)")
        trades = trades[~bad].reset_index(drop=True)

    if trades.empty:
        print("    no trades remain after calendar match.")
        return {"equity_curve": pd.DataFrame(),
                "trades_done": pd.DataFrame(),
                "slots_full_skips": 0,
                "summary": {}}

    # Iterate over trading days chronologically, maintaining a list of
    # OPEN positions. Each open position is a dict:
    #   {trade_id, entry_idx, exit_idx, slot, entry_nav,
    #    allocated_dollars, daily_pnl_pct_cols[]}
    open_positions: list[dict] = []
    next_trade_id = 0
    cash = float(initial_nav)
    realized_dollars = 0.0
    equity_rows = []
    trades_done = []
    skips = 0

    # Sort trades by entry_idx
    trades_sorted = trades.sort_values("entry_idx").reset_index(drop=True)

    # Walk calendar from first trade's entry to last trade's exit
    min_d = int(trades_sorted["entry_idx"].min())
    max_d = int(trades_sorted["exit_idx"].max())
    print(f"    [sim] simulating days {min_d} -> {max_d}  ({max_d-min_d+1} trading days)")

    # Pre-bucket potential trades by their entry date so we can search
    # efficiently
    trades_by_entry: dict[int, list[int]] = {}
    for i, row in trades_sorted.iterrows():
        e = int(row["entry_idx"])
        trades_by_entry.setdefault(e, []).append(i)

    # Pre-extract path_pnl columns
    path_cols = [f"path_pnl_t{t}_pct" for t in range(12)]

    for d in range(min_d, max_d + 1):
        # Mark-to-market all open positions for their current snap-day
        # A trade that entered on day D (its entry_idx) is on snap-day
        # t = (d - D). For t <= 0 -> still 0 (not yet open / just opened).
        # t == 11 -> exit close; realized at day d.
        new_open = []
        for pos in open_positions:
            t = d - pos["entry_idx"]
            if t < 0:
                # Not yet open (shouldn't happen since we add at entry)
                cur_pnl_pct = 0.0
            elif t > 11:
                # Should have been closed already, but defensive
                cur_pnl_pct = float(pos["path_arr"][11])
            else:
                cur_pnl_pct = float(pos["path_arr"][t])
            pos["current_pnl_pct"] = cur_pnl_pct
            new_open.append(pos)
        open_positions = new_open

        # Try to open new trades for those whose entry fires today
        if d in trades_by_entry:
            for ti in trades_by_entry[d]:
                row = trades_sorted.iloc[ti]
                # How many slots are free?
                free_slots = n_slots - len(open_positions)
                if free_slots <= 0:
                    skips += 1
                    continue
                # Allocate dollars = f * (cash + sum of open positions
                # mark-to-market contribution). NAV at this moment
                # = cash + sum( allocated * (1 + current_pnl_pct) ).
                nav_now = cash
                for pos in open_positions:
                    nav_now += pos["allocated_dollars"] * (
                        1.0 + pos["current_pnl_pct"])
                alloc_d = (1.0 / n_slots) * nav_now
                # Hold-back logic: don't borrow. If alloc exceeds cash,
                # reduce to cash, unless we receive proceeds later.
                # Actually each slot has its OWN cash pool... Expect this
                # to be tight if a lot of slots fill simultaneously. For
                # baseline simplicity we just skip if cash < alloc AND
                # we can't draw down further.
                if alloc_d > cash:
                    # Reduce allocation to current cash available
                    # (this is a no-leverage assumption)
                    alloc_d_eff = cash
                    if alloc_d_eff < nav_now * 0.01:
                        # Skip if even 1% of NAV can't be allocated
                        skips += 1
                        continue
                    alloc_d = alloc_d_eff
                cash -= alloc_d
                path_arr = row[path_cols].to_numpy(dtype=float)
                open_positions.append({
                    "trade_id": next_trade_id,
                    "permaTicker": row["permaTicker"],
                    "report_date": row["report_date"],
                    "calendar_week_group": row["calendar_week_group"],
                    "entry_idx": d,
                    "exit_idx": int(row["exit_idx"]),
                    "slot": n_slots - free_slots + 1,
                    "entry_nav": nav_now,
                    "allocated_dollars": alloc_d,
                    "path_arr": path_arr,
                    "current_pnl_pct": 0.0,
                })
                next_trade_id += 1

        # Process exits for positions whose exit_idx == d (t == 11 today)
        still_open = []
        for pos in open_positions:
            if d == pos["exit_idx"]:
                # Realize at exit close (snap t=11 == current_pnl_pct for t=11)
                realized_pct = float(pos["path_arr"][11])
                realized_d = pos["allocated_dollars"] * (1.0 + realized_pct)
                cash += realized_d
                trades_done.append({
                    "trade_id": pos["trade_id"],
                    "permaTicker": pos["permaTicker"],
                    "entry_date": calendar[pos["entry_idx"]],
                    "exit_date": calendar[pos["exit_idx"]],
                    "calendar_week_group": pos["calendar_week_group"],
                    "allocated_dollars": pos["allocated_dollars"],
                    "entry_nav": pos["entry_nav"],
                    "realized_arith_pct": realized_pct,
                    "pnl_dollars": pos["allocated_dollars"] * realized_pct,
                    "n_open_at_entry": len([p for p in open_positions
                                            if p is not pos
                                            and p["entry_idx"] <= pos["entry_idx"]
                                            and p["exit_idx"] >= pos["entry_idx"]])
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # Daily equity row
        nav_eod = cash
        for pos in open_positions:
            nav_eod += pos["allocated_dollars"] * (
                1.0 + pos["current_pnl_pct"])
        equity_rows.append({
            "date": calendar[d],
            "day_idx": d,
            "cash": cash,
            "n_open_positions": len(open_positions),
            "realized_dollars_so_far": realized_dollars,
            "nav": nav_eod,
        })

    eq = pd.DataFrame(equity_rows)
    td = pd.DataFrame(trades_done)
    print(f"    [sim] EXIT: trades executed={len(td)}  "
          f"slots_full_skips={skips}  "
          f"final NAV=${eq['nav'].iloc[-1]:,.2f}")

    # Summary metrics
    summary = compute_summary(eq, td, n_slots, skips, initial_nav)
    return {
        "equity_curve": eq,
        "trades_done": td,
        "slots_full_skips": skips,
        "summary": summary,
    }


def compute_summary(eq: pd.DataFrame, td: pd.DataFrame,
                    n_slots: int, skips: int,
                    initial_nav: float) -> dict:
    """Compute headline portfolio metrics."""
    if eq.empty or td.empty:
        return {}
    final_nav = float(eq["nav"].iloc[-1])
    n_days = int(eq["day_idx"].iloc[-1] - eq["day_idx"].iloc[0]) + 1
    # Annualization: ~252 trading days per year
    n_years = n_days / 252.0
    irr = (final_nav / initial_nav) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    # Daily log returns of NAV
    eq["log_ret"] = np.log(eq["nav"] / eq["nav"].shift(1)).fillna(0.0)
    daily_mean = float(eq["log_ret"].mean())
    daily_std = float(eq["log_ret"].std())
    sharpe_liq = (daily_mean / (daily_std + 1e-9)) * np.sqrt(252) if daily_std > 0 else 0.0
    # Max drawdown
    nav = eq["nav"].values
    running_max = np.maximum.accumulate(nav)
    drawdowns = nav / running_max - 1.0
    max_dd = float(np.min(drawdowns))
    # Per-trade stats
    hit_rate = float((td["realized_arith_pct"] > 0).mean()) if len(td) else 0.0
    avg_pnl_pct = float(td["realized_arith_pct"].mean()) if len(td) else 0.0
    return {
        "n_slots": n_slots,
        "n_trades_executed": int(len(td)),
        "n_slots_full_skips": int(skips),
        "initial_nav": initial_nav,
        "final_nav": final_nav,
        "n_trading_days": n_days,
        "n_years": n_years,
        "irr_pct": irr * 100,
        "sharpe_liq_annualized": sharpe_liq,
        "max_drawdown_pct": max_dd * 100,
        "hit_rate_pct": hit_rate * 100,
        "avg_trade_pnl_pct": avg_pnl_pct * 100,
        "daily_log_return_mean_pct": daily_mean * 100,
        "daily_log_return_std_pct": daily_std * 100,
        "first_trade_date": str(td["entry_date"].min()) if not td.empty else None,
        "last_trade_date": str(td["exit_date"].max()) if not td.empty else None,
    }


# ------------------ main / CLI --------------------------
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY,
                        choices=list(STRATEGIES.keys()))
    parser.add_argument("--n-slots", type=int, default=4,
                        help="Max simultaneous positions (capital budget)")
    parser.add_argument("--initial-nav", type=float, default=100_000.0)
    args = parser.parse_args(argv)

    print("=" * 78)
    print(f"PHASE G -- Multi-period position simulator")
    print(f"Strategy: {args.strategy}    n_slots={args.n_slots}    "
          f"initial_nav=${args.initial_nav:,.0f}")
    print("=" * 78)

    val_df = load_universe()
    val_df = compute_trade_paths(val_df)

    trades = select_trades(val_df, args.strategy)

    result = simulate_portfolio(trades, n_slots=args.n_slots,
                                initial_nav=args.initial_nav)
    eq, td, skips, summary = (result["equity_curve"], result["trades_done"],
                              result["slots_full_skips"], result["summary"])

    if summary:
        print("\n" + "=" * 78)
        print("PORTFOLIO SIMULATION RESULTS")
        print("=" * 78)
        print(f"  Strategy              : {args.strategy}")
        print(f"  Description           : {STRATEGIES[args.strategy][4]}")
        print(f"  n_slots               : {summary['n_slots']}")
        print(f"  Trades executed       : {summary['n_trades_executed']}")
        print(f"  Slots-full skips      : {summary['n_slots_full_skips']}")
        print(f"  Initial NAV           : ${summary['initial_nav']:,.2f}")
        print(f"  Final NAV             : ${summary['final_nav']:,.2f}")
        print(f"  Trading days          : {summary['n_trading_days']}")
        print(f"  Years                 : {summary['n_years']:.2f}")
        print(f"  IRR (annualized)      : {summary['irr_pct']:+.2f}%")
        print(f"  Sharpe (liq annual.)  : {summary['sharpe_liq_annualized']:+.2f}")
        print(f"  Max drawdown          : {summary['max_drawdown_pct']:.2f}%")
        print(f"  Per-trade hit rate    : {summary['hit_rate_pct']:.1f}%")
        print(f"  Avg trade pnl (arith): {summary['avg_trade_pnl_pct']:+.3f}%")
        print(f"  Daily log ret (mean)  : {summary['daily_log_return_mean_pct']:+.4f}%")
        print(f"  Daily log ret (std)   : {summary['daily_log_return_std_pct']:.4f}%")
        print("=" * 78)

    # Persist outputs
    out_dir = HERE / f"phase_g_portfoliosim_{args.strategy}_n{args.n_slots}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not eq.empty:
        eq.to_csv(out_dir / "equity_curve.csv", index=False)
    if not td.empty:
        td.to_csv(out_dir / "trades.csv", index=False)
    if summary:
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

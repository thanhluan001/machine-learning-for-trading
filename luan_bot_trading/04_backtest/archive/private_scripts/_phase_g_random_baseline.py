"""Multi-trial random baseline for the portfolio simulator.

Reuses the same infrastructure as 04_phase_g_portfolio.py but reruns
N_RNG_TRIALS times with different seeds for the random_universe
strategy. Reports the distribution of IRR / Sharpe / MaxDD / hit%
across trials to test whether the seed=42 baseline result (+16.5%
IRR, n_slots=4) is unusually favorable or typical.

NO DB WRITES.
"""
from __future__ import annotations
import sys, importlib.util
import numpy as np, pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

HERE = __import__('pathlib').Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "sim", HERE / "04_phase_g_portfolio.py")
sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)

N_RNG_TRIALS = 100
N_SLOTS = 4


def _simulate_with_cached_calendar(trades, n_slots, initial_nav, calendar):
    """A version of simulate_portfolio that takes a pre-built trading
    calendar so we don't rebuild it 100 times. Mirrors sim.simulate_portfolio
    but with calendar injected."""
    import numpy as _np
    cal_idx = {d: i for i, d in enumerate(calendar)}
    if trades.empty:
        return {"equity_curve": sim.pd.DataFrame(), "trades_done": sim.pd.DataFrame(),
                "slots_full_skips": 0, "summary": {}}
    trades = trades.copy()
    trades["entry_idx"] = trades["entry_date"].map(cal_idx)
    trades["exit_idx"] = trades["exit_date"].map(cal_idx)
    bad = trades["entry_idx"].isna() | trades["exit_idx"].isna()
    trades = trades[~bad].reset_index(drop=True) if not bad.all() else trades
    if trades.empty or trades["entry_idx"].isna().all():
        return {"equity_curve": sim.pd.DataFrame(), "trades_done": sim.pd.DataFrame(),
                "slots_full_skips": 0, "summary": {}}
    path_cols = [f"path_pnl_t{t}_pct" for t in range(12)]
    trades_sorted = trades.sort_values("entry_idx").reset_index(drop=True)
    open_positions = []
    next_trade_id = 0
    cash = float(initial_nav)
    realized_dollars = 0.0
    equity_rows = []
    trades_done = []
    skips = 0
    min_d = int(trades_sorted["entry_idx"].min())
    max_d = int(trades_sorted["exit_idx"].max())
    trades_by_entry = {}
    for i, row in trades_sorted.iterrows():
        e = int(row["entry_idx"])
        trades_by_entry.setdefault(e, []).append(i)
    for d in range(min_d, max_d + 1):
        for pos in open_positions:
            t = d - pos["entry_idx"]
            if t < 0:
                pos["current_pnl_pct"] = 0.0
            elif t > 11:
                pos["current_pnl_pct"] = float(pos["path_arr"][11])
            else:
                pos["current_pnl_pct"] = float(pos["path_arr"][t])
        if d in trades_by_entry:
            for ti in trades_by_entry[d]:
                row = trades_sorted.iloc[ti]
                free_slots = n_slots - len(open_positions)
                if free_slots <= 0:
                    skips += 1
                    continue
                nav_now = cash
                for pos in open_positions:
                    nav_now += pos["allocated_dollars"] * (1.0 + pos["current_pnl_pct"])
                alloc_d = (1.0 / n_slots) * nav_now
                if alloc_d > cash:
                    alloc_d_eff = cash
                    if alloc_d_eff < nav_now * 0.01:
                        skips += 1
                        continue
                    alloc_d = alloc_d_eff
                cash -= alloc_d
                path_arr = row[path_cols].to_numpy(dtype=float)
                open_positions.append({
                    "trade_id": next_trade_id,
                    "permaTicker": row["permaTicker"],
                    "calendar_week_group": row["calendar_week_group"],
                    "entry_idx": d,
                    "exit_idx": int(row["exit_idx"]),
                    "entry_nav": nav_now,
                    "allocated_dollars": alloc_d,
                    "path_arr": path_arr,
                    "current_pnl_pct": 0.0,
                })
                next_trade_id += 1
        still_open = []
        for pos in open_positions:
            if d == pos["exit_idx"]:
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
                })
            else:
                still_open.append(pos)
        open_positions = still_open
        nav_eod = cash
        for pos in open_positions:
            nav_eod += pos["allocated_dollars"] * (1.0 + pos["current_pnl_pct"])
        equity_rows.append({
            "date": calendar[d], "day_idx": d, "cash": cash,
            "n_open_positions": len(open_positions),
            "realized_dollars_so_far": realized_dollars, "nav": nav_eod,
        })
    eq = sim.pd.DataFrame(equity_rows)
    td = sim.pd.DataFrame(trades_done)
    summary = sim.compute_summary(eq, td, n_slots, skips, initial_nav)
    return {"equity_curve": eq, "trades_done": td, "slots_full_skips": skips, "summary": summary}


def select_random(val_df, seed):
    """Random 1-per-week selection with explicit seed."""
    rng = np.random.default_rng(seed)
    rows = []
    for week, g in val_df.groupby("calendar_week_group", sort=True):
        g_ok = g.dropna(subset=["path_pnl_t11_pct"])
        if g_ok.empty:
            continue
        idx = rng.integers(len(g_ok))
        rows.append(g_ok.iloc[idx])
    return pd.DataFrame(rows).reset_index(drop=True)


def main():
    print("=" * 78, flush=True)
    print(f"Multi-trial random baseline: N_RNG_TRIALS={N_RNG_TRIALS}, "
          f"n_slots={N_SLOTS}", flush=True)
    print("=" * 78, flush=True)

    val_df = sim.load_universe()
    val_df = sim.compute_trade_paths(val_df)
    # Pre-build the trading calendar ONCE and stash it inside the sim
    # module so subsequent simulate_portfolio calls don't rebuild it.
    with sim.pd.HDFStore(sim.DB, mode="r") as s:
        all_dates = set()
        for key in s.keys():
            if not key.startswith("/sp400/"):
                continue
            d = s[key]
            all_dates.update(sim.pd.to_datetime(d["Date"]).tolist())
    calendar = sorted(all_dates)
    print(f"  [pre-cached calendar] {len(calendar)} trading days", flush=True)
    # Monkey-patch simulate_portfolio to skip its calendar-build step:
    orig_simulate = sim.simulate_portfolio
    def fast_simulate(trades, n_slots, initial_nav=100_000.0):
        # Inject cached calendar and call a simplified version
        return _simulate_with_cached_calendar(
            trades, n_slots, initial_nav, calendar)
    # Replace in module for the duration of this script
    sim.simulate_portfolio = fast_simulate

    rows = []
    import sys as _sys
    for trial in range(N_RNG_TRIALS):
        # Each trial uses a different rng seed
        seed = trial * 7 + 100
        trades = select_random(val_df, seed)
        result = sim.simulate_portfolio(trades, n_slots=N_SLOTS,
                                        initial_nav=100_000.0)
        s = result.get("summary", {})
        if not s:
            continue
        if not s:
            continue
        rows.append({
            "trial": trial,
            "seed": seed,
            "n_trades": s.get("n_trades_executed", 0),
            "irr": s.get("irr_pct", float("nan")),
            "sharpe": s.get("sharpe_liq_annualized", float("nan")),
            "max_dd": s.get("max_drawdown_pct", float("nan")),
            "hit_pct": s.get("hit_rate_pct", float("nan")),
            "avg_pnl_pct": s.get("avg_trade_pnl_pct", float("nan")),
        })
        if trial < 5 or trial % 10 == 0:
            print(f"  trial {trial:3d}  trades={rows[-1]['n_trades']:>3d}  "
                  f"IRR={rows[-1]['irr']:>+6.2f}%  Sharpe={rows[-1]['sharpe']:>+5.2f}",
                  flush=True)

    df = pd.DataFrame(rows)
    out_csv = HERE / f"phase_g_random_baseline_dist_n{N_SLOTS}.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n{N_RNG_TRIALS}-trial random baseline summary (n_slots={N_SLOTS}):")
    print(f"  IRR:    mean={df['irr'].mean():+.2f}%  median={df['irr'].median():+.2f}%  "
          f"std={df['irr'].std():.2f}   5%-95% CI=[{df['irr'].quantile(0.05):+.2f}%, "
          f"{df['irr'].quantile(0.95):+.2f}%]")
    print(f"  Sharpe: mean={df['sharpe'].mean():+.2f}  median={df['sharpe'].median():+.2f}  "
          f"std={df['sharpe'].std():.2f}   5%-95% CI=[{df['sharpe'].quantile(0.05):+.2f}, "
          f"{df['sharpe'].quantile(0.95):+.2f}]")
    print(f"  MaxDD:  mean={df['max_dd'].mean():+.2f}%  median={df['max_dd'].median():+.2f}%  "
          f"std={df['max_dd'].std():.2f}   5%-95% CI=[{df['max_dd'].quantile(0.05):+.2f}%, "
          f"{df['max_dd'].quantile(0.95):+.2f}%]")
    print(f"  Hit:    mean={df['hit_pct'].mean():+.2f}%  median={df['hit_pct'].median():+.2f}%")

    # Compare v1.1 IRR (+21.3%) to the distribution
    v11_irr = 21.29  # known from the run
    frac_above = (df["irr"] > v11_irr).mean()
    print(f"\n  v1.1 IRR ({v11_irr:+.2f}%) exceeds random baseline by "
          f"1 - {frac_above:.3f} of N_RNG_TRIALS trials.")
    print(f"  Median IRR + Sharpe: {df['sharpe'].median():+.2f}, "
          f"v1.1 Sharpe = +1.92 -> ",
          end='')
    frac_above_sharpe = (df["sharpe"] > 1.92).mean()
    print(f"  v1.1 Sharpe exceeds {frac_above_sharpe:.3f} of trials.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

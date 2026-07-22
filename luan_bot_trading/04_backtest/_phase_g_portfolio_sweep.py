"""Sweep n_slots across {1, 2, 3, 4, 8, 16} for the v1.1 two-stage
strategy plus a random baseline at n=4.

Outputs a comparison table to stdout and saves a CSV summary.
"""
from __future__ import annotations
import sys, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "sim", HERE / "04_phase_g_portfolio.py")
sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)


def main():
    print("=" * 90)
    print("PHASE G -- n_slots sweep for v1_1_two_stage strategy + random baseline")
    print("=" * 90)
    # Reuse global load + path compute ONCE (expensive)
    val_df = sim.load_universe()
    val_df = sim.compute_trade_paths(val_df)

    strategies_to_test = [
        ("v1_1_two_stage", "v1.1 P>=0.20 + gap[+2%,+15%]"),
        ("v1_two_stage", "v1 P>=0.20 + gap[+2%,+15%]"),
        ("v1_1_sunday_passthru", "v1.1 P>=0.20 (no gap)"),
        ("random_universe", "random 1-per-week baseline"),
    ]
    n_slots_list = [1, 2, 3, 4, 8, 16]
    rows = []
    for strat_key, strat_label in strategies_to_test:
        print(f"\n[*] Strategy: {strat_key} ({strat_label})")
        trades = sim.select_trades(val_df, strat_key)
        for n_slots in n_slots_list:
            result = sim.simulate_portfolio(trades, n_slots=n_slots,
                                            initial_nav=100_000.0)
            s = result.get("summary", {})
            if not s:
                continue
            row = {
                "strategy": strat_key,
                "strategy_label": strat_label,
                "n_slots": s["n_slots"],
                "n_trades_executed": s["n_trades_executed"],
                "n_skips": s["n_slots_full_skips"],
                "final_nav": s["final_nav"],
                "irr_pct": s["irr_pct"],
                "sharpe": s["sharpe_liq_annualized"],
                "max_dd_pct": s["max_drawdown_pct"],
                "hit_rate_pct": s["hit_rate_pct"],
                "avg_trade_pnl_pct": s["avg_trade_pnl_pct"],
                "daily_log_mean_pct": s["daily_log_return_mean_pct"],
                "daily_log_std_pct": s["daily_log_return_std_pct"],
            }
            rows.append(row)
            print(f"   n={n_slots:>2d}  trades={row['n_trades_executed']:>3d}  "
                  f"skips={row['n_skips']:>3d}  "
                  f"final=${row['final_nav']:>9,.0f}  "
                  f"IRR={row['irr_pct']:>+6.2f}%  "
                  f"Sharpe={row['sharpe']:>+5.2f}  "
                  f"MaxDD={row['max_dd_pct']:>5.2f}%  "
                  f"hit={row['hit_rate_pct']:>5.1f}%  "
                  f"avg_pnl={row['avg_trade_pnl_pct']:>+6.3f}%")
    df = pd.DataFrame(rows)
    out_csv = HERE / "phase_g_portfolio_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved sweep table to {out_csv}")

    # Pretty print the final comparison table sorted by IRR within each strategy
    print(f"\n{'='*100}")
    print("FINAL n_slots sweep comparison")
    print(f"{'='*100}")
    print(df[["strategy", "n_slots", "n_trades_executed", "final_nav",
             "irr_pct", "sharpe", "max_dd_pct", "hit_rate_pct",
             "avg_trade_pnl_pct"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

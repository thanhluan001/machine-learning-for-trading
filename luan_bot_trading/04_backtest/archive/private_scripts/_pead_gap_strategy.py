"""
PEAD-v3 Strategy Exploration: Gap-Driven Entry + PEAD-Gate-Correlated Confidence

Goal: validate the practical Saturday-walking-strategy hypothesis:
  1. Earnings events pre-traded:
  2. T+1 open: compute opening_gap_t1 ranking.
  3. Enter top-N by gap; hold 9 more trading days (T+1 close -> T+11 close).
  4. Realized PnL = 9-day remaining drift (close_T+1 -> close_T+11).

Hard rule (no labels): Do NOT use the 3 PEAD-Gates as ex ante filters
for TRADE SELECTION (that's forward-looking). Use them only as ex post
AFTER-THE-FACT PERFORMANCE ATTRIBUTION:
  - bin realized pnls by "would this event have been classified as PEAD?"
  - show long-only profile by gap size

Also compute a Sunday-feasible simulator: pre-T forecast capability.
At Sunday, gap is unknown. The Sunday ranker can only use pre-T features
(SUEs, mom, vol -- the non-leak features). Compute Sunday-feasible
predictors' top-N realized PnL.

NO DB WRITES.
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "tm", HERE.parent / "03_model" / "01_train_model.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)

DB = tm.DB_FILE


def load_and_align_val():
    print("[*] Loading train_matrix ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    train_df, val_df = tm.split_walk_forward(df, tm.DEFAULT_SPLIT_DATE)
    print(f"  VAL rows: {len(val_df)}  weeks: {val_df['calendar_week_group'].nunique()}")
    return val_df


def compute_remaining_drift(val_df: pd.DataFrame) -> pd.DataFrame:
    """For each val event, compute:
       - ret_t1_open_to_close: log return of (Open[T+1] / Close[T]) up to 
pra   (Close[T+1]/Close[T]) -- captured by the gap bool, but realized properly as
         log(Close[T+1] / Close[T]) for full-day drift from T close to T+1 close.
         Approximate Open[T+1] / Open[T] to skip intraday noise.
       - ret_close_2_to_close_11: 9-day remaining log CAR from T+1 close to T+11 close.
       - car_remaining = car remaining 9-day (Close[T+1] -> Close[T+11], stat vs IJH)
       - opening_gap_t1 (already in df).
    """
    print(f"\n[*] Computing per-event realized 9-day remaining drift over VAL ...")
    val_df = val_df.copy()
    # We use Adj_Close prices for car estimation:
    val_df["car_remaining_9d_log"] = np.nan
    val_df["ret_t1_full_open_close"] = np.nan
    val_df["ret_t1_close_close"] = np.nan

    with pd.HDFStore(DB, mode="r") as s:
        ijh_df = s["/macros/IJH"]
        ijh_index = pd.to_datetime(ijh_df["Date"]).values
        ijh_close = ijh_df["Close"].values  # raw Close (no adjClose in IJH node)

        unique_pts = val_df["permaTicker"].unique()
        print(f"  distinct permaTickers to scan: {len(unique_pts)}")
        for i, pt in enumerate(unique_pts):
            key = f"/sp400/{pt}"
            if key not in s: continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_close = p["Adj_Close"].values
            p_open = p["Adj_Open"].values
            sub = val_df[val_df["permaTicker"] == pt]
            for idx, row in sub.iterrows():
                rdate = pd.to_datetime(row["report_date"]).to_datetime64()
                t_mask = p_index >= rdate
                if not t_mask.any(): continue
                t_idx = int(np.argmax(t_mask))
                # Need T+1 close to T+11 close (10 trading-day remaining after T+1).
                if t_idx + 12 >= len(p_close) - 1: continue
                # T+1 open-to-close (we do NOT recompute 'opening_gap_t1'; we use the
                # stored value in the train_matrix). Here we compute close-to-close:
                if t_idx + 1 >= len(p_close): continue
                close_T = p_close[t_idx]
                close_T1 = p_close[t_idx + 1]
                close_T11 = p_close[t_idx + 11] if t_idx+11 < len(p_close) else np.nan
                if pd.isna(close_T11): continue
                # T+1 close-to-close return (the closed day-1 of a 10-day drift).
                retT1_close = float(np.log(close_T1 / close_T))
                # Remaining 9-day drift (Close[T+1] -> Close[T+11])
                ret_9d_remaining = float(np.log(close_T11 / close_T1))
                val_df.loc[idx, "ret_t1_close_close"] = retT1_close
                val_df.loc[idx, "car_remaining_9d_log"] = ret_9d_remaining
                val_df.loc[idx, "ret_t1_full_open_close"] = retT1_close
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(unique_pts)} done")
    return val_df


def _backtest_top_n_simple(df_subset, top_n, sort_col="opening_gap_t1", 
                            ascending=False, ret_col="car_remaining_9d_log"):
    """Walker: for each week, pick top-N by sort_col, return avg ret_col
    (week-per-week arithmetic mean of selected events)."""
    per_week = []
    for week, g in df_subset.groupby("calendar_week_group", sort=True):
        if len(g) < 1: continue
        g_ok = g.dropna(subset=[sort_col, ret_col])
        if g_ok.empty: continue
        n_pick = min(top_n, len(g_ok))
        sorted_g = g_ok.sort_values(sort_col, ascending=ascending)
        sel = sorted_g.head(n_pick)
        rem = sel[ret_col].to_numpy()
        if len(rem) == 0: continue
        per_week.append({
            "week": week,
            "n_events": len(g),
            "n_pick": n_pick,
            "mean_rem_arith_return": float(np.mean(np.expm1(rem))),
            "mean_rem_log_return": float(np.mean(rem)),
            "hit_rate": (rem > 0).mean() if len(rem) else float("nan"),
        })
    return pd.DataFrame(per_week)


def _summarize(weeks_df, name, top_n, ret_col):
    if weeks_df.empty:
        print(f"  {name}: empty")
        return
    n_w = len(weeks_df)
    n_ev = int(weeks_df["n_pick"].sum())
    arith_per_week = weeks_df["mean_rem_arith_return"]
    log_per_week = weeks_df["mean_rem_log_return"]
    mean_ar = float(arith_per_week.mean())
    std_ar = float(arith_per_week.std(ddof=1))
    sharpe = (mean_ar / std_ar) * np.sqrt(52) if std_ar > 0 else float("nan")
    cum_log = float(log_per_week.sum())
    cum_arith = float(np.expm1(cum_log))
    win_rate = float((arith_per_week > 0).mean())
    pe_h = (weeks_df["hit_rate"] * weeks_df["n_pick"]).sum() / weeks_df["n_pick"].sum()
    print(f"  {name}")
    print(f"    weeks traded: {n_w}   events picked: {n_ev}")
    print(f"    per-week arith mean: {mean_ar*100:+.3f}%  std: {std_ar*100:+.3f}%")
    print(f"    per-event hit rate (pos car_remaining): {pe_h*100:.1f}%")
    print(f"    win rate (weeks positive): {win_rate*100:.1f}%")
    print(f"    Sharpe (weekly, sqrt(52)-annualized): {sharpe:+.3f}")
    print(f"    Cumulative log PnL (naive): {cum_log*100:+.2f}%  arith {cum_arith*100:+.2f}%")
    print()


def main():
    val_df = load_and_align_val()
    val_df = compute_remaining_drift(val_df)
    # basic coverage
    print(f"  Total val rows: {len(val_df)}")
    print(f"  with car_remaining_9d_log notnull: {val_df['car_remaining_9d_log'].notna().sum()}")
    print(f"  with opening_gap_t1 notnull: {val_df['opening_gap_t1'].notna().sum()}")

    # === THE PROPER BACKTEST (no-lookahead-strict) ===
    # Strategy: each week, after T+1 open is known, long top-N events
    # whose realized `opening_gap_t1` (computed from T+1 open) is the highest
    # (positive gaps only); entry price = Close[T+1] (we use open-on-close);
    # holding 9 trading days.
    # Realized PnL = log(Close[T+11] / Close[T+1]) - log(IJH[T+11] / IJH[T+1]).
    # For the simpler PnL we just use car_remaining_9d_log (the unconditioned
    # 9-day log close-to-close). It's a near proxy.
    print("\n" + "=" * 78)
    print("STRATEGY: Top-N long entry at T+1 close, holding 9 days to T+11 close.")
    print("Rank selection: by realized opening_gap_t1 (descending, positive gaps only).")
    print("=" * 78)
    for top_n in [1, 3, 5, 10]:
        print(f"\n--- Top-{top_n} long by opening_gap_t1 (positive only) ---")
        # Positive gap filter to avoid short-mechanical = long-only
        posg = val_df[val_df["opening_gap_t1"] > 0]
        # Only positive gaps taken
        r = _backtest_top_n_simple(posg, top_n,
                                    sort_col="opening_gap_t1",
                                    ascending=False,
                                    ret_col="car_remaining_9d_log")
        _summarize(r, "Gap-driven (pos gaps only, top-N long)",
                   top_n, "car_remaining_9d_log")

    # === RANDOM BASELINE ===
    print("=" * 78)
    print("RANDOM BASELINE (3-trial controlled, same per-week n)")
    print("=" * 78)
    rng = np.random.default_rng(42)
    for top_n in [3, 5]:
        print(f"\n--- top-{top_n} random pick from pos-gap events ---")
        posg = val_df[val_df["opening_gap_t1"] > 0]
        # random selection (but only events with finite opening_gap_t1 -- same universe)
        per_week = []
        for week, g in posg.groupby("calendar_week_group", sort=True):
            g_ok = g.dropna(subset=["car_remaining_9d_log", "opening_gap_t1"])
            if len(g_ok) < top_n: continue
            idx = rng.choice(len(g_ok), top_n, replace=False)
            sel = g_ok.iloc[idx]
            rem = sel["car_remaining_9d_log"].to_numpy()
            per_week.append({"week": week, "n_events": len(g), "n_pick": top_n,
                              "mean_rem_arith_return": float(np.mean(np.expm1(rem))),
                              "mean_rem_log_return": float(np.mean(rem)),
                              "hit_rate": (rem > 0).mean()})
        r_df = pd.DataFrame(per_week)
        _summarize(r_df, "Random from positive-gap universe", top_n, "car_remaining_9d_log")

    # === STRATIFICATION: did we drop the open-of-T+1 jump by computing remaining? ===
    print("=" * 78)
    print("STRATIFICATION: 9-day REMAINING drift (T+1 close -> T+11 close) by opening_gap_t1")
    print("=" * 78)
    bins = [-np.inf, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, np.inf]
    labels = ["<-5%","-5..-2","-2..-1","-1..0","0..+1","+1..+2","+2..+5",">+5%"]
    val_df["gap_bucket"] = pd.cut(val_df["opening_gap_t1"], bins=bins, labels=labels)
    by_gap = val_df.groupby("gap_bucket", observed=False).agg(
        n=("car_remaining_9d_log","count"),
        rem_log_mean=("car_remaining_9d_log","mean"),
        rem_log_median=("car_remaining_9d_log","median"),
        gap_mean=("opening_gap_t1","mean"),
    )
    by_gap["rem_arith_mean_pct"] = np.expm1(by_gap["rem_log_mean"]) * 100
    print(by_gap.to_string())
    # Compare to the 10-day full drift we showed before:
    print(f"\n  Notice: with car_remaining (T+1 close -> T+11 close),")
    print(f"  the T+1 open-to-close day has been stripped out.")
    print(f"  Bucket >+5% opening gap shows +{np.expm1(by_gap.loc['>+5%','rem_log_mean'])*100:.2f}% remaining drift (vs +{np.expm1(val_df[val_df['opening_gap_t1']>0.05]['car_10d']).replace([np.inf,-np.inf], np.nan).dropna().mean()*100:.2f}% for the full 10-day).")

    # === Oracles: PEAD-gated subset ===
    print("\n" + "=" * 78)
    print("ORACLE CHECK: would the 3 PEAD Gates have helped us?")
    print("(Caveat: gates use post-event data -- these are EX POST trackers, not trade-time signal)")
    print("=" * 78)
    # Re-run the 3 gate computation on val
    from importlib.util import spec_from_file_location as sfalt
    # Hacky: just recompute gate signals via the exploration script helper
    exp_path = HERE / "_pead_exploration.py"
    e_spec = sfalt("explor", exp_path)
    em = importlib.util.module_from_spec(e_spec); e_spec.loader.exec_module(em)
    print("  Computing 3 PEAD gates on val (for post-hoc stratification only)...")
    g = em.compute_gates_on_subset(val_df)
    # Gate flags
    G1 = em.GATE1_CAR_MIN; G2 = em.GATE2_VOL_RATIO_MIN; G3 = em.GATE3_MAXDD_MIN
    g["pass_g1"] = (g["car_10d"].fillna(-9) > G1)
    g["pass_g2"] = (g["inst_vol_ratio"] > G2)
    g["pass_g3"] = (g["maxdd_ma"] > G3)
    g["pass_all"] = g["pass_g1"] & g["pass_g2"] & g["pass_g3"]

    # Stratify: in the PEAD-event universe, what's the realized remaining 9-day drift?
    # merge g into val_df by (permaTicker, report_date)
    g["report_date"] = pd.to_datetime(g["report_date"])
    val_df["report_date"] = pd.to_datetime(val_df["report_date"])
    merged = val_df.merge(g[["permaTicker","report_date","pass_all","pass_g1","pass_g2","pass_g3"]],
                          on=["permaTicker","report_date"], how="left")
    print("\n  9-day remaining drift, stratified by PEAD-gate outcome:")
    print(f"  {'PEAD_passed':>14s}  n_rows  car_rem_9d_arith_mean  hit_rate")
    for sub, mask in [("All events", pd.Series([True]*len(merged))),
                       ("PEAD-passed", merged["pass_all"]==True),
                       ("Failed any gate", merged["pass_all"]!=True)]:
        s = merged[mask]
        s_ok = s.dropna(subset=["car_remaining_9d_log"])
        if s_ok.empty: continue
        mean_rem = float(np.expm1(s_ok["car_remaining_9d_log"]).mean())
        hit_rate = float((s_ok["car_remaining_9d_log"] > 0).mean())
        print(f"  {sub:>14s}  {len(s_ok):>5d}  {mean_rem*100:>+8.3f}%             {hit_rate*100:>5.1f}%")

    # Real tradeable alpha: top-N by gap, AFTER excluding all PEADs that failed 
    # the gate (but this is forward-looking, IF the gates could be known at T+1). 
    # But the gates use post-T+1 data so this is just oracular upper bound.
    print("\n  ORACLE STRATEGY: top-N by gap, restricted to PEAD-passed events (forward-looking):")
    pead_passed = merged[merged["pass_all"]==True]
    if len(pead_passed) > 0:
        for top_n in [3, 5]:
            r = _backtest_top_n_simple(pead_passed, top_n,
                                        sort_col="opening_gap_t1",
                                        ascending=False,
                                        ret_col="car_remaining_9d_log")
            _summarize(r, f"ORACLE: top-{top_n} gap among PEAD-passed",
                       top_n, "car_remaining_9d_log")
    return 0


if __name__ == "__main__":
    sys.exit(main())

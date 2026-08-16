"""
PEAD-v3 Exploration:

Three questions to settle architecture empirically:
1. BASE RATE - Run the 3 PEAD-Verification Gates across all val (and maybe
   train) events. What's the PEAD event rate per gate? Combined?
2. ALPHA SOURCE - Confirm that opening_gap_t1 is a *real* signal even 
   after the 3-gate filter is applied. I.e., filter val rows to 
   PEAD-candidate events (or events that just barely pass gate #1), and 
   look at the realized car_10d / 9-day-remaining-drift distribution 
   stratified by the gap size. (Gap-persistence check.)
3. REMAINING DRIFT - Compute the post-T+1 remaining-[T+2..T+11] drift 
   (car_10d without the T+1 open-to-close leg) for top-N picks, with 
   the 3-PEAD-gate filter active. We want to confirm that the alpha 
   the model captures lives in 9-day REMAINING drift, not just the 
   T+1-day mechanical return that we've already seen by T+1 open.

Backtest ind: long top-N by (raw opening_gap_t1 descending) within the 
PEAD-gated subset. Compare to long top-N by (raw opening_gap_t1 desc) 
in the UNGATED universe. This is model-free (Path 3 baseline).

NO DB WRITES. This script is READ-ONLY on db.h5.
"""
from __future__ import annotations
import sys, os, importlib.util
import numpy as np, pandas as pd
from pathlib import Path

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

# Load the stage-3 train module to reuse data paths + helpers
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "tm", HERE.parent / "03_model" / "01_train_model.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)

DB_FILE = tm.DB_FILE

GATE1_CAR_MIN = 0.03      # CAR > 3.0%   (idiosyncratic alpha)
GATE2_VOL_RATIO_MIN = 2.0 # (Vol_T + Vol_T+1 + Vol_T+2)/3 > 2.0 * vma20
GATE3_MAXDD_MIN = -0.015  # MaxDD_MA > -1.5%

CAR_10D_END_OFFSET = 11   # T+1..T+11 (10 trading days)


def load_3_pead_gate_table():
    """Compute the 3 PEAD gates raw ingredients from /sp400/{pt} prices.

    For each row in /features/train_matrix returns:
      permaTicker, report_date, T, calendar_week_group,
      car_10d (already stored),
      install_vma20_ratio_T_to_T2: (Vol_T + Vol_T+1 + Vol_T+2)/3 / vma20,
      maxdd_ma: min over t in T+1..T+11 of (stock_cum_ret - index_cum_ret)

    Returns a DataFrame aligned to /features/train_matrix rows.
    """
    print("[*] Loading train_matrix ...")
    with pd.HDFStore(DB_FILE, mode="r") as s:
        m = s["/features/train_matrix"]
    print(f"  rows: {len(m)}")
    return m


def compute_gates_on_subset(m_subset: pd.DataFrame) -> pd.DataFrame:
    """For a subset of train_matrix rows, compute Gate 2 and Gate 3 raw
    signals (Gate 1 is already in 'car_10d'). SLOWS because we hit the HDF5
    nodes per permaTicker; selected subset should be small (e.g. VAL only).
    """
    # We index prices by permaTicker once
    needed_pts = m_subset["permaTicker"].unique()
    print(f"  distinct permaTickers needing gate calc: {len(needed_pts)}")
    out_rows = []
    with pd.HDFStore(DB_FILE, mode="r") as s:
        ijh_path = "/macros/IJH"  # phase B stored at uppercase key
        ijh_df = None
        if ijh_path in s:
            ijh_df = s[ijh_path]
        else:
            print(f"  WARN: IJH price node {ijh_path} not in DB")
            return m_subset.assign(inst_vol_ratio=np.nan, maxdd_ma=np.nan)
        ijh_index = pd.to_datetime(ijh_df["Date"]).values
        # /macros/IJH stores raw Close only (no Adj_Close). Use raw Close for
        # market adjustment -- the ETF has only minor split/dividend discrepancy,
        # acceptable for a 0..11 trading-day window in a drawdown metric.
        close_col = "Adj_Close" if "Adj_Close" in ijh_df.columns else "Close"
        ijh_close = ijh_df[close_col].values

        for i, pt in enumerate(needed_pts):
            key = f"/sp400/{pt}"
            if key not in s:
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_close = p["Adj_Close"].values
            p_vol = p["Adj_Volume"].values
            # Sub-rows for this permaTicker
            sub = m_subset[m_subset["permaTicker"] == pt].sort_values("report_date")

            for _, row in sub.iterrows():
                report_date = pd.to_datetime(row["report_date"])
                # T = first trading day >= report_date (the announcement day);
                # the stored 'T' column should match. We'll just use Date index
                # lookup since the train_matrix T column was already aligned.
                t_mask = p_index >= report_date.to_datetime64()
                if not t_mask.any():
                    continue
                t_idx = int(np.argmax(t_mask))
                # T-20 to T-1 for vma20
                if t_idx < 20:
                    continue
                vma20 = float(np.mean(p_vol[t_idx-20:t_idx]))
                if vma20 <= 0:
                    continue
                # T, T+1, T+2 volume average
                end_t2 = min(t_idx + 3, len(p_vol))
                if end_t2 - t_idx < 3:
                    continue
                inst_vol_avg = float(np.mean(p_vol[t_idx:t_idx+3]))
                vol_ratio = inst_vol_avg / vma20

                # MaxDD_MA over T+1..T+11
                end_t11 = min(t_idx + 12, len(p_close))
                if end_t11 - (t_idx+1) < 11:
                    # Window truncated; per the NaN policy proceed partial.
                    # For gate-3 it means we can compute over the partial.
                    pass
                stock_path = p_close[t_idx+1:end_t11] / p_close[t_idx] - 1.0
                # Align IJH by Date
                # find ijh idx for t_idx
                try:
                    ijh_t_idx = int(np.searchsorted(ijh_index, p_index[t_idx]))
                except Exception:
                    continue
                if ijh_t_idx >= len(ijh_close):
                    continue
                ijh_end = min(ijh_t_idx + (end_t11 - (t_idx+1)) + 1, len(ijh_close))
                if ijh_end - ijh_t_idx < 2:
                    continue
                # NOTE: this is approximate alignment - IJH and stock trading 
                # days may not match exactly, but both are US exchanges so 
                # trading-calendar should align for >99% of days.
                ijh_path_ret = ijh_close[ijh_t_idx+1:ijh_end] / ijh_close[ijh_t_idx] - 1.0
                # Pair elementwise (truncate to min length)
                n = min(len(stock_path), len(ijh_path_ret))
                stock_path = stock_path[:n]
                ijh_path_ret = ijh_path_ret[:n]
                ma_dd = (stock_path - ijh_path_ret)
                if len(ma_dd) == 0:
                    continue
                maxdd_ma = float(np.min(ma_dd))
                out_rows.append({
                    "permaTicker": pt,
                    "report_date": row["report_date"],
                    "calendar_week_group": row["calendar_week_group"],
                    "car_10d": row["car_10d"],
                    "inst_vol_ratio": vol_ratio,
                    "maxdd_ma": maxdd_ma,
                })
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{len(needed_pts)} permaTickers processed")
    return pd.DataFrame(out_rows)


def main(argv=None):
    val_df = tm.load_train_matrix()
    val_df = tm.apply_priming_cutoff(val_df, tm.PRIMING_RUNWAY_START)
    train_df, val_df = tm.split_walk_forward(val_df, tm.DEFAULT_SPLIT_DATE)
    # Note: don't apply sparse-week cutoff here, so we see ALL val events.
    val_df = val_df.sort_values(["calendar_week_group","permaTicker","report_date"]).reset_index(drop=True)
    print(f"\n[*] VAL universe: {len(val_df)} rows, {val_df['permaTicker'].nunique()} permaTickers, "
          f"{val_df['calendar_week_group'].nunique()} weeks")

    # Compute the 3 gates
    print("\n[*] Computing 3 PEAD-Gate signals over VAL ...")
    g = compute_gates_on_subset(val_df)
    print(f"  gate rows computed: {len(g)} (centile coverage: {len(g)/len(val_df)*100:.1f}%)")

    # Gate pass masks
    g["pass_g1"] = (g["car_10d"].fillna(-9) > GATE1_CAR_MIN)  # arith/log?
    # NOTE: car_10d is in log units; >+3% gate is +0.03 log ~= +3.045% arith.
    # Use log threshold faithfully:
    g["pass_g1"] = (g["car_10d"].fillna(-9) > GATE1_CAR_MIN * 1.0)
    g["pass_g2"] = (g["inst_vol_ratio"] > GATE2_VOL_RATIO_MIN)
    g["pass_g3"] = (g["maxdd_ma"] > GATE3_MAXDD_MIN)
    g["pass_all"] = g["pass_g1"] & g["pass_g2"] & g["pass_g3"]

    print(f"\n[*] GATE STATISTICS (VAL {val_df['report_date'].min()} -> {val_df['report_date'].max()}):")
    n = len(g)
    for nm, col in [("Gate1 (CAR > +3%)", "pass_g1"),
                    ("Gate2 (Inst Vol > 2x vma20)", "pass_g2"),
                    ("Gate3 (MaxDD_MA > -1.5%)", "pass_g3"),
                    ("All 3 gates combined", "pass_all")]:
        k = int(g[col].sum())
        print(f"  {nm:35s}: {k:5d}/{n} = {k/n*100:5.2f}%")

    # Spatial: how many weeks have at least 1 PEAD event?
    pead_weeks = g[g["pass_all"]]["calendar_week_group"].nunique()
    print(f"\n  Weeks containing >= 1 PEAD event: {pead_weeks}/{g['calendar_week_group'].nunique()}")

    # For confidently-PEAD events, compute arithmetic CAR
    p_clusters = g[g["pass_all"]].copy()
    print(f"\n[*] {len(p_clusters)} PEAD-clusters found in VAL:")
    if len(p_clusters):
        print(f"  car_10d (log) distribution: mean={p_clusters['car_10d'].mean():.4f} std={p_clusters['car_10d'].std():.4f}")
        arith = np.expm1(p_clusters["car_10d"])
        print(f"  car_10d (arith): mean={arith.mean()*100:.2f}% std={arith.std()*100:.2f}% min={arith.min()*100:.2f}% max={arith.max()*100:.2f}%")
        print(f"  inst_vol_ratio: mean={p_clusters['inst_vol_ratio'].mean():.2f} median={p_clusters['inst_vol_ratio'].median():.2f}")
        print(f"  maxdd_ma: mean={p_clusters['maxdd_ma'].mean()*100:.2f}% (min dd = {p_clusters['maxdd_ma'].min()*100:.2f}%)")

    # STRATIFY alpha source: car_10d by opening_gap_t1 strata (val universe)
    # Note this is under UNGATED universe to see whether the gap alone identifies 
    # the higher returns. This is the "Path 3 assertion" baseline.
    print("\n[*] FORWARD-LOOKING FEATURE STRATIFICATION (Path 3 sanity):")
    val_w_gap = val_df.copy()
    val_w_gap = val_w_gap[val_w_gap["opening_gap_t1"].notna()]
    print(f"  val rows with opening_gap_t1 notna: {len(val_w_gap)}")
    # Stratify
    bins = [-np.inf, -0.05, -0.02, -0.01, -0.005, -0.001, 0.001, 0.005, 0.01, 0.02, 0.05, np.inf]
    labels = ["<-5%", "-5..-2", "-2..-1", "-1..-0.5", "-0.5..-0.1", "-0.1..+0.1", "+0.1..+0.5", "+0.5..+1", "+1..+2", "+2..+5", ">+5%"]
    val_w_gap["gap_bucket"] = pd.cut(val_w_gap["opening_gap_t1"], bins=bins, labels=labels)
    by_gap = val_w_gap.groupby("gap_bucket", observed=False).agg(
        n=("car_10d","count"),
        car_10d_log_mean=("car_10d","mean"),
        car_10d_log_median=("car_10d","median"),
    )
    by_gap["car_10d_arith_mean_pct"] = np.expm1(by_gap["car_10d_log_mean"]) * 100
    print(by_gap.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())

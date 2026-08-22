"""rc9_undecided_state_detector.py — monthly market-state classifier (advisory).

RC-9 (Design.md §18): classify the megatrend regime into one of four states
using ONLY prior-month data, so the panel can say WHICH policy posture
(fractional rotation vs concentration vs de-risk advisory) applies now.

Pre-registered design (2026-08-23, BEFORE any backfill was computed):

  Metrics (monthly, from cached db_megatrend.h5 /mt series):
    P = Spearman rank autocorrelation of theme 3m relative returns vs SPY,
        lag 1 month. Leadership persistence.
        high P  = same leaders keep leading (a winner exists)
        low  P  = leadership churns month to month (undecided)
    B = fraction of theme-proxy series above their own MA10 AND with
        positive 6m relative strength vs SPY. Theme breadth.
        high B  = everything is being bid at once
    C = average pairwise 60d correlation of theme daily returns,
        expressed as a percentile of its own trailing history. Bloc vs
        differentiation.
        high C-pct = themes trade as ONE asset (risk-on tide)

  Fixed thresholds (pre-registered; DO NOT tune after seeing the backfill):
    P >= 0.50            -> persistent leadership
    B >= 50%             -> broad theme bid
    C >= 60th percentile -> bloc trading

  State map (deterministic from the three flags):
    UNDECIDED       B>=50% & P<0.5 & C>=60th   (broad, churning, one bloc)
    DIFFERENTIATING otherwise, if P >= 0.5 and P rose over the prior 2 months
                                          (winner emerging / recently emerged)
    CONCENTRATED    P>=0.5 & C<60th            (leader holds, differentiated)
    DISPERSAL       B<50% & C>=60th & P<0.5    (bid withdrawn, still one bloc)

Validation gate (episode classification, NOT NAV):
  using prior-month data only, the map must call
    2020 through mid-2021  UNDECIDED (clean energy/SPAC/EV/WFH everything-bid)
    late 2021              DIFFERENTIATING
    2023-2025              CONCENTRATED (AI leadership)
    2018 Q4                DISPERSAL
  Any threshold changed after seeing the backfill = mining; reject instead.

Output: monthly state table + episode verdicts. Advisory only — this script
changes no allocation, writes no plan, and is logged as panel section [13]
material by the human reading it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB = PROJECT_ROOT / "01_data" / "db_megatrend.h5"

# Theme proxies (same list as the operational watcher's THEME_PROXIES).
THEMES = {
    "AI/hyperscale": ["SMH"],
    "clean_energy": ["ICLN", "TAN"],
    "crypto": ["MSTR", "COIN"],
    "biotech": ["XBI", "IBB"],
    "metals": ["GDX", "LIT"],
    "uranium": ["URA"],
}
ALL_THEME_SERIES = sorted({s for v in THEMES.values() for s in v})

# Pre-registered thresholds (2026-08-23).
P_THR = 0.50
B_THR = 0.50
C_PCT_THR = 60
P_RISE_WINDOW = 2  # months of rising P for DIFFERENTIATING


def load_series() -> pd.DataFrame:
    """Close prices for all needed series, wide frame indexed by date."""
    need = ALL_THEME_SERIES + ["SPY"]
    frames = {}
    with pd.HDFStore(DB, "r") as store:
        for sym in need:
            key = f"/mt/{sym}"
            if key not in store.keys():
                print(f"  [WARN] missing {key}")
                continue
            d = store[key]
            dt = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
            frames[sym] = pd.Series(d["adjClose"].astype(float).values, index=dt)
    return pd.DataFrame(frames).sort_index()


def month_end(daily: pd.Series) -> pd.Series:
    return daily.groupby(daily.index.to_period("M")).last()


def compute_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly P, B, C-pct using ONLY data available at each month end."""
    # ---- month-end resample, DataFrame-level (no .apply traps) ----
    me = daily.groupby(daily.index.to_period("M")).last()

    # --- P: Spearman autocorr of 3m relative returns (theme level), lag 1m ---
    theme_cols = {}
    for theme, proxies in THEMES.items():
        cols = [c for c in proxies if c in me.columns]
        if cols:
            theme_cols[theme] = me[cols].mean(axis=1)
    theme_me = pd.DataFrame(theme_cols)
    rel1 = np.log(theme_me).diff(1).sub(np.log(me["SPY"]).diff(1), axis=0)
    rel3 = rel1.rolling(3).sum()

    P = {}
    idx = list(rel3.index)
    for i in range(1, len(idx)):
        prev = rel3.iloc[i - 1]
        cur = rel3.iloc[i]
        mask = cur.notna() & prev.notna()
        if mask.sum() >= 4 and cur[mask].std() > 0 and prev[mask].std() > 0:
            P[idx[i]] = float(cur[mask].rank().corr(prev[mask].rank(), method="spearman"))
    P = pd.Series(P)

    # --- B: fraction of theme proxies above own MA10 (month-end) with 6m
    #     relative momentum positive ---
    ma10_me = daily.rolling(10).mean().groupby(daily.index.to_period("M")).last()
    rel1_px = np.log(me).diff(1).sub(np.log(me["SPY"]).diff(1), axis=0)
    rel6 = rel1_px.rolling(6).sum()
    B = {}
    for m in me.index[6:]:
        flags = []
        for s_ in ALL_THEME_SERIES:
            if s_ not in me.columns:
                continue
            px, ma, r6 = me.loc[m, s_], ma10_me.loc[m, s_], rel6.loc[m, s_]
            if pd.isna(px) or pd.isna(ma) or pd.isna(r6):
                continue
            flags.append(bool(px > ma and r6 > 0))
        if len(flags) >= 4:
            B[m] = float(np.mean(flags))
    B = pd.Series(B)

    # --- C: avg pairwise 60d correlation of theme daily returns, expanding
    #     percentile of own history (>=36 months) ---
    rets = np.log(daily[ALL_THEME_SERIES]).diff()
    C_hist = []
    for m in me.index:
        m_end_ts = m.to_timestamp(how="end")
        win = rets.loc[:m_end_ts].tail(60)
        if len(win) < 40:
            continue
        # use themes that are fully valid IN THIS WINDOW (COIN launches
        # 2021-04; requiring all 10 would blank 2014-2021). Min 6 themes.
        valid = [c for c in win.columns if win[c].notna().all()]
        if len(valid) < 6:
            continue
        corr = win[valid].corr()
        if corr.isna().any().any():
            continue
        n = corr.shape[0]
        vals = corr.values[np.triu_indices(n, k=1)]
        C_hist.append((m, float(np.nanmean(vals))))
    Cpct = {}
    hist = []
    for m, c in C_hist:
        hist.append(c)
        if len(hist) >= 36:
            Cpct[m] = float((np.array(hist) <= c).mean() * 100)
    Cpct = pd.Series(Cpct)

    out = pd.concat(
        {"P": P, "B": B, "C_pct": Cpct}, axis=1
    ).dropna()
    return out


def classify(m: pd.Series) -> str:
    if any(pd.isna(m[c]) for c in ["P", "B", "C_pct"]):
        return "n/a"
    p_hi = m["P"] >= P_THR
    b_hi = m["B"] >= B_THR
    c_hi = m["C_pct"] >= C_PCT_THR
    if p_hi and not c_hi:
        return "CONCENTRATED"
    if p_hi and c_hi:
        return "CONCENTRATED (bloc)"  # persistent leaders, still one bloc
    if b_hi and c_hi and not p_hi:
        return "UNDECIDED"
    if not b_hi and c_hi and not p_hi:
        return "DISPERSAL"
    return "DIFFERENTIATING"  # rising leadership / mixed remainder


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, RuntimeError):
        pass
    print("=" * 76)
    print("RC-9 undecided-state detector — monthly backfill 2015-2026 (advisory)")
    print("=" * 76)
    daily = load_series()
    print(f"series loaded: {daily.shape[1]-1} themes + SPY, {daily.index[0].date()}..{daily.index[-1].date()}")
    M = compute_metrics(daily)
    M["state"] = M.apply(classify, axis=1)

    # P-rise refinement for DIFFERENTIATING: require P rising over prior 2m
    states = []
    for i, (m, row) in enumerate(M.iterrows()):
        s = row["state"]
        if s == "CONCENTRATED" and i >= P_RISE_WINDOW:
            p_now = row["P"]
            p_past = M["P"].iloc[i - P_RISE_WINDOW]
            if pd.notna(p_now) and pd.notna(p_past) and (p_now - p_past) >= 0.25:
                s = "DIFFERENTIATING"
        states.append(s)
    M["state"] = states

    print(f"\nthresholds (pre-registered): P>={P_THR}  B>={B_THR:.0%}  C>={C_PCT_THR}th pct")
    print("\n=== monthly state map ===")
    with pd.option_context("display.max_rows", 200):
        print(M[["P", "B", "C_pct", "state"]].round(2).to_string())

    # episode validation verdicts
    print("\n=== episode validation (prior-month data only) ===")
    def slice_states(a: str, b: str) -> pd.Series:
        return M.loc[a:b, "state"]
    checks = [
        ("2020-02..2021-06 everything-bid era -> expect UNDECIDED-heavy", "2020-02", "2021-06"),
        ("2021 H2 winner emergence -> expect DIFFERENTIATING appears", "2021-07", "2021-12"),
        ("2023-2025 AI era -> expect CONCENTRATED-heavy", "2023-01", "2025-12"),
        ("2018 Q4 selloff -> expect DISPERSAL appears", "2018-09", "2019-01"),
    ]
    for label, a, b in checks:
        ss = slice_states(a, b).value_counts()
        print(f"  {label}:")
        for k, v in ss.items():
            print(f"      {k:24s} {v} months")

    M.to_csv(PROJECT_ROOT / "04_backtest" / "archive" / "experiments" / "rc9_state_backfill.csv")
    print("\nsaved: 04_backtest/archive/experiments/rc9_state_backfill.csv (advisory only)")


if __name__ == "__main__":
    main()

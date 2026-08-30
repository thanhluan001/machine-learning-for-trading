"""rc9_state_portfolio_sim.py — advisory-state portfolio simulation (RESEARCH ONLY).

Simulates the RC-9 §[13] advisory postures as a mechanical monthly rebalance,
2022-01 .. 2026-08 (per user request "starting 2022"), vs SPY buy-and-hold
and a static equal-weight theme basket.

IMPORTANT HONESTY NOTES (read before quoting numbers):
 1. This is a HYPOTHETICAL overlay backtest. The RC-9 backfill was validated
    as a CLASSIFIER (episode labels), never as a portfolio input. Any result
    here is post-hoc, informed by having seen the full state history
    (2026-04 UNDECIDED spell etc.) — treat as descriptive, not predictive.
 2. Costs: 50bps per side on turnover (conservative for monthly ETF trades).
 3. The "winner" is proxied by the best 3m-relative-return theme of the
    PRIOR completed month (no look-ahead: ranking known at rebalance).
 4. UNDECIDED posture = cash (SPY-shorted? no — flat cash). DISPERSAL =
    staggered 50% deployment into equal-weight themes. DIFFERENTIATING =
    build to 100% equal-weight (fractional steps of 25%). CONCENTRATED =
    100% in prior-month leader theme. CONCENTRATED (bloc) = 100% leader
    but we haircut it to 80% (size like beta) with 20% cash.
 5. Uses raw monthly states + 2m hysteresis exactly as the panel prints them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB = PROJECT_ROOT / "01_data" / "db_megatrend.h5"

THEMES = {
    "AI/hyperscale": ["SMH"],
    "clean_energy": ["ICLN", "TAN"],
    "crypto": ["MSTR", "COIN"],
    "biotech": ["XBI", "IBB"],
    "metals": ["GDX", "LIT"],
    "uranium": ["URA"],
}
THEME_ETFS = sorted({s for v in THEMES.values() for s in v})
COST = 0.005  # 50bps per side


def load_daily() -> pd.DataFrame:
    frames = {}
    with pd.HDFStore(DB, "r") as s:
        for sym in THEME_ETFS + ["SPY"]:
            d = s[f"/mt/{sym}"]
            dt = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
            frames[sym] = pd.Series(d["adjClose"].astype(float).values, index=dt)
    return pd.DataFrame(frames).sort_index()


def rc9_state(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.dropna(how="all", axis=1)
    me = d.groupby(d.index.to_period("M")).last()
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
        prev, cur = rel3.iloc[i - 1], rel3.iloc[i]
        mask = cur.notna() & prev.notna()
        if mask.sum() >= 4 and cur[mask].std() > 0 and prev[mask].std() > 0:
            P[idx[i]] = float(cur[mask].rank().corr(prev[mask].rank(), method="spearman"))
    P = pd.Series(P)
    ma10_me = d.rolling(10).mean().groupby(d.index.to_period("M")).last()
    rel1_px = np.log(me).diff(1).sub(np.log(me["SPY"]).diff(1), axis=0)
    rel6 = rel1_px.rolling(6).sum()
    B = {}
    for m in me.index[6:]:
        flags = []
        for s_ in THEME_ETFS:
            if s_ not in me.columns:
                continue
            px, ma, r6 = me.loc[m, s_], ma10_me.loc[m, s_], rel6.loc[m, s_]
            if pd.isna(px) or pd.isna(ma) or pd.isna(r6):
                continue
            flags.append(bool(px > ma and r6 > 0))
        if len(flags) >= 4:
            B[m] = float(np.mean(flags))
    B = pd.Series(B)
    rets = np.log(d[THEME_ETFS]).diff()
    C_hist = []
    for m in me.index:
        win = rets.loc[: m.to_timestamp(how="end")].tail(60)
        if len(win) < 40:
            continue
        valid = [c for c in win.columns if win[c].notna().all()]
        if len(valid) < 6:
            continue
        corr = win[valid].corr()
        if corr.isna().any().any():
            continue
        n = corr.shape[0]
        C_hist.append((m, float(corr.values[np.triu_indices(n, k=1)].mean())))
    Cpct, hist = {}, []
    for m, c in C_hist:
        hist.append(c)
        if len(hist) >= 36:
            Cpct[m] = float((np.array(hist) <= c).mean() * 100)
    Cpct = pd.Series(Cpct)
    Mx = pd.concat({"P": P, "B": B, "C_pct": Cpct}, axis=1).dropna()
    states = []
    for m, row in Mx.iterrows():
        p_hi, b_hi = row["P"] >= 0.50, row["B"] >= 0.50
        c_hi = row["C_pct"] >= 60.0
        if p_hi and c_hi:
            states.append("CONCENTRATED (bloc)")
        elif p_hi:
            states.append("CONCENTRATED")
        elif b_hi and c_hi:
            states.append("UNDECIDED")
        elif (not b_hi) and c_hi:
            states.append("DISPERSAL")
        else:
            states.append("DIFFERENTIATING")
    Mx["state"] = states
    # 2m hysteresis
    raw = list(Mx["state"])
    hyp, cur = [raw[0]], raw[0]
    for i in range(1, len(raw)):
        if raw[i] == cur:
            hyp.append(cur)
        elif i + 1 < len(raw) and raw[i + 1] == raw[i]:
            cur = raw[i]
            hyp.append(cur)
        else:
            hyp.append(cur)
    Mx["regime"] = hyp
    return Mx


def target_weights(regime: str, leader_theme: str | None, prev: dict) -> dict:
    """Posture -> portfolio weights (theme ETFs + CASH)."""
    w = {"CASH": 0.0}
    for s in THEME_ETFS:
        w[s] = 0.0
    if regime == "UNDECIDED":
        # no new buys: keep existing fractional, else 50% EW max
        held = {k: v for k, v in prev.items() if k != "CASH" and v > 0}
        if not held:
            return w  # all cash
        tot = sum(held.values())
        return {**{k: v / tot * 0.5 for k, v in held.items()}, "CASH": 0.5}
    if regime == "DISPERSAL":
        # staggered entry: 50% equal-weight basket now
        live = [s for s in THEME_ETFS]
        per = 0.5 / len(live)
        for s in live:
            w[s] = per
        w["CASH"] = 0.5
        return w
    if regime == "DIFFERENTIATING":
        # build to 100% EW in 25% steps from current exposure
        target = {s: 1.0 / len(THEME_ETFS) for s in THEME_ETFS}
        cur_exp = sum(v for k, v in prev.items() if k != "CASH")
        new_exp = min(1.0, cur_exp + 0.25)
        for s in THEME_ETFS:
            w[s] = target[s] * new_exp
        w["CASH"] = 1.0 - new_exp
        return w
    if regime == "CONCENTRATED":
        if leader_theme:
            for s in THEMES[leader_theme]:
                w[s] = 1.0 / len(THEMES[leader_theme])
        return w
    if regime == "CONCENTRATED (bloc)":
        # 100% leader haircut to 80% (size like beta)
        if leader_theme:
            for s in THEMES[leader_theme]:
                w[s] = 0.8 / len(THEMES[leader_theme])
        w["CASH"] = 0.2
        return w
    return w


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, RuntimeError):
        pass
    daily = load_daily()
    me = daily.groupby(daily.index.to_period("M")).last()
    st = rc9_state(daily)

    # theme 3m relative-return ranking for leader selection (prior month)
    theme_me = pd.DataFrame({t: me[[p for p in ps if p in me.columns]].mean(axis=1)
                             for t, ps in THEMES.items()})
    rel3 = (np.log(theme_me).diff(1)
            .sub(np.log(me["SPY"]).diff(1), axis=0)).rolling(3).sum()

    start = pd.Period("2022-01", "M")
    months = [m for m in st.index if m >= start]
    me = me.loc[me.index.isin(months)]

    # simulate
    w = {s: 0.0 for s in THEME_ETFS}
    w["CASH"] = 1.0
    nav = 1.0
    rows = []
    prev_regime = None
    for m in months:
        # month return applied to previous weights
        if m > months[0]:
            ret = 0.0
            for s in THEME_ETFS:
                if w.get(s, 0) > 0 and s in me.columns and m in me.index:
                    r = me.loc[m, s] / me.loc[m - 1, s] - 1 if pd.notna(me.loc[m, s]) and pd.notna(me.loc[m - 1, s]) else 0.0
                    ret += w[s] * r
            nav *= (1 + ret)
        regime = st.loc[m, "regime"]
        raw_state = st.loc[m, "state"]
        # leader = best 3m rel ret theme of PRIOR month (no look-ahead)
        if m - 1 in rel3.index:
            lead = rel3.loc[m - 1].idxmax()
        else:
            lead = None
        tw = target_weights(regime, lead, w)
        # turnover cost
        turn = sum(abs(tw.get(k, 0) - w.get(k, 0)) for k in set(tw) | set(w))
        nav *= (1 - turn * COST)
        w = tw
        spy_r = me.loc[m, "SPY"] / me.loc[m - 1, "SPY"] - 1 if (m - 1) in me.index else np.nan
        rows.append({"month": str(m), "regime": regime, "raw": raw_state, "leader": lead,
                     "cash": round(w["CASH"], 2), "nav": nav, "spy_ret": spy_r})

    res = pd.DataFrame(rows).set_index("month")
    res["state_nav_ret"] = res["nav"].pct_change().fillna(res["nav"].iloc[0] - 1)
    res["cum_spy"] = (1 + res["spy_ret"].fillna(0)).cumprod()

    # equal-weight theme basket benchmark
    tw_r = me[THEME_ETFS].pct_change().mean(axis=1)
    ew_nav = (1 + tw_r.fillna(0)).cumprod()

    total_spy = res["cum_spy"].iloc[-1]
    peak = res["nav"].cummax()
    dd = (res["nav"] / peak - 1).min()
    ew_peak = ew_nav.cummax()
    ew_dd = (ew_nav / ew_peak - 1).min()

    print("=" * 78)
    print("RC-9 advisory-posture simulation 2022-01 .. 2026-08 (RESEARCH ONLY)")
    print("=" * 78)
    print(f"state-overlay NAV:   {res['nav'].iloc[-1]:.3f}x   maxDD {dd:.1%}")
    print(f"SPY buy&hold:        {total_spy:.3f}x   maxDD {(res['cum_spy']/res['cum_spy'].cummax()-1).min():.1%}")
    print(f"EW theme basket:     {ew_nav.iloc[-1]:.3f}x   maxDD {ew_dd:.1%}")
    print("\nmonthly detail (state regime, leader, cash%, NAV):")
    print(res[["regime", "raw", "leader", "cash", "nav"]].round(3).to_string())
    res.to_csv(PROJECT_ROOT / "04_backtest" / "archive" / "experiments" / "rc9_portfolio_sim.csv")


if __name__ == "__main__":
    main()

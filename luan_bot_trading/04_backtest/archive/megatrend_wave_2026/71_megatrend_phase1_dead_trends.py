#!/usr/bin/env python3
"""71_megatrend_phase1_dead_trends.py — RC-4 Phase 1: the dead-trends exit test.

KILL TEST A (from Design.md §18 RC-4)
-------------------------------------
A megatrend watcher is only useful if its exit machinery gets OUT of dying
trends before the full drawdown, while still riding the live ones. Testing on
survivors only (NVDA/TSM) would be a eulogy written in advance — so this test
is dominated by the DEAD megatrends of 2015-2026.

Rule under test (deliberately trivial — this is the floor, Kill test C bar):
    state machine per asset:
      entry : close > N-day MA (fast in)
      exit  : M consecutive closes < N-day MA (whipsaw-damped out)
    variants: MA in {150, 200, 250} x confirm in {1, 3, 5}, plus a monthly-
    cadence variant (state evaluated only at month-end closes — the operationally
    realistic cadence for a core-book overlay).

PASS CRITERIA (fixed before running):
  A1 dead-class: average giveback-from-peak at rule exit <= 35%,
      AND rule maxDD < B&H maxDD for EVERY dead trend.
  A2 live-class: average log-return capture >= 70% of B&H.
  A3 at least one variant satisfies A1 and A2 simultaneously.

UNIVERSE (Tiingo daily bars, cached in 01_data/db_megatrend.h5):
  live_2026 : SMH, NVDA, TSM            (AI complex — the motivating case)
  dead_2021 : ARKK, KWEB, XBI, TAN, ICLN, COIN, MSTR, ZM, PTON, SHOP, PYPL,
              XYZ(Block), SPAK           (pandemic-era megatrends that died)
  cycles    : GDX, XLE, EEM, XLU, URA, LIT, IBB, VNQ  (died AND revived — the
              harder test: exit the death, re-enter the revival)
  reference : SPY, QQQ

Future data sources noted (NOT used in Phase 1 per doctrine — narratives
arrive last; detection is price/breadth only): FMP /stable/news, Tiingo /news/.

No transaction costs modeled (turnover is months-scale; whipsaw is the enemy,
not friction). Production db.h5 untouched.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
load_dotenv(ROOT / ".env")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

DB_MT = ROOT / "01_data" / "db_megatrend.h5"
START = "2014-01-01"

UNIVERSE = {
    "live_2026": ["SMH", "NVDA", "TSM"],
    "dead_2021": ["ARKK", "KWEB", "XBI", "TAN", "ICLN", "COIN", "MSTR",
                  "ZM", "PTON", "SHOP", "PYPL", "XYZ", "SPAK"],
    "cycles":    ["GDX", "XLE", "EEM", "XLU", "URA", "LIT", "IBB", "VNQ"],
    "reference": ["SPY", "QQQ"],
}

VARIANTS = [
    # (ma_len, exit_confirm, cadence)
    (150, 1, "daily"), (200, 1, "daily"), (200, 3, "daily"),
    (200, 5, "daily"), (250, 5, "daily"), (200, 3, "monthly"),
]


def fetch_tiingo(ticker: str) -> pd.DataFrame | None:
    url = f"https://api.tiingo.com/tiingo/daily/{requests.utils.quote(ticker)}/prices"
    params = {"token": TIINGO_API_KEY, "startDate": START,
              "columns": "date,adjClose"}
    for _ in range(2):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                df = pd.DataFrame(r.json())
                if df.empty or "adjClose" not in df.columns:
                    return None
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
                df["adjClose"] = pd.to_numeric(df["adjClose"], errors="coerce")
                df = df.dropna().sort_values("date").reset_index(drop=True)
                return df
            elif r.status_code == 429:
                time.sleep(1.0)
        except Exception:
            time.sleep(1.0)
    return None


def ensure_data():
    with pd.HDFStore(DB_MT, mode="a") as store:
        keys = set(store.keys())
        n = 0
        all_syms = [s for v in UNIVERSE.values() for s in v]
        for sym in all_syms:
            if f"/mt/{sym}" in keys:
                continue
            df = fetch_tiingo(sym)
            # Block Inc: SQ renamed to XYZ (2025); keep whichever history is longer
            if sym == "XYZ":
                alt = fetch_tiingo("SQ")
                if alt is not None and (df is None or len(alt) > len(df)):
                    df = alt
            if df is not None and len(df) >= 300:
                store.put(f"/mt/{sym}", df, format="table")
                n += 1
            else:
                print(f"    !! no usable history for {sym}"
                      f" ({0 if df is None else len(df)} bars) — dropped")
            time.sleep(0.1)
        print(f"    fetched {n} new series; cache nodes: {len(store.keys())}")


def run_rule2(dates: np.ndarray, close: np.ndarray, ma_len: int, confirm: int, cadence: str):
    ma = pd.Series(close).rolling(ma_len).mean().to_numpy()
    n = len(close)
    states = np.zeros(n, dtype=float)
    state, below = 0, 0
    month_end = np.zeros(n, dtype=bool)
    if cadence == "monthly":
        m = pd.Series(dates).dt.to_period("M")
        month_end[np.append(m.values[1:] != m.values[:-1], True)] = True
    held = 0
    for i in range(n):
        if np.isnan(ma[i]):
            states[i] = 0 if cadence != "monthly" else held
            continue
        if close[i] > ma[i]:
            state, below = 1, 0
        elif state == 1:
            below += 1
            if below >= confirm:
                state, below = 0, 0
        if cadence == "monthly":
            if month_end[i]:
                held = state
            states[i] = held
        else:
            states[i] = state
    return states, ma


def metrics(dates, close, states):
    rets = np.diff(np.log(close))
    strat = rets * states[:-1]
    bh_log = float(rets.sum())
    st_log = float(strat.sum())
    nav = np.exp(np.cumsum(strat))
    dd = float((nav / np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:] - 1).min()) if len(nav) else 0.0
    bhnav = np.exp(np.cumsum(rets))
    bhdd = float((bhnav / np.maximum.accumulate(np.concatenate([[1.0], bhnav]))[1:] - 1).min())
    time_in = float(states.mean())
    rt = int(((states[1:] == 1) & (states[:-1] == 0)).sum())
    return dict(bh_log=bh_log, st_log=st_log, capture=(st_log / bh_log if bh_log > 0 else np.nan),
                dd=dd, bhdd=bhdd, time_in=time_in, round_trips=rt)


def giveback_from_peak(close, states):
    """B&H global peak -> first rule exit AFTER the peak. Return % given back."""
    peak_i = int(np.argmax(close))
    if states[peak_i] == 0:
        return "already_out"
    for i in range(peak_i + 1, len(close)):
        if states[i] == 0:
            return float(close[i] / close[peak_i] - 1)
    return float(close[-1] / close[peak_i] - 1)


def gb_val(g):
    return 0.0 if g == "already_out" else float(g)


def main():
    print("=" * 92)
    print("RC-4 PHASE 1 — DEAD-TRENDS EXIT TEST (Kill test A)   MA state machine, 2014/2020-2026")
    print("=" * 92)
    print("[1] Data ...")
    ensure_data()

    with pd.HDFStore(DB_MT, mode="r") as store:
        keys = set(store.keys())
        data = {}
        for cls, syms in UNIVERSE.items():
            for s in syms:
                if f"/mt/{s}" in keys:
                    df = store[f"/mt/{s}"]
                    data[s] = (cls, df["date"].to_numpy(), df["adjClose"].to_numpy(float))
    print(f"    {len(data)} series loaded")

    # ---------- headline variant: MA200, confirm 3, daily ----------
    HEAD = (200, 3, "daily")
    print(f"\n[2] HEADLINE RULE: close>MA{HEAD[0]}, exit after {HEAD[1]} consecutive closes below (daily)")
    rows = []
    for s, (cls, dates, close) in data.items():
        states, _ = run_rule2(dates, close, *HEAD)
        m = metrics(dates, close, states)
        gb = giveback_from_peak(close, states)
        rows.append({"sym": s, "class": cls, **m, "gb_peak": gb})
    t = pd.DataFrame(rows)
    t["dd_avoid"] = t.bhdd - t.dd
    for cls in ["dead_2021", "cycles", "live_2026", "reference"]:
        sub = t[t["class"] == cls]
        print(f"\n  --- {cls} ---")
        print(f"  {'sym':>5} {'B&H%':>8} {'rule%':>8} {'cap%':>6} {'B&Hdd':>7} {'ruleDD':>7} "
              f"{'avoid':>6} {'in%':>5} {'RT':>3} {'giveback@peak':>14}")
        for _, r in sub.iterrows():
            gb = r.gb_peak if isinstance(r.gb_peak, str) else f"{r.gb_peak*100:+.1f}%"
            print(f"  {r.sym:>5} {np.expm1(r.bh_log)*100:>+7.0f}% {np.expm1(r.st_log)*100:>+7.0f}% "
                  f"{(r.capture*100 if not np.isnan(r.capture) else float('nan')):>5.0f}% "
                  f"{r.bhdd*100:>6.1f}% {r.dd*100:>6.1f}% {r.dd_avoid*100:>+5.0f}% "
                  f"{r.time_in*100:>4.0f}% {r.round_trips:>3} {gb:>14}")

    # ---------- kill-test evaluation ----------
    dead = t[t["class"] == "dead_2021"]
    live = t[t["class"] == "live_2026"]
    a1_gb = dead.gb_peak.map(gb_val).mean()
    # NOTE: drawdowns are negative; rule DD "smaller" (better) = numerically GREATER.
    a1_dd = bool((dead.dd > dead.bhdd).all())
    a2_cap = live.capture.mean()
    n_dd_wins = int((dead.dd > dead.bhdd).sum())
    print(f"\n[3] KILL TEST A @ headline rule:")
    print(f"    A1 dead-class: mean giveback-from-peak = {a1_gb*100:+.1f}% (pass if >= -35%)")
    print(f"       rule maxDD < B&H maxDD for {n_dd_wins}/{len(dead)} dead trends (pass if 13/13)")
    print(f"    A2 live-class: mean capture = {a2_cap*100:.0f}% (need >= 70%)")

    # ---------- variant sweep (aggregates) ----------
    print(f"\n[4] VARIANT SWEEP (aggregate pass metrics)")
    print(f"  {'variant':>22} {'dead mean GB':>13} {'dead DD-wins':>12} {'live cap%':>9} "
          f"{'cycles cap%':>11} {'avg in%':>7}")
    for v in VARIANTS:
        gbs, dds, caps, ccaps, tins = [], [], [], [], []
        for s, (cls, dates, close) in data.items():
            states, _ = run_rule2(dates, close, *v)
            m = metrics(dates, close, states)
            tins.append(m["time_in"])
            if cls == "dead_2021":
                gbs.append(gb_val(giveback_from_peak(close, states)))
                dds.append(m["dd"] > m["bhdd"])  # negative DDs: greater = smaller loss
            if cls == "live_2026":
                caps.append(m["capture"])
            if cls == "cycles":
                ccaps.append(m["capture"])
        print(f"  MA{v[0]}/{v[1]}cf/{v[2]:>7} {np.mean(gbs)*100:>+12.1f}% {sum(dds):>6}/{len(dds):<5} "
              f"{np.mean(caps)*100:>8.0f}% {np.nanmean(ccaps)*100:>10.0f}% {np.mean(tins)*100:>6.0f}%")

    print(f"\n[5] VERDICT vs criteria (A1: GB>=-35% AND 13/13 DD-reduced; A2: capture>=70%; A3: any variant meets both):")
    ok = (a1_gb >= -0.35) and a1_dd and (a2_cap >= 0.70)
    print(f"    HEADLINE PASS = {ok}")
    print(f"    (see [4] sweep — variant-level A3 evaluated from corrected DD-wins column)")


if __name__ == "__main__":
    main()

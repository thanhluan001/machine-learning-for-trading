#!/usr/bin/env python3
"""76_megatrend_phase2c_capex.py — RC-4 Phase 2c: capex-flow confirmation probe.

USER HYPOTHESIS (2026-08-16)
---------------------------
"At the end, no one spends money on Bitcoin but massive money flows to AI" —
i.e., capital expenditure AGGREGATES by theme are a fundamental confirmation
signal for which megatrend is real. Proposed as the tilt variable for the
slow-pivot allocation (Phase 2b failed with price momentum: procyclical at
trend death — fattest at the top).

THIS SCRIPT (probe, not a backtest)
-----------------------------------
1. Build trailing-4Q capex aggregates per theme from FMP cash-flow statements
   for a bellwether panel:
     AI/hyperscale : MSFT GOOGL AMZN META NVDA AVGO ORCL
     clean energy  : FSLR ENPH SEDG NEE RUN PLUG
     crypto        : MSTR COIN RIOT MARA CLSK
2. Verify the user's divergence claim (AI vs crypto capex scale).
3. THE NATURAL EXPERIMENT (discriminating-power check): clean-energy capex
   into 2021-23 while TAN/ICLN bled ~70%. If capex CONFIRMED the trend at
   its death, a capex tilt inherits the same procyclicality that killed 2b.
   If capex rolled over BEFORE the ETFs, it has genuine lead information.

Decision rule for Phase 2c continuation: capex-tilt only earns a backtest if
the natural experiment shows capex LEADS (rolls before price), not LAGS.

FMP: /stable/cash-flow-statement?symbol=X&period=quarter (paid tier).
Cached to 01_data/db_capex.h5. Production db.h5 untouched.
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
FMP_KEY = os.getenv("FMP_API_KEY")

DB_CAPEX = ROOT / "01_data" / "db_capex.h5"
FMP = "https://financialmodelingprep.com/stable/cash-flow-statement"

THEMES = {
    "AI/hyperscale": ["MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL"],
    "clean_energy":  ["FSLR", "ENPH", "SEDG", "NEE", "RUN", "PLUG"],
    "crypto":        ["MSTR", "COIN", "RIOT", "MARA", "CLSK"],
}


def fetch_capex(sym: str):
    try:
        r = requests.get(FMP, params={"symbol": sym, "period": "quarter",
                                      "limit": 60, "apikey": FMP_KEY}, timeout=30)
        if r.status_code != 200 or not isinstance(r.json(), list):
            return None
        rows = []
        for b in r.json():
            d = b.get("date")
            cx = b.get("capitalExpenditure")
            if d and cx is not None:
                rows.append({"date": pd.to_datetime(d), "capex": -float(cx)})  # FMP: negative = spend
        if not rows:
            return None
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def ensure_data():
    with pd.HDFStore(DB_CAPEX, mode="a") as store:
        keys = set(store.keys())
        n = 0
        for theme, syms in THEMES.items():
            for sym in syms:
                k = f"/capex/{sym}"
                if k in keys:
                    continue
                df = fetch_capex(sym)
                if df is not None and len(df) >= 8:
                    store.put(k, df, format="table")
                    n += 1
                else:
                    print(f"    !! no capex for {sym}")
                time.sleep(0.1)
        print(f"    fetched {n} new capex series; nodes: {len(store.keys())}")


def theme_ttm(theme):
    with pd.HDFStore(DB_CAPEX, mode="r") as store:
        keys = set(store.keys())
        series = []
        for sym in THEMES[theme]:
            k = f"/capex/{sym}"
            if k not in keys:
                continue
            df = store[k].set_index("date")["capex"].sort_index()
            # quarterly -> TTM (4Q rolling sum) on quarter-end dates
            q = df.resample("QE").last()
            ttm = q.rolling(4).sum()
            series.append(ttm.rename(sym))
    if not series:
        return None
    X = pd.concat(series, axis=1)
    return X.sum(axis=1)  # theme aggregate TTM capex (missing member = 0 contribution, noted)


def main():
    print("=" * 92)
    print("RC-4 PHASE 2c — CAPEX-FLOW CONFIRMATION PROBE (FMP cash-flow statements)")
    print("=" * 92)
    print("[1] Gathering capex ...")
    ensure_data()

    agg = {t: theme_ttm(t) for t in THEMES}
    print(f"\n[2] Theme TTM capex ($B), year-ends:")
    years = range(2018, 2027)
    hdr = f"  {'year':>5}" + "".join(f"{t:>16}" for t in THEMES)
    print(hdr)
    for y in years:
        row = f"  {y:>5}"
        for t in THEMES:
            s = agg[t]
            v = s.asof(pd.Timestamp(f"{y}-12-31")) if s is not None else np.nan
            row += f"{v/1e9:>14,.0f}B" if v and not np.isnan(v) else f"{'—':>16}"
        print(row)

    print(f"\n[3] USER HYPOTHESIS CHECK — AI vs crypto capex scale:")
    ai, cr = agg["AI/hyperscale"], agg["crypto"]
    for d in ["2022-12-31", "2024-12-31", "2026-06-30"]:
        a, c = ai.asof(pd.Timestamp(d)), cr.asof(pd.Timestamp(d))
        print(f"  {d}: AI {a/1e9:,.0f}B vs crypto {c/1e9:,.0f}B  -> ratio {a/c:,.0f}x")

    print(f"\n[4] NATURAL EXPERIMENT — clean-energy capex vs TAN/ICLN price:")
    with pd.HDFStore(ROOT / "01_data" / "db_megatrend.h5", mode="r") as s:
        tan = s["/mt/TAN"].set_index("date")["adjClose"].resample("QE").last()
        icln = s["/mt/ICLN"].set_index("date")["adjClose"].resample("QE").last()
    ce = agg["clean_energy"]
    peak_i = icln.idxmax()
    print(f"  ICLN price peak: {peak_i.date()}  (then {((icln.loc['2024-12-31']/icln.loc[peak_i])-1)*100:+.0f}% by 2024)")
    print(f"  {'quarter':>9} {'cleanCapex$B':>12} {'capex vs peak':>13} {'ICLN vs peak':>12}")
    capex_peak = ce.loc[:peak_i].max()
    for d in ce.loc["2020-06-30":].index:
        if d < pd.Timestamp("2020-06-30") or (d - pd.Timestamp("2020-06-30")).days % 365 > 100:
            pass
    for d in [pd.Timestamp(x) for x in ["2020-12-31","2021-06-30","2021-12-31","2022-06-30",
                                        "2022-12-31","2023-06-30","2023-12-31","2024-06-30",
                                        "2024-12-31","2025-06-30","2025-12-31","2026-06-30"]]:
        cx = ce.asof(d); ip = icln.asof(d)
        if pd.isna(cx) or pd.isna(ip):
            continue
        print(f"  {d.date():>9} {cx/1e9:>11,.0f}B {(cx/capex_peak-1)*100:>+12.0f}% "
              f"{(ip/icln.loc[peak_i]-1)*100:>+11.0f}%")

    print(f"""
[5] READING (decision rule):
    - If clean-energy capex kept RISING through 2022-23 while ICLN fell -60..70%,
      capex CONFIRMS at trend death -> capex tilt inherits 2b's procyclicality
      -> Phase 2c REJECTED (no discriminating power at death, only at scale).
    - If capex rolled over 2+ quarters BEFORE the ETF peak, it has lead info
      -> full capex-tilt backtest is earned (Phase 2c full).
""")


if __name__ == "__main__":
    main()

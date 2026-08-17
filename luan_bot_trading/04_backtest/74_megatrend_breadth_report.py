#!/usr/bin/env python3
"""74_megatrend_breadth_report.py — RC-4 deployed component: monthly breadth
warning indicator (validated in Phase 3; strategy role closed).

WHAT IT IS
----------
The single surviving component of the megatrend watcher: a market-trend
health dashboard for the CORE book (real estate + index + blue chips), read
at month-end, zero capital deployed on it. It answers one question:

    "How broad is the current uptrend — and is it narrowing?"

MECHANIC
--------
Breadth = fraction of a fixed 26-asset category universe (11 GICS sector
ETFs, broad/international/bond/gold, standard theme-ETF menu — chosen by
CATEGORY, not by performance) trading above their own 10-month mean.

Historical calibration (2006-2026, see Phase 3 findings):
  q05 17% | q25 54% | median 77% | q75 89% | q95 96%
Stress-entry signatures: 2008 → 29% (Jan), 2020 → 31% (Feb) then 15% (Mar),
2022 → 30% (Jan) then 0-4% by summer. Healthy regimes hold 75%+.

READING GUIDE (guidance, not rules — this is an overlay for judgment):
  >= 75%   healthy breadth — trend is broad, no action
  50-75%   narrowing — rotation underneath; check which clusters dropped
  < 50%    stress signature (q25 boundary) — historically preceded/deepened
           every major drawdown; de-risk carve-out / tighten core adds
  < 25%    crisis regime (q05 boundary) — defenses already tested

MONTHLY CADENCE ONLY (Phase 1 validated; intraday/weekly = noise + narrative
temptation, which the doctrine excludes).

DATA: Tiingo daily bars, cached in 01_data/db_megatrend.h5. This script
refreshes prices, prints the report, and appends a dated row to
04_backtest/archive/experiments/megatrend_breadth_log.json (history of
readings for future calibration).

USAGE:
    conda run -n trading python luan_bot_trading/04_backtest/74_megatrend_breadth_report.py
    (run at month-end, after the close; ~30s)
"""
from __future__ import annotations
import os, sys, time, json
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
LOG = HERE / "archive" / "experiments" / "megatrend_breadth_log.json"

PHASE1_ASSETS = {'ARKK','COIN','EEM','GDX','IBB','ICLN','KWEB','LIT','MSTR','NVDA',
                 'PTON','PYPL','QQQ','SHOP','SMH','SPAK','SPY','TAN','TSM','URA',
                 'VNQ','XBI','XLE','XLU','XYZ','ZM'}


def refresh_prices():
    """Refresh the 26 broad-universe bars through the latest close."""
    with pd.HDFStore(DB_MT, mode="a") as store:
        keys = set(store.keys())
        for k in keys:
            sym = k.split("/")[-1]
            if sym in PHASE1_ASSETS:
                continue  # report universe only; trend proxies not refreshed
            r = requests.get(f"https://api.tiingo.com/tiingo/daily/{sym}/prices",
                             params={"token": TIINGO_API_KEY, "startDate": "2023-01-01",
                                     "columns": "date,adjClose"}, timeout=60)
            if r.status_code != 200:
                print(f"  !! refresh failed for {sym} (using cache)")
                continue
            df = pd.DataFrame(r.json())
            if df.empty or "adjClose" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            df["adjClose"] = pd.to_numeric(df["adjClose"], errors="coerce")
            df = df.dropna().sort_values("date").reset_index(drop=True)
            if len(df) >= 200:
                store.put(k, df, format="table")
            time.sleep(0.08)


def build_report():
    with pd.HDFStore(DB_MT, mode="r") as s:
        px = {k.split("/")[-1]: s[k].set_index("date")["adjClose"] for k in s.keys()}
    M = pd.DataFrame({sym: ser.resample("ME").last() for sym, ser in px.items()})
    broad = [c for c in M.columns if c not in PHASE1_ASSETS and c != "SPY"]
    ma10 = M.rolling(10).mean()

    asof = M.index[-1]
    active = [c for c in broad if pd.notna(ma10[c].iloc[-1]) and M[c].iloc[-1] > ma10[c].iloc[-1]]
    inactive = [c for c in broad if c not in active]
    frac = len(active) / len(broad)

    # trailing 12 months of breadth for the trend-of-the-trend
    hist = pd.Series({d: np.mean([M[c].loc[d] > ma10[c].loc[d] for c in broad
                                  if pd.notna(ma10[c].loc[d])]) for d in M.index[-13:]})
    return asof, broad, active, inactive, frac, hist.dropna()


def main():
    print("=" * 78)
    print("MEGATREND BREADTH REPORT — core-book warning indicator (RC-4 salvage)")
    print("=" * 78)
    print("[1] refreshing prices ...")
    refresh_prices()
    asof, broad, active, inactive, frac, hist = build_report()

    zone = ("HEALTHY" if frac >= 0.75 else
            "NARROWING" if frac >= 0.50 else
            "STRESS" if frac >= 0.25 else "CRISIS")
    print(f"\n[2] READING @ {asof.date()}:  {len(active)}/{len(broad)} above 10m mean "
          f"= {frac*100:.0f}%  ->  {zone}")
    print("    trailing 13 months:")
    for d, v in hist.items():
        print(f"      {d.date()}  {v*100:3.0f}%  {'#' * int(v * 40)}")

    print(f"\n[3] ABOVE (trend participants): {sorted(active)}")
    print(f"    BELOW (laggards):            {sorted(inactive)}")

    print(f"""
[4] CALIBRATION (2006-2026): q05 17% | q25 54% | median 77% | q75 89% | q95 96%
    stress-entry history: 2008 Jan 29% | 2020 Feb 31% (Mar 15%) | 2022 Jan 30% (0-4% by summer)

[5] READING GUIDE:
    >= 75% healthy — no action          50-75% narrowing — check rotation
    < 50%  stress signature — de-risk carve-out / tighten core adds
    < 25%  crisis regime — defenses tested

    NOTE: guidance for judgment, not mechanical rules. Monthly cadence only.
""")

    # log it
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(LOG.read_text()) if LOG.exists() else {}
    log[str(asof.date())] = {"frac": round(frac, 3), "n_above": len(active),
                             "n_total": len(broad), "zone": zone,
                             "below": sorted(inactive)}
    LOG.write_text(json.dumps(log, indent=1))
    print(f"[6] logged -> {LOG.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()

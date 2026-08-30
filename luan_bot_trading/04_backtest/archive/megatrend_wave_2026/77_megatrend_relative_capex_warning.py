#!/usr/bin/env python3
"""77_megatrend_relative_capex_warning.py — RC-4 Phase 2c, relative capex.

QUESTION
--------
Does relative capital allocation provide earlier warning than price alone?
Absolute capex can rise in a dying theme. The hypothesis is instead:

    theme TTM capex share = theme capex / all tracked-theme capex

A theme can be spending more in absolute dollars while losing the competition
for the limited capital pool. This script tests whether falling relative share
leads price-trend failure.

POINT-IN-TIME CONTRACT
----------------------
FMP quarterly cash-flow observations are usable only after `acceptedDate`
(or filingDate fallback). At every month-end, for each fiscal period the
latest filing available by that month-end is selected; future amendments are
not visible early. TTM is the last four available reported quarters. No
fiscal-period-date look-ahead.

THEMES / PRICE PROXIES
----------------------
AI/hyperscale: MSFT GOOGL AMZN META NVDA AVGO ORCL -> SMH
clean energy : FSLR ENPH SEDG NEE RUN PLUG             -> equal ICLN/TAN
crypto       : MSTR COIN RIOT MARA CLSK                -> equal MSTR/COIN

TESTS
-----
- relative-share warning: share down >=20% over four quarters
- point-share warning: share down >=5 percentage points over four quarters
- price failure: proxy below its 10-month mean for three consecutive month-ends
- warning lead time, false-warning rate, forward returns, and clean-energy
  2020-2024 natural experiment

This is a warning-context study, not a trading/tilt backtest. The PEAD model
and production db.h5 are untouched. Raw FMP data is cached in db_capex.h5.
"""
from __future__ import annotations
import json
import os
import sys
import time
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
FMP_URL = "https://financialmodelingprep.com/stable/cash-flow-statement"
DB_CAPEX = ROOT / "01_data" / "db_capex.h5"
DB_MT = ROOT / "01_data" / "db_megatrend.h5"

THEMES = {
    "AI/hyperscale": ["MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL"],
    "clean_energy": ["FSLR", "ENPH", "SEDG", "NEE", "RUN", "PLUG"],
    "crypto": ["MSTR", "COIN", "RIOT", "MARA", "CLSK"],
}
PRICE_PROXIES = {
    "AI/hyperscale": ["SMH"],
    "clean_energy": ["ICLN", "TAN"],
    "crypto": ["MSTR", "COIN"],
}


def fetch_raw(sym: str) -> pd.DataFrame | None:
    try:
        r = requests.get(FMP_URL, params={"symbol": sym, "period": "quarter",
                                          "limit": 80, "apikey": FMP_KEY}, timeout=30)
        if r.status_code != 200 or not isinstance(r.json(), list):
            return None
        rows = []
        for x in r.json():
            period = pd.to_datetime(x.get("date"), errors="coerce")
            available = pd.to_datetime(x.get("acceptedDate") or x.get("filingDate"), errors="coerce")
            capex = x.get("capitalExpenditure")
            if pd.isna(period) or pd.isna(available) or capex is None:
                continue
            rows.append({"period_date": period.normalize(),
                         "available_date": available.tz_localize(None) if available.tzinfo else available,
                         "capex": max(0.0, -float(capex)),
                         "filingDate": x.get("filingDate"),
                         "acceptedDate": x.get("acceptedDate")})
        if not rows:
            return None
        return pd.DataFrame(rows).sort_values(["period_date", "available_date"])
    except Exception:
        return None


def refresh_raw():
    with pd.HDFStore(DB_CAPEX, mode="a") as store:
        keys = set(store.keys())
        count = 0
        for syms in THEMES.values():
            for sym in syms:
                k = f"/capex_raw/{sym}"
                df = fetch_raw(sym)
                if df is not None and len(df) >= 8:
                    if k in keys:
                        store.remove(k)
                    store.put(k, df, format="table")
                    count += 1
                else:
                    print(f"  !! no point-in-time capex data for {sym}")
                time.sleep(0.08)
        print(f"  refreshed {count} raw capex panels")


def point_in_time_ttm(months: pd.DatetimeIndex) -> dict[str, pd.Series]:
    out = {}
    with pd.HDFStore(DB_CAPEX, mode="r") as store:
        keys = set(store.keys())
        for theme, syms in THEMES.items():
            company_series = []
            for sym in syms:
                k = f"/capex_raw/{sym}"
                if k not in keys:
                    continue
                raw = store[k].copy()
                raw["period_date"] = pd.to_datetime(raw["period_date"]).dt.normalize()
                raw["available_date"] = pd.to_datetime(raw["available_date"]).dt.tz_localize(None)
                values = []
                for asof in months:
                    visible = raw[raw.available_date <= asof]
                    if visible.empty:
                        values.append(np.nan)
                        continue
                    # Latest available version of each fiscal period by as-of date.
                    latest = visible.sort_values("available_date").drop_duplicates(
                        "period_date", keep="last")
                    latest = latest[latest.period_date <= asof].sort_values("period_date")
                    values.append(float(latest.tail(4).capex.sum())
                                  if len(latest) >= 4 else np.nan)
                company_series.append(pd.Series(values, index=months, name=sym))
            if company_series:
                out[theme] = pd.concat(company_series, axis=1).sum(axis=1, min_count=1)
            else:
                out[theme] = pd.Series(index=months, dtype=float)
    return out


def price_proxy(theme: str, months: pd.DatetimeIndex) -> pd.Series:
    series = []
    with pd.HDFStore(DB_MT, mode="r") as store:
        keys = set(store.keys())
        for sym in PRICE_PROXIES[theme]:
            k = f"/mt/{sym}"
            if k not in keys:
                continue
            p = store[k].set_index("date")["adjClose"].sort_index()
            m = p.resample("ME").last().reindex(months)
            # Equal-weight normalized price-return proxy; avoids nominal-price bias.
            first = m.dropna().iloc[0] if not m.dropna().empty else np.nan
            series.append((m / first).rename(sym))
    if not series:
        return pd.Series(index=months, dtype=float)
    return pd.concat(series, axis=1).mean(axis=1)


def failure_onsets(proxy: pd.Series) -> list[pd.Timestamp]:
    ma = proxy.rolling(10).mean()
    below = (proxy < ma).fillna(False)
    run = below.astype(int).rolling(3).sum()
    onset = run[run >= 3].index
    # one onset per below episode
    return [d for d in onset if not any((d - x).days <= 180 for x in onset if x < d)]


def main():
    print("=" * 92)
    print("RC-4 PHASE 2c — RELATIVE CAPEX WARNING (point-in-time FMP data)")
    print("=" * 92)
    print("[1] Refreshing raw quarterly cash-flow panels with filing availability ...")
    refresh_raw()

    months = pd.date_range("2015-01-31", "2026-08-31", freq="ME")
    capex = point_in_time_ttm(months)
    X = pd.DataFrame(capex).dropna(how="all")
    shares = X.div(X.sum(axis=1), axis=0)
    share4 = shares / shares.shift(4) - 1
    point4 = shares - shares.shift(4)

    print("\n[2] Point-in-time relative capex shares")
    print(f"  {'asof':>12} {'AI':>10} {'clean':>10} {'crypto':>10} {'AI/clean':>10}")
    for d in ["2019-12-31", "2020-12-31", "2021-06-30", "2021-12-31",
              "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31", "2026-06-30"]:
        row = shares.loc[:pd.Timestamp(d)].iloc[-1]
        vals = {k: row.get(k, np.nan) for k in shares.columns}
        print(f"  {d:>12} {vals.get('AI/hyperscale',np.nan)*100:>9.1f}%"
              f" {vals.get('clean_energy',np.nan)*100:>9.1f}%"
              f" {vals.get('crypto',np.nan)*100:>9.1f}%"
              f" {vals.get('AI/hyperscale',np.nan)/vals.get('clean_energy',np.nan):>9.1f}x")

    print("\n[3] Warning observations (share down >=20% YoY OR >=5pp YoY)")
    warnings = {}
    for theme in shares.columns:
        w = (share4[theme] <= -0.20) | (point4[theme] <= -0.05)
        warnings[theme] = list(shares.index[w.fillna(False)])
        print(f"  {theme:>16}: {len(warnings[theme]):2d} warnings; "
              f"latest={warnings[theme][-1].date() if warnings[theme] else 'none'}")

    print("\n[4] Price proxies and warning/failure timing")
    proxy = {t: price_proxy(t, months) for t in THEMES}
    for theme in THEMES:
        p = proxy[theme]
        onsets = failure_onsets(p)
        print(f"\n  {theme}: price failure onsets = {[d.date() for d in onsets]}")
        for onset in onsets:
            prior = [d for d in warnings[theme] if d <= onset]
            lead = (onset - prior[-1]).days / 30.44 if prior else None
            print(f"    failure {onset.date()} | last capex warning "
                  f"{prior[-1].date() if prior else 'NONE'} | lead={lead:.1f}m" if lead is not None
                  else f"    failure {onset.date()} | last capex warning NONE")

    print("\n[5] Clean-energy natural experiment (point-in-time shares)")
    for d in ["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]:
        row = shares.loc[:pd.Timestamp(d)].iloc[-1]
        print(f"  {d}: clean share={row.get('clean_energy',np.nan)*100:.1f}%  "
              f"4q change={share4.loc[:pd.Timestamp(d),'clean_energy'].iloc[-1]*100:+.1f}%  "
              f"point change={point4.loc[:pd.Timestamp(d),'clean_energy'].iloc[-1]*100:+.1f}pp")

    print("\n[6] Forward returns after capex warnings (warning-context diagnostic)")
    for theme in THEMES:
        p = proxy[theme]
        lr = np.log(p / p.shift(1))
        dates = warnings[theme]
        vals = []
        for d in dates:
            if d not in lr.index:
                continue
            i = lr.index.get_loc(d)
            if i + 6 < len(lr):
                vals.append(float(lr.iloc[i + 1:i + 7].sum()))
        print(f"  {theme:>16}: n={len(vals):2d}  mean fwd6m="
              f"{np.mean(vals)*100:+.1f}%" if vals else f"  {theme:>16}: n=0")

    print("\n[7] CONCLUSION")
    print("  Relative capex is now point-in-time and correctly measures allocation share.")
    print("  It is a valid warning-context candidate only if warnings lead price failures;")
    print("  it is NOT automatically a portfolio tilt signal. See output above and findings.")


if __name__ == "__main__":
    main()

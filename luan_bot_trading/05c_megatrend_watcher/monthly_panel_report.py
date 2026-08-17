#!/usr/bin/env python3
"""monthly_panel_report.py — RC-4 operational monthly warning dashboard.

ROLE
----
This is the only surviving megatrend component. It is a CORE-book warning and
context report, not a trading strategy and not an automatic allocation rule.
It is read once at month-end after the close.

PANELS
------
1. EQUITY breadth: SPY, QQQ, IWM, IJR, EFA, EEM and 11 GICS sector ETFs.
2. THEME breadth: AI/semis, clean energy, crypto, biotech, metals, uranium.
3. CROSS-ASSET context: AGG, TLT, GLD, SHY.
4. RELATIVE CAPEX: point-in-time TTM capex share by AI/clean/crypto theme.
5. INSIDER context: full-market theme-company Form 4 net material flow.
6. NEWS context: timestamped FMP operational keyword balance (diagnostic only).

The panel split prevents normal theme rotation from being confused with broad
equity stress. Price breadth remains the timing signal. Capex/insider/news are
slow context and confirmation candidates; none triggers a sell or buy.

READING GUIDE
-------------
Equity breadth: >=75% healthy; 50-75% narrowing; <50% stress; <25% crisis.
Theme and cross-asset panels are descriptive. A narrow theme does not equal a
systemic equity warning. Current context is logged for future calibration.

DATA CONTRACT
-------------
Tiingo daily bars: db_megatrend.h5 (refreshed and merged through latest close).
Relative capex: db_capex.h5, FMP cash-flow observations point-in-time aligned
by acceptedDate/filingDate. Insider/news: full theme panels, not S&P-400-only
caches. Production db.h5, V6, PEAD plans, and positions are untouched.

USAGE
-----
    conda run -n trading python 05c_megatrend_watcher/monthly_panel_report.py
"""
from __future__ import annotations
import json
import os
import re
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
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
FMP_KEY = os.getenv("FMP_API_KEY")

DB_MT = ROOT / "01_data" / "db_megatrend.h5"
DB_CAPEX = ROOT / "01_data" / "db_capex.h5"
DB_INSIDER = ROOT / "01_data" / "db_insider_megatrend.h5"
DB_NEWS = ROOT / "01_data" / "db_news.h5"
LOG = HERE / "logs" / "megatrend_breadth_log.json"

EQUITY = ["SPY", "QQQ", "IWM", "IJR", "EFA", "EEM", "XLB", "XLF", "XLI",
          "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "XLC", "XLE"]
CROSS_ASSET = ["AGG", "TLT", "GLD", "SHY"]
THEME_PROXIES = {
    "AI/hyperscale": ["SMH"],
    "clean_energy": ["ICLN", "TAN"],
    "crypto": ["MSTR", "COIN"],
    "biotech": ["XBI", "IBB"],
    "metals": ["GDX", "LIT"],
    "uranium": ["URA"],
}
THEME_MEMBERS = {
    "AI/hyperscale": ["MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL"],
    "clean_energy": ["FSLR", "ENPH", "SEDG", "NEE", "RUN", "PLUG"],
    "crypto": ["MSTR", "COIN", "RIOT", "MARA", "CLSK"],
}
NEGATIVE_TERMS = re.compile(
    r"(?:guidance\s+(?:cut|lower|reduc)|lower(?:ed|ing)?\s+guidance|"
    r"capex\s+(?:cut|reduc|slash|lower)|weak\s+(?:demand|orders?|backlog)|"
    r"demand\s+(?:slow|weak|collapse|declin)|order\s+(?:cancel|cut|delay|declin)|"
    r"project\s+(?:cancel|delay|scrap)|inventory\s+(?:build|glut|excess|surplus)|"
    r"oversupply|overcapacity|pricing\s+pressure|price\s+(?:cut|declin|pressure)|"
    r"layoff|bankrupt|restructur|shutdown|production\s+cut|slowdown|deteriorat)", re.I)
POSITIVE_TERMS = re.compile(
    r"(?:raise[sd]?\s+guidance|record\s+(?:orders?|backlog|demand|revenue)|"
    r"strong\s+(?:demand|orders?|backlog)|order\s+book|capacity\s+expan|"
    r"capex\s+(?:increase|boost|raise|plan)|pricing\s+power)", re.I)


def zone(frac: float) -> str:
    return "HEALTHY" if frac >= .75 else "NARROWING" if frac >= .50 else "STRESS" if frac >= .25 else "CRISIS"


def refresh_prices() -> None:
    """Refresh every cached megatrend series, merging new rows with history."""
    with pd.HDFStore(DB_MT, mode="a") as store:
        keys = list(store.keys())
        for k in keys:
            sym = k.split("/")[-1]
            try:
                r = requests.get(
                    f"https://api.tiingo.com/tiingo/daily/{sym}/prices",
                    params={"token": TIINGO_API_KEY, "startDate": "2023-01-01",
                            "columns": "date,adjClose"}, timeout=60)
                if r.status_code != 200:
                    print(f"  !! price refresh failed for {sym}; using cache")
                    continue
                new = pd.DataFrame(r.json())
                if new.empty or "adjClose" not in new.columns:
                    continue
                new["date"] = pd.to_datetime(new["date"]).dt.tz_localize(None).dt.normalize()
                new["adjClose"] = pd.to_numeric(new["adjClose"], errors="coerce")
                new = new.dropna().sort_values("date")[["date", "adjClose"]]
                old = store[k][["date", "adjClose"]]
                old["date"] = pd.to_datetime(old["date"]).dt.tz_localize(None).dt.normalize()
                merged = pd.concat([old, new]).drop_duplicates("date", keep="last").sort_values("date")
                store.put(k, merged.reset_index(drop=True), format="table")
            except Exception as exc:
                print(f"  !! {sym}: {exc}")
            time.sleep(0.05)


def monthly_matrix() -> pd.DataFrame:
    with pd.HDFStore(DB_MT, mode="r") as s:
        px = {k.split("/")[-1]: s[k].set_index("date")["adjClose"] for k in s.keys()}
    return pd.DataFrame({sym: ser.resample("ME").last() for sym, ser in px.items()})


def panel_breadth(M: pd.DataFrame, names: list[str]) -> tuple[pd.Series, list[str], list[str]]:
    available = [c for c in names if c in M.columns]
    p = M[available]
    ma = p.rolling(10).mean()
    above = p > ma
    frac = above.mean(axis=1)
    latest = frac.index[-1]
    active = [c for c in available if pd.notna(ma[c].loc[latest]) and p[c].loc[latest] > ma[c].loc[latest]]
    return frac, active, [c for c in available if c not in active]


def theme_matrix(M: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for theme, proxies in THEME_PROXIES.items():
        series = []
        for sym in proxies:
            if sym in M.columns:
                p = M[sym].ffill()
                if p.notna().any():
                    series.append((p / p.dropna().iloc[0]).rename(sym))
        if series:
            out[theme] = pd.concat(series, axis=1).mean(axis=1)
    return pd.DataFrame(out)


def latest_capex(asof: pd.Timestamp) -> dict:
    result = {}
    if not DB_CAPEX.exists():
        return result
    with pd.HDFStore(DB_CAPEX, mode="r") as s:
        keys = set(s.keys())
        raw_by_theme = {}
        for theme, members in THEME_MEMBERS.items():
            vals = []
            for sym in members:
                k = f"/capex_raw/{sym}"
                if k not in keys:
                    continue
                d = s[k].copy()
                d["period_date"] = pd.to_datetime(d["period_date"]).dt.normalize()
                d["available_date"] = pd.to_datetime(d["available_date"]).dt.tz_localize(None)
                visible = d[d.available_date <= asof].sort_values("available_date").drop_duplicates("period_date", keep="last")
                visible = visible[visible.period_date <= asof].sort_values("period_date")
                if len(visible) >= 4:
                    vals.append(float(visible.tail(4).capex.sum()))
            raw_by_theme[theme] = sum(vals) if vals else np.nan
        total = sum(v for v in raw_by_theme.values() if pd.notna(v))
        shares = {t: v / total if total else np.nan for t, v in raw_by_theme.items()}
        result["values_b"] = {t: v / 1e9 for t, v in raw_by_theme.items()}
        result["shares"] = shares
    return result


def latest_insider(asof: pd.Timestamp) -> dict:
    result = {}
    if not DB_INSIDER.exists():
        return result
    with pd.HDFStore(DB_INSIDER, mode="r") as s:
        keys = set(s.keys())
        for theme, members in THEME_MEMBERS.items():
            buys = sells = 0.0
            companies = set()
            for sym in members:
                k = f"/insider/{sym}"
                if k not in keys:
                    continue
                d = s[k].copy()
                d["filingDate"] = pd.to_datetime(d["filingDate"], errors="coerce").dt.tz_localize(None).dt.normalize()
                d["value"] = pd.to_numeric(d.get("value", 0), errors="coerce").fillna(0)
                d = d[(d.filingDate <= asof) & (d.filingDate > asof - pd.Timedelta(days=90)) & (d.value >= 50000)]
                typ = d.transactionType.astype(str)
                if (typ.str.startswith("P")).any():
                    buys += d.loc[typ.str.startswith("P"), "value"].sum(); companies.update(d.loc[typ.str.startswith("P"), "symbol"].dropna() if "symbol" in d else [sym])
                sells += d.loc[typ.str.startswith("S"), "value"].sum()
            result[theme] = {"buy_b": buys / 1e6, "sell_b": sells / 1e6,
                             "net_b": (buys - sells) / 1e6, "warning": sells > buys}
    return result


def latest_news(asof: pd.Timestamp) -> dict:
    result = {}
    if not DB_NEWS.exists():
        return result
    with pd.HDFStore(DB_NEWS, mode="r") as s:
        keys = set(s.keys())
        for theme, members in THEME_MEMBERS.items():
            rows = []
            for sym in members:
                k = f"/news/{sym}"
                if k in keys:
                    d = s[k].copy(); d["symbol"] = sym; rows.append(d)
            if not rows:
                result[theme] = {"articles": 0, "neg_days": 0, "pos_days": 0, "warning": False}; continue
            d = pd.concat(rows, ignore_index=True)
            d["publishedDate"] = pd.to_datetime(d["publishedDate"], errors="coerce").dt.tz_localize(None)
            d = d[(d.publishedDate < asof + pd.Timedelta(days=1)) & (d.publishedDate >= asof - pd.Timedelta(days=90))].copy()
            d["txt"] = d.title.fillna("") + " " + d.text.fillna("")
            d["day"] = d.publishedDate.dt.normalize()
            d = d.drop_duplicates(["symbol", "day", "txt"])
            d["neg"] = d.txt.str.contains(NEGATIVE_TERMS); d["pos"] = d.txt.str.contains(POSITIVE_TERMS)
            neg_days = d.loc[d.neg, ["symbol", "day"]].drop_duplicates().shape[0]
            pos_days = d.loc[d.pos, ["symbol", "day"]].drop_duplicates().shape[0]
            result[theme] = {"articles": len(d), "neg_days": int(neg_days), "pos_days": int(pos_days),
                             "warning": bool(neg_days >= 2 and neg_days > pos_days)}
    return result


def main():
    print("=" * 82)
    print("MEGATREND MONTHLY PANEL REPORT — RC-4 operational warning/context dashboard")
    print("=" * 82)
    print("[1] refreshing Tiingo prices ...")
    refresh_prices()
    M = monthly_matrix()
    asof = M.index[-1]
    equity, eq_active, eq_inactive = panel_breadth(M, EQUITY)
    cross, cross_active, cross_inactive = panel_breadth(M, CROSS_ASSET)
    themeM = theme_matrix(M)
    theme_breadth, theme_active, theme_inactive = panel_breadth(themeM, list(themeM.columns))
    capex = latest_capex(asof); insider = latest_insider(asof); news = latest_news(asof)
    eq_frac = float(equity.loc[asof]); cross_frac = float(cross.loc[asof]); theme_frac = float(theme_breadth.loc[asof])
    print(f"\n[2] AS OF {asof.date()}")
    print(f"  EQUITY breadth:      {len(eq_active)}/{len(EQUITY)} = {eq_frac*100:.0f}% -> {zone(eq_frac)}")
    print(f"  THEME breadth:       {len(theme_active)}/{len(themeM.columns)} = {theme_frac*100:.0f}% -> {zone(theme_frac)}")
    print(f"  CROSS-ASSET breadth:  {len(cross_active)}/{len(CROSS_ASSET)} = {cross_frac*100:.0f}%")
    print(f"\n[3] EQUITY ACTIVE: {sorted(eq_active)}\n    LAGGING: {sorted(eq_inactive)}")
    print(f"\n[4] THEME ACTIVE: {sorted(theme_active)}\n    LAGGING: {sorted(theme_inactive)}")
    print(f"\n[5] CROSS-ASSET ACTIVE: {sorted(cross_active)}\n    BELOW MA10m: {sorted(cross_inactive)}")
    print("\n[6] FUNDAMENTAL CONTEXT")
    print("  Relative TTM capex shares:")
    for t, sh in capex.get("shares", {}).items(): print(f"    {t:>16}: {sh*100:5.1f}%  ({capex['values_b'].get(t, np.nan):,.0f}B)")
    print("  Insider trailing 90d material net flow ($M; negative = selling):")
    for t, x in insider.items(): print(f"    {t:>16}: net={x['net_b']:+,.1f}  buy={x['buy_b']:,.1f} sell={x['sell_b']:,.1f} warning={x['warning']}")
    print("  News trailing 90d unique company-day diagnostic:")
    for t, x in news.items(): print(f"    {t:>16}: articles={x['articles']:4d} neg_days={x['neg_days']:3d} pos_days={x['pos_days']:3d} warning={x['warning']}")
    print("\n[7] INTERPRETATION (not an automatic trading instruction)")
    print("  Equity breadth is the timing panel. Theme breadth describes rotation.")
    print("  Cross-asset/capex/insider/news are context only; no automatic exit or buy.")
    print("  Historical calibration: equity q05=17%, q25=54%, median=77%, q75=89%.")
    print("\n[8] TRAILING 13-MONTH EQUITY BREADTH")
    for d, v in equity.dropna().tail(13).items(): print(f"  {d.date()} {v*100:3.0f}% {'#'*int(v*40)}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(LOG.read_text()) if LOG.exists() else {}
    log[str(asof.date())] = {"equity": {"frac": round(eq_frac,3), "zone": zone(eq_frac), "active": sorted(eq_active)},
        "theme": {"frac": round(theme_frac,3), "active": sorted(theme_active)},
        "cross_asset": {"frac": round(cross_frac,3), "active": sorted(cross_active)},
        "capex": capex, "insider": insider, "news": news}
    LOG.write_text(json.dumps(log, indent=1, default=str))
    print(f"\n[9] logged -> {LOG.relative_to(ROOT.parent)}")

if __name__ == "__main__": main()

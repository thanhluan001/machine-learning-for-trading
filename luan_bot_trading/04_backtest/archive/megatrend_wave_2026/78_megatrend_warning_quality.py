#!/usr/bin/env python3
"""78_megatrend_warning_quality.py — RC-4 multi-source warning-quality study.

This is NOT a trading strategy and does not change production allocation.
It tests whether slow fundamental/public channels improve the monthly
megatrend breadth warning:

  PRICE  : proxy below its 10-month mean / trend-failure state
  CAPEX  : point-in-time relative TTM capex share down >=20% YoY or >=5pp
  INSIDER: material net selling in available S&P-400 coverage, trailing 90d
  NEWS   : operationally negative FMP stock-news articles, trailing 90d

Themes and proxies:
  AI/hyperscale -> SMH; clean_energy -> equal ICLN/TAN; crypto -> equal MSTR/COIN

POINT-IN-TIME CONTRACT
----------------------
- Capex uses FMP acceptedDate/filingDate, never fiscal-period date as knowledge.
- News uses publishedDate; articles are visible only after publication.
- Insider activity uses filingDate from cached Form 4 data, not transactionDate.

NEWS CLASSIFIER
---------------
This first pass deliberately uses a transparent operational-keyword screen,
not a fitted NLP model. Negative terms cover demand weakness, guidance cuts,
order/project cancellations, inventory/oversupply, pricing pressure, layoffs,
bankruptcy/restructuring, and capex reductions. It is a research diagnostic;
keyword thresholds are not production parameters.

OUTPUT
------
Lead-time, false-warning, forward-six-month return, coverage, and channel
ablation tables. Production db.h5 and frozen V6 are untouched. Raw news is
cached in 01_data/db_news.h5.
"""
from __future__ import annotations
import os, re, sys, time
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
DB_MT = ROOT / "01_data" / "db_megatrend.h5"
DB_INSIDER = ROOT / "01_data" / "db_insider.h5"  # S&P 400-only legacy cache
DB_INSIDER_MT = ROOT / "01_data" / "db_insider_megatrend.h5"  # full theme panels
DB_PROD = ROOT / "01_data" / "db.h5"
DB_NEWS = ROOT / "01_data" / "db_news.h5"
NEWS_URL = "https://financialmodelingprep.com/stable/news/stock"

THEMES = {
    "AI/hyperscale": {"members": ["MSFT","GOOGL","AMZN","META","NVDA","AVGO","ORCL"], "proxies": ["SMH"]},
    "clean_energy": {"members": ["FSLR","ENPH","SEDG","NEE","RUN","PLUG"], "proxies": ["ICLN","TAN"]},
    "crypto": {"members": ["MSTR","COIN","RIOT","MARA","CLSK"], "proxies": ["MSTR","COIN"]},
}
NEGATIVE_TERMS = re.compile(
    r"(?:guidance\s+(?:cut|lower|reduc)|lower(?:ed|ing)?\s+guidance|"
    r"capex\s+(?:cut|reduc|slash|lower)|capital expenditure.{0,30}(?:cut|reduc|lower)|"
    r"weak\s+(?:demand|orders?|backlog)|demand\s+(?:slow|weak|collapse|declin)|"
    r"order\s+(?:cancel|cut|delay|declin)|project\s+(?:cancel|delay|scrap)|"
    r"inventory\s+(?:build|glut|excess|surplus)|oversupply|overcapacity|"
    r"pricing\s+pressure|price\s+(?:cut|declin|pressure)|"
    r"layoff|bankrupt|restructur|shutdown|production\s+cut|slowdown|deteriorat)", re.I)
POSITIVE_TERMS = re.compile(
    r"(?:raise[sd]?\s+guidance|record\s+(?:orders?|backlog|demand|revenue)|"
    r"strong\s+(?:demand|orders?|backlog)|order\s+book|capacity\s+expan|"
    r"capex\s+(?:increase|boost|raise|plan)|pricing\s+power)", re.I)


def refresh_news():
    """Fetch historical FMP news pages for all theme members; cache compact rows."""
    with pd.HDFStore(DB_NEWS, mode="a") as store:
        keys = set(store.keys())
        n = 0
        for theme, cfg in THEMES.items():
            for sym in cfg["members"]:
                k = f"/news/{sym}"
                if k in keys:
                    continue
                rows = []
                for page in range(20):
                    try:
                        r = requests.get(NEWS_URL, params={"symbols": sym, "from": "2015-01-01",
                            "to": "2026-08-31", "limit": 250, "page": page, "apikey": FMP_KEY}, timeout=30)
                        data = r.json() if r.status_code == 200 else []
                        if not isinstance(data, list) or not data:
                            break
                        for x in data:
                            rows.append({"publishedDate": x.get("publishedDate"),
                                         "title": x.get("title", ""), "text": x.get("text", "")})
                        if len(data) < 250:
                            break
                    except Exception:
                        break
                    time.sleep(0.05)
                if rows:
                    df = pd.DataFrame(rows).drop_duplicates(subset=["publishedDate", "title"])
                    df["publishedDate"] = pd.to_datetime(df["publishedDate"], errors="coerce").dt.tz_localize(None)
                    df = df.dropna(subset=["publishedDate"]).sort_values("publishedDate")
                    store.put(k, df, format="table")
                    n += 1
                time.sleep(0.08)
        print(f"  refreshed {n} news panels; cache nodes: {len(store.keys())}")


def load_point_capex(months):
    out = {}
    with pd.HDFStore(DB_CAPEX, mode="r") as store:
        keys = set(store.keys())
        for theme, cfg in THEMES.items():
            companies = []
            for sym in cfg["members"]:
                k = f"/capex_raw/{sym}"
                if k not in keys:
                    continue
                raw = store[k].copy()
                raw["period_date"] = pd.to_datetime(raw["period_date"]).dt.normalize()
                raw["available_date"] = pd.to_datetime(raw["available_date"]).dt.tz_localize(None)
                vals = []
                for asof in months:
                    visible = raw[raw.available_date <= asof]
                    latest = visible.sort_values("available_date").drop_duplicates("period_date", keep="last")
                    latest = latest[latest.period_date <= asof].sort_values("period_date")
                    vals.append(float(latest.tail(4).capex.sum()) if len(latest) >= 4 else np.nan)
                companies.append(pd.Series(vals, index=months, name=sym))
            out[theme] = pd.concat(companies, axis=1).sum(axis=1, min_count=1) if companies else pd.Series(index=months, dtype=float)
    X = pd.DataFrame(out)
    return X.div(X.sum(axis=1), axis=0)


def load_prices(months):
    with pd.HDFStore(DB_MT, mode="r") as store:
        result = {}
        for theme, cfg in THEMES.items():
            arr = []
            for sym in cfg["proxies"]:
                k = f"/mt/{sym}"
                if k not in store.keys(): continue
                p = store[k].set_index("date")["adjClose"].sort_index().resample("ME").last()
                p = p.reindex(months).ffill()
                if p.notna().any(): arr.append((p / p.dropna().iloc[0]).rename(sym))
            result[theme] = pd.concat(arr, axis=1).mean(axis=1) if arr else pd.Series(index=months, dtype=float)
    return pd.DataFrame(result)


def refresh_insider_theme():
    """Fetch Form 4 data directly by theme ticker, not through S&P 400 IDs."""
    with pd.HDFStore(DB_INSIDER_MT, mode="a") as store:
        keys = set(store.keys())
        count = 0
        for cfg in THEMES.values():
            for sym in cfg["members"]:
                k = f"/insider/{sym}"
                if k in keys:
                    continue
                rows = []
                for page in range(5):
                    try:
                        r = requests.get(
                            "https://financialmodelingprep.com/stable/insider-trading/search",
                            params={"symbol": sym, "limit": 1000, "page": page, "apikey": FMP_KEY},
                            timeout=30)
                        data = r.json() if r.status_code == 200 else []
                        if not isinstance(data, list) or not data:
                            break
                        rows.extend(data)
                        if len(data) < 1000:
                            break
                    except Exception:
                        break
                    time.sleep(0.05)
                if rows:
                    d = pd.DataFrame(rows).drop_duplicates()
                    d["filingDate"] = pd.to_datetime(d["filingDate"], errors="coerce").dt.tz_localize(None).dt.normalize()
                    d["price"] = pd.to_numeric(d.get("price", 0), errors="coerce").fillna(0)
                    d["securitiesTransacted"] = pd.to_numeric(d.get("securitiesTransacted", 0), errors="coerce").fillna(0)
                    d["value"] = d["price"] * d["securitiesTransacted"]
                    d.to_hdf(DB_INSIDER_MT, key=k, mode="a", format="table")
                    count += 1
                time.sleep(0.08)
        print(f"  refreshed {count} full-market theme insider panels; cache nodes: {len(store.keys())}")


def load_insider(months):
    """Theme-level trailing 90d net material insider dollars from full panels."""
    rows = []
    with pd.HDFStore(DB_INSIDER_MT, mode="r") as store:
        keys = set(store.keys())
        for theme, cfg in THEMES.items():
            theme_events = []
            for sym in cfg["members"]:
                k = f"/insider/{sym}"
                if k not in keys:
                    continue
                d = store[k].copy()
                d["filingDate"] = pd.to_datetime(d["filingDate"], errors="coerce").dt.tz_localize(None).dt.normalize()
                d["buy"] = d.transactionType.astype(str).str.startswith("P")
                d["sell"] = d.transactionType.astype(str).str.startswith("S")
                d = d[(d.value >= 50000) & (d.buy | d.sell)]
                theme_events.append(d[["filingDate", "value", "buy", "sell"]])
            if theme_events:
                ev = pd.concat(theme_events, ignore_index=True)
                vals = []
                for asof in months:
                    x = ev[(ev.filingDate <= asof) & (ev.filingDate > asof - pd.Timedelta(days=90))]
                    vals.append(float(x.loc[x.buy, "value"].sum() - x.loc[x.sell, "value"].sum()))
                rows.append(pd.Series(vals, index=months, name=theme))
            else:
                rows.append(pd.Series(np.nan, index=months, name=theme))
    return pd.concat(rows, axis=1) if rows else pd.DataFrame(index=months)


def load_news(months):
    out = {}
    with pd.HDFStore(DB_NEWS, mode="r") as store:
        keys = set(store.keys())
        for theme, cfg in THEMES.items():
            allrows = []
            for sym in cfg["members"]:
                k = f"/news/{sym}"
                if k in keys:
                    d = store[k].copy(); d["sym"] = sym; allrows.append(d)
            rows = []
            for asof in months:
                if not allrows: rows.append((0,0,0)); continue
                d = pd.concat(allrows, ignore_index=True)
                d = d[(d.publishedDate < asof + pd.Timedelta(days=1)) & (d.publishedDate >= asof - pd.Timedelta(days=90))]
                txt = d.title.fillna("") + " " + d.text.fillna("")
                neg = txt.str.contains(NEGATIVE_TERMS).sum()
                pos = txt.str.contains(POSITIVE_TERMS).sum()
                rows.append((len(d), int(neg), int(pos)))
            out[theme] = pd.DataFrame(rows, index=months, columns=["news_n","news_neg","news_pos"])
    return out


def failure_onsets(proxy):
    ma = proxy.rolling(10).mean()
    below = (proxy < ma).fillna(False)
    run = below.astype(int).rolling(3).sum()
    onset = list(run[run >= 3].index)
    return [d for d in onset if not any((d-x).days <= 180 for x in onset if x < d)]


def main():
    print("=" * 92)
    print("RC-4 WARNING-QUALITY STUDY — PRICE + RELATIVE CAPEX + INSIDER + NEWS")
    print("=" * 92)
    months = pd.date_range("2015-01-31", "2026-08-31", freq="ME")
    print("[1] Refreshing full-market FMP news and insider panels ...")
    refresh_news()
    refresh_insider_theme()
    shares = load_point_capex(months)
    prices = load_prices(months)
    insider = load_insider(months)
    news = load_news(months)

    print("\n[2] Channel coverage")
    for theme in THEMES:
        print(f"  {theme:>16}: capex={shares[theme].notna().sum():3d} months  "
              f"insider={insider[theme].notna().sum():3d} months  "
              f"news={sum(news[theme].news_n>0):3d} months")

    # Fixed signal definitions; capex uses four-quarter changes.
    capex_yoy = shares / shares.shift(4) - 1
    capex_pp = shares - shares.shift(4)
    capex_warn = (capex_yoy <= -0.20) | (capex_pp <= -0.05)
    price_ma = prices.rolling(10).mean()
    price_below = prices < price_ma
    price_slope = prices.pct_change(3) < 0
    price_warn = price_below & price_slope
    insider_warn = insider < 0
    news_warn = pd.DataFrame({t: (news[t].news_neg >= 2) & (news[t].news_neg > news[t].news_pos)
                              for t in THEMES}, index=months)

    print("\n[3] Warning counts by channel")
    for t in THEMES:
        print(f"  {t:>16}: price={price_warn[t].sum():2d} capex={capex_warn[t].sum():2d} "
              f"insider={insider_warn[t].sum():2d} news={news_warn[t].sum():2d}  "
              f"capex+price={(capex_warn[t]&price_warn[t]).sum():2d}")

    print("\n[4] Price-failure lead/lag test")
    for t in THEMES:
        onsets = failure_onsets(prices[t])
        print(f"\n  {t}: failures={[d.date() for d in onsets]}")
        for onset in onsets:
            line=[]
            for nm, sig in [("capex",capex_warn[t]),("price",price_warn[t]),("insider",insider_warn[t]),("news",news_warn[t])]:
                prev=[d for d in sig.index[sig.fillna(False)] if d<=onset]
                if prev:
                    lead=(onset-prev[-1]).days/30.44; line.append(f"{nm}:{lead:.0f}m")
                else: line.append(f"{nm}:none")
            print(f"    {onset.date()}: " + " ".join(line))

    print("\n[5] Ablation: forward 6-month proxy return after warnings")
    combos = {
        "price": lambda t: price_warn[t],
        "capex": lambda t: capex_warn[t],
        "price+capex": lambda t: price_warn[t]&capex_warn[t],
        "price+capex+insider": lambda t: price_warn[t]&capex_warn[t]&insider_warn[t],
        "price+capex+news": lambda t: price_warn[t]&capex_warn[t]&news_warn[t],
        "all4": lambda t: price_warn[t]&capex_warn[t]&insider_warn[t]&news_warn[t],
    }
    for cname, fun in combos.items():
        vals=[]; n=0
        for t in THEMES:
            sig=fun(t); lr=np.log(prices[t]/prices[t].shift(1))
            for i,d in enumerate(months):
                if bool(sig.iloc[i]) and i+6<len(lr):
                    x=lr.iloc[i+1:i+7].dropna()
                    if len(x)>=4: vals.append(float(x.sum())); n+=1
        print(f"  {cname:>25}: n={n:3d} mean fwd6m={np.mean(vals)*100:+6.1f}% "
              f"median={np.median(vals)*100:+6.1f}% win={np.mean(np.array(vals)>0)*100:4.1f}%" if vals else f"  {cname:>25}: n=0")

    print("\n[6] Current context (latest month)")
    for t in THEMES:
        d=months[-1]
        print(f"  {t:>16}: price_warn={bool(price_warn.loc[d,t])} capex_warn={bool(capex_warn.loc[d,t])} "
              f"insider_warn={bool(insider_warn.loc[d,t])} news_warn={bool(news_warn.loc[d,t])} "
              f"capex_share={shares.loc[d,t]*100:.1f}% news90d={int(news[t].loc[d,'news_n'])}")

    print("\n[7] CONCLUSION")
    print("  This is a warning-quality diagnostic. No allocation rule is changed.")
    print("  Interpret capex as early sponsorship context; price as market timing;")
    print("  insider/news as confirmation only where coverage is adequate.")


if __name__ == "__main__":
    main()

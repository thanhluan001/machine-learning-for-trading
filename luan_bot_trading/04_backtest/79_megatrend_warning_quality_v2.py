#!/usr/bin/env python3
"""79_megatrend_warning_quality_v2.py — RC-4 second warning-quality study.

FOCUS
-----
Phase 3's failure was not inability to exit; it was:
    exit -> temporary bear-market recovery -> re-entry -> renewed decline.

This study tests whether additional slow channels improve warning states and
especially reduce false re-entry. It does NOT place trades and does not modify
PEAD or core allocations.

PANELS
------
1. EQUITY breadth: SPY, QQQ, IWM, IJR, EFA, EEM and 11 GICS ETFs.
2. THEME breadth: Phase-1/theme proxies.
3. CROSS-ASSET context: AGG, TLT, GLD, SHY (not counted as equity breadth).
4. Relative capex: point-in-time TTM theme-capex shares.
5. Full-market insider: direct FMP ticker panels in db_insider_megatrend.h5.
6. Operational news: unique company-day article counts and negative/positive
   keyword balance from timestamped FMP news. This is a transparent diagnostic,
   not a validated NLP model.

FIXED SIGNALS
-------------
- Price stress: equity breadth < 50% AND 3-month SPY return < 0.
- Price recovery: equity breadth >= 50% after a prior stress month.
- Capex stress: >=2 of 3 themes have relative share warning.
- Insider stress: >=2 of 3 themes have material net selling in trailing 90d.
- News stress: >=2 of 3 themes have negative operational article balance,
  requiring at least two unique negative company-days.
- Confirmed stress: price stress plus at least one slow confirmation.
- Confirmed recovery: price recovery, breadth >=60%, and no confirmed capex /
  insider / news stress. This is evaluated only as a diagnostic.

No threshold is promoted to production. Results must be interpreted with the
small theme count and historical news/insider coverage limitations.
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
load_dotenv(ROOT / ".env")
DB_MT = ROOT / "01_data" / "db_megatrend.h5"
DB_CAPEX = ROOT / "01_data" / "db_capex.h5"
DB_INSIDER = ROOT / "01_data" / "db_insider_megatrend.h5"
DB_NEWS = ROOT / "01_data" / "db_news.h5"

THEMES = {
    "AI/hyperscale": {"members": ["MSFT","GOOGL","AMZN","META","NVDA","AVGO","ORCL"], "proxies": ["SMH"]},
    "clean_energy": {"members": ["FSLR","ENPH","SEDG","NEE","RUN","PLUG"], "proxies": ["ICLN","TAN"]},
    "crypto": {"members": ["MSTR","COIN","RIOT","MARA","CLSK"], "proxies": ["MSTR","COIN"]},
}
EQUITY = ["SPY","QQQ","IWM","IJR","EFA","EEM","XLB","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY","XLC","XLE"]
CROSS_ASSET = ["AGG","TLT","GLD","SHY"]
NEG = re.compile(r"(?:guidance\s+(?:cut|lower|reduc)|lower(?:ed|ing)?\s+guidance|capex\s+(?:cut|reduc|slash|lower)|capital expenditure.{0,30}(?:cut|reduc|lower)|weak\s+(?:demand|orders?|backlog)|demand\s+(?:slow|weak|collapse|declin)|order\s+(?:cancel|cut|delay|declin)|project\s+(?:cancel|delay|scrap)|inventory\s+(?:build|glut|excess|surplus)|oversupply|overcapacity|pricing\s+pressure|price\s+(?:cut|declin|pressure)|layoff|bankrupt|restructur|shutdown|production\s+cut|slowdown|deteriorat)", re.I)
POS = re.compile(r"(?:raise[sd]?\s+guidance|record\s+(?:orders?|backlog|demand|revenue)|strong\s+(?:demand|orders?|backlog)|order\s+book|capacity\s+expan|capex\s+(?:increase|boost|raise|plan)|pricing\s+power)", re.I)


def monthly_prices():
    with pd.HDFStore(DB_MT, "r") as s:
        raw = {k.split("/")[-1]: s[k].set_index("date")["adjClose"] for k in s.keys()}
    return pd.DataFrame({k: v.resample("ME").last() for k, v in raw.items()})


def point_capex(months):
    out = {}
    with pd.HDFStore(DB_CAPEX, "r") as s:
        keys = set(s.keys())
        for theme, cfg in THEMES.items():
            vals = []
            for sym in cfg["members"]:
                k = f"/capex_raw/{sym}"
                if k not in keys: continue
                d = s[k].copy()
                d["period_date"] = pd.to_datetime(d.period_date).dt.normalize()
                d["available_date"] = pd.to_datetime(d.available_date).dt.tz_localize(None)
                a = []
                for asof in months:
                    x = d[d.available_date <= asof].sort_values("available_date").drop_duplicates("period_date", keep="last")
                    x = x[x.period_date <= asof].sort_values("period_date")
                    a.append(float(x.tail(4).capex.sum()) if len(x) >= 4 else np.nan)
                vals.append(pd.Series(a, index=months, name=sym))
            out[theme] = pd.concat(vals, axis=1).sum(axis=1, min_count=1) if vals else pd.Series(np.nan, index=months)
    x = pd.DataFrame(out)
    return x.div(x.sum(axis=1), axis=0)


def price_panel(M, names, months):
    return pd.DataFrame({c: M[c].reindex(months).ffill() for c in names if c in M.columns})


def theme_proxies(M, months):
    out = {}
    for theme, cfg in THEMES.items():
        vals = []
        for c in cfg["proxies"]:
            if c in M.columns:
                x = M[c].reindex(months).ffill()
                if x.notna().any(): vals.append((x / x.dropna().iloc[0]).rename(c))
        out[theme] = pd.concat(vals, axis=1).mean(axis=1) if vals else pd.Series(np.nan, index=months)
    return pd.DataFrame(out)


def insider_signal(months):
    out = []
    with pd.HDFStore(DB_INSIDER, "r") as s:
        keys = set(s.keys())
        for theme, cfg in THEMES.items():
            pieces = []
            for sym in cfg["members"]:
                k = f"/insider/{sym}"
                if k not in keys: continue
                d = s[k].copy()
                d["filingDate"] = pd.to_datetime(d.filingDate, errors="coerce").dt.tz_localize(None).dt.normalize()
                d["value"] = pd.to_numeric(d.value, errors="coerce").fillna(0)
                d["buy"] = d.transactionType.astype(str).str.startswith("P")
                d["sell"] = d.transactionType.astype(str).str.startswith("S")
                pieces.append(d[(d.value >= 50000) & (d.buy | d.sell)][["filingDate","value","buy","sell"]])
            vals = []
            ev = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
            for asof in months:
                x = ev[(ev.filingDate <= asof) & (ev.filingDate > asof - pd.Timedelta(days=90))] if not ev.empty else ev
                vals.append(float(x.loc[x.buy,"value"].sum() - x.loc[x.sell,"value"].sum()) if not x.empty else np.nan)
            out.append(pd.Series(vals, index=months, name=theme))
    return pd.concat(out, axis=1)


def news_signal(months):
    out = {}
    with pd.HDFStore(DB_NEWS, "r") as s:
        keys = set(s.keys())
        for theme, cfg in THEMES.items():
            pieces=[]
            for sym in cfg["members"]:
                k=f"/news/{sym}"
                if k in keys:
                    d=s[k].copy(); d["symbol"]=sym; pieces.append(d)
            d=pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["publishedDate","title","text","symbol"])
            d["publishedDate"]=pd.to_datetime(d.publishedDate, errors="coerce").dt.tz_localize(None)
            vals=[]
            for asof in months:
                x=d[(d.publishedDate < asof+pd.Timedelta(days=1)) & (d.publishedDate >= asof-pd.Timedelta(days=90))].copy()
                if x.empty: vals.append((0,0,0)); continue
                x["txt"]=x.title.fillna("")+" "+x.text.fillna("")
                # One company-day counts once, preventing syndicated article floods.
                x["day"]=x.publishedDate.dt.normalize()
                x=x.sort_values("publishedDate").drop_duplicates(["symbol","day","txt"])
                x["neg"]=x.txt.str.contains(NEG); x["pos"]=x.txt.str.contains(POS)
                neg_days=x.loc[x.neg,["symbol","day"]].drop_duplicates().shape[0]
                pos_days=x.loc[x.pos,["symbol","day"]].drop_duplicates().shape[0]
                vals.append((x.symbol.nunique(), int(neg_days), int(pos_days)))
            out[theme]=pd.DataFrame(vals,index=months,columns=["companies","neg_days","pos_days"])
    return out


def main():
    print("="*92); print("RC-4 SECOND WARNING-QUALITY STUDY — SPLIT BREADTH + CONFIRMATIONS"); print("="*92)
    M=monthly_prices(); months=pd.date_range("2015-01-31","2026-08-31",freq="ME")
    eq=price_panel(M,EQUITY,months); cross=price_panel(M,CROSS_ASSET,months); theme=theme_proxies(M,months)
    shares=point_capex(months); ins=insider_signal(months); news=news_signal(months)
    eq_ma=eq.rolling(10).mean(); eq_breadth=(eq>eq_ma).mean(axis=1)
    theme_ma=theme.rolling(10).mean(); theme_breadth=(theme>theme_ma).mean(axis=1)
    cross_ma=cross.rolling(10).mean(); cross_breadth=(cross>cross_ma).mean(axis=1)
    spy_ret=eq["SPY"]/eq["SPY"].shift(3)-1
    price_stress=(eq_breadth<.50)&(spy_ret<0)
    price_recovery=(eq_breadth>=.50)&price_stress.shift(1,fill_value=False).rolling(6).max().astype(bool)
    cap_warn=((shares/shares.shift(4)-1<=-.20)|(shares-shares.shift(4)<=-.05)).sum(axis=1)>=2
    ins_warn=(ins<0).sum(axis=1)>=2
    news_warn=pd.Series({d: sum((news[t].loc[d,"neg_days"]>=2) and (news[t].loc[d,"neg_days"]>news[t].loc[d,"pos_days"]) for t in THEMES)>=2 for d in months})
    confirmed=price_stress&(cap_warn|ins_warn|news_warn)
    confirmed_recovery=price_recovery&(eq_breadth>=.60)&~(cap_warn|ins_warn|news_warn)
    print("\n[1] Breadth panels, latest")
    print(f"  equity: {eq_breadth.iloc[-1]*100:.0f}% ({int(eq_breadth.iloc[-1]*len(eq.columns))}/{len(eq.columns)})")
    print(f"  themes: {theme_breadth.iloc[-1]*100:.0f}% ({theme_breadth.iloc[-1]:.0f} of {len(theme.columns)})")
    print(f"  cross-assets: {cross_breadth.iloc[-1]*100:.0f}%")
    print("\n[2] Signal counts")
    for n,x in [("price stress",price_stress),("capex",cap_warn),("insider",ins_warn),("news",news_warn),("confirmed",confirmed),("confirmed recovery",confirmed_recovery)]: print(f"  {n:>20}: {int(x.sum()):3d}")
    print("\n[3] Re-entry episodes")
    rec=list(price_recovery[price_recovery].index); print(f"  recoveries after prior stress: {len(rec)}")
    for d in rec:
        i=months.get_loc(d); future=spy_ret.iloc[i+1:i+7].dropna(); relapse=bool(price_stress.iloc[i+1:i+7].any())
        print(f"  {d.date()} breadth={eq_breadth.loc[d]*100:.0f}% capex={bool(cap_warn.loc[d])} insider={bool(ins_warn.loc[d])} news={bool(news_warn.loc[d])} fwd6m={future.mean()*100 if len(future) else np.nan:+.1f}% relapse6m={relapse}")
    print("\n[4] Warning lead/follow test")
    # episodes: first month of a 3-month price-stress run
    starts=[d for d in months if bool(price_stress.loc[d]) and not bool(price_stress.shift(1).fillna(False).loc[d])]
    for d in starts:
        prior={}
        for n,x in [("capex",cap_warn),("insider",ins_warn),("news",news_warn)]:
            p=[z for z in months if z<=d and bool(x.loc[z])]
            prior[n]=round((d-p[-1]).days/30.44,1) if p else None
        print(f"  stress {d.date()} breadth={eq_breadth.loc[d]*100:.0f}% leads(months): {prior}")
    print("\n[5] Channel ablation: false re-entry rate")
    # On recovery dates, block if selected confirmation; blocked recovery is measured as avoided relapse.
    for nm,sig in [("none",pd.Series(False,index=months)),("capex",cap_warn),("insider",ins_warn),("news",news_warn),("any",cap_warn|ins_warn|news_warn)]:
        allowed=[d for d in rec if not bool(sig.loc[d])]
        rel=sum(bool(price_stress.iloc[months.get_loc(d)+1:months.get_loc(d)+7].any()) for d in allowed)
        print(f"  {nm:>8}: allowed={len(allowed):2d} relapses={rel:2d} rate={(rel/len(allowed)*100 if allowed else np.nan):.1f}%")
    print("\n[6] Conclusion: diagnostic only; no allocation rule changed.")
    print("  Use equity breadth for market timing; capex/insider/news for context and")
    print("  recovery hysteresis only after larger, cleaner out-of-sample evidence.")

if __name__ == "__main__": main()

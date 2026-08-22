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
7. ADVISORY allocation references: absolute capex ratio and the fixed Cycle-1
   price+capex bounded-rotation reference ratio.
8. THEME CANDIDATES: cached non-panel series that recently entered their own
   uptrend with rising relative strength. Descriptive watchlist for manual
   review only; nothing is auto-added to the theme panels.

The panel split prevents normal theme rotation from being confused with broad
equity stress. Price breadth remains the timing signal. Capex/insider/news are
slow context and confirmation candidates; none triggers a sell or buy. The
allocation references are advisory ratios for a separately sized high-risk
thematic sleeve; they are not instructions for the core book.

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

# Candidate pool for the new-theme watchlist: series already cached in
# db_megatrend.h5 but NOT part of any panel. Category-diverse by construction
# (Phase-1 universe), so the watchlist cannot silently become a winners list.
CANDIDATE_POOL = [
    "ARKK", "COIN", "EEM", "GDX", "IBB", "ICLN", "KWEB", "LIT", "MSTR",
    "NVDA", "PTON", "PYPL", "QQQ", "SHOP", "SMH", "SPAK", "SPY", "TAN",
    "TSM", "URA", "VNQ", "XBI", "XLE", "XLU", "XYZ", "ZM",  # Phase-1 trend universe
    "EPP", "GXTG", "ILF", "ITA", "ITB", "IYT", "KWEB", "LIT", "PAVE",
    "PBW", "REMX", "WOOD", "XME", "VNQ", "AGG", "TLT", "GLD", "SHY",
    "EFA", "IWM", "IJR", "XL*",
]
PANEL_SERIES = set(
    EQUITY + CROSS_ASSET
    + [s for v in THEME_PROXIES.values() for s in v]
)
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
    M = pd.DataFrame({sym: ser.resample("ME").last() for sym, ser in px.items()})
    # Never label a partial current month as a completed month-end reading.
    # The watcher is explicitly month-end/manual; during the month, retain the
    # latest completed calendar month.
    latest_daily = max(ser.index.max() for ser in px.values() if len(ser))
    cutoff = latest_daily.to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)
    return M.loc[M.index <= cutoff]


# ---------------- RC-9 undecided-state detector (advisory) ----------------
# Pre-registered design + thresholds (Design.md §18 RC-9, 2026-08-23):
#   P  = Spearman autocorr of theme 3m relative returns vs SPY, lag 1m
#   B  = fraction of theme proxies above own MA10 with 6m relative momentum > 0
#   C  = avg pairwise 60d correlation of theme daily returns,
#        expanding percentile of own history (>=36m)
# Thresholds: P>=0.50, B>=50%, C>=60th pct. States (deterministic):
#   CONCENTRATED (P high) / CONCENTRATED (bloc) (P high, C high) /
#   UNDECIDED (B>=50%, P low, C high) / DISPERSAL (B low, P low, C high) /
#   DIFFERENTIATING (remainder).
RC9_THEMES = {
    "AI/hyperscale": ["SMH"],
    "clean_energy": ["ICLN", "TAN"],
    "crypto": ["MSTR", "COIN"],
    "biotech": ["XBI", "IBB"],
    "metals": ["GDX", "LIT"],
    "uranium": ["URA"],
}
RC9_P, RC9_B, RC9_CPCT = 0.50, 0.50, 60.0


def rc9_state(daily: pd.DataFrame) -> pd.Series:
    """Monthly {P,B,C_pct,state} using only data available at each month end."""
    d = daily.dropna(how="all", axis=1)
    me = d.groupby(d.index.to_period("M")).last()
    theme_cols = {}
    for theme, proxies in RC9_THEMES.items():
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
    syms = sorted({s_ for v in RC9_THEMES.values() for s_ in v})
    for m in me.index[6:]:
        flags = []
        for s_ in syms:
            if s_ not in me.columns:
                continue
            px, ma, r6 = me.loc[m, s_], ma10_me.loc[m, s_], rel6.loc[m, s_]
            if pd.isna(px) or pd.isna(ma) or pd.isna(r6):
                continue
            flags.append(bool(px > ma and r6 > 0))
        if len(flags) >= 4:
            B[m] = float(np.mean(flags))
    B = pd.Series(B)
    rets = np.log(d[syms]).diff()
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
        p_hi, b_hi = row["P"] >= RC9_P, row["B"] >= RC9_B
        c_hi = row["C_pct"] >= RC9_CPCT
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
    return Mx


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


def capex_share_history(months: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time TTM capex shares for each requested month-end."""
    raw = {}
    if not DB_CAPEX.exists():
        return pd.DataFrame(index=months, columns=list(THEME_MEMBERS), dtype=float)
    with pd.HDFStore(DB_CAPEX, mode="r") as s:
        keys = set(s.keys())
        for theme, members in THEME_MEMBERS.items():
            vals = []
            for sym in members:
                k = f"/capex_raw/{sym}"
                if k not in keys: continue
                d = s[k].copy()
                d["period_date"] = pd.to_datetime(d["period_date"], errors="coerce").dt.normalize()
                d["available_date"] = pd.to_datetime(d["available_date"], errors="coerce").dt.tz_localize(None)
                d["capex"] = pd.to_numeric(d["capex"], errors="coerce")
                company = []
                for asof in months:
                    visible = d[d.available_date <= asof].sort_values("available_date")
                    visible = visible.drop_duplicates("period_date", keep="last")
                    visible = visible[visible.period_date <= asof].sort_values("period_date")
                    company.append(float(visible.tail(4).capex.sum()) if len(visible) >= 4 else np.nan)
                vals.append(pd.Series(company, index=months, name=sym))
            raw[theme] = pd.concat(vals, axis=1).sum(axis=1, min_count=1) if vals else pd.Series(np.nan, index=months)
    X = pd.DataFrame(raw, index=months)
    return X.div(X.sum(axis=1), axis=0)


def latest_capex(asof: pd.Timestamp) -> dict:
    result = {}
    if not DB_CAPEX.exists(): return result
    history = capex_share_history(pd.DatetimeIndex([asof]))
    with pd.HDFStore(DB_CAPEX, mode="r") as s:
        keys = set(s.keys()); raw_by_theme = {}
        for theme, members in THEME_MEMBERS.items():
            vals = []
            for sym in members:
                k = f"/capex_raw/{sym}"
                if k not in keys: continue
                d = s[k].copy()
                d["period_date"] = pd.to_datetime(d["period_date"], errors="coerce").dt.normalize()
                d["available_date"] = pd.to_datetime(d["available_date"], errors="coerce").dt.tz_localize(None)
                d["capex"] = pd.to_numeric(d["capex"], errors="coerce")
                visible = d[d.available_date <= asof].sort_values("available_date").drop_duplicates("period_date", keep="last")
                visible = visible[visible.period_date <= asof].sort_values("period_date")
                if len(visible) >= 4: vals.append(float(visible.tail(4).capex.sum()))
            raw_by_theme[theme] = sum(vals) if vals else np.nan
    result["values_b"] = {t: v / 1e9 for t, v in raw_by_theme.items()}
    result["shares"] = history.iloc[0].to_dict() if not history.empty else {}
    return result


def project_simplex(x: np.ndarray, floor: float, cap: float) -> np.ndarray:
    """Project target onto sum=1 and floor<=weight<=cap."""
    x = np.asarray(x, dtype=float); lo, hi = -2.0, 2.0
    for _ in range(80):
        mid = (lo + hi) / 2; w = np.clip(x + mid, floor, cap)
        if w.sum() > 1: hi = mid
        else: lo = mid
    w = np.clip(x + (lo + hi) / 2, floor, cap)
    return w / w.sum()


def bounded_step(previous: np.ndarray, target: np.ndarray, step: float, floor: float, cap: float) -> np.ndarray:
    """Exact Cycle-1/D bounded transition, kept identical for reproducibility."""
    previous = np.asarray(previous, dtype=float); target = np.asarray(target, dtype=float)
    w = previous + np.clip(target - previous, -step, step)
    for _ in range(5):
        residual = 1 - w.sum()
        if abs(residual) < 1e-10: break
        if residual > 0:
            for i in np.argsort(-(target - w)):
                room = min(step - max(0, w[i] - previous[i]), cap - w[i])
                add = min(residual, max(0, room)); w[i] += add; residual -= add
                if residual <= 1e-10: break
        else:
            for i in np.argsort(target - w):
                room = min(step - max(0, previous[i] - w[i]), w[i] - floor)
                sub = min(-residual, max(0, room)); w[i] -= sub; residual += sub
                if residual >= -1e-10: break
    return w / w.sum()


def algorithmic_reference(M: pd.DataFrame, asof: pd.Timestamp) -> dict:
    """Fixed Cycle-1/D: 50% price + 50% capex, floor 10%, cap 70%, step 10%."""
    names = [t for t in THEME_MEMBERS if t in THEME_PROXIES]
    series = {}
    for theme in names:
        components = [M[s].pct_change().rename(s) for s in THEME_PROXIES[theme] if s in M.columns]
        series[theme] = pd.concat(components, axis=1).mean(axis=1, skipna=True) if components else pd.Series(dtype=float)
    R = pd.DataFrame(series).sort_index(); months = R.index[R.index <= asof]
    shares = capex_share_history(months); nav = (1 + R.fillna(0)).cumprod(); trailing = nav / nav.shift(12) - 1
    prev = np.ones(len(names)) / len(names); weights = []
    for d in months:
        scores = trailing.loc[d].dropna(); raw_price = np.ones(len(names)) / len(names)
        if len(scores) >= 2:
            order = list(scores.sort_values(ascending=False).index); mapping = {t: v for t, v in zip(order, [.60, .30, .10])}
            raw_price = np.array([mapping.get(t, 1 / len(names)) for t in names]); raw_price /= raw_price.sum()
        price_target = project_simplex(raw_price, .10, .70)
        raw_capex = shares.loc[d].reindex(names).fillna(0).to_numpy(dtype=float) if d in shares.index else np.ones(len(names))/len(names)
        raw_capex = raw_capex / raw_capex.sum() if raw_capex.sum() else np.ones(len(names))/len(names)
        raw_capex = np.clip(raw_capex, .10, .70); capex_target = raw_capex / raw_capex.sum()
        target = .5 * price_target + .5 * capex_target
        prev = bounded_step(prev, target, .10, .10, .70)
        weights.append(pd.Series(prev, index=names, name=d))
    W = pd.DataFrame(weights); recent = W.tail(24)
    return {"asof": str(asof.date()), "signal_month": str(W.index[-1].date()),
            "config": {"price_weight": .5, "capex_weight": .5, "floor": .10, "cap": .70, "monthly_step": .10},
            "current": W.iloc[-1].to_dict(), "trailing_24m_average": recent.mean().to_dict(),
            "trailing_24m_months": int(len(recent))}


def theme_candidates(M: pd.DataFrame, asof: pd.Timestamp) -> list[dict]:
    """Descriptive watchlist: cached non-panel series newly in their own uptrend.

    Flags a candidate when ALL hold at the latest completed month:
      1. >=10 months of price history (valid 10m mean),
      2. price above its own 10-month mean,
      3. 12-month relative return vs SPY is positive and improving vs prior month.
    This is a review list only: nothing is auto-added to any panel or ratio.
    """
    with pd.HDFStore(DB_MT, mode="r") as s:
        cached = {k.rsplit("/", 1)[-1] for k in s.keys()}
    spy = M["SPY"].dropna() if "SPY" in M.columns else None
    if spy is None:
        return []
    spy_rel_base = spy / spy.shift(12)
    out = []
    for sym in sorted(cached):
        if sym in PANEL_SERIES or sym not in M.columns:
            continue
        p = M[sym].dropna()
        if len(p) < 11:  # need >=10 months history + a prior month to compare
            continue
        ma = M[sym].rolling(10).mean()
        last, prior = asof, M.index[M.index < asof][-1]
        if pd.isna(ma.loc[last]) or pd.isna(M[sym].loc[last]):
            continue
        above = M[sym].loc[last] > ma.loc[last]
        rel = (M[sym] / M["SPY"]).dropna()
        rel12 = rel / rel.shift(12)
        if pd.isna(rel12.loc[last]) or pd.isna(rel12.loc[prior]):
            continue
        rising = rel12.loc[last] > rel12.loc[prior] and rel12.loc[last] > 1.0
        if above and rising:
            out.append({"symbol": sym,
                        "rel12": float(rel12.loc[last] - 1.0),
                        "rel12_change": float(rel12.loc[last] - rel12.loc[prior]),
                        "months_history": int(len(p))})
    out.sort(key=lambda x: -x["rel12"])
    return out


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
    capex = latest_capex(asof)
    advisory_ratio = algorithmic_reference(M, asof)
    candidates = theme_candidates(M, asof)
    insider = latest_insider(asof); news = latest_news(asof)
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
    print("\n[7] ADVISORY HIGH-RISK THEMATIC SLEEVE REFERENCES")
    print("  These ratios are references only; they do not size the core book or issue orders.")
    print("  Absolute TTM capex ratio:")
    for t, sh in capex.get("shares", {}).items(): print(f"    {t:>16}: {sh*100:5.1f}%")
    print("  Algorithmic current reference (fixed 50% price + 50% capex; floor 10%, cap 70%, step 10%):")
    for t, w in advisory_ratio.get("current", {}).items(): print(f"    {t:>16}: {w*100:5.1f}%")
    print(f"  Algorithmic trailing-{advisory_ratio.get('trailing_24m_months', 0)}-month average:")
    for t, w in advisory_ratio.get("trailing_24m_average", {}).items(): print(f"    {t:>16}: {w*100:5.1f}%")
    print("  Interpretation: use as a high-risk sleeve reference, not a 90%-core allocation.")
    print(f"\n[8] THEME CANDIDATES (review list only; not auto-added to any panel)")
    print("  Criteria: >=10m history, above own 10m mean, 12m relative strength vs SPY positive and rising.")
    if candidates:
        for c in candidates[:10]: print(f"    {c['symbol']:>6}: rel12 {c['rel12']*100:+6.1f}%  d_rel12 {c['rel12_change']*100:+5.1f}%  history {c['months_history']}m")
    else:
        print("    (none)")
    print("\n[9] Insider trailing 90d material net flow ($M; negative = selling):")
    for t, x in insider.items(): print(f"    {t:>16}: net={x['net_b']:+,.1f}  buy={x['buy_b']:,.1f} sell={x['sell_b']:,.1f} warning={x['warning']}")
    print("  News trailing 90d unique company-day diagnostic:")
    for t, x in news.items(): print(f"    {t:>16}: articles={x['articles']:4d} neg_days={x['neg_days']:3d} pos_days={x['pos_days']:3d} warning={x['warning']}")
    print("\n[10] INTERPRETATION (not an automatic trading instruction)")
    print("  Equity breadth is the timing panel. Theme breadth describes rotation.")
    print("  Cross-asset/capex/insider/news are context only; no automatic exit or buy.")
    print("  Historical calibration: equity q05=17%, q25=54%, median=77%, q75=89%.")
    print("\n[11] TRAILING 13-MONTH EQUITY BREADTH")
    for d, v in equity.dropna().tail(13).items(): print(f"  {d.date()} {v*100:3.0f}% {'#'*int(v*40)}")
    print("\n[13] RC-9 STATE DETECTOR (advisory; pre-registered thresholds P>=0.5 B>=50% C>=60th)")
    try:
        with pd.HDFStore(DB_MT, mode="r") as s:
            pxr = {k.split("/")[-1]: s[k].set_index("date")["adjClose"] for k in s.keys()}
        daily_all = pd.DataFrame({sym: ser.astype(float) for sym, ser in pxr.items()})
        daily_all.index = pd.to_datetime(daily_all.index)
        st = rc9_state(daily_all)
        last = st.iloc[-1]
        print(f"  as of {st.index[-1]}:  P={last['P']:.2f}  B={last['B']:.0%}  C-pct={last['C_pct']:.0f}")
        print(f"  STATE: {last['state']}")
        print("  postures: UNDECIDED=fractional+rotation | DIFFERENTIATING=winner emerging |")
        print("            CONCENTRATED=hold leader | DISPERSAL=de-risk advisory (context only)")
        for m, row in st.tail(6).iloc[:-1].iterrows():
            print(f"    {m}: P={row['P']:.2f} B={row['B']:.0%} C={row['C_pct']:.0f} -> {row['state']}")
        rc9 = {"P": round(float(last['P']), 3), "B": round(float(last['B']), 3),
               "C_pct": round(float(last['C_pct']), 1), "state": last['state']}
    except Exception as exc:
        print(f"  !! state detector failed: {exc}")
        rc9 = {"error": str(exc)[:80]}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(LOG.read_text()) if LOG.exists() else {}
    log[str(asof.date())] = {"equity": {"frac": round(eq_frac,3), "zone": zone(eq_frac), "active": sorted(eq_active)},
        "theme": {"frac": round(theme_frac,3), "active": sorted(theme_active)},
        "cross_asset": {"frac": round(cross_frac,3), "active": sorted(cross_active)},
        "capex": capex, "advisory_ratio": advisory_ratio, "theme_candidates": candidates,
        "insider": insider, "news": news, "rc9_state": rc9}
    LOG.write_text(json.dumps(log, indent=1, default=str))
    print(f"\n[12] logged -> {LOG.relative_to(ROOT.parent)}")

if __name__ == "__main__": main()

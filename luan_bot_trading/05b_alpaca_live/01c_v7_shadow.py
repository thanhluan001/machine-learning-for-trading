#!/usr/bin/env python3
"""01c — V7 combined-universe PAPER SHADOW (no orders, ever).

Runs as script 01's step [8] (or standalone). Pipeline:

  1. Load SP400 scored candidates from candidates.json (written by
     script 01 this run) -> is_sp400 = 1.
  2. Discover actionable SP600 events (FMP calendar, 2 weeks, current
     SP600 members only, EXCLUDING tickers already scored from the
     SP400 side).
  3. Freshness contract on the SP600 side: Tiingo price refresh with
     last-session hard-fail (IJR session anchor); incremental FMP
     grades append ([4.6] pattern); lazy earnings back-fill skip
     (earnings_full history is backward-looking; the calendar row
     carries the upcoming event).
  4. Features per event (phase-3 formulas, IJR benchmark + sector ETF;
     latest-bar anchor = same provisional-then-final doctrine as the
     SP400 path; grades cutoff < report_date per the 2026-09-03 fix).
  5. Score with frozen V7 (phase_g_v7_combined), min-gate >= 0.33,
     XLF excluded, ADV >= $10M on SP600 rows.
  6. Write v7_plan.json + v7_shadow_trades.json (V4-ledger format).

Non-fatal by contract: exceptions bubble to the caller's try/except;
the live V6/V4 pipeline never depends on this script.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
DB = ROOT / "01_data" / "db.h5"
V7_DIR = ROOT / "03_model" / "models" / "phase_g_v7_combined"
PLAN_OUT = HERE / "v7_plan.json"
LEDGER_OUT = HERE / "v7_shadow_trades.json"
CAND_JSON = HERE / "candidates.json"

V7_THRESHOLD = 0.33
ADV_MIN = 1e7
XLF = {"XLF"}
GICS_ETF = {
    "Energy": "XLE", "Materials": "XLB", "Industrials": "XLI",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
    "Health Care": "XLV", "Financials": "XLF", "Information Technology": "XLK",
    "Communication Services": "XLC", "Real Estate": "XLRE", "Utilities": "XLU",
}
GRADE_ORDINAL = {
    "strong sell": 1, "sell": 1, "underweight": 1, "underperform": 1,
    "reduce": 1, "reduce in price": 1,
    "negative": 2, "below average": 2, "below market": 2,
    "market underperform": 2, "market underperformer": 2,
    "hold": 3, "neutral": 3, "sector weight": 3, "sector perform": 3,
    "in-line": 3, "in-line market perform": 3, "sector weight (equal-weight)": 3,
    "market perform": 3, "equal-weight": 3, "equal weight": 3,
    "positive": 4, "above average": 4, "outperform": 4, "market outperform": 4,
    "add": 4, "accumulate": 4, "overweight": 4, "market overweight": 4,
    "sector outperform": 4, "strong buy": 5, "buy": 5,
}


def _env(key: str) -> str:
    v = os.environ.get(key, "")
    if not v:
        p = ROOT / ".env"
        if p.exists():
            for ln in p.read_text().splitlines():
                if ln.strip().startswith(key):
                    v = ln.split("=", 1)[1].strip().strip('"').strip("'")
    return v


TIINGO_KEY = _env("TIINGO_API_KEY")
FMP_KEY = _env("FMP_API_KEY")
FMP_BASE = "https://financialmodelingprep.com/stable"


def load_v7():
    import xgboost as xgb
    meta = json.load(open(V7_DIR / "meta.json", encoding="utf-8"))
    feats = meta["features"]
    models = {}
    for g in ("pass_g1", "pass_g2", "pass_g3"):
        m = xgb.Booster()
        m.load_model(str(V7_DIR / g / "classifier.json"))
        models[g] = m
    return feats, models


def current_sp600_members() -> pd.DataFrame:
    with pd.HDFStore(DB_SP600, "r") as s:
        md = s["/metadata/sp600"].copy()
        pm = s["/metadata/sp600_ptmap"].copy()
    rows = []
    for r in md.itertuples(index=False):
        try:
            ivs = json.loads(str(r.intervals).replace("'", '"').replace("None", "null"))
        except Exception:
            ivs = []
        if ivs and ivs[-1].get("removed") in (None, "null", ""):
            rows.append({"ticker": str(r.ticker), "gics_sector": r.gics_sector})
    cur = pd.DataFrame(rows)
    cur["sector_etf"] = cur.gics_sector.map(GICS_ETF)
    pm = pm.drop_duplicates("ticker")
    return cur.merge(pm[["ticker", "permaTicker"]], on="ticker", how="inner")


def fetch_calendar(weeks=2) -> pd.DataFrame:
    today = pd.Timestamp.now().normalize()
    start = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (today + timedelta(days=weeks * 7)).strftime("%Y-%m-%d")
    r = requests.get(f"{FMP_BASE}/earnings-calendar",
                     params={"from": start, "to": end, "includeReportTimes": "true",
                             "apikey": FMP_KEY}, timeout=30)
    r.raise_for_status()
    cal = pd.DataFrame(r.json())
    cal = cal[cal["time"].isin(["bmo", "amc"]) & cal["epsActual"].isna()].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    return cal


def refresh_prices(tickers: list[tuple[str, str]], ijr_last: pd.Timestamp):
    """Tiingo incremental refresh into db_sp600 /sp600/{pt}. Hard-fail if a
    scored ticker's last bar predates the IJR session."""
    lagged = []
    with pd.HDFStore(DB_SP600, "a") as s:
        for sym, pt in tickers:
            k = f"/sp600/{pt}"
            if k in s.keys():
                d = s[k]
                d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
                d = d.sort_values("Date")
            else:
                d = None
            last = d["Date"].iloc[-1] if d is not None and len(d) else None
            if last is not None and last >= ijr_last:
                continue
            url = f"https://api.tiingo.com/tiingo/daily/{sym}/prices"
            r = requests.get(url, params={"token": TIINGO_KEY,
                                          "startDate": (last or pd.Timestamp("2015-01-01")).strftime("%Y-%m-%d"),
                                          "format": "json"}, timeout=60)
            if r.status_code == 200 and r.json():
                nd = pd.DataFrame(r.json())
                nd["date"] = pd.to_datetime(nd["date"]).dt.tz_localize(None).dt.normalize()
                nd = nd.rename(columns={"date": "Date", "adjClose": "Adj_Close",
                                        "adjVolume": "Adj_Volume"})
                keep = [c for c in ["Date", "Close", "Adj_Close", "Volume", "Adj_Volume"]
                        if c in nd.columns]
                nd = nd[keep].sort_values("Date")
                if d is not None and len(d):
                    nd = pd.concat([d, nd[nd.Date > d.Date.iloc[-1]]], ignore_index=True)
                s.put(k, nd, format="table")
                last = nd.Date.iloc[-1]
            if last is None or last < ijr_last:
                lagged.append(sym)
    return lagged


def refresh_grades(tickers: list[str]):
    with pd.HDFStore(DB_SP600, "a") as s:
        for sym in tickers:
            k = f"/sp600/grades/{sym}"
            old = s[k] if k in s.keys() else pd.DataFrame(
                columns=["date", "grading_company", "previous_grade", "new_grade", "action"])
            if len(old):
                old_dates = pd.to_datetime(old["date"], errors="coerce")
                since = old_dates.max().strftime("%Y-%m-%d")
            else:
                since = "2015-01-01"
            r = requests.get(f"{FMP_BASE}/grades",
                             params={"symbol": sym, "apikey": FMP_KEY}, timeout=30)
            if r.status_code != 200 or not r.json():
                continue
            rows = [{"date": pd.to_datetime(x.get("date"), errors="coerce"),
                     "grading_company": x.get("gradingCompany"),
                     "previous_grade": x.get("previousGrade"),
                     "new_grade": x.get("newGrade"),
                     "action": x.get("action")} for x in r.json() if x.get("date")]
            if not rows:
                continue
            new = pd.DataFrame(rows)
            new["date"] = pd.to_datetime(new["date"])
            merged = pd.concat([old.assign(date=old_dates), new], ignore_index=True)
            merged = merged.drop_duplicates(
                subset=["date", "grading_company", "previous_grade", "new_grade", "action"])
            s.put(k, merged.sort_values("date"), format="table")


def grade_to_ordinal(g):
    if g is None or (isinstance(g, float) and np.isnan(g)):
        return None
    return GRADE_ORDINAL.get(str(g).strip().lower())


def revision_features(gdf, rd):
    nan8 = {k: np.nan for k in
            ["revision_momentum_30d", "revision_momentum_60d", "revision_momentum_90d",
             "revision_ordinal_momentum_90d", "revision_intensity_90d",
             "grade_dispersion_90d", "n_analysts_covering",
             "last_action_days_before_earnings"]}
    if gdf is None or len(gdf) == 0:
        return nan8
    g = gdf.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    pre = g[g["date"] < rd]
    if pre.empty:
        return {**{k: 0 for k in nan8 if k != "last_action_days_before_earnings"},
                "last_action_days_before_earnings": np.nan}
    pre["prev_o"] = pre["previous_grade"].apply(grade_to_ordinal)
    pre["new_o"] = pre["new_grade"].apply(grade_to_ordinal)
    pre["delta"] = pre.apply(lambda r: (r["new_o"] - r["prev_o"])
                             if (r["prev_o"] is not None and r["new_o"] is not None) else None, axis=1)
    w = {d: pre[pre["date"] >= rd - pd.Timedelta(days=d)] for d in (30, 60, 90)}
    net = lambda df: int((df.action == "upgrade").sum() - (df.action == "downgrade").sum()) if len(df) else 0
    inten = lambda df: int(((df.action == "upgrade") | (df.action == "downgrade")).sum()) if len(df) else 0
    disp = lambda df: int(df["new_o"].dropna().nunique()) if len(df) and df["new_o"].notna().any() else 0
    return {
        "revision_momentum_30d": net(w[30]),
        "revision_momentum_60d": net(w[60]),
        "revision_momentum_90d": net(w[90]),
        "revision_ordinal_momentum_90d": float(w[90]["delta"].dropna().sum()) if len(w[90]) else 0.0,
        "revision_intensity_90d": inten(w[90]),
        "grade_dispersion_90d": disp(w[90]),
        "n_analysts_covering": int(w[90]["grading_company"].nunique()) if len(w[90]) else 0,
        "last_action_days_before_earnings": float((rd - pre["date"].max()).days),
    }


def compute_sp600_features(sym, pt, etf, rdate, px, ev, gdf, bench_s, etf_px, macros):
    d = px
    dates = d["Date"].to_numpy()
    close = d["Adj_Close"].to_numpy(float)
    vol = d["Adj_Volume"].to_numpy(float)
    n = len(d)
    if n < 60:
        return None
    s_ = pd.Series(close, index=d["Date"])
    bench_al = bench_s.reindex(s_.index, method="ffill")
    slr = np.diff(np.log(np.where(close > 0, close, np.nan)))
    blr = np.diff(np.log(bench_al.to_numpy(float)))
    t = n  # latest-bar anchor (provisional until final evening — same doctrine)
    f = {}
    ev = ev.sort_values("report_date").reset_index(drop=True)
    ev = ev[ev.actual.notna() & ev.estimate.notna()].reset_index(drop=True)
    if ev.empty:
        return None
    diff = ev["actual"] - ev["estimate"]
    roll = diff.rolling(12, min_periods=12).std(ddof=1)
    sue = diff / roll
    beat = ev.actual > ev.estimate
    consec = 0
    for b in beat:
        consec = consec + 1 if b else 0
    f["sue_lag_1"] = float(sue.iloc[-1]) if pd.notna(sue.iloc[-1]) else np.nan
    f["sue_lag_2"] = float(sue.iloc[-2]) if len(sue) > 1 and pd.notna(sue.iloc[-2]) else np.nan
    f["consecutive_surprises_pre"] = consec
    # car_drift_q1: 60d CAR after the last COMPLETED quarter
    rd_last = pd.Timestamp(ev["report_date"].iloc[-1])
    tl = int(np.searchsorted(dates, np.datetime64(rd_last), side="left"))
    if tl + 61 < len(close):
        f["car_drift_historical_q1"] = float(np.nansum(slr[tl + 1:tl + 61] - blr[tl + 1:tl + 61]))
    else:
        f["car_drift_historical_q1"] = np.nan
    idio = slr[t - 20:t] - blr[t - 20:t]
    f["pre_event_idiosyncratic_vol"] = float(np.std(idio, ddof=1)) if len(idio) == 20 else np.nan
    lv = np.log(vol[t - 10:t])
    f["pre_event_volume_trend"] = (float(np.polyfit(np.arange(10, dtype=float), lv, 1)[0])
                                   if len(lv) == 10 and np.all(np.isfinite(lv)) else np.nan)
    for h in (3, 5, 10, 20, 30):
        lo = t - 1 - h
        f[f"rel_ret_{h}d"] = (np.log(close[t - 1] / close[lo])
                              - np.log(bench_al.iloc[t - 1] / bench_al.iloc[lo])) if lo >= 0 else np.nan
    if etf_px is not None and t - 21 >= 0:
        ea = pd.Series(etf_px["Adj_Close"].to_numpy(float),
                       index=pd.to_datetime(etf_px["Date"]).dt.tz_localize(None).dt.normalize())
        ea_al = ea.reindex(s_.index, method="ffill")
        f["sector_adjusted_ret_20d"] = (np.log(close[t - 1] / close[t - 21])
                                        - np.log(ea_al.iloc[t - 1] / ea_al.iloc[t - 21]))
    else:
        f["sector_adjusted_ret_20d"] = np.nan
    f["adv20"] = float(np.nanmean(close[t - 20:t] * vol[t - 20:t]))
    f.update(revision_features(gdf, rdate))
    f.update(macros)
    return f


def macro_snapshot(feature_date) -> dict:
    with pd.HDFStore(DB, "r") as s:
        def series(key, col):
            df = s[key]
            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
            return df.set_index("Date")[col].astype(float)
        vix = series("/macros/fred_vix_close", "vix_close")
        ff = series("/macros/fred_fed_funds_rate", "fed_funds_rate")
        un = series("/macros/fred_unemployment_rate", "unemployment_rate").sort_index()
    return {
        "vix": float(vix.asof(feature_date)),
        "fed_funds": float(ff.asof(feature_date)),
        "unemployment_roc21": float(un.asof(feature_date)) if pd.notna(un.asof(feature_date)) else np.nan,
    }


def entry_exit(rdate: pd.Timestamp, time_str: str):
    entry = rdate - pd.Timedelta(days=1) if time_str == "bmo" else rdate
    while entry.weekday() >= 5:
        entry = entry - pd.Timedelta(days=1)
    exit_date = rdate + pd.Timedelta(days=7)
    while exit_date.weekday() >= 5:
        exit_date = exit_date + pd.Timedelta(days=1)
    return entry.strftime("%Y-%m-%d"), exit_date.strftime("%Y-%m-%d")


def record_shadow(picks, generated_at):
    if LEDGER_OUT.exists():
        try:
            payload = json.load(open(LEDGER_OUT, encoding="utf-8"))
        except Exception:
            payload = {"records": []}
    else:
        payload = {"records": []}
    records = payload.get("records", [])
    by_key = {r.get("event_key"): r for r in records if r.get("event_key")}
    # calendar-shift dedupe: FMP revises report_date by a day sometimes —
    # a new pick matching permaTicker+time within +-4d UPDATES the record
    # under the new key instead of duplicating it.
    def _shifted_key(p):
        try:
            rd_new = pd.Timestamp(p.get("report_date"))
        except Exception:
            return None
        for k, r in by_key.items():
            if r.get("permaTicker") != p.get("permaTicker") or r.get("time") != p.get("time"):
                continue
            try:
                if abs((pd.Timestamp(r.get("report_date")) - rd_new).days) <= 4:
                    return k
            except Exception:
                continue
        return None
    for p in picks:
        key = "|".join(str(p.get(k, "")) for k in ("permaTicker", "report_date", "time"))
        sk = _shifted_key(p)
        if sk and sk != key:
            old = by_key.pop(sk)
            old.update({"event_key": key, "report_date": p.get("report_date"),
                        "entry_date": p.get("entry_date"), "exit_date": p.get("exit_date"),
                        "p_v7_min": p.get("p_v7_min"),
                        "p_v7_min_noflag": p.get("p_v7_min_noflag"),
                        "flag_decisive": p.get("flag_decisive", False),
                        "calendar_shifted_from": sk,
                        # entry not yet real under the new date — refill
                        "entry_price": None, "entry_fill_date": None,
                        "exit_price": None, "exit_fill_date": None,
                        "return_pct": None,
                        "outcome_status": "pending" if old.get("outcome_status") != "complete" else "complete"})
            by_key[key] = old
            continue
        old = by_key.get(key, {})
        rec = {**old, "event_key": key, "model": "phase_g_v7_combined",
               "hypothetical": True, "canonical_ticker": p["canonical_ticker"],
               "permaTicker": p["permaTicker"], "report_date": p["report_date"],
               "time": p["time"], "entry_date": p["entry_date"], "exit_date": p["exit_date"],
               "p_v7_min": p["p_v7_min"], "is_sp400": p["is_sp400"], "sector": p["sector"],
               "p_v7_min_noflag": p.get("p_v7_min_noflag"),
               "flag_decisive": p.get("flag_decisive", False),
               "features": p["features"], "last_seen_at": generated_at}
        rec.setdefault("first_seen_at", generated_at)
        rec.setdefault("outcome_status", "pending")
        rec.setdefault("entry_price", None)
        rec.setdefault("exit_price", None)
        rec.setdefault("return_pct", None)
        by_key[key] = rec
    payload["records"] = list(by_key.values())
    payload["model"] = "phase_g_v7_combined"
    payload["status"] = "hypothetical_comparison_only"
    payload["updated_at"] = generated_at
    with open(LEDGER_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def fill_outcomes():
    """V4-tracker pattern for V7: fill entry/exit prices from the freshest
    Tiingo closes in the local stores; complete records whose exit passed."""
    if not LEDGER_OUT.exists():
        return
    payload = json.load(open(LEDGER_OUT, encoding="utf-8"))
    recs = payload.get("records", [])

    def close_on(pt, day):
        """Last close <= day from /sp600/{pt} or db.h5 /sp400/{pt}."""
        for store, key in ((DB_SP600, f"/sp600/{pt}"), (DB, f"/sp400/{pt}")):
            try:
                with pd.HDFStore(store, "r") as s:
                    if key not in s.keys():
                        continue
                    d = s[key]
                dt = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
                sub = d[dt <= pd.Timestamp(day)]
                if len(sub):
                    return float(sub["Adj_Close"].iloc[-1]), str(dt[sub.index[-1]].date())
            except Exception:
                continue
        return None, None

    changed = 0
    for r in recs:
        if r.get("outcome_status") in ("dropped_pre_entry", "complete"):
            continue
        pt = r.get("permaTicker")
        if not pt:
            continue
        if r.get("entry_price") is None and r.get("entry_date"):
            p, d = close_on(pt, r["entry_date"])
            if p is not None and d == str(r["entry_date"]):   # only the actual entry-day close
                r["entry_price"], r["entry_fill_date"] = p, d
                changed += 1
        if r.get("entry_price") is not None and r.get("exit_price") is None \
                and r.get("exit_date"):
            p, d = close_on(pt, r["exit_date"])
            if p is not None and d >= str(r["exit_date"]):
                r["exit_price"], r["exit_fill_date"] = p, d
                r["return_pct"] = round(p / r["entry_price"] - 1.0, 6)
                r["outcome_status"] = "complete"
                changed += 1
    if changed:
        payload["records"] = recs
        with open(LEDGER_OUT, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        print(f"[V7 shadow] outcome-filler updated {changed} fields")


def main():
    print("[V7 shadow] start")
    feats, models = load_v7()
    try:
        cand = json.load(open(CAND_JSON, encoding="utf-8"))
        sp400_rows = cand.get("rows", [])
        cand_at = cand.get("generated_at", "?")
    except FileNotFoundError:
        print("[V7 shadow] WARN candidates.json missing (standalone run before any "
              "script-01 run?) — proceeding with EMPTY SP400 side")
        sp400_rows, cand_at = [], None
    sp400_pts = {r.get("permaTicker") for r in sp400_rows}
    sp400_syms = {r.get("canonical_ticker") for r in sp400_rows}
    print(f"[V7 shadow] SP400 candidates from script 01: {len(sp400_rows)}")

    cur = current_sp600_members()
    cal = fetch_calendar(2)
    m = cal[cal.symbol.isin(set(cur.ticker) - sp400_syms)].copy()
    m = m.sort_values("date")
    print(f"[V7 shadow] actionable SP600 events (2w): {len(m)} "
          f"({m.symbol.nunique()} tickers)")
    if m.empty:
        picks = []
    else:
        need = cur[cur.ticker.isin(m.symbol)]
        with pd.HDFStore(DB_SP600, "r") as s:
            ijr = s["/sp600/benchmark_IJR"].copy()
        ijr["Date"] = pd.to_datetime(ijr["Date"]).dt.tz_localize(None).dt.normalize()
        ijr_last = ijr.Date.iloc[-1]
        lagged = refresh_prices(list(zip(need.ticker, need.permaTicker)), ijr_last)
        if lagged:
            print(f"[V7 shadow] FRESHNESS HARD-FAIL: {lagged} have no {ijr_last.date()} bar; "
                  f"dropping those events (never scoring on stale closes)")
            m = m[~m.symbol.isin(lagged)]
        refresh_grades(sorted(set(m.symbol)))

        bench_s = pd.Series(ijr["Adj_Close"].to_numpy(float), index=ijr["Date"])
        etf_cache = {}
        with pd.HDFStore(DB_SP600, "r") as s:
            for etf in set(need.sector_etf.dropna()):
                k = f"/sp600/etf_{etf}"
                if k in s.keys():
                    d = s[k].copy()
                    d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
                    etf_cache[etf] = d.sort_values("Date")
        fdate = ijr_last
        macros = macro_snapshot(fdate)

        rows = []
        with pd.HDFStore(DB_SP600, "r") as s:
            for r in m.itertuples(index=False):
                sub = need[need.ticker == r.symbol]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                pk = f"/sp600/{row.permaTicker}"
                ek = f"/sp600/earnings_full/{r.symbol}"
                gk = f"/sp600/grades/{r.symbol}"
                if pk not in s.keys() or ek not in s.keys():
                    continue
                px = s[pk].copy()
                px["Date"] = pd.to_datetime(px["Date"]).dt.tz_localize(None).dt.normalize()
                px = px.sort_values("Date")
                ev = s[ek].copy()
                ev["report_date"] = pd.to_datetime(ev["report_date"])
                gdf = s[gk].copy() if gk in s.keys() else None
                f = compute_sp600_features(r.symbol, row.permaTicker, row.sector_etf,
                                           pd.Timestamp(r.date), px, ev, gdf,
                                           bench_s, etf_cache.get(row.sector_etf), macros)
                if f is None:
                    continue
                entry, exit_d = entry_exit(pd.Timestamp(r.date), r.time)
                rows.append({"permaTicker": row.permaTicker, "canonical_ticker": r.symbol,
                             "report_date": pd.Timestamp(r.date).strftime("%Y-%m-%d"),
                             "time": r.time, "entry_date": entry, "exit_date": exit_d,
                             "sector": row.sector_etf, "is_sp400": 0,
                             "features": {k: (float(v) if v is not None and pd.notna(v) else None)
                                          for k, v in f.items()}})
        print(f"[V7 shadow] SP600 features computed: {len(rows)}")

        # merge sides + score with V7
        # 2026-09-06 bugfix: is_sp400 must be tagged on the ROW, not read
        # from the features dict (script 01's features don't carry it;
        # missing -> 0.0 silently labeled every SP400 event as SP600).
        for r in sp400_rows:
            r["is_sp400"] = 1
        all_rows = sp400_rows + rows
        picks = []
        cand_summary = []   # Amendment B runway: both scores for every candidate
        import xgboost as xgb
        for r in all_rows:
            f = r.get("features", {})
            x = {k: (f.get(k) if f.get(k) is not None else 0.0) for k in feats}
            x["is_sp400"] = float(r.get("is_sp400", 0))
            X = xgb.DMatrix(pd.DataFrame([x])[feats])
            probs = {g: float(models[g].predict(X)[0]) for g in models}
            r["p_v7_min"] = round(min(probs.values()), 4)
            r["gates"] = {k: round(v, 4) for k, v in probs.items()}
            # counterfactual no-flag score (Amendment B: eligibility probe)
            x0 = dict(x)
            x0["is_sp400"] = 0.0
            X0 = xgb.DMatrix(pd.DataFrame([x0])[feats])
            probs0 = {g: float(models[g].predict(X0)[0]) for g in models}
            r["p_v7_min_noflag"] = round(min(probs0.values()), 4)
            r["flag_decisive"] = bool(r["is_sp400"] == 1
                                       and r["p_v7_min_noflag"] < V7_THRESHOLD
                                       and r["p_v7_min"] >= V7_THRESHOLD)
            cand_summary.append({"canonical_ticker": r["canonical_ticker"],
                                 "report_date": r["report_date"], "time": r["time"],
                                 "is_sp400": r.get("is_sp400", 1),
                                 "p_v7_min": r["p_v7_min"],
                                 "p_v7_min_noflag": r["p_v7_min_noflag"],
                                 "flag_decisive": r["flag_decisive"]})
            adv = f.get("adv20")
            adv_pass = True if r.get("is_sp400", 1) == 1 else (adv is not None and adv >= ADV_MIN)
            if (r["p_v7_min"] >= V7_THRESHOLD and r.get("sector") not in XLF and adv_pass):
                picks.append(r)
        picks.sort(key=lambda z: -z["p_v7_min"])

    generated_at = pd.Timestamp.now().isoformat()
    plan = {"model": "phase_g_v7_combined", "status": "paper_shadow_not_live",
            "generated_at": generated_at, "threshold": V7_THRESHOLD,
            "total_candidates": len(sp400_rows) + (len(rows) if m is not None and not m.empty else 0),
            "candidates": cand_summary if (m is not None and not m.empty) else [],
            "picks": picks}
    with open(PLAN_OUT, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False, default=str)
    record_shadow(picks, generated_at)
    fill_outcomes()
    print(f"[V7 shadow] picks: {[(p['canonical_ticker'], p['p_v7_min'], 'SP4' if p.get('is_sp400',1)==1 else 'SP6') for p in picks[:8]]}")
    print(f"[V7 shadow] wrote {PLAN_OUT.name} + {LEDGER_OUT.name}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()

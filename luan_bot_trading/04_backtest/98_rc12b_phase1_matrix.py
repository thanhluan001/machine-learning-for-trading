"""98_rc12b_phase1_matrix.py — RC-12b Phase 1: SP600 23-feature matrix.

Pre-registration: 04_backtest/rc12b_pre_registration.md (2026-08-31).
Adapts script 67 (SP500 transfer) machinery to the SP600 universe:

  --gather   FMP /stable/earnings (includeReportTimes) + /stable/grades
             for the 601 current members -> db_sp600.h5; copies sector
             ETF frames from db_sp500.h5; fetches IJR benchmark from Tiingo.
             Resumable (skips existing keys).
  --build    /features/train_matrix_sp600: the 23 DEPLOY_FEATURES with
             production formulas, benchmark leg = IJR, window 2019-2026
             (SPSM changes history starts 2019; 120d lead-in for lags),
             timing contract v4 (BMO Close[T-1], AMC Close[T], exit T+5),
             pregap_return outcome, adv20 column (ADV>=10M filter is
             applied at evaluation per Phase 0 binding note).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
load_dotenv(ROOT / ".env")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

DB_PROD = ROOT / "01_data" / "db.h5"
DB_SP500 = ROOT / "01_data" / "db_sp500.h5"
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
FMP_BASE = "https://financialmodelingprep.com/stable"
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
    "market perform": 3, "in-line": 3, "peer perform": 3,
    "equal-weight": 3, "equal weight": 3, "market weight": 3,
    "fair value": 3, "average": 3, "maintain": 3,
    "accumulate": 4, "overweight": 4, "outperform": 4, "market outperform": 4,
    "add": 4, "add position": 4, "positive": 4, "above average": 4,
    "above market": 4, "mild buy": 4, "long-term buy": 4, "long-term buy ": 4,
    "strong buy": 5, "buy": 5, "trading buy": 5, "market outperformer": 5,
}
WIN_START = pd.Timestamp("2019-01-01")
WIN_END = pd.Timestamp("2026-06-30")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _fmp_get(url, params):
    for _ in range(2):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                d = r.json()
                return d if isinstance(d, list) else None
        except Exception:
            pass
    return None


def grade_to_ordinal(g):
    if not isinstance(g, str):
        return None
    g = g.strip().lower()
    if g in GRADE_ORDINAL:
        return GRADE_ORDINAL[g]
    for k, v in GRADE_ORDINAL.items():
        if k in g:
            return v
    return None


def revision_features(gdf, rd):
    nan_r = {k: np.nan for k in [
        "revision_momentum_30d", "revision_momentum_60d", "revision_momentum_90d",
        "revision_ordinal_momentum_90d", "revision_intensity_90d",
        "grade_dispersion_90d", "n_analysts_covering", "last_action_days_before_earnings"]}
    if gdf is None or gdf.empty:
        return nan_r
    pre = gdf[gdf["date"] < rd].copy()
    if pre.empty:
        return {**{k: (np.nan if k == "last_action_days_before_earnings" else 0) for k in nan_r}}
    pre["prev_ordinal"] = pre["previous_grade"].apply(grade_to_ordinal)
    pre["new_ordinal"] = pre["new_grade"].apply(grade_to_ordinal)
    pre["ordinal_delta"] = pre.apply(
        lambda r: (r["new_ordinal"] - r["prev_ordinal"])
        if r["prev_ordinal"] is not None and r["new_ordinal"] is not None else None, axis=1)
    d30, d60, d90 = (rd - pd.Timedelta(days=d) for d in (30, 60, 90))
    w30, w60, w90 = pre[pre.date >= d30], pre[pre.date >= d60], pre[pre.date >= d90]

    def net(df):
        return 0 if df.empty else int((df.action == "upgrade").sum() - (df.action == "downgrade").sum())

    def ordinal_mom(df):
        d = df.ordinal_delta.dropna()
        return float(d.sum()) if len(d) else 0.0

    def intensity(df):
        return 0 if df.empty else int((df.action == "upgrade").sum() + (df.action == "downgrade").sum())

    def dispersion(df):
        o = df.new_ordinal.dropna()
        return int(o.nunique()) if len(o) else 0

    last = pre["date"].max()
    return {
        "revision_momentum_30d": net(w30), "revision_momentum_60d": net(w60),
        "revision_momentum_90d": net(w90),
        "revision_ordinal_momentum_90d": ordinal_mom(w90),
        "revision_intensity_90d": intensity(w90),
        "grade_dispersion_90d": dispersion(w90),
        "n_analysts_covering": int(w90.grading_company.nunique()) if not w90.empty else 0,
        "last_action_days_before_earnings": float((rd - last).days) if pd.notna(last) else np.nan,
    }


# ---------------------------------------------------------------- gather

def fetch_earnings_full(sym):
    d = _fmp_get(f"{FMP_BASE}/earnings", {"symbol": sym, "apikey": FMP_API_KEY,
                                          "includeReportTimes": "true"})
    if not d:
        return None
    df = pd.DataFrame(d)
    if "date" not in df.columns:
        return None
    out = pd.DataFrame({
        "report_date": pd.to_datetime(df["date"]).dt.normalize(),
        "actual": pd.to_numeric(df.get("epsActual"), errors="coerce"),
        "estimate": pd.to_numeric(df.get("epsEstimated"), errors="coerce"),
        "time": df.get("time"),
    })
    return out.sort_values("report_date").reset_index(drop=True)


def fetch_grades(sym):
    d = _fmp_get(f"{FMP_BASE}/grades", {"symbol": sym, "apikey": FMP_API_KEY})
    if not d:
        return None
    rows = []
    for r in d:
        if r.get("date") is None:
            continue
        rows.append({
            "date": pd.to_datetime(r["date"], errors="coerce"),
            "grading_company": r.get("gradingCompany"),
            "previous_grade": r.get("previousGrade"),
            "new_grade": r.get("newGrade"),
            "action": r.get("action"),
        })
    if not rows:
        return None
    return pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def fetch_tiingo_cols(ticker, cols=("Date", "Adj_Close", "Adj_Volume"), start="2014-01-01"):
    url = f"https://api.tiingo.com/tiingo/daily/{requests.utils.quote(ticker)}/prices"
    try:
        resp = requests.get(url, params={"token": TIINGO_API_KEY, "startDate": start}, timeout=60)
        if resp.status_code != 200:
            return None
        df = pd.DataFrame(resp.json())
        ren = {"date": "Date", "adjClose": "Adj_Close", "adjVolume": "Adj_Volume", "close": "Close"}
        df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
        if df.empty or not set(cols).issubset(df.columns):
            return None
        df = df[list(cols)].dropna()
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        return df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return None


def gather(limit: int | None = None):
    pm = pd.read_hdf(DB_SP600, "/metadata/sp600_ptmap")
    with pd.HDFStore(DB_SP600, "a") as store:
        keys = set(store.keys())
        # ETF frames from db_sp500
        with pd.HDFStore(DB_SP500, "r") as s5:
            for etf in GICS_ETF.values():
                k = f"/sp600/etf_{etf}"
                if k not in keys and f"/sp500/etf_{etf}" in s5.keys():
                    store.put(k, s5[f"/sp500/etf_{etf}"].copy(), format="table")
        if "/sp600/benchmark_IJR" not in keys:
            b = fetch_tiingo_cols("IJR")
            if b is not None:
                store.put("/sp600/benchmark_IJR", b, format="table")
                print("fetched IJR benchmark")
        keys = set(store.keys())
        todo = [(r.ticker, r.permaTicker) for r in pm.itertuples()
                if f"/sp600/earnings_full/{r.ticker}" not in keys]
        if limit:
            todo = todo[:limit]
        print(f"gathering earnings+grades for {len(todo)} tickers")
        t0 = time.time()
        for i, (sym, pt) in enumerate(todo, 1):
            e = fetch_earnings_full(sym)
            if e is not None and not e.empty:
                store.put(f"/sp600/earnings_full/{sym}", e, format="table")
            g = fetch_grades(sym)
            if g is not None and not g.empty:
                store.put(f"/sp600/grades/{sym}", g, format="table")
            if i % 50 == 0:
                print(f"  [{i}/{len(todo)}] {time.time()-t0:.0f}s")
            time.sleep(0.05)
        n_e = len([k for k in store.keys() if k.startswith("/sp600/earnings_full")])
        n_g = len([k for k in store.keys() if k.startswith("/sp600/grades")])
    print(f"earnings frames: {n_e} | grades frames: {n_g}")


# ---------------------------------------------------------------- build

def load_price(pt: str):
    with pd.HDFStore(DB_SP600, "r") as s6:
        k = f"/sp600/{pt}"
        if k in s6.keys():
            df = s6[k][["Date", "Adj_Close", "Adj_Volume"]].copy()
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
            return df
    with pd.HDFStore(DB_PROD, "r") as sp:
        k = f"/sp400/{pt}"
        if k in sp.keys():
            df = sp[k][["Date", "Adj_Close", "Adj_Volume"]].copy()
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
            return df
    return None


def build():
    pm = pd.read_hdf(DB_SP600, "/metadata/sp600_ptmap")
    md = pd.read_hdf(DB_SP600, "/metadata/sp600")
    md["ticker"] = md["ticker"].astype(str)
    sector_of = dict(zip(md.ticker, md.gics_sector.map(GICS_ETF)))

    with pd.HDFStore(DB_SP600, "r") as store:
        keys = set(store.keys())
        bench = store["/sp600/benchmark_IJR"].copy()
        etf_px = {}
        for etf in set(sector_of.values()):
            k = f"/sp600/etf_{etf}"
            if k in keys:
                d = store[k].copy()
                d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
                etf_px[etf] = d
    bench["Date"] = pd.to_datetime(bench["Date"]).dt.tz_localize(None).dt.normalize()
    bench_s = pd.Series(bench["Adj_Close"].to_numpy(float), index=bench["Date"])

    with pd.HDFStore(DB_SP600, "r") as store:
        ekeys = {k.split("/")[-1] for k in store.keys() if k.startswith("/sp600/earnings_full")}
        gkeys = {k.split("/")[-1] for k in store.keys() if k.startswith("/sp600/grades")}

    rows = []
    for i, r in enumerate(pm.itertuples(), 1):
        sym, pt = r.ticker, r.permaTicker
        if sym not in ekeys:
            continue
        px = load_price(pt)
        if px is None or px.empty:
            continue
        with pd.HDFStore(DB_SP600, "r") as store:
            ev = store[f"/sp600/earnings_full/{sym}"].copy()
            gdf = store[f"/sp600/grades/{sym}"].copy() if sym in gkeys else None
        dates = px["Date"].to_numpy()
        close = px["Adj_Close"].to_numpy(float)
        vol = px["Adj_Volume"].to_numpy(float)
        s = pd.Series(close, index=px["Date"])
        bench_al = bench_s.reindex(s.index, method="ffill")
        etf = sector_of.get(sym)
        etf_al = (pd.Series(etf_px[etf]["Adj_Close"].to_numpy(float),
                            index=etf_px[etf]["Date"]).reindex(s.index, method="ffill")
                  if etf in etf_px else None)
        slr = np.diff(np.log(np.where(close > 0, close, np.nan)))
        blr = np.diff(np.log(bench_al.to_numpy(float)))
        ev = ev.sort_values("report_date").reset_index(drop=True)
        ev = ev[ev.actual.notna() & ev.estimate.notna()]
        if ev.empty:
            continue
        diff = ev["actual"] - ev["estimate"]
        roll = diff.rolling(12, min_periods=12).std(ddof=1)
        sue = diff / roll
        beat = (ev.actual > ev.estimate)
        consec = np.zeros(len(ev), dtype=int)
        run = 0
        for k in range(len(ev)):
            run = run + 1 if beat.iloc[k] else 0
            consec[k] = run
        ev["sue_lag_1"] = sue.shift(1)
        ev["sue_lag_2"] = sue.shift(2)
        ev["consec_pre"] = pd.Series(consec).shift(1)
        car60 = np.full(len(ev), np.nan)
        tpos = np.searchsorted(dates, ev["report_date"].to_numpy().astype("datetime64[D]").astype(dates.dtype), side="left")
        for k, rd in enumerate(ev["report_date"]):
            t = int(tpos[k])
            if t + 61 < len(close):
                car60[k] = float(np.nansum(slr[t + 1:t + 61] - blr[t + 1:t + 61]))
        ev["car_drift_q1"] = pd.Series(car60).shift(1)
        for k, rd in enumerate(ev["report_date"]):
            if not (WIN_START - pd.Timedelta(days=120) <= rd <= WIN_END):
                continue
            t = int(tpos[k])
            if t < 31 or t + 6 >= len(close):
                continue
            f = {"ticker": sym, "permaTicker": pt, "report_date": rd, "sector": etf}
            f["sue_lag_1"] = ev["sue_lag_1"].iloc[k]
            f["sue_lag_2"] = ev["sue_lag_2"].iloc[k]
            f["consecutive_surprises_pre"] = ev["consec_pre"].iloc[k]
            f["car_drift_historical_q1"] = ev["car_drift_q1"].iloc[k]
            idio = slr[t - 20:t] - blr[t - 20:t]
            f["pre_event_idiosyncratic_vol"] = float(np.std(idio, ddof=1)) if len(idio) == 20 else np.nan
            lv = np.log(vol[t - 10:t])
            if len(lv) == 10 and np.all(np.isfinite(lv)):
                f["pre_event_volume_trend"] = float(np.polyfit(np.arange(10, dtype=float), lv, 1)[0])
            else:
                f["pre_event_volume_trend"] = np.nan
            for h in (3, 5, 10, 20, 30):
                lo = t - 1 - h
                f[f"rel_ret_{h}d"] = (np.log(close[t - 1] / close[lo])
                                      - np.log(bench_al.iloc[t - 1] / bench_al.iloc[lo])) if lo >= 0 else np.nan
            if etf_al is not None and t - 21 >= 0:
                f["sector_adjusted_ret_20d"] = (np.log(close[t - 1] / close[t - 21])
                                                - np.log(etf_al.iloc[t - 1] / etf_al.iloc[t - 21]))
            else:
                f["sector_adjusted_ret_20d"] = np.nan
            f["adv20"] = float(np.nanmean(close[t - 20:t] * vol[t - 20:t]))
            f.update(revision_features(gdf, rd))
            tm = (ev["time"].iloc[k] or "").lower() if isinstance(ev["time"].iloc[k], str) else ""
            entry_idx = exit_idx = None
            if tm == "bmo" and t >= 1:
                entry_idx, exit_idx = t - 1, t + 5
            elif tm == "amc":
                entry_idx, exit_idx = t, t + 5
            if entry_idx is not None and exit_idx < len(close):
                ep, xp = close[entry_idx], close[exit_idx]
                if ep > 0 and np.isfinite(ep) and np.isfinite(xp):
                    f["entry_date"] = pd.Timestamp(dates[entry_idx])
                    f["exit_date"] = pd.Timestamp(dates[exit_idx])
                    f["pregap_return"] = float(xp / ep - 1.0)
            rows.append(f)
        if i % 100 == 0:
            print(f"  [{i}/{len(pm)}] events so far: {len(rows):,}")

    d = pd.DataFrame(rows)
    print(f"\nraw events: {len(d):,}")
    d = d[d.pregap_return.notna()].copy()
    # macros from production db (universe-independent)
    with pd.HDFStore(DB_PROD, "r") as sp:
        def macro(key):
            df = sp[key]
            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
            return df
        def series(key, col):
            df = sp[key]
            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
            return df.set_index("Date")[col].astype(float)
        vix = series("/macros/fred_vix_close", "vix_close")
        ff = series("/macros/fred_fed_funds_rate", "fed_funds_rate")
        un = series("/macros/fred_unemployment_rate", "unemployment_rate")
    rd = pd.to_datetime(d.report_date)
    d["vix"] = vix.reindex(rd, method="ffill").values
    d["fed_funds"] = ff.reindex(rd, method="ffill").values
    un_s = un.sort_index()
    d["unemployment_roc21"] = (un_s.reindex(rd, method="ffill").pct_change(21).values)

    with pd.HDFStore(DB_SP600, "a") as store:
        if "/features/train_matrix_sp600" in store.keys():
            store.remove("/features/train_matrix_sp600")
        store.put("/features/train_matrix_sp600", d, format="table",
                  data_columns=["permaTicker", "report_date"])
    print(f"wrote /features/train_matrix_sp600: {len(d):,} events, "
          f"{d.permaTicker.nunique()} tickers, "
          f"{d.report_date.min().date()} .. {d.report_date.max().date()}")
    print(f"adv20 >= $10M share: {(d.adv20 >= 1e7).mean():.0%}")


if __name__ == "__main__":
    if "--gather" in sys.argv:
        gather()
    elif "--build" in sys.argv:
        build()
    else:
        print("usage: --gather | --build")

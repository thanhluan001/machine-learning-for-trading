#!/usr/bin/env python3
"""67_sp500_model_transfer.py — Apply the frozen V6 model (trained on S&P 400)
to the S&P 500 universe, 2023-07-01 -> 2026-06-30 (3 years), and compare
against the same policy on the S&P 400 universe.

RESEARCH QUESTION
-----------------
Does the S&P 400-trained V6 gate model transfer to the S&P 500 with better or
worse performance than on its home universe? (Follow-up to 66_sp500_pead_
comparison.py, which showed sp500 PEAD events are rarer: 8.5% vs 10.7%.)

METHOD
------
- Features: the 23 DEPLOY_FEATURES computed exactly per 02_build_feature_matrix
  formulas (Sunday-safe: all pre-event windows end at T-1 close):
    Block1 : sue_lag_1/2 (SUE = diff/rolling-12Q-std), consecutive_surprises_pre
             (strict-beat counter, lagged), car_drift_historical_q1 (previous
             event's car_60d_pass1, lagged)
    Block2 : pre_event_idiosyncratic_vol (std stock-vs-SPY ret, [T-20,T-1],
             ddof=1), pre_event_volume_trend (OLS slope log(vol), [T-10,T-1])
    Block3 : rel_ret_{3,5,10,20,30}d vs SPY, sector_adjusted_ret_20d vs GICS
             sector SPDR ETF
    Grades : 8 revision-momentum features from FMP /stable/grades (exact
             compute_revision_momentum formulas incl. ordinal map)
    Macros : vix, fed_funds, unemployment_roc21 (merge_asof from db.h5 macros)
- Model: frozen phase_g_v6_gate_decomposition gates; score = min(g1,g2,g3);
  threshold 0.33; XLF (Financials) excluded — same as live policy.
- Timing contract (v4 timing-correct, from 22_bmo_amc_pregap.py):
    BMO: entry Close[T-1], exit Close[T+5]   AMC: entry Close[T], exit Close[T+5]
- Execution sim: imported from 63_force_refresh_backtest.py — weekly top-4
  slate by score, force_refresh displacement with mh=4 guard, 10% stop on
  force-sells, weekly 1/4-slot NAV compounding.

CAVEATS (stated up front)
-------------------------
1. In-sample bias ASYMMETRY, favoring sp400: the frozen V6 gates were trained
   on sp400 data through 2025-H1 (dev folds). 2023-2025 sp400 results are
   partially in-sample; 2026-H1 is holdout. sp500 results are pure transfer
   (out-of-universe for every period).
2. sp500 membership = current constituents applied backward (survivorship,
   flatters sp500); sp400 uses historical intervals.
3. SUE/grades sources (FMP) differ from the sp400 matrix's (EODHD-era
   earnings, FMP grades) — feature distribution shift possible.

DATA
----
Extends 01_data/db_sp500.h5 (from script 66) with:
    /sp500/earnings_full/{SYM}  (report_date, actual, estimate, time)
    /sp500/grades/{SYM}         (date, grading_company, previous_grade, new_grade, action)
    /sp500/sectors              (symbol -> sector ETF)
    /sp500/etf_{SYM}            sector ETF prices (copied from db.h5 /macros or Tiingo)
Production db.h5 is read-only here.

USAGE
-----
    conda run -n trading python luan_bot_trading/04_backtest/67_sp500_model_transfer.py
"""
from __future__ import annotations
import importlib.util, io, json, os, sys, time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # luan_bot_trading/ (2 up from archive/edge_search_2026)
load_dotenv(ROOT / ".env")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

DB_PROD = ROOT / "01_data" / "db.h5"
DB_SP500 = ROOT / "01_data" / "db_sp500.h5"
MODEL_DIR = ROOT / "03_model" / "models" / "phase_g_v6_gate_decomposition"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FMP_BASE = "https://financialmodelingprep.com/stable"
BENCH = "SPY"
THRESH = 0.33
N_SLOTS = 4
MIN_HOLD = 4
WIN_START = pd.Timestamp("2023-07-01")
WIN_END = pd.Timestamp("2026-06-30")

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

# ---- import shared machinery from 51 (features) and 63 (simulator) --------
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

# ---- import shared machinery -------------------------------------------
# NOTE: load 63 FIRST (it pulls in 51 as its own 'bt' submodule). 51 replaces
# sys.stdout/stderr with fresh wrappers at import; exec-ing it twice closes
# the shared buffer ("I/O operation on closed file" crash).
sim63 = _load("sim63", HERE.parents[1] / "63_force_refresh_backtest.py")  # 04_backtest/ (1 up)
bt = sim63.bt
DEPLOY_FEATURES = bt.DEPLOY_FEATURES


def grade_to_ordinal(g):
    if g is None or not isinstance(g, str):
        return None
    g = g.strip().lower()
    if not g:
        return None
    if g in GRADE_ORDINAL:
        return GRADE_ORDINAL[g]
    for k, v in GRADE_ORDINAL.items():
        if k in g:
            return v
    return None


def revision_features(gdf: pd.DataFrame | None, rd: pd.Timestamp) -> dict:
    nan_r = {k: np.nan for k in [
        "revision_momentum_30d", "revision_momentum_60d", "revision_momentum_90d",
        "revision_ordinal_momentum_90d", "revision_intensity_90d",
        "grade_dispersion_90d", "n_analysts_covering",
        "last_action_days_before_earnings"]}
    if gdf is None or gdf.empty:
        return nan_r
    pre = gdf[gdf["date"] < rd].copy()
    if pre.empty:
        return {**{k: (np.nan if k == "last_action_days_before_earnings" else 0)
                   for k in nan_r}}
    pre["prev_ordinal"] = pre["previous_grade"].apply(grade_to_ordinal)
    pre["new_ordinal"] = pre["new_grade"].apply(grade_to_ordinal)
    pre["ordinal_delta"] = pre.apply(
        lambda r: (r["new_ordinal"] - r["prev_ordinal"])
        if r["prev_ordinal"] is not None and r["new_ordinal"] is not None else None, axis=1)
    d30, d60, d90 = (rd - pd.Timedelta(days=d) for d in (30, 60, 90))
    w30, w60, w90 = pre[pre.date >= d30], pre[pre.date >= d60], pre[pre.date >= d90]

    def net(df):
        if df.empty: return 0
        return int((df.action == "upgrade").sum() - (df.action == "downgrade").sum())
    def ordinal_mom(df):
        d = df.ordinal_delta.dropna()
        return float(d.sum()) if len(d) else 0.0
    def intensity(df):
        if df.empty: return 0
        return int((df.action == "upgrade").sum() + (df.action == "downgrade").sum())
    def dispersion(df):
        o = df.new_ordinal.dropna()
        return int(o.nunique()) if len(o) else 0
    last = pre["date"].max()
    return {
        "revision_momentum_30d": net(w30),
        "revision_momentum_60d": net(w60),
        "revision_momentum_90d": net(w90),
        "revision_ordinal_momentum_90d": ordinal_mom(w90),
        "revision_intensity_90d": intensity(w90),
        "grade_dispersion_90d": dispersion(w90),
        "n_analysts_covering": int(w90.grading_company.nunique()) if not w90.empty else 0,
        "last_action_days_before_earnings": float((rd - last).days) if pd.notna(last) else np.nan,
    }


# ============================== DATA GATHERING ==============================

def _fmp_get(url, params):
    for _ in range(2):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                d = r.json()
                if isinstance(d, list):
                    return d
                return None
        except Exception:
            time.sleep(2)
    return None


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


def fetch_tiingo_cols(ticker, cols=("Date", "Adj_Close", "Adj_Volume")):
    url = f"https://api.tiingo.com/tiingo/daily/{requests.utils.quote(ticker)}/prices"
    params = {"token": TIINGO_API_KEY, "startDate": "2014-01-01"}
    try:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code != 200:
            return None
        df = pd.DataFrame(resp.json())
        ren = {"date": "Date", "adjClose": "Adj_Close", "adjVolume": "Adj_Volume",
               "close": "Close"}
        df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
        if df.empty or not set(cols).issubset(df.columns):
            return None
        df = df[list(cols)].dropna()
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        return df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return None


def gather(symbols, sector_of):
    """Extend db_sp500.h5 with full earnings, grades, sectors, ETF prices."""
    with pd.HDFStore(DB_SP500, mode="a") as store:
        keys = set(store.keys())
        # sectors table
        if "/sp500/sectors" in keys:
            sec = store["/sp500/sectors"]
        else:
            sec = pd.DataFrame({"symbol": list(sector_of),
                                "sector": [sector_of[s] for s in sector_of]})
            store.put("/sp500/sectors", sec, format="table")
            print("    wrote /sp500/sectors")
        # sector ETF prices (from db.h5 /macros if present, else Tiingo)
        for etf in sorted(set(sector_of.values())):
            ek = f"/sp500/etf_{etf}"
            if ek in keys:
                continue
            df = None
            try:
                with pd.HDFStore(DB_PROD, mode="r") as ps:
                    if f"/macros/{etf}" in ps.keys():
                        p = ps[f"/macros/{etf}"]
                        df = pd.DataFrame({
                            "Date": pd.to_datetime(p["Date"]).dt.normalize(),
                            "Adj_Close": pd.to_numeric(p["Close"], errors="coerce")})
                        df = df.dropna()
            except Exception:
                df = None
            if df is None or df.empty:
                df = fetch_tiingo_cols(etf, cols=("Date", "Adj_Close"))
            if df is not None and not df.empty:
                store.put(ek, df, format="table")
                print(f"    wrote {ek} ({len(df)} rows)")
            else:
                print(f"    !! no data for sector ETF {etf}")
        keys = set(store.keys())
        n_e = n_g = 0
        for i, sym in enumerate(symbols, 1):
            if f"/sp500/earnings_full/{sym}" not in keys:
                e = fetch_earnings_full(sym)
                if e is not None:
                    store.put(f"/sp500/earnings_full/{sym}", e, format="table")
                    n_e += 1
                time.sleep(0.12)
            if f"/sp500/grades/{sym}" not in keys:
                g = fetch_grades(sym)
                if g is not None:
                    store.put(f"/sp500/grades/{sym}", g, format="table")
                    n_g += 1
                time.sleep(0.12)
            if i % 100 == 0:
                print(f"    [{i}/{len(symbols)}] new this run: {n_e} earnings, {n_g} grades")


# ============================ FEATURE COMPUTATION ===========================

def build_sp500_events(symbols, sector_of):
    """Compute the 23 deploy features + timing contract for every sp500 event."""
    with pd.HDFStore(DB_SP500, mode="r") as store:
        bench = store[f"/sp500/benchmark_{BENCH}"].copy()
        bench["Date"] = pd.to_datetime(bench["Date"]).dt.tz_localize(None).dt.normalize()
        etf_px = {}
        for etf in set(sector_of.values()):
            k = f"/sp500/etf_{etf}"
            if k in store.keys():
                d = store[k].copy()
                d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
                etf_px[etf] = d
    bench_s = pd.Series(bench["Adj_Close"].to_numpy(float), index=bench["Date"])
    rows = []
    with pd.HDFStore(DB_SP500, mode="r") as store:
        keys = set(store.keys())
        for i, sym in enumerate(symbols, 1):
            pk = f"/sp500/prices/{sym}"
            ek = f"/sp500/earnings_full/{sym}"
            gk = f"/sp500/grades/{sym}"
            if pk not in keys or ek not in keys:
                continue
            px = store[pk]; ev = store[ek]
            if px.empty or ev.empty:
                continue
            px = px.copy()
            px["Date"] = pd.to_datetime(px["Date"]).dt.tz_localize(None).dt.normalize()
            gdf = store[gk] if gk in keys else None
            etf = sector_of.get(sym)
            # aligned price arrays
            dates = px["Date"].to_numpy()
            close = px["Adj_Close"].to_numpy(float)
            vol = px["Adj_Volume"].to_numpy(float)
            s = pd.Series(close, index=px["Date"])
            bench_al = bench_s.reindex(s.index, method="ffill")
            etf_al = (pd.Series(etf_px[etf]["Adj_Close"].to_numpy(float),
                                index=etf_px[etf]["Date"]).reindex(s.index, method="ffill")
                      if etf in etf_px else None)
            slr = np.diff(np.log(close))
            blr = np.diff(np.log(bench_al.to_numpy(float)))
            # Block 1 (full earnings history)
            ev = ev.sort_values("report_date").reset_index(drop=True)
            diff = ev["actual"] - ev["estimate"]
            roll = diff.rolling(12, min_periods=12).std(ddof=1)
            sue = diff / roll
            beat = ((ev.actual.notna() & ev.estimate.notna()) & (ev.actual > ev.estimate))
            consec = np.zeros(len(ev), dtype=int)
            run = 0
            for k in range(len(ev)):
                run = run + 1 if beat.iloc[k] else 0
                consec[k] = run
            ev["sue_lag_1"] = sue.shift(1)
            ev["sue_lag_2"] = sue.shift(2)
            ev["consec_pre"] = pd.Series(consec).shift(1)
            # car_60d_pass1 for all events, then lag
            car60 = np.full(len(ev), np.nan)
            tpos = np.searchsorted(dates, ev["report_date"].to_numpy().astype("datetime64[D]").astype(dates.dtype), side="left")
            for k, rd in enumerate(ev["report_date"]):
                t = int(tpos[k])
                if t + 61 < len(close):
                    car60[k] = float(np.sum(slr[t + 1:t + 61] - blr[t + 1:t + 61]))
            ev["car_drift_q1"] = pd.Series(car60).shift(1)
            # per-event features for events in window
            for k, rd in enumerate(ev["report_date"]):
                if not (WIN_START - pd.Timedelta(days=120) <= rd <= WIN_END):
                    continue
                t = int(tpos[k])
                if t < 31 or t + 6 >= len(close):
                    continue
                f = {"ticker": sym, "report_date": rd, "sector": etf}
                f["sue_lag_1"] = ev["sue_lag_1"].iloc[k]
                f["sue_lag_2"] = ev["sue_lag_2"].iloc[k]
                f["consecutive_surprises_pre"] = ev["consec_pre"].iloc[k]
                f["car_drift_historical_q1"] = ev["car_drift_q1"].iloc[k]
                # Block 2 (windows end at T-1)
                idio = slr[t - 20:t] - blr[t - 20:t]
                f["pre_event_idiosyncratic_vol"] = float(np.std(idio, ddof=1)) if len(idio) == 20 else np.nan
                lv = np.log(vol[t - 10:t])
                if len(lv) == 10 and np.all(np.isfinite(lv)):
                    x = np.arange(len(lv), dtype=float)
                    f["pre_event_volume_trend"] = float(np.polyfit(x, lv, 1)[0])
                else:
                    f["pre_event_volume_trend"] = np.nan
                # Block 3 (levels at T-1)
                for h in (3, 5, 10, 20, 30):
                    lo = t - 1 - h
                    if lo < 0:
                        f[f"rel_ret_{h}d"] = np.nan
                    else:
                        f[f"rel_ret_{h}d"] = (np.log(close[t - 1] / close[lo])
                                             - np.log(bench_al.iloc[t - 1] / bench_al.iloc[lo]))
                if etf_al is not None and t - 21 >= 0:
                    f["sector_adjusted_ret_20d"] = (np.log(close[t - 1] / close[t - 21])
                                                    - np.log(etf_al.iloc[t - 1] / etf_al.iloc[t - 21]))
                else:
                    f["sector_adjusted_ret_20d"] = np.nan
                f.update(revision_features(gdf, rd))
                # timing contract
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
                        f["entry_idx"], f["exit_idx"] = entry_idx, exit_idx
                        f["pregap_return"] = float(xp / ep - 1.0)
                rows.append(f)
            if i % 100 == 0:
                print(f"    [{i}/{len(symbols)}] events in window so far: {len(rows):,}")
    return pd.DataFrame(rows)


def load_macros():
    frames = {}
    with pd.HDFStore(DB_PROD, mode="r") as s:
        for name, key in bt.MACRO_KEYS.items():
            if key not in s.keys():
                continue
            m = s[key].copy()
            m["Date"] = pd.to_datetime(m["Date"])
            m = m.sort_values("Date").rename(columns={"Date": "report_date"})
            cc = "Close" if "Close" in m.columns else m.columns[1]
            m = m[["report_date", cc]].rename(columns={cc: name})
            m[name] = pd.to_numeric(m[name], errors="coerce")
            m[f"{name}_roc21"] = m[name].pct_change(21)
            frames[name] = m.sort_values("report_date")
    return frames


def add_macros(df, frames):
    df = df.sort_values("report_date").copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    for name, m in frames.items():
        df = pd.merge_asof(df, m[["report_date", name, f"{name}_roc21"]],
                           on="report_date", direction="backward")
    return df


# ================================= SCORING ==================================

def score_v6(df):
    XF = df[DEPLOY_FEATURES]
    probs = {}
    for g in ("pass_g1", "pass_g2", "pass_g3"):
        clf = xgb.XGBClassifier()
        clf.load_model(str(MODEL_DIR / g / "classifier.json"))
        probs[g] = clf.predict_proba(XF)[:, 1]
    df = df.copy()
    df["score"] = np.minimum.reduce(list(probs.values()))
    return df


# =================================== RUN ====================================

def attach_entry_idx(cands, cache_populate=True):
    """63's _slot needs ev.entry_idx; compute from the price cache."""
    rows = []
    for _, r in cands.iterrows():
        pl = sim63._PCACHE.get(r.permaTicker)
        if pl is None:
            continue
        vdates, _ = pl
        eidx = int(np.searchsorted(vdates, np.datetime64(pd.Timestamp(r.entry_date)), side="left"))
        rows.append({**r.to_dict(), "entry_idx": eidx})
    return pd.DataFrame(rows)


def run_sim(cands, mode="force_refresh"):
    slate = sim63.weekly_slate(cands)
    trades = sim63.simulate(slate, mode, MIN_HOLD)
    return sim63.stats(trades), len(slate)


def main():
    print("=" * 78)
    print("S&P 500 MODEL TRANSFER — frozen V6 (sp400-trained), 3y window")
    print(f"{WIN_START.date()} -> {WIN_END.date()}   theta={THRESH} mh={MIN_HOLD} force_refresh")
    print("=" * 78)

    # sp500 symbols + sectors from Wikipedia
    print("[1] Wikipedia S&P 500 constituents ...")
    r = requests.get(WIKI_URL, headers={"User-Agent": "research-script"}, timeout=60)
    r.raise_for_status()
    tbl = next(t for t in pd.read_html(io.StringIO(r.text)) if "Symbol" in t.columns)
    tbl = tbl.dropna(subset=["Symbol"])
    syms = sorted({s.strip().replace(".", "-") for s in tbl["Symbol"].astype(str)})
    sector_of = {}
    for _, row in tbl.iterrows():
        s = str(row["Symbol"]).strip().replace(".", "-")
        g = row.get("GICS Sector")
        if isinstance(g, str):
            sector_of[s] = GICS_ETF.get(g.strip(), "SPY")
    print(f"    {len(syms)} symbols, {len(set(sector_of.values()))} sector ETFs")

    print("[2] Gathering sp500 data (earnings_full, grades, ETFs) ...")
    gather(syms, sector_of)

    print("[3] Computing sp500 features ...")
    sp5 = build_sp500_events(syms, sector_of)
    print(f"    {len(sp5):,} events with features")
    sp5 = add_macros(sp5, load_macros())
    sp5 = score_v6(sp5)
    # diagnostics: is the low pass-rate genuine transfer degradation or a bug?
    print("    sp500 min-gate score deciles:",
          np.round(np.nanpercentile(sp5.score, [10, 25, 50, 75, 90]), 3).tolist())
    nan_top = sp5[DEPLOY_FEATURES].isna().mean().sort_values(ascending=False)
    print("    worst-NaN features:", {k: round(v, 2) for k, v in nan_top.head(6).items()})

    # ---- sp500 candidates ----
    m5 = ((sp5.score >= THRESH) & (sp5.sector != "XLF") & sp5.pregap_return.notna()
          & (sp5.report_date >= WIN_START))
    c5 = sp5[m5 & (sp5.report_date <= WIN_END)].copy()
    c5 = c5.rename(columns={"ticker": "permaTicker"})
    print(f"    sp500 candidates (theta>= {THRESH}, non-XLF, timed): {len(c5):,}")

    # ---- sp400 side over identical window ----
    print("[4] S&P 400 side (frozen V6 scores, identical window) ...")
    sp4 = pd.read_hdf(DB_PROD, "/features/train_matrix_v4_timing_correct")
    sp4 = score_v6(sp4)
    m4 = ((sp4.score >= THRESH) & (~sp4.sector.isin(["XLF"])) & sp4.pregap_return.notna())
    rd4 = pd.to_datetime(sp4.report_date)
    c4 = sp4[m4 & (rd4 >= WIN_START) & (rd4 <= WIN_END)].copy()
    c4["entry_date"] = pd.to_datetime(c4.pregap_entry_date)
    c4["exit_date"] = pd.to_datetime(c4.pregap_exit_date)
    print(f"    sp400 candidates: {len(c4):,}")

    # ---- run sims (populate price cache per universe) ----
    print("[5] Simulating (weekly top-4, force_refresh mh=4) ...")
    sim63._PCACHE.clear()
    with pd.HDFStore(DB_PROD, mode="r") as s:
        keys = set(s.keys())
        for pt in c4.permaTicker.unique():
            sim63.get_prices(s, keys, pt)
    s4, slate4 = run_sim(attach_entry_idx(c4))
    c4h = c4[pd.to_datetime(c4.report_date) >= pd.Timestamp("2026-01-01")]
    s4h, _ = run_sim(attach_entry_idx(c4h))

    sim63._PCACHE.clear()
    with pd.HDFStore(DB_SP500, mode="r") as s:
        for _, rr in c5.iterrows():
            p = s[f"/sp500/prices/{rr.permaTicker}"]
            sim63._PCACHE[rr.permaTicker] = (pd.to_datetime(p["Date"]).values,
                                             p["Adj_Close"].to_numpy(float))
    c5a = attach_entry_idx(c5)
    s5, slate5 = run_sim(c5a)
    c5h = c5a[c5a.report_date >= pd.Timestamp("2026-01-01")]
    s5h, _ = run_sim(c5h)

    print("\n" + "=" * 78)
    print(f"RESULTS — frozen V6 transfer, {WIN_START.date()} -> {WIN_END.date()}")
    print("=" * 78)
    hdr = f"{'':24s}{'trades':>7}{'win%':>7}{'avgW%':>8}{'avgL%':>8}{'maxDD%':>8}{'NAV%':>9}{'forced':>7}"
    print(hdr)
    for nm, st in [("S&P 400 (home)", s4), ("S&P 500 (transfer)", s5)]:
        print(f"{nm:24s}{st['trades']:>7}{st['win_rate_pct']:>7.1f}{st['avg_win_pct']:>8.1f}"
              f"{st['avg_loss_pct']:>8.1f}{st['max_dd_pct']:>8.1f}{st['nav_pct']:>9.1f}"
              f"{st['force_sold']:>7}")
    print("\n--- 2026 H1 only (pure holdout for BOTH sides) ---")
    print(hdr)
    for nm, st in [("S&P 400 (home)", s4h), ("S&P 500 (transfer)", s5h)]:
        print(f"{nm:24s}{st['trades']:>7}{st['win_rate_pct']:>7.1f}{st['avg_win_pct']:>8.1f}"
              f"{st['avg_loss_pct']:>8.1f}{st['max_dd_pct']:>8.1f}{st['nav_pct']:>9.1f}"
              f"{st['force_sold']:>7}")
    print("\nCaveats: sp400 side is partially IN-SAMPLE (V6 dev folds cover 2023-2025);"
          "\nsp500 is pure out-of-universe transfer. sp500 membership has survivorship"
          "\nbias (current constituents applied backward) which flatters sp500.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
01_fetch_and_predict.py — v6 Default / v4 Comparison Inference (STATELESS)
====================================================================

PURPOSE
-------
Predict P(PEAD) for upcoming earnings and output the top 4 picks to
`plan.json`. This script is STATELESS: it does NOT manage slots, does
NOT track positions, does NOT talk to Alpaca. Script 02 reads plan.json
and handles execution.

FLOW
----
  1. Fetch FMP earnings calendar (next N weeks, FUTURE dates only)
  2. Filter to CURRENT S&P 400 members (via wikipedia_intervals, not stale isActive)
  3. For each concerned ticker: fetch Tiingo prices up to today (fill gaps)
  4. Compute 23 honest features per event (NO look-ahead — see audit below)
  5. Load frozen v4 comparison classifier and V6 gate classifiers
  6. Build V6 executable plan and separate V4 comparison plan
  7. Apply deployable filters:
     a. V6 min-gate score >= 0.33
     b. exclude XLF (Financials)
     c. LEG-like exclusion (neg SUE & no streak & oversold) — backtest money-loser
  7. Sort by P(PEAD) descending, output top 4 to plan.json

LOOK-AHEAD AUDIT (per feature — ALL verified safe)
===================================================
All 23 features use ONLY data available BEFORE the earnings event:

  sue_lag_1, sue_lag_2:
      Prior 1-2 quarters' SUE scores. Computed from HISTORICAL earnings
      results already in the DB. Current quarter EPS is NOT used.

  car_drift_historical_q1:
      60-day CAR following the PRIOR quarter's earnings (months ago).
      All 60 days of post-prior-quarter price data exist.

  consecutive_surprises_pre:
      Prior quarter's beat-streak count. Historical only.

  pre_event_idiosyncratic_vol, pre_event_volume_trend:
      20-day/10-day stats through the executable daily cutoff:
      T-1 for AMC and T-2 for BMO.

  rel_ret_3d/5d/10d/20d/30d, sector_adjusted_ret_20d:
      Returns through the same T-1/T-2 executable daily cutoff.

  revision_momentum_* (8 features):
      Analyst grades strictly before the same executable cutoff.

  unemployment_roc21, fed_funds, vix:
      Last macro observations available at the same executable cutoff.

DROPPED (look-ahead — NOT computed):
  sue_score, eps_surprise_pct, consecutive_surprises,
  sue_acceleration, sue_abs_x_inverse_vol
  These require the current earnings result — NOT available at entry time.

USAGE
-----
    python 05b_alpaca_live/01_fetch_and_predict.py
    python 05b_alpaca_live/01_fetch_and_predict.py --weeks 3
    python 05b_alpaca_live/01_fetch_and_predict.py --dry-run       # no writes (no fetch, no plan.json)

NOTE: This script ALWAYS refreshes Tiingo prices (the whole point is fresh
data). There is intentionally NO --skip-tiingo or --due-soon option — those
defeated the purpose by using stale data and narrowing the calendar window.
Use --weeks to widen/narrow the lookahead (default 2).

OUTPUT: plan.json (see README.md for schema)
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from dotenv import load_dotenv

# ==============================================================================
# CONFIGURATION
# ==============================================================================
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DB_FILE = PROJECT_ROOT / "01_data" / "db.h5"

FROZEN_CLASSIFIER = PROJECT_ROOT / "03_model" / "models" / "phase_g_v4_timing_correct" / "classifier.json"
FROZEN_META = PROJECT_ROOT / "03_model" / "models" / "phase_g_v4_timing_correct" / "meta.json"
V6_MODEL_DIR = PROJECT_ROOT / "03_model" / "models" / "phase_g_v6_gate_decomposition"
PLAN_JSON = HERE / "plan.json"                 # V6: executed by Script 02
V4_PLAN_JSON = HERE / "v4_plan.json"           # V4: comparison only
V4_SHADOW_TRADES_JSON = HERE / "v4_shadow_trades.json"

# Deployable operating point
THETA = 0.20
MAX_PICKS = 8  # Output all theta-passers as a ranked bench for the slot manager
EXCLUDE_SECTORS = ["XLF"]
# Min-gate threshold. Raised 0.30 -> 0.33 on 2026-08-13 after bootstrap
# validation (04_backtest/61_v6_threshold_bootstrap.py): at 0.30 the DEV OOS
# per-trade avg-return CI included zero (edge not statistically reliable);
# at 0.33 it excludes zero (CI [0.56, 5.22]). Win rate 57->59%, avg trade
# 1.46->2.84%, holds on 2026 H1 holdout (70% win). Border-band negativity
# was NOT confirmed by the bootstrap (point estimate only).
V6_THETA = 0.33
V6_MAX_PICKS = 8
V6_GATES = ["pass_g1", "pass_g2", "pass_g3"]

# 23 honest features (must match 05_freeze_honest_model.py)
DEPLOY_FEATURES = [
    "sue_lag_1", "sue_lag_2", "car_drift_historical_q1",
    "pre_event_idiosyncratic_vol", "pre_event_volume_trend",
    "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d", "rel_ret_30d",
    "sector_adjusted_ret_20d",
    "revision_momentum_30d", "revision_momentum_60d", "revision_momentum_90d",
    "revision_ordinal_momentum_90d", "revision_intensity_90d",
    "grade_dispersion_90d", "n_analysts_covering", "last_action_days_before_earnings",
    "consecutive_surprises_pre",
    "unemployment_roc21", "fed_funds", "vix",
]

MACRO_KEYS = {
    "vix": "/macros/fred_vix_close",
    "fed_funds": "/macros/fred_fed_funds_rate",
    "unemployment": "/macros/fred_unemployment_rate",
}

# FRED series IDs for the 3 macro features. Maps the DB column name (the
# /macros/fred_{name} suffix) -> FRED series_id. Used by refresh_macros().
#   vix_close        <- VIXCLS  (daily, ~1-day lag)
#   fed_funds_rate   <- DFF     (daily, ~1-day lag)
#   unemployment_rate<- UNRATE  (monthly, released mid-month for prior month)
FRED_SERIES = {
    "vix_close": "VIXCLS",
    "fed_funds_rate": "DFF",
    "unemployment_rate": "UNRATE",
}
FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE = "https://api.stlouisfed.org/fred"

FMP_BASE = "https://financialmodelingprep.com/stable"
TIINGO_TIMEOUT = 60
FMP_TIMEOUT = 30

PERMATICKERS_KEY = "/metadata/sp400_permatickers"
EARNINGS_KEY = "/earnings/fmp"
SP400_GROUP = "sp400"

TIINGO_OUTPUT_COLS = [
    "Date", "Open", "High", "Low", "Close", "Volume",
    "Adj_Open", "Adj_High", "Adj_Low", "Adj_Close", "Adj_Volume",
]

# ==============================================================================
# IMPORT STAGE 2 HELPERS (reuse exact feature computation)
# ==============================================================================
_s2_path = PROJECT_ROOT / "02_features" / "02_build_feature_matrix.py"
_s2_spec = importlib.util.spec_from_file_location("stage2", _s2_path)
s2 = importlib.util.module_from_spec(_s2_spec)
_s2_spec.loader.exec_module(s2)


# ==============================================================================
# PART 1: FMP EARNINGS CALENDAR FETCH
# ==============================================================================
def fetch_earnings_calendar(weeks_ahead: int = 2) -> pd.DataFrame:
    """Fetch FMP earnings calendar for the next N weeks.

    Uses /stable/earnings-calendar?from=..&to=..&includeReportTimes=true.
    Returns FUTURE dates only (epsActual is None for unreported events).
    """
    today = pd.Timestamp.now().normalize()
    # Start at TODAY-1, not today. FMP's bulk calendar silently DROPS an event
    # when the query 'from' date equals/after that event's earnings date (e.g.
    # ROKU 2026-08-06 is omitted from from=2026-08-06 but present in
    # from=2026-08-05). Starting at today-1 guarantees today's events are always
    # captured, AND returns the fuller 4000-record set. Already-reported events
    # (yesterday) are dropped below by the epsActual.isna() filter.
    start = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (today + timedelta(days=weeks_ahead * 7)).strftime("%Y-%m-%d")

    url = f"{FMP_BASE}/earnings-calendar"
    params = {"from": start, "to": end, "includeReportTimes": "true", "apikey": FMP_API_KEY}
    r = requests.get(url, params=params, timeout=FMP_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"FMP calendar {r.status_code}: {r.text[:200]}")

    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"FMP calendar unexpected response: {str(data)[:200]}")

    cal = pd.DataFrame(data)
    print(f"  Calendar {start}..{end} ({weeks_ahead} weeks): {len(cal)} total records")

    # Filter to events with BMO/AMC timing
    cal = cal[cal["time"].isin(["bmo", "amc"])].copy()
    print(f"  With BMO/AMC timing: {len(cal)}")

    # Keep only future events (no epsActual yet)
    cal = cal[cal["epsActual"].isna()].copy()
    print(f"  Future (epsActual=None): {len(cal)}")

    cal["date"] = pd.to_datetime(cal["date"])
    return cal


def _get_current_sp400_members(meta: pd.DataFrame) -> pd.DataFrame:
    """Filter metadata to only CURRENT S&P 400 members.

    The isActive flag is stale (294 tickers graduated to S&P 500 but still
    flagged active). wikipedia_intervals is the source of truth.
    A company is current if its latest interval has no 'removed' date.
    """
    is_current = []
    for _, row in meta.iterrows():
        intervals_raw = row.get("wikipedia_intervals", "[]")
        try:
            intervals = json.loads(intervals_raw) if isinstance(intervals_raw, str) else (intervals_raw or [])
        except (json.JSONDecodeError, TypeError):
            intervals = []
        if not intervals:
            is_current.append(True)  # no intervals = keep (defensive)
            continue
        last = intervals[-1]
        removed = last.get("removed")
        is_current.append(removed is None or (isinstance(removed, str) and removed.strip() == ""))
    return meta[pd.Series(is_current, index=meta.index)]


def filter_to_sp400(cal: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Filter calendar to CURRENT S&P 400 members only.

    Uses wikipedia_intervals (not the stale isActive flag) to determine
    current membership. This excludes graduated stocks like AMD, ETSY, ENPH
    that moved to the S&P 500.
    """
    current_meta = _get_current_sp400_members(meta)
    our_canonical = set(current_meta["canonical_ticker"].dropna().unique())
    cal_sp400 = cal[cal["symbol"].isin(our_canonical)].copy()
    print(f"  SP400 current members: {len(our_canonical)} tickers "
          f"({len(meta) - len(current_meta)} excluded as graduated/delisted)")
    print(f"  SP400 overlap: {len(cal_sp400)} events ({cal_sp400['symbol'].nunique()} tickers)")
    return cal_sp400


# ==============================================================================
# PART 2: TIINGO INCREMENTAL FETCH (concerned tickers only)
# ==============================================================================
def fetch_tiingo_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch Tiingo daily prices for one ticker."""
    url = f"https://api.tiingo.com/tiingo/daily/{requests.utils.quote(ticker)}/prices"
    params = {"token": TIINGO_API_KEY, "startDate": start, "endDate": end}
    try:
        resp = requests.get(url, params=params, timeout=TIINGO_TIMEOUT)
    except Exception:
        return pd.DataFrame()
    if resp.status_code != 200:
        return pd.DataFrame()
    try:
        data = resp.json()
    except Exception:
        return pd.DataFrame()
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    required = ["date", "open", "high", "low", "close", "volume",
                "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    out["Open"] = df["open"].astype(float)
    out["High"] = df["high"].astype(float)
    out["Low"] = df["low"].astype(float)
    out["Close"] = df["close"].astype(float)
    out["Volume"] = df["volume"].astype(float)
    out["Adj_Open"] = df["adjOpen"].astype(float)
    out["Adj_High"] = df["adjHigh"].astype(float)
    out["Adj_Low"] = df["adjLow"].astype(float)
    out["Adj_Close"] = df["adjClose"].astype(float)
    out["Adj_Volume"] = df["adjVolume"].astype(float)
    out = out[TIINGO_OUTPUT_COLS]
    out = out.dropna(subset=["Adj_Close", "Adj_Volume"]).reset_index(drop=True)
    out = out.sort_values("Date", kind="mergesort").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    return out


def refresh_tiingo_for_ticker(permaTicker: str, dry_run: bool = False) -> bool:
    """Fetch incremental Tiingo prices for one permaTicker. Write back to DB.
    Returns True if new data was fetched (or would be in dry-run)."""
    h5_path = f"/{SP400_GROUP}/{permaTicker}"

    with pd.HDFStore(DB_FILE, mode="r") as store:
        if h5_path in store.keys():
            existing = store[h5_path]
            latest = pd.Timestamp(existing["Date"].max()) if not existing.empty else None
        else:
            latest = None

    # End-date: fetch through TODAY's close when the market is already closed
    # (after-hours / weekend) — today's bar is final and is the T-1 price we
    # need. During live market hours today's bar is incomplete, so use yesterday.
    now_utc = datetime.utcnow()
    # ET = UTC-4 (summer DST). Fixed -4 is fine for the after-hours refresh use-case.
    et_hour = (now_utc.hour - 4) % 24
    is_weekday = datetime.now().weekday() < 5
    market_closed = (not is_weekday) or et_hour >= 16 or et_hour < 9
    end = datetime.now() if market_closed else (datetime.now() - timedelta(days=1))
    end_str = end.strftime("%Y-%m-%d")

    if latest is not None and latest >= pd.Timestamp(end_str):
        return False  # already up to date

    if latest is not None:
        start = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start = (datetime.now() - timedelta(days=15 * 365)).strftime("%Y-%m-%d")

    if dry_run:
        print(f"    [DRY] Would fetch {permaTicker} {start}..{end_str} "
              f"({'closed' if market_closed else 'open'} market)")
        return True

    new_data = fetch_tiingo_prices(permaTicker, start, end_str)
    if new_data.empty:
        return False

    # Append to DB
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if h5_path in store.keys():
            existing = store[h5_path]
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.sort_values("Date", kind="mergesort")
            combined = combined.drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
            store.remove(h5_path)
            store.put(h5_path, combined, format="table", data_columns=["Date"])
        else:
            store.put(h5_path, new_data, format="table", data_columns=["Date"])

    return True


def refresh_concerned_tickers(concerned_pts: list[str], dry_run: bool = False) -> dict:
    """Refresh Tiingo prices for all concerned tickers."""
    if dry_run:
        print(f"  [DRY] Would refresh {len(concerned_pts)} tickers")
        return {"refreshed": 0, "skipped": len(concerned_pts), "failed": 0}

    stats = {"refreshed": 0, "skipped": 0, "failed": 0}
    t0 = time.time()
    for i, pt in enumerate(concerned_pts):
        try:
            did = refresh_tiingo_for_ticker(pt)
            if did:
                stats["refreshed"] += 1
            else:
                stats["skipped"] += 1
        except Exception as e:
            stats["failed"] += 1
            print(f"    [ERROR] {pt}: {e}")
        if (i + 1) % 10 == 0 or (i + 1) == len(concerned_pts):
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(concerned_pts)}] refreshed={stats['refreshed']} "
                  f"skipped={stats['skipped']} failed={stats['failed']} ({elapsed:.1f}s)")
    return stats


def refresh_benchmark(ticker: str, dry_run: bool = False) -> bool:
    """Fetch incremental Tiingo prices for ONE benchmark ETF (IJH/IJJ/etc).

    Stores under /macros/{ticker} with the adjusted-column schema used by
    04_index_data_gathering.py (Date, Open=adjOpen, ..., Close=adjClose,
    Volume=adjVolume). Stale benchmarks corrupt every rel_ret / car_drift
    feature, so this MUST run before feature computation.

    Returns True if new data was fetched (or would be in dry-run).
    """
    h5_path = f"/macros/{ticker}"

    with pd.HDFStore(DB_FILE, mode="r") as store:
        if h5_path in store.keys():
            existing = store[h5_path]
            latest = pd.Timestamp(existing["Date"].max()) if not existing.empty else None
        else:
            latest = None

    # End-date: same market-closed logic as refresh_tiingo_for_ticker.
    now_utc = datetime.utcnow()
    et_hour = (now_utc.hour - 4) % 24
    is_weekday = datetime.now().weekday() < 5
    market_closed = (not is_weekday) or et_hour >= 16 or et_hour < 9
    end = datetime.now() if market_closed else (datetime.now() - timedelta(days=1))
    end_str = end.strftime("%Y-%m-%d")

    if latest is not None and latest >= pd.Timestamp(end_str):
        return False  # already up to date

    if latest is not None:
        start = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start = (datetime.now() - timedelta(days=15 * 365)).strftime("%Y-%m-%d")

    if dry_run:
        print(f"    [DRY] Would fetch benchmark {ticker} {start}..{end_str}")
        return True

    # Reuse the stock fetcher (full 11-col schema), then map to the
    # adjusted-only /macros schema used by 04_index_data_gathering.
    raw = fetch_tiingo_prices(ticker, start, end_str)
    if raw.empty:
        return False

    bench = pd.DataFrame({
        "Date": raw["Date"],
        "Open": raw["Adj_Open"],
        "High": raw["Adj_High"],
        "Low": raw["Adj_Low"],
        "Close": raw["Adj_Close"],
        "Volume": raw["Adj_Volume"],
    })

    with pd.HDFStore(DB_FILE, mode="a") as store:
        if h5_path in store.keys():
            existing = store[h5_path]
            combined = pd.concat([existing, bench], ignore_index=True)
            combined = combined.sort_values("Date", kind="mergesort")
            combined = combined.drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
            store.remove(h5_path)
            store.put(h5_path, combined, format="table", data_columns=["Date"])
        else:
            store.put(h5_path, bench, format="table", data_columns=["Date"])
    return True


def refresh_benchmarks(tickers: set[str], dry_run: bool = False) -> dict:
    """Refresh Tiingo prices for the benchmark ETFs needed for inference.

    Benchmarks (IJH + sector ETFs) are few and fetch in seconds, but stale
    benchmarks silently corrupt every rel_ret / car_drift feature. Always
    refresh before loading.
    """
    ordered = sorted(tickers | {"IJH"})
    if dry_run:
        print(f"  [DRY] Would refresh {len(ordered)} benchmarks: {ordered}")
        return {"refreshed": 0, "skipped": len(ordered), "failed": 0}
    stats = {"refreshed": 0, "skipped": 0, "failed": 0}
    for tk in ordered:
        try:
            if refresh_benchmark(tk):
                stats["refreshed"] += 1
            else:
                stats["skipped"] += 1
        except Exception as e:
            stats["failed"] += 1
            print(f"    [ERROR] benchmark {tk}: {e}")
    return stats


# ----------------------------------------------------------------------
# FRED macro refresh (vix, fed_funds, unemployment)
# ----------------------------------------------------------------------
def fetch_fred_series(series_id: str, start: str, end: str) -> pd.DataFrame:
    """Fetch FRED series observations via REST API. Returns DataFrame[Date, value].

    FRED uses "." for missing values (holidays / unreleased days); those are
    dropped. Numeric values are coerced to float.
    """
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    try:
        resp = requests.get(f"{FRED_BASE}/series/observations",
                            params=params, timeout=30)
    except Exception:
        return pd.DataFrame()
    if resp.status_code != 200:
        return pd.DataFrame()
    try:
        obs = resp.json().get("observations", [])
    except Exception:
        return pd.DataFrame()
    if not obs:
        return pd.DataFrame()
    df = pd.DataFrame(obs)
    df = df[df["value"] != "."]                      # drop FRED missing marker
    df["Date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["Date", "value"]].dropna(subset=["value"]).reset_index(drop=True)
    df = df.sort_values("Date", kind="mergesort").drop_duplicates(
        subset="Date", keep="last").reset_index(drop=True)
    return df


def refresh_macro(name: str, dry_run: bool = False) -> bool:
    """Incremental FRED fetch for one macro series.

    Stores under /macros/fred_{name} with columns [Date, {name}], matching
    05_fed_data_gathering.py. FRED data always lags (daily series ~1 day,
    unemployment ~monthly); feature computation uses a backward-lookup cutoff
    at the latest price bar, so there is no look-ahead.

    Returns True if new data was fetched (or would be in dry-run).
    """
    if not FRED_API_KEY:
        return False
    series_id = FRED_SERIES.get(name)
    if series_id is None:
        return False
    h5_path = f"/macros/fred_{name}"

    with pd.HDFStore(DB_FILE, mode="r") as store:
        if h5_path in store.keys():
            existing = store[h5_path]
            latest = pd.Timestamp(existing["Date"].max()) if not existing.empty else None
        else:
            latest = None

    end_str = datetime.now().strftime("%Y-%m-%d")
    if latest is not None and latest >= pd.Timestamp(end_str):
        return False  # already up to date

    start = (latest + timedelta(days=1)).strftime("%Y-%m-%d") if latest is not None else "2009-01-01"

    if dry_run:
        print(f"    [DRY] Would fetch FRED {series_id} ({name}) {start}..{end_str}")
        return True

    new = fetch_fred_series(series_id, start, end_str)
    if new.empty:
        return False
    new = new.rename(columns={"value": name})

    with pd.HDFStore(DB_FILE, mode="a") as store:
        if h5_path in store.keys():
            existing = store[h5_path]
            combined = pd.concat([existing, new], ignore_index=True)
            combined = combined.sort_values("Date", kind="mergesort")
            combined = combined.drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
            store.remove(h5_path)
            store.put(h5_path, combined, format="table", data_columns=["Date"])
        else:
            store.put(h5_path, new, format="table", data_columns=["Date"])
    return True


def refresh_macros(dry_run: bool = False) -> dict:
    """Refresh the 3 FRED macro series used by the 23-feature set.

    Macro staleness corrupts vix / fed_funds / unemployment_roc21 directly.
    Only 3 API calls, so always refresh before loading.
    """
    ordered = list(FRED_SERIES.keys())
    if dry_run:
        print(f"  [DRY] Would refresh {len(ordered)} FRED series: {ordered}")
        return {"refreshed": 0, "skipped": len(ordered), "failed": 0}
    stats = {"refreshed": 0, "skipped": 0, "failed": 0}
    for name in ordered:
        try:
            if refresh_macro(name):
                stats["refreshed"] += 1
            else:
                stats["skipped"] += 1
        except Exception as e:
            stats["failed"] += 1
            print(f"    [ERROR] macro {name}: {e}")
    return stats


# ==============================================================================
# PART 3: FEATURE COMPUTATION (23 honest features — NO look-ahead)
# ==============================================================================
def compute_block1_live(earnings_df: pd.DataFrame, stock_ret: pd.Series,
                        ijh_ret: pd.Series, stock_dates: np.ndarray) -> dict:
    """Compute sue_lag_1, sue_lag_2, car_drift_historical_q1.

    For a FUTURE event, these use only PRIOR quarters' data:
      - sue_lag_1 = most recent REPORTED quarter's SUE
      - sue_lag_2 = the quarter before that
      - car_drift_historical_q1 = 60-day CAR after the prior quarter's report

    IMPORTANT: Filter to REPORTED earnings only (eps_actual not NaN).
    The FMP earnings table includes future scheduled events with
    eps_actual=NaN — these must be excluded.
    """
    result = {"sue_lag_1": np.nan, "sue_lag_2": np.nan, "car_drift_historical_q1": np.nan}

    if earnings_df is None or earnings_df.empty:
        return result

    # Filter to REPORTED earnings only (exclude future scheduled events)
    reported = earnings_df[earnings_df["actual"].notna()].sort_values("report_date").reset_index(drop=True)
    if reported.empty:
        return result

    # SUE: difference / rolling_12Q_std(difference)
    diff = reported["difference"].astype(float)
    sue = s2._compute_sue_score(diff)

    # For a future event, sue_lag_1 = last REPORTED SUE, sue_lag_2 = second-to-last
    if len(sue) >= 1 and pd.notna(sue.iloc[-1]):
        result["sue_lag_1"] = float(sue.iloc[-1])
    if len(sue) >= 2 and pd.notna(sue.iloc[-2]):
        result["sue_lag_2"] = float(sue.iloc[-2])

    # car_drift_historical_q1: 60-day CAR after the PRIOR REPORTED quarter's report_date
    prior_rdate = pd.Timestamp(reported["report_date"].iloc[-1])
    t_pos = s2.match_T(stock_dates, prior_rdate)
    if t_pos is not None:
        car60 = s2.compute_car_window(stock_ret, ijh_ret, t_pos, +1, s2.CAR_60D_END_OFFSET)
        if pd.notna(car60):
            result["car_drift_historical_q1"] = float(car60)

    return result


def compute_consecutive_surprises_pre(earnings_df: pd.DataFrame) -> float:
    """Prior quarter's beat-streak count.

    For a future event, this is the consecutive_surprises value at the
    most recent REPORTED quarter.

    IMPORTANT: Filter to REPORTED earnings only (eps_actual not NaN).
    """
    if earnings_df is None or earnings_df.empty:
        return np.nan
    # Filter to REPORTED earnings only
    reported = earnings_df[earnings_df["actual"].notna()].sort_values("report_date").reset_index(drop=True)
    if reported.empty:
        return np.nan
    consecutive = s2._compute_consecutive_surprises(
        reported["actual"].astype(float), reported["estimate"].astype(float)
    )
    if len(consecutive) >= 1:
        return float(consecutive.iloc[-1])
    return np.nan


def compute_macro_features(report_date: pd.Timestamp, macro_cache: dict) -> dict:
    """Compute unemployment_roc21, fed_funds, vix via backward lookup.

    Uses pre-loaded macro_cache for speed.
    No look-ahead (FRED data always lags).
    """
    result = {"unemployment_roc21": np.nan, "fed_funds": np.nan, "vix": np.nan}
    rd = pd.Timestamp(report_date)

    for name in ["vix", "fed_funds", "unemployment"]:
        if name not in macro_cache:
            continue
        m = macro_cache[name]
        mask = m["Date"] <= rd
        if not mask.any():
            continue
        if name == "unemployment":
            val = m.loc[mask, "unemployment_roc21"].iloc[-1]
            result["unemployment_roc21"] = float(val) if pd.notna(val) else np.nan
        else:
            val = m.loc[mask, name].iloc[-1]
            result[name] = float(val) if pd.notna(val) else np.nan

    return result


def preload_macro_cache() -> dict:
    """Pre-load and pre-process all macro data once."""
    cache = {}
    with pd.HDFStore(DB_FILE, mode="r") as store:
        for name, key in MACRO_KEYS.items():
            if key not in store:
                continue
            m = store[key].copy()
            m["Date"] = pd.to_datetime(m["Date"])
            m = m.sort_values("Date")
            close_col = m.columns[1]
            m[name] = pd.to_numeric(m[close_col], errors="coerce")
            if name == "unemployment":
                m["unemployment_roc21"] = m[name].pct_change(21).replace([np.inf, -np.inf], np.nan)
            cache[name] = m
    return cache


def _filter_actionable_events(cal: pd.DataFrame) -> pd.DataFrame:
    """Keep only events whose entry is still actionable now.

    During a trading day, today's entry is still actionable (AMC today and
    BMO tomorrow). After the close, today's entry has passed, so the next
    actionable set starts with tomorrow's entry date. This prevents the plan
    from displaying stale BMO events such as Aug-6 BMO after its Aug-5 MOC
    entry window has already closed.
    """
    if cal.empty:
        return cal
    now_et = datetime.now(ZoneInfo("America/New_York"))
    today = pd.Timestamp(now_et.date())
    # Before/within the regular session, today's close can still be ordered.
    # After 16:00 ET, today's MOC window is over. Premarket also allows a
    # same-day AMC decision, so only the after-close boundary is strict.
    after_close = now_et.weekday() < 5 and now_et.hour >= 16
    entry_dates = cal.apply(
        lambda r: pd.Timestamp(_compute_entry_date(pd.Timestamp(r["date"]), str(r["time"]).lower())),
        axis=1,
    )
    keep = entry_dates > today if after_close else entry_dates >= today
    dropped = int((~keep).sum())
    if dropped:
        print(f"  Actionable entry filter: dropped {dropped} events whose entry passed "
              f"(now ET={now_et:%Y-%m-%d %H:%M})")
    out = cal.loc[keep].copy()
    out["entry_date"] = entry_dates.loc[keep].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def compute_all_features(
    permaTicker: str,
    canonical_ticker: str,
    index_ref: str,
    report_date: pd.Timestamp,
    time_str: str,
    earnings_df: pd.DataFrame | None,
    benchmark_cache: dict[str, pd.DataFrame],
    stock_cache: dict[str, pd.DataFrame],
    grades_cache: dict[str, pd.DataFrame | None],
    macro_cache: dict,
) -> dict | None:
    """Compute all 23 honest features for one future earnings event.

    Returns None if price data is insufficient.
    Uses pre-loaded benchmark/stock/macro caches for speed.
    """
    # Load stock prices (from cache or DB)
    if permaTicker in stock_cache:
        stock_df = stock_cache[permaTicker]
    else:
        stock_key = f"/{SP400_GROUP}/{permaTicker}"
        with pd.HDFStore(DB_FILE, mode="r") as store:
            if stock_key not in store.keys():
                print(f"    [SKIP] {canonical_ticker}: no price node {stock_key}")
                return None
            stock_df = store[stock_key]
        stock_cache[permaTicker] = stock_df

    if stock_df is None or stock_df.empty or len(stock_df) < 35:
        print(f"    [SKIP] {canonical_ticker}: insufficient price data ({len(stock_df) if stock_df is not None else 0} rows)")
        return None

    # Get benchmark + sector ETF from cache
    ijh_df = benchmark_cache["/macros/IJH"]
    sector_key = f"/macros/{index_ref}" if isinstance(index_ref, str) and index_ref else "/macros/IJH"
    if sector_key not in benchmark_cache:
        sector_key = "/macros/IJH"
    sector_df = benchmark_cache[sector_key]

    # Build aligned frame
    aligned = s2.build_aligned_price_frame(stock_df, ijh_df, sector_df)
    stock_ret = s2._log_ret(pd.Series(aligned["stock_Adj_Close"].values, index=aligned.index))
    ijh_ret = s2._log_ret(pd.Series(aligned["ijh_Close"].values, index=aligned.index))

    stock_dates_np = stock_df["Date"].values

    # === Block 2 + 3: freshest actionable daily features ===
    # The plan is an as-of-NOW execution list, not a list of all future events.
    # For every event that survives _filter_actionable_events(), use the latest
    # fully completed daily bar currently in DB. At the final decision point
    # this matches v4's training contract automatically:
    #   AMC today -> latest bar is T-1;
    #   BMO tomorrow -> latest bar is T-2.
    # For events farther away this is a provisional fresh ranking and will be
    # refreshed again before their entry date.
    n = len(aligned)
    t_position = n  # feature formulas end at aligned[n-1], the latest bar
    if t_position < 21:
        return None
    feature_date = pd.Timestamp(aligned.index[t_position - 1])

    block2 = s2.compute_block2_features(aligned, stock_ret, ijh_ret, t_position, time_str)
    block3 = s2.compute_block3_features(aligned, t_position)

    # === Revision momentum (8 features) ===
    grades_df = grades_cache.get(permaTicker)
    # Revision and macro observations must use the same point-in-time cutoff
    # as prices. Prior earnings history remains keyed to the future event.
    rev = s2.compute_revision_momentum(grades_df, feature_date)

    # === Block 1: sue_lag_1, sue_lag_2, car_drift_historical_q1 ===
    block1 = compute_block1_live(earnings_df, stock_ret, ijh_ret, stock_dates_np)

    # === consecutive_surprises_pre ===
    consec_pre = compute_consecutive_surprises_pre(earnings_df)

    # === Macros (3 features) ===
    macros = compute_macro_features(feature_date, macro_cache)

    # Assemble all 23 features
    features = {
        "sue_lag_1": block1["sue_lag_1"],
        "sue_lag_2": block1["sue_lag_2"],
        "car_drift_historical_q1": block1["car_drift_historical_q1"],
        "pre_event_idiosyncratic_vol": block2["pre_event_idiosyncratic_vol"],
        "pre_event_volume_trend": block2["pre_event_volume_trend"],
        "rel_ret_3d": block3["rel_ret_3d"],
        "rel_ret_5d": block3["rel_ret_5d"],
        "rel_ret_10d": block3["rel_ret_10d"],
        "rel_ret_20d": block3["rel_ret_20d"],
        "rel_ret_30d": block3["rel_ret_30d"],
        "sector_adjusted_ret_20d": block3["sector_adjusted_ret_20d"],
        "revision_momentum_30d": rev["revision_momentum_30d"],
        "revision_momentum_60d": rev["revision_momentum_60d"],
        "revision_momentum_90d": rev["revision_momentum_90d"],
        "revision_ordinal_momentum_90d": rev["revision_ordinal_momentum_90d"],
        "revision_intensity_90d": rev["revision_intensity_90d"],
        "grade_dispersion_90d": rev["grade_dispersion_90d"],
        "n_analysts_covering": rev["n_analysts_covering"],
        "last_action_days_before_earnings": rev["last_action_days_before_earnings"],
        "consecutive_surprises_pre": consec_pre,
        "unemployment_roc21": macros["unemployment_roc21"],
        "fed_funds": macros["fed_funds"],
        "vix": macros["vix"],
    }
    return features


# ==============================================================================
# PART 4: MAIN PIPELINE
# ==============================================================================
def _record_v4_shadow_picks(picks: list[dict], generated_at: str) -> None:
    """Persist V4 hypothetical entries/exits without placing orders.

    Records are keyed by event identity. Re-running inference updates the
    latest ranking/observation while preserving any later outcome fields.
    """
    if V4_SHADOW_TRADES_JSON.exists():
        try:
            with open(V4_SHADOW_TRADES_JSON, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {"model": "phase_g_v4_timing_correct", "records": []}
    else:
        payload = {"model": "phase_g_v4_timing_correct", "records": []}
    records = payload.get("records", [])
    by_key = {r.get("event_key"): r for r in records if r.get("event_key")}
    for pick in picks:
        key = "|".join(str(pick.get(k, "")) for k in
                        ("permaTicker", "report_date", "time"))
        old = by_key.get(key, {})
        record = {
            **old,
            "event_key": key,
            "model": "phase_g_v4_timing_correct",
            "hypothetical": True,
            "canonical_ticker": pick.get("canonical_ticker"),
            "permaTicker": pick.get("permaTicker"),
            "report_date": pick.get("report_date"),
            "time": pick.get("time"),
            "entry_date": pick.get("entry_date"),
            "exit_date": pick.get("exit_date"),
            "p_v4": pick.get("p_pead"),
            "sector": pick.get("sector"),
            "features": pick.get("features", {}),
            "last_seen_at": generated_at,
        }
        record.setdefault("first_seen_at", generated_at)
        record.setdefault("outcome_status", "pending")
        record.setdefault("entry_price", None)
        record.setdefault("exit_price", None)
        record.setdefault("return_pct", None)
        by_key[key] = record
    payload["model"] = "phase_g_v4_timing_correct"
    payload["status"] = "hypothetical_comparison_only"
    payload["updated_at"] = generated_at
    payload["records"] = sorted(by_key.values(), key=lambda r: (r.get("entry_date", ""), r.get("canonical_ticker", "")))
    with open(V4_SHADOW_TRADES_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"    → Recorded {len(picks)} V4 hypothetical picks; ledger has {len(payload['records'])} events")


def main(weeks: int = 2, dry_run: bool = False, limit: int | None = None,
         shadow_v6: bool = True):
    bar = "=" * 70
    print(bar)
    print("  01_fetch_and_predict.py — v6 Default / v4 Comparison Inference")
    print(bar)
    print(f"  DB:          {DB_FILE}")
    print(f"  Classifier:  {FROZEN_CLASSIFIER}")
    print(f"  Theta:       P(PEAD) >= {THETA}")
    print(f"  Exclude:     {EXCLUDE_SECTORS}")
    print(f"  Max picks:   {MAX_PICKS}")
    print(f"  Weeks ahead: {weeks}")
    print(f"  Limit:       {limit if limit else 'none'}")
    print(f"  Dry run:     {dry_run}")
    print("  V6 default:  True")
    print(f"  V4 plan:     {V4_PLAN_JSON.name} + {V4_SHADOW_TRADES_JSON.name}")
    print(bar)

    if not FROZEN_CLASSIFIER.exists():
        print(f"\n  ERROR: Frozen classifier not found at {FROZEN_CLASSIFIER}")
        return

    # --- Step 1: Load metadata + model ---
    print("\n[1] Loading metadata and model ...")
    with pd.HDFStore(DB_FILE, mode="r") as store:
        meta = store[PERMATICKERS_KEY]
    meta_lookup = meta.set_index("canonical_ticker", drop=False)

    clf = xgb.XGBClassifier()
    clf.load_model(str(FROZEN_CLASSIFIER))
    v6_clfs = {}
    for gate in V6_GATES:
        path = V6_MODEL_DIR / gate / "classifier.json"
        if not path.exists():
            raise FileNotFoundError(f"V6 classifier missing: {path}")
        gate_clf = xgb.XGBClassifier()
        gate_clf.load_model(str(path))
        v6_clfs[gate] = gate_clf
    print("  Models loaded: V4 comparison + V6 default")

    # --- Step 2: Fetch FMP earnings calendar ---
    print(f"\n[2] Fetching FMP earnings calendar ({weeks} weeks ahead) ...")
    cal = fetch_earnings_calendar(weeks)
    cal_sp400 = filter_to_sp400(cal, meta)

    if cal_sp400.empty:
        print("\n  *** No SP400 earnings in the calendar window. ***")
        return

    if limit:
        cal_sp400 = cal_sp400.head(limit)
        print(f"  [--limit] Processing only {len(cal_sp400)} events")

    # --- Step 3: Refresh Tiingo prices for concerned tickers (THE WHOLE POINT — fresh data) ---
    # Build the base set from the calendar window.
    concerned_pts = []
    for _, row in cal_sp400.iterrows():
        ct = row["symbol"]
        if ct in meta_lookup.index:
            pt = meta_lookup.loc[ct, "permaTicker"]
            if isinstance(pt, pd.Series):
                pt = pt.iloc[0]
            concerned_pts.append(str(pt))
    concerned_pts = list(dict.fromkeys(concerned_pts))  # dedup preserving order

    # Union calendar tickers with currently-held positions + prior plan picks so
    # no candidate is silently dropped by the FMP bulk-calendar's known gap (it
    # misses some stocks like ROKU). We ALWAYS track what we hold / what we've
    # already flagged.
    safe_extra = set()
    # Persistent known-candidates: every permaTicker ever predicted. This survives
    # plan.json rewrites — once a stock is a candidate (e.g. ROKU), it stays tracked
    # even if a later run drops it. Fixes the round-trip loss where a stock falls
    # out of the prior plan and is never re-discovered.
    KNOWN_FILE = HERE / "known_candidates.json"
    known = set()
    if KNOWN_FILE.exists():
        try:
            with open(KNOWN_FILE, "r", encoding="utf-8") as kf:
                known = set(json.load(kf))
        except Exception:
            known = set()
    if known:
        safe_extra |= known
    # Prior plan picks (may include stocks the calendar dropped this run)
    prior_plan = HERE / "plan.json"
    if prior_plan.exists():
        try:
            with open(prior_plan, "r", encoding="utf-8") as pf:
                prior_picks = json.load(pf).get("picks", [])
            for p in prior_picks:
                s = p.get("canonical_ticker")
                if s in meta_lookup.index:
                    safe_extra.add(str(meta_lookup.loc[s, "permaTicker"])
                                   if not isinstance(meta_lookup.loc[s, "permaTicker"], pd.Series)
                                   else str(meta_lookup.loc[s, "permaTicker"].iloc[0]))
        except Exception:
            pass
    # Currently held positions (always keep prices fresh for tracking exits/stops)
    pos_file = HERE / "positions.json"
    if pos_file.exists():
        try:
            with open(pos_file, "r", encoding="utf-8") as pf:
                pos = json.load(pf)
            for p in pos.get("active", []) + pos.get("pending", []):
                if p.get("permaTicker"):
                    safe_extra.add(str(p["permaTicker"]))
        except Exception:
            pass
    if safe_extra:
        concerned_pts = list(dict.fromkeys(concerned_pts + sorted(safe_extra)))
        print(f"  + {len(safe_extra)} safety tickers (held/prior picks) added to refresh")

    if dry_run:
        print(f"\n[3] [DRY] Would refresh Tiingo for {len(concerned_pts)} tickers")
    else:
        print(f"\n[3] Refreshing Tiingo prices for {len(concerned_pts)} concerned tickers ...")
        tiingo_stats = refresh_concerned_tickers(concerned_pts, dry_run)
        print(f"  Done: refreshed={tiingo_stats['refreshed']} "
              f"skipped={tiingo_stats['skipped']} failed={tiingo_stats['failed']}")

    if dry_run:
        print(f"\n  [DRY] Would compute features for {len(cal_sp400)} events and predict.")
        return

    # --- Step 4: Load shared data (earnings, benchmarks, macros) ---
    print(f"\n[4] Loading shared data (earnings, benchmarks, macros) ...")
    t_load = time.time()

    # 4a. Determine benchmark ETFs needed and refresh them from Tiingo BEFORE
    #     loading. Index benchmarks (IJH + sector ETFs) must be current: a
    #     stale benchmark corrupts every rel_ret / car_drift feature (the
    #     2026-08-08 stale-IJH incident showed IJH a month behind silently
    #     biasing all candidates). Benchmarks are few, so this is fast.
    sector_tickers_needed = {"IJH"}
    for _, row in cal_sp400.iterrows():
        ct = row["symbol"]
        if ct in meta_lookup.index:
            pt_row = meta_lookup.loc[ct]
            if isinstance(pt_row, pd.DataFrame):
                pt_row = pt_row.iloc[0]
            iref = pt_row.get("index_ref", "IJH")
            if isinstance(iref, str) and iref:
                sector_tickers_needed.add(iref)
    if dry_run:
        print(f"\n  [DRY] Would refresh {len(sector_tickers_needed)} benchmarks: {sorted(sector_tickers_needed)}")
    else:
        bench_stats = refresh_benchmarks(sector_tickers_needed, dry_run)
        print(f"  Benchmarks refreshed: {bench_stats['refreshed']} "
              f"skipped={bench_stats['skipped']} failed={bench_stats['failed']}")

    # 4b. Refresh FRED macro series (vix, fed_funds, unemployment). These
    #     feed the vix / fed_funds / unemployment_roc21 features directly;
    #     stale values bias every candidate. Only 3 API calls.
    if dry_run:
        print(f"\n  [DRY] Would refresh {len(FRED_SERIES)} FRED macros: {list(FRED_SERIES.keys())}")
    else:
        macro_stats = refresh_macros(dry_run)
        print(f"  FRED macros refreshed: {macro_stats['refreshed']} "
              f"skipped={macro_stats['skipped']} failed={macro_stats['failed']}")

    with pd.HDFStore(DB_FILE, mode="r") as store:
        earnings_all = store[EARNINGS_KEY]
        # Benchmark cache (IJH + sector ETFs) — now freshly updated above.
        benchmark_cache: dict[str, pd.DataFrame] = {}
        for st in sector_tickers_needed:
            sk = f"/macros/{st}"
            if sk in store.keys():
                benchmark_cache[sk] = store[sk].sort_values("Date").reset_index(drop=True)
    earnings_all = earnings_all.rename(columns={
        "eps_actual": "actual", "eps_estimated": "estimate",
        "eps_difference": "difference", "eps_surprise_pct": "percent",
        "period_ending": "fiscal_period_end",
    })
    earnings_by_pt = {pt: g for pt, g in earnings_all.groupby("permaTicker")}
    macro_cache = preload_macro_cache()
    print(f"  Earnings: {len(earnings_all):,} rows | Benchmarks: {sorted(benchmark_cache.keys())} | "
          f"Macros: {list(macro_cache.keys())} | ({time.time()-t_load:.1f}s)")
    if "/macros/IJH" not in benchmark_cache:
        print("  WARNING: /macros/IJH missing!")

    # Per-ticker caches (populated lazily by compute_all_features)
    stock_cache: dict[str, pd.DataFrame] = {}
    grades_cache: dict[str, pd.DataFrame | None] = {}

    # --- Step 4.5: Add safe-extra tickers' events (held/prior picks the bulk
    # calendar missed) via the per-ticker /stable/earnings endpoint. The bulk
    # /earnings-calendar drops some stocks (e.g. ROKU); the per-ticker endpoint
    # is reliable. This guarantees we still predict what we already track. ---
    if safe_extra:
        print(f"\n[4.5] Fetching future earnings for {len(safe_extra)} safety tickers ...")
        extra_events = []
        with pd.HDFStore(DB_FILE, mode="r") as store:
            if PERMATICKERS_KEY in store.keys():
                safe_meta = store[PERMATICKERS_KEY]
            else:
                safe_meta = pd.DataFrame()
        pt_to_canonical = {}
        for _, rr in safe_meta.iterrows():
            pt_to_canonical[str(rr["permaTicker"])] = rr["canonical_ticker"]
        for pt in safe_extra:
            canonical = pt_to_canonical.get(pt)
            if not canonical:
                continue
            url = f"{FMP_BASE}/earnings"
            try:
                rsp = requests.get(url, params={"symbol": canonical, "includeReportTimes": "true",
                                                 "apikey": FMP_API_KEY}, timeout=FMP_TIMEOUT)
                if rsp.status_code != 200:
                    continue
                rows = rsp.json()
                if not isinstance(rows, list):
                    continue
            except Exception:
                continue
            # Safety fallback must obey the same lookahead window as the
            # bulk calendar. Never append a full future earnings history here
            # (that was how Nov 4 appeared during a --weeks 1 run).
            lookahead_end = (pd.Timestamp.now().normalize() +
                             timedelta(days=weeks * 7))
            for d in rows:
                ddate = d.get("date", "")
                if not ddate:
                    continue
                ddate_ts = pd.Timestamp(ddate)
                if ddate_ts < pd.Timestamp.now().normalize() or ddate_ts > lookahead_end:
                    continue  # outside current actionable lookahead window
                # only if not already in calendar
                if (cal_sp400["symbol"] == canonical).any():
                    continue
                extra_events.append({
                    "symbol": canonical,
                    "date": pd.Timestamp(ddate),
                    "time": d.get("time", "amc"),
                    "epsActual": d.get("epsActual"),
                    "epsEstimated": d.get("epsEstimated"),
                })
        if extra_events:
            extra_df = pd.DataFrame(extra_events)
            extra_df = extra_df[(extra_df["time"].isin(["bmo", "amc"]))]
            cal_sp400 = pd.concat([cal_sp400, extra_df], ignore_index=True)
            cal_sp400 = cal_sp400.drop_duplicates(subset=["symbol", "date", "time"])
            print(f"    + {len(extra_df)} safety-ticker events added to prediction set")

    # Keep only events whose entry is still actionable as of this run.
    # This is applied after the safety-calendar union so stale/passed events
    # cannot leak into plan.json.
    cal_sp400 = _filter_actionable_events(cal_sp400)
    if cal_sp400.empty:
        print("\n  *** No actionable earnings entries remain. ***")
        return

    # --- Step 5: Compute features + predict for each event ---
    print(f"\n[5] Computing features + predicting for {len(cal_sp400)} events ...")
    print(f"    (HDF5 reads ~2s each — expect ~{len(cal_sp400)*4//60}min total)")
    results = []
    t_feat = time.time()
    # Open ONE persistent DB handle for lazy per-ticker reads
    with pd.HDFStore(DB_FILE, mode="r") as lazy_store:
        for idx, (_, row) in enumerate(cal_sp400.iterrows()):
            ct = row["symbol"]
            rdate = pd.Timestamp(row["date"])
            time_str = row["time"]  # 'bmo' or 'amc'

            if ct not in meta_lookup.index:
                continue
            pt_row = meta_lookup.loc[ct]
            if isinstance(pt_row, pd.DataFrame):
                pt_row = pt_row.iloc[0]
            permaTicker = str(pt_row["permaTicker"])
            index_ref = pt_row.get("index_ref", "IJH")
            if not isinstance(index_ref, str) or not index_ref:
                index_ref = "IJH"

            # Lazy-load stock + grades from persistent handle (cached after first read)
            if permaTicker not in stock_cache:
                sk = f"/{SP400_GROUP}/{permaTicker}"
                stock_cache[permaTicker] = lazy_store[sk] if sk in lazy_store.keys() else None
            if permaTicker not in grades_cache:
                gk = f"/analyst/grades/{permaTicker}"
                grades_cache[permaTicker] = lazy_store[gk] if gk in lazy_store.keys() else None

            # Get earnings history for this permaTicker
            earnings_df = earnings_by_pt.get(permaTicker)

            features = compute_all_features(
                permaTicker, ct, index_ref, rdate, time_str, earnings_df,
                benchmark_cache, stock_cache, grades_cache, macro_cache
            )
            if features is None:
                continue

            # Predict P(PEAD)
            X = pd.DataFrame([features])[DEPLOY_FEATURES]
            proba = float(clf.predict_proba(X)[0, 1])
            v6_scores = {}
            for gate in V6_GATES:
                v6_scores[gate] = float(v6_clfs[gate].predict_proba(X)[0, 1])
            v6_scores["p_v6_min"] = min(v6_scores.values())

            results.append({
                "permaTicker": permaTicker,
                "canonical_ticker": ct,
                "report_date": rdate.strftime("%Y-%m-%d"),
                "time": time_str,
                "entry_date": _compute_entry_date(rdate, time_str),
                "exit_date": _compute_exit_date(rdate),
                "p_pead": round(proba, 4),
                **{k: round(v, 4) for k, v in v6_scores.items()},
                "p_v4_pead": round(proba, 4),
                "eps_estimated": row.get("epsEstimated"),
                "sector": index_ref,
                "features": {k: (float(v) if pd.notna(v) else None) for k, v in features.items()},
            })

            status = "✓" if proba >= THETA else " "
            elapsed = time.time() - t_feat
            eta = elapsed / (idx + 1) * (len(cal_sp400) - idx - 1)
            print(f"  [{idx+1}/{len(cal_sp400)}] {ct:6s} {rdate.strftime('%Y-%m-%d')} "
                  f"{time_str.upper()} P={proba:.3f} {status}  ({elapsed:.0f}s, ETA {eta:.0f}s)",
                  flush=True)

    if not results:
        print("\n  *** No events with computable features. ***")
        return

    # --- Step 6: Build separate V4 comparison and V6 executable plans ---
    print(f"\n[6] Building V4 comparison plan and V6 executable plan")
    accepted_v4 = [r for r in results if r["p_v4_pead"] >= THETA]
    print(f"  V4 after theta >= {THETA}: {len(accepted_v4)} of {len(results)}")
    accepted_v4 = [r for r in accepted_v4 if r["sector"] not in EXCLUDE_SECTORS]
    print(f"  V4 after sector exclusion: {len(accepted_v4)}")

    # LEG-like exclusion filter (backtest-validated money-loser, -1.31% avg / 38% win):
    #   prior SUE < -0.5 AND no beat streak AND oversold (20d < -5%)
    # These picks pass theta but LOSE money. Drop them regardless of P(PEAD).
    n_before_lelike = len(accepted_v4)
    def _is_leg_like(f):
        sue = f.get("sue_lag_1") or 0
        streak = f.get("consecutive_surprises_pre") or 0
        r20 = f.get("rel_ret_20d") or 0
        return sue < -0.5 and streak == 0 and r20 < -0.05
    accepted_v4 = [r for r in accepted_v4 if not _is_leg_like(r.get("features", {}))]
    print(f"  V4 LEG-like exclusion dropped {n_before_lelike - len(accepted_v4)}")
    accepted_v4.sort(key=lambda r: r["p_v4_pead"], reverse=True)
    picks_v4 = accepted_v4[:MAX_PICKS]

    accepted_v6 = [r for r in results if r["p_v6_min"] >= V6_THETA]
    print(f"  V6 after min-gate threshold >= {V6_THETA}: {len(accepted_v6)} of {len(results)}")
    accepted_v6 = [r for r in accepted_v6 if r["sector"] not in EXCLUDE_SECTORS]
    print(f"  V6 after sector exclusion: {len(accepted_v6)}")
    # The frozen V6 historical policy does not include the V4 LEG-like filter.
    accepted_v6.sort(key=lambda r: r["p_v6_min"], reverse=True)
    picks_v6 = accepted_v6[:V6_MAX_PICKS]

    def _make_plan(model, picks, threshold, accepted_count, extra):
        return {
            "generated_at": datetime.now().isoformat(), "model": model,
            "model_features": len(DEPLOY_FEATURES), "theta": threshold,
            "lookahead_audit": "All features use only pre-event data; no current-quarter earnings result used.",
            "calendar_window_weeks": weeks, "total_candidates": len(results),
            "accepted_after_filters": accepted_count, "picks": picks, **extra,
        }

    generated_at = datetime.now().isoformat()
    plan_v4 = _make_plan("phase_g_v4_timing_correct", picks_v4, THETA, len(accepted_v4), {
        "status": "comparison_only_not_executed", "accepted_after_theta": sum(r["p_v4_pead"] >= THETA for r in results),
        "accepted_after_xlf": len([r for r in results if r["p_v4_pead"] >= THETA and r["sector"] not in EXCLUDE_SECTORS]),
    })
    plan_v6 = _make_plan("phase_g_v6_gate_decomposition", picks_v6, V6_THETA, len(accepted_v6), {
        "status": "default_executable", "policy": f"min(p_pass_g1, p_pass_g2, p_pass_g3) >= {V6_THETA}",
        "accepted_after_threshold": sum(r["p_v6_min"] >= V6_THETA for r in results),
        "accepted_after_xlf": len([r for r in results if r["p_v6_min"] >= V6_THETA and r["sector"] not in EXCLUDE_SECTORS]),
    })
    # Script 02 expects the generic `p_pead` ranking field. For V6 it must
    # represent the frozen min-gate score rather than the V4 score.
    plan_v4["picks"] = [{**p, "p_pead": p["p_v4_pead"]} for p in picks_v4]
    plan_v6["picks"] = [{**p, "p_pead": p["p_v6_min"]} for p in picks_v6]
    with open(V4_PLAN_JSON, "w", encoding="utf-8") as f:
        json.dump(plan_v4, f, indent=2, ensure_ascii=False, default=str)
    with open(PLAN_JSON, "w", encoding="utf-8") as f:
        json.dump(plan_v6, f, indent=2, ensure_ascii=False, default=str)
    _record_v4_shadow_picks(picks_v4, generated_at)

    print(f"\n  V4 comparison picks ({len(picks_v4)}):")
    for i, p in enumerate(picks_v4):
        print(f"    V4 #{i+1} {p['canonical_ticker']:6s} | {p['report_date']} {p['time'].upper()} | P={p['p_v4_pead']:.3f}")
    print(f"\n  V6 executable picks ({len(picks_v6)}):")
    for i, p in enumerate(picks_v6):
        print(f"    V6 #{i+1} {p['canonical_ticker']:6s} | {p['report_date']} {p['time'].upper()} | min(Pgates)={p['p_v6_min']:.3f}")

    # Persist known-candidates: union of every permaTicker we've ever predicted,
    # so a stock dropped by the flaky FMP bulk-calendar (e.g. ROKU) stays tracked
    # and is re-checked via the per-ticker endpoint on future runs. THIS is what
    # makes the refresh safety-net actually durable across plan.json rewrites.
    try:
        known |= {r["permaTicker"] for r in results}
        known = {k for k in known if isinstance(k, str) and k}
        with open(KNOWN_FILE, "w", encoding="utf-8") as kf:
            json.dump(sorted(known), kf, indent=1)
    except Exception:
        pass

    print(f"\n[7] Wrote V6 executable {PLAN_JSON}")
    print(f"    Wrote V4 comparison {V4_PLAN_JSON}")
    print(f"    Recorded V4 hypothetical ledger {V4_SHADOW_TRADES_JSON}")
    print(f"    {len(picks_v6)} V6 picks available for execution by 02_paper_trade.py")
    print(bar)


def _compute_entry_date(report_date: pd.Timestamp, time_str: str) -> str:
    """Entry date: T-1 for BMO (buy at prior close), T for AMC (buy at same-day close)."""
    if time_str == "bmo":
        entry = report_date - pd.Timedelta(days=1)
    else:  # amc
        entry = report_date
    # Roll back to last business day if entry falls on weekend
    while entry.weekday() >= 5:
        entry = entry - pd.Timedelta(days=1)
    return entry.strftime("%Y-%m-%d")


def _compute_exit_date(report_date: pd.Timestamp) -> str:
    """Exit date: T+5 trading days. Approximate with 7 calendar days."""
    exit_date = report_date + pd.Timedelta(days=7)
    while exit_date.weekday() >= 5:
        exit_date = exit_date + pd.Timedelta(days=1)
    return exit_date.strftime("%Y-%m-%d")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V6 default live inference with V4 comparison plan")
    parser.add_argument("--weeks", type=int, default=2, help="Calendar look-ahead weeks")
    parser.add_argument("--dry-run", action="store_true", help="No writes, no API calls")
    parser.add_argument("--limit", type=int, default=None, help="Max events to process (testing)")
    args = parser.parse_args()
    main(weeks=args.weeks, dry_run=args.dry_run, limit=args.limit, shadow_v6=True)

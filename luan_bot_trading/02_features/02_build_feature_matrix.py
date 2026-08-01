#!/usr/bin/env python3
"""
Stage 2: Feature Matrix Builder  (Phase E v2 -- permaTicker-keyed)
==================================================================

PURPOSE
-------
Expand the Stage-1 gated events table (`/features/gated_events`, 20,742 rows
post Phase E v2, 7 columns including permaTicker) into the full listwise-
ranker training matrix at `/features/train_matrix`:

  * 21 active features in `X` (Block 1-4 per `features.md`  1).
  * 7 mandatory non-feature metadata columns (Phase E v2 keys by permaTicker).
  * 1 target label `car_10d` (continuous 10-day LOG CAR; NDCG target).
  * 1 Pass-1 audit column `car_60d_pass1` (used by Pass B for
    `car_drift_historical_q1`; also live-inference audit).

Total: 7 + 1 + 1 + 21 = 30 columns (same count as v1; permaTicker replaces
perm_id).

INPUTS (from `01_data/db.h5`; READ-ONLY; zero external API calls)
-----------------------------------------------------------------
  * `/features/gated_events`        -- Stage-1 v2 output (permaTicker-keyed):
        permaTicker, canonical_ticker, cik, report_date, added, removed,
        calendar_week_group                                 (7 cols)
  * `/earnings/raw`                    -- Phase D migrated schema. Required
        for Block 1 (12Q rolling std over ALL prior quarters per permaTicker).
        Columns include `permaTicker` (migrated from legacy `perm_id`).
  * `/sp400/{permaTicker}`             -- 928 individual price-series nodes
        (Tiingo-source, Phase B v2 storage). 11 cols: Date + Adj_Close /
        Adj_Volume etc. Loaded ONCE per permaTicker in the processing loop.
  * `/metadata/sp400_permatickers`      -- permaTicker metadata (Phase A v2):
        permaTicker, canonical_ticker, cik, index_ref, wikipedia_intervals
        (One row per permaTicker.)
  * `/macros/IJH`                      -- mid-cap benchmark for
        pre_event_idiosyncratic_vol and all rel_ret_*d features. Loaded ONCE.
  * `/macros/{index_ref}`              -- sector ETF nodes (per-permaTicker's
        `index_ref` field, defaults to IJH). 8 distinct ETFs total.

OUTPUTS
-------
  * `/features/train_matrix` in `01_data/db.h5` (HDF5 table).
  * stdout build report including per-feature non-NaN coverage, T-match
    failure count, distinct group count, pass timings.

PASS STRATEGY (Phase E v2)
--------------------------
  Pass A -- Per-permaTicker price-frame alignment + per-permaTicker
           Pass-A CAR windows (car_10d for gated events; car_60d_pass1 for
           ALL per-permaTicker earnings rows so car_drift_historical_q1
           shift stays accurate for gated events whose prior event was
           non-gated). Also per-event Block 2/3 features computed here.

  Pass B -- Per-permaTicker Block 1 (rolling 12Q std over full earnings
           timeline) + Block 4 interaction (needs Pass A vol + Pass B sue).

  Pass C -- Assemble row-major dataframe. Sort by
           ['calendar_week_group', 'permaTicker', 'report_date']. Write.

v2 CHANGES vs v1 (per Phase A+B migration report)
--------------------------------------------------
  * Identity: perm_id -> permaTicker (Tiingo-issued, identity-stable).
  * Price node path: /sp400/{canonical_ticker} -> /sp400/{permaTicker}.
  * Orchestration: per-canonical-ticker outer loop REMOVED; each permaTicker
    is its own iteration (Phase B v2 stores one price node per permaTicker).
  * Section-7.7 disambiguation REMOVED -- permaTicker is the storage key, no
    two active permaTickers can collide on a single canonical_ticker.
  * Stable mergesort for Date sort (eliminates the Phase B v2.1 quicksort
    contamination root cause).

LOG UNITS (LOCKED -- Design.md  17.4, features.md  1)
------------------------------------------------------
`car_10d` and `car_60d_pass1` (therefore `car_drift_historical_q1`) are
stored in LOG units. The arithmetic-percentage conversion happens at the
Stage-3 isotonic calibration bridge (`np.expm1` on the calibrator fit
target). NO conversion in Stage 2.

12 PRIMING CUTOFF (features.md  0)
------------------------------------
Stage 2 stores ALL gated events (2012-03-31 onward, post v2). The 12 train-
time cutoff (report_date >= 2015-01-01) is the training script's
responsibility (Stage 3), not here. Same for the sparse-week cutoff
(<3 events/week).

CLI
---
    python luan_bot_trading/02_features/02_build_feature_matrix.py
    python luan_bot_trading/02_features/02_build_feature_matrix.py --dry-run
    python luan_bot_trading/02_features/02_build_feature_matrix.py --limit N
        # Process only first N permaTickers (smoke testing). Default = all.

HDF5 WRITE SAFETY
-----------------
NEVER `mode='w'` on existing DB. Use `HDFStore(mode='a')` + `store.remove()`
if the target key already exists (per STOP_DOING_EXTRA_SHIT.md).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure unicode characters in errors/messages don't crash Windows cp1252 console.
# Per Windows constraint: this is the cleanest pattern (Python 3.7+).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, RuntimeError):
    pass  # not all streams support reconfigure (eg test capture)


# ==============================================================================
# CONFIGURATION
# ==============================================================================
DB_FILE = Path(__file__).resolve().parent.parent / "01_data" / "db.h5"

GATED_EVENTS_KEY = "/features/gated_events"
TRAIN_MATRIX_KEY = "/features/train_matrix"
EARNINGS_KEY = "/earnings/fmp"
PERMATICKERS_KEY = "/metadata/sp400_permatickers"
IJH_KEY = "/macros/IJH"
DEFAULT_SECTOR_INDEX = "IJH"

# CAR windows (trading days, relative to T):
CAR_10D_END_OFFSET = 11    # T+1 .. T+11  (10 trading-day holding; target)
CAR_60D_END_OFFSET = 60    # T+1 .. T+60  (60-day post-event drift; pass 1)

# Lookback windows (trading days, relative to T):
VWAP_LOOKBACK = 20          # volume / suv rolling window
IDIO_VOL_LOOKBACK = 20      # pre_event_idiosyncratic_vol window
VOLUME_TREND_LOOKBACK = 10  # pre_event_volume_trend OLS window
REL_RET_HORIZONS = [3, 5, 10, 20, 30]
SECTOR_RET_HORIZON = 20

# SUE rolling-12Q configuration
SUE_ROLLING_WINDOW = 12     # 12 quarters
SUE_MIN_PERIODS = 12        # see features.md sec 4 (rolling 12Q std denominator)

# Hard cap on eps_surprise_pct to suppress EODHD's divide-by-tiny-denominator
# outliers (~0.5% of rows). EODHD percent = (actual - estimate) / |estimate| * 100;
# when estimate is tiny (e.g. $0.001), small $ differences explode to absurd
# values (e.g. -320M% for SUNE 2015-Q1). Capping at +/-300% preserves true
# large surprises (99th percentile raw EODHD = +500%) while eliminating
# nonsense outliers. Inspection 2026-07-15: 99 rows (0.49%) clipped, 18
# rows (0.089%) have |val| > 5000% before cap.
EPS_SURPRISE_PCT_CAP = 300.0


def _sector_key(ticker: str) -> str:
    """Map a sector ticker (e.g. 'IJK') to its HDF5 key under /macros."""
    return f"/macros/{ticker}"


# ==============================================================================
# PART 1 — LOADERS
# ==============================================================================
def load_gated_events() -> pd.DataFrame:
    """Load Stage-1 /features/gated_events (Phase E 7-col schema):
        perm_id, canonical_ticker, cik, report_date, added, removed,
        calendar_week_group
    """
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"DB file not found: {DB_FILE} — run 01_features_gate_events.py first."
        )
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if GATED_EVENTS_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {GATED_EVENTS_KEY} not in {DB_FILE} — run 01_features_gate_events.py first."
            )
        return store.get(GATED_EVENTS_KEY)


def load_permatickers_map() -> pd.DataFrame:
    """Load /metadata/sp400_permatickers (Phase A v2 schema). Returns a
    DataFrame indexed by permaTicker with columns: canonical_ticker, cik,
    index_ref, wikipedia_intervals (raw JSON str -- Stage 2 does not parse).
    """
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if PERMATICKERS_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {PERMATICKERS_KEY} not found. Run 02b_build_company_map.py "
                "(Phase A v2) first."
            )
        df = store.get(PERMATICKERS_KEY)
    # Use only the columns we need.
    return df[["permaTicker", "canonical_ticker", "cik", "index_ref", "wikipedia_intervals"]]

def load_earnings_full() -> pd.DataFrame:
    """Load the FULL /earnings/fmp (FMP schema, replaces EODHD /earnings/raw).
    Renames FMP columns to match the builder's expected schema.

    Required columns after rename:
        report_date, fiscal_period_end, actual, estimate, difference,
        percent, before_after_market, permaTicker
    """
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if EARNINGS_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {EARNINGS_KEY} not found. Run 06b_fmp_earnings_gathering.py first."
            )
        df = store.get(EARNINGS_KEY)
    # Rename FMP columns to match the builder's expected schema
    rename_map = {
        "eps_actual": "actual",
        "eps_estimated": "estimate",
        "eps_difference": "difference",
        "eps_surprise_pct": "percent",
        "period_ending": "fiscal_period_end",
    }
    df = df.rename(columns=rename_map)
    return df


def load_stock_prices(permaTicker: str) -> pd.DataFrame:
    """Load /sp400/{permaTicker} (Phase B v2 storage path). Returns columns (11):
        Date, Open, High, Low, Close, Volume,
        Adj_Open, Adj_High, Adj_Low, Adj_Close, Adj_Volume
    Date is tz-naive datetime64[us]; sorted ascending.
    """
    key = f"/sp400/{permaTicker}"
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if key not in store.keys():
            raise KeyError(
                f"Price node {key} missing. "
                "This should never happen post Stage 1 (price_unavailable "
                "permaTickers are excluded at the gate). Investigate."
            )
        df = store.get(key)
    # Defensive: ensure sorted by Date ascending (stable sort).
    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)
    return df


def load_benchmark_prices(macro_key: str) -> pd.DataFrame:
    """Generic loader for an index/benchmark OHLCV table under /macros/{name}.
    Returns Date, Open, High, Low, Close, Volume (6 cols), sorted by Date asc.
    """
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if macro_key not in store.keys():
            raise KeyError(
                f"Benchmark node {macro_key} not found in {DB_FILE}. "
                f"Run 04_index_data_gathering.py first."
            )
        df = store.get(macro_key)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ==============================================================================
# PART 2 — ALIGNMENT & RETURN HELPERS
# ==============================================================================
def build_aligned_price_frame(
    stock_df: pd.DataFrame,
    ijh_df: pd.DataFrame,
    sector_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a single indexed frame aligned on the STOCK's Date index.

    Stock's Date is the master index (LEFT join stock.Date -> benchmarks.Date).
    Missing benchmark rows are filled from the most recent PRIOR benchmark row
    (ffill); this guarantees we never look forward for a price on a holiday.

    Returns: DataFrame indexed by Date, with these columns:
        stock_Open, stock_High, stock_Low, stock_Close, stock_Volume,
        stock_Adj_Close, stock_Adj_Volume,
        ijh_Close,
        sector_Close
    """
    s = stock_df[["Date", "Open", "High", "Low", "Close", "Volume",
                  "Adj_Close", "Adj_Volume"]].copy()
    s = s.rename(columns={
        "Open": "stock_Open", "High": "stock_High", "Low": "stock_Low",
        "Close": "stock_Close", "Volume": "stock_Volume",
        "Adj_Close": "stock_Adj_Close", "Adj_Volume": "stock_Adj_Volume",
    })
    s = s.set_index("Date")

    ijh = ijh_df[["Date", "Close"]].rename(columns={"Close": "ijh_Close"}).set_index("Date")
    sec = sector_df[["Date", "Close"]].rename(columns={"Close": "sector_Close"}).set_index("Date")

    out = s.join(ijh, how="left")
    out = out.join(sec, how="left")
    # ffill benchmark gaps (holiday -> most recent prior benchmark close).
    out["ijh_Close"] = out["ijh_Close"].ffill()
    out["sector_Close"] = out["sector_Close"].ffill()
    return out


def _log_ret(series: pd.Series) -> pd.Series:
    """log(s[t] / s[t-1])  — natural log return.  NaN-safe; returns NaN
    at lead position.

    Per Design §6 / features.md §1: returns are in log space (continuity,
    additivity across horizons)."""
    s = series.astype(float)
    return np.log(s / s.shift(1))


# ==============================================================================
# PART 3 — T-MATCH + PER-EVENT FEATURE COMPUTERS (PASS A)
# ==============================================================================
def match_T(stock_dates: np.ndarray, report_date: pd.Timestamp) -> int | None:
    """Return the integer positional index of the first Date >= report_date in
    the stock's sorted Date array (roll FORWARD per features.md §1 Block 2).
    Return None if no such date exists (event beyond the price-series end);
    caller drops the row and logs it.

    O(log N) per lookup via searchsorted(side='left', report_date). Assumes
    stock_dates is sorted ascending (enforced by load_stock_prices()).
    """
    # Convert report_date to numpy datetime64 matching the column dtype.
    target = pd.Timestamp(report_date)
    # searchsorted requires same-dtype comparison.
    ts_array = stock_dates
    pos = int(np.searchsorted(ts_array, np.datetime64(target), side="left"))
    if pos >= len(ts_array):
        return None
    return pos


def compute_car_window(
    stock_ret: pd.Series,
    benchmark_ret: pd.Series,
    t_position: int,
    start_offset: int,
    end_offset: int,
) -> float:
    """CAR = SUM[ log_stock_ret - log_benchmark_ret ] over trading days
    [t_position + start_offset, t_position + end_offset]. Returns NaN if
    the window extends beyond the price-series end (no row drop per NaN
    policy).

    UNITS: returns value is in LOG units (log CAR). Do NOT convert inside
    this function — Stage 3 isotonic bridge handles the conversion.
    """
    lo = t_position + start_offset
    hi = t_position + end_offset
    # Series bounds-check: if lo is already past the end, return NaN.
    if lo < 0 or lo >= len(stock_ret):
        return float("nan")
    if hi >= len(stock_ret):
        # Clip hi to end-1; partial-window CAR is fine (returns NaN if even
        # the partial cookie is empty). Per NaN policy, partial-window result
        # is acceptable:CAR of an incomplete window is the sum-so-far.
        # But per features.md §4, CAR for events where T+11/T+60 falls
        # beyond series end is NaN (not partial). Let's honor that explicit
        # policy: full-window-or-NaN.
        return float("nan")
    window_excess = (stock_ret.iloc[lo: hi + 1]
                     - benchmark_ret.iloc[lo: hi + 1])
    return float(window_excess.sum())


def compute_block2_features(
    aligned: pd.DataFrame,
    stock_ret: pd.Series,
    ijh_ret: pd.Series,
    t_position: int,
    before_after_market_str: object,
) -> dict:
    """Compute the 7 Block-2 features for a single event at t_position.

    Feature formulas (features.md §1 Block 2):
        is_bmo                       : 1 if bam == 'Bmo' else 0
        volume_vma20_ratio_pre_event : Volume[T] / mean(Volume[T-20:T-1])
        suv_day_1                    : Adj_Volume[T] / mean(Adj_Volume[T-20:T-1])
        pre_event_idiosyncratic_vol  : std(stock_ret - ijh_ret, ddof=1) at [T-20:T-1]
        opening_gap_t1                : (Open[T+1] - Close[T]) / Close[T]
        intraday_range_t             : (High[T] - Low[T]) / Close[T]
        pre_event_volume_trend       : OLS slope of log(Adj_Volume) over [T-10:T-1]

    NaN-safe per features.md §4. Returns dict with the 7 keys; missing
    lookback data yields NaN.
    """
    n = len(aligned)
    i = t_position

    def _safe_slice(series, lo, hi):
        # Pandas inclusive slicing convention: [lo, hi] when lo <= hi.
        if lo < 0 or hi >= n or lo > hi:
            return None
        return series.iloc[lo: hi + 1]

    # is_bmo: 1 if event was announced before market open. EODHD encodes as
    # 'BeforeMarket' / 'AfterMarket' (CamelCase concatenated). We match the
    # 'before' prefix (case-insensitive) to be robust to any encoding variant.
    bam = before_after_market_str
    is_bmo = 1 if (isinstance(bam, str) and "bmo" in bam.strip().lower()[:6]) else 0
    # Defensive alternative matchers: 'beforemarket' starts with 'before'.
    if is_bmo == 0 and isinstance(bam, str) and bam.strip().lower().startswith("before"):
        is_bmo = 1

    # volume_vma20_ratio_pre_event: Volume[T] / mean(Volume[T-20:T-1])
    if i >= 0 and i < n:
        vol_t = aligned["stock_Volume"].iloc[i]
    else:
        vol_t = np.nan
    pre_vol = _safe_slice(aligned["stock_Volume"], i - VWAP_LOOKBACK, i - 1)
    if pre_vol is not None and not pre_vol.empty and not pre_vol.isna().all():
        pre_vol_mean = float(pre_vol.mean())
        volume_vma20_ratio = float(vol_t / pre_vol_mean) if pre_vol_mean not in (0.0,) and not pd.isna(pre_vol_mean) else np.nan
    else:
        volume_vma20_ratio = np.nan

    # suv_day_1: Adj_Volume[T] / mean(Adj_Volume[T-20:T-1])
    adj_vol_t = aligned["stock_Adj_Volume"].iloc[i] if (0 <= i < n) else np.nan
    pre_adj_vol = _safe_slice(aligned["stock_Adj_Volume"], i - VWAP_LOOKBACK, i - 1)
    if pre_adj_vol is not None and not pre_adj_vol.empty and not pre_adj_vol.isna().all():
        pre_adj_vol_mean = float(pre_adj_vol.mean())
        suv_day_1 = float(adj_vol_t / pre_adj_vol_mean) if pre_adj_vol_mean not in (0.0,) and not pd.isna(pre_adj_vol_mean) else np.nan
    else:
        suv_day_1 = np.nan

    # pre_event_idiosyncratic_vol: std(stock_ret - ijh_ret, ddof=1) [T-20:T-1]
    residual = (stock_ret - ijh_ret)
    pre_residual = _safe_slice(residual, i - IDIO_VOL_LOOKBACK, i - 1)
    if pre_residual is not None and pre_residual.notna().sum() >= 2:
        pre_event_idio_vol = float(pre_residual.std(ddof=1))
    else:
        pre_event_idio_vol = np.nan

    # opening_gap_t1: (Open[T+1] - Close[T]) / Close[T]
    if (i + 1) < n and aligned["stock_Close"].iloc[i] not in (0, np.nan):
        open_t1 = aligned["stock_Open"].iloc[i + 1]
        close_t = aligned["stock_Close"].iloc[i]
        opening_gap = float((open_t1 - close_t) / close_t)
    else:
        opening_gap = np.nan

    # intraday_range_t: (High[T] - Low[T]) / Close[T]
    if 0 <= i < n and aligned["stock_Close"].iloc[i] not in (0, np.nan):
        high_t = aligned["stock_High"].iloc[i]
        low_t = aligned["stock_Low"].iloc[i]
        close_t = aligned["stock_Close"].iloc[i]
        intraday_range = float((high_t - low_t) / close_t)
    else:
        intraday_range = np.nan

    # pre_event_volume_trend: OLS slope over [T-10:T-1] of LOG(Adj_Volume).
    # Log-transformed to normalize across stocks of different liquidity levels.
    # Raw volume slope (shares/day) was incomparable between AAPL (~50M vol)
    # and a small mid-cap (~500K vol). Log slope is in units of log(shares)/day,
    # typically in [-0.1, +0.1] range, cross-stock comparable.
    #   Positive slope = volume increasing into earnings (anticipation)
    #   Negative slope = volume decreasing (pre-earnings lull, de-risking)
    pre_trend = _safe_slice(aligned["stock_Adj_Volume"], i - VOLUME_TREND_LOOKBACK, i - 1)
    if pre_trend is not None and pre_trend.notna().sum() >= 2:
        # Drop NaNs; use numeric index for day_index.
        valid = pre_trend.dropna()
        if len(valid) >= 2:
            x = np.arange(len(valid), dtype=float)
            # Log-transform volume (safe: Adj_Volume should always be > 0;
            # defensive clipping for any zero/negative values).
            vol_vals = valid.values.astype(float)
            vol_vals = np.where(vol_vals > 0, vol_vals, np.nan)
            y = np.log(vol_vals)
            # Drop any NaN from the log transform (zero/negative volumes).
            finite_mask = np.isfinite(y)
            if finite_mask.sum() >= 2:
                x = x[finite_mask]
                y = y[finite_mask]
                # polyfit deg=1 returns [slope, intercept]
                try:
                    slope = float(np.polyfit(x, y, 1)[0])
                except (np.linalg.LinAlgError, TypeError):
                    slope = np.nan
            else:
                slope = np.nan
        else:
            slope = np.nan
    else:
        slope = np.nan
    pre_event_volume_trend = slope

    return {
        "is_bmo": is_bmo,
        "volume_vma20_ratio_pre_event": volume_vma20_ratio,
        "suv_day_1": suv_day_1,
        "pre_event_idiosyncratic_vol": pre_event_idio_vol,
        "opening_gap_t1": opening_gap,
        "intraday_range_t": intraday_range,
        "pre_event_volume_trend": pre_event_volume_trend,
    }


def compute_block3_features(
    aligned: pd.DataFrame,
    t_position: int,
) -> dict:
    """Compute the 6 Block-3 relative-return features for a single event.

    Formulas (features.md §1 Block 3):
        rel_ret_{h}d for h in {3,5,10,20,30}:
            log(stock_Adj_Close[T-1] / stock_Adj_Close[T-1-h])
            - log(ijh_Close[T-1] / ijh_Close[T-1-h])
        sector_adjusted_ret_20d:
            log(stock_Adj_Close[T-1] / stock_Adj_Close[T-21])
            - log(sector_Close[T-1] / sector_Close[T-21])
    """
    n = len(aligned)
    i = t_position
    out = {}

    # Stock and IJH relative returns.
    for h in REL_RET_HORIZONS:
        lo = i - 1 - h
        hi = i - 1
        if lo < 0 or hi < 0 or hi >= n:
            out[f"rel_ret_{h}d"] = np.nan
            continue
        s_lo = aligned["stock_Adj_Close"].iloc[lo]
        s_hi = aligned["stock_Adj_Close"].iloc[hi]
        b_lo = aligned["ijh_Close"].iloc[lo]
        b_hi = aligned["ijh_Close"].iloc[hi]
        if (s_lo is None or pd.isna(s_lo) or s_lo == 0
            or b_lo is None or pd.isna(b_lo) or b_lo == 0):
            out[f"rel_ret_{h}d"] = np.nan
        else:
            out[f"rel_ret_{h}d"] = float(
                (np.log(s_hi / s_lo) - np.log(b_hi / b_lo))
            )

    # Sector-adjusted 20d.
    h = SECTOR_RET_HORIZON
    lo = i - 1 - h
    hi = i - 1
    if lo < 0 or hi < 0 or hi >= n:
        out["sector_adjusted_ret_20d"] = np.nan
    else:
        s_lo = aligned["stock_Adj_Close"].iloc[lo]
        s_hi = aligned["stock_Adj_Close"].iloc[hi]
        sec_lo = aligned["sector_Close"].iloc[lo]
        sec_hi = aligned["sector_Close"].iloc[hi]
        if (s_lo is None or pd.isna(s_lo) or s_lo == 0
            or sec_lo is None or pd.isna(sec_lo) or sec_lo == 0):
            out["sector_adjusted_ret_20d"] = np.nan
        else:
            out["sector_adjusted_ret_20d"] = float(
                (np.log(s_hi / s_lo) - np.log(sec_hi / sec_lo))
            )

    return out


# ==============================================================================
# PART 4 — BLOCK 1 + HISTORICAL CAR DRIFT (PASS B)
# ==============================================================================
def _compute_sue_score(
    difference: pd.Series,
) -> pd.Series:
    """sue_score = difference / rolling_12Q_std(difference, min_periods=12, ddof=1).

    Per features.md §1 / Design.md §15 Option B:
        - NaN estimates -> EODHD sets difference=0.0 already.
        - Include the 0.0 values in the rolling std denominator.
        - min_periods=12 (first 11 quarters per perm_id -> NaN).
        - Use ALL prior quarters (no SP400-window restriction).
    """
    roll = difference.rolling(
        window=SUE_ROLLING_WINDOW, min_periods=SUE_MIN_PERIODS
    ).std(ddof=1)
    # Suppress pandas division warning (NaN / NaN yielded a NaN already).
    with np.errstate(invalid="ignore", divide="ignore"):
        sue = difference / roll
    return sue


def _compute_consecutive_surprises(
    actual: pd.Series,
    estimate: pd.Series,
) -> pd.Series:
    """Strict-beat running counter per features.md §1:
        +1 if actual > estimate, else reset to 0.
    NaN actual or estimate -> reset to 0 (treated as non-beat).
    """
    act = actual.values
    est = estimate.values
    out = np.zeros(len(act), dtype=np.int64)
    run = 0
    for k in range(len(act)):
        a = act[k]
        e = est[k]
        try:
            if (a is not None and e is not None
                and not (isinstance(a, float) and np.isnan(a))
                and not (isinstance(e, float) and np.isnan(e))
                and a > e):
                run += 1
            else:
                run = 0
        except TypeError:
            run = 0
        out[k] = run
    return pd.Series(out, index=actual.index)


# ==============================================================================
# PART 4B -- ANALYST REVISION MOMENTUM FEATURES (Phase H, FMP data)
# ==============================================================================
GRADES_GROUP_KEY = "analyst/grades"

# Ordinal mapping for analyst grade labels.
# Maps string grade labels to a numeric scale so we can compute revision
# MAGNITUDE (not just count). A "Sell -> Strong Buy" (+5) is vastly more
# informative than a "Buy -> Outperform" (+0), but a simple upgrade count
# treats them identically.
#
# Scale: 1 (most bearish) -> 6 (most bullish)
# Based on standard Wall Street rating scales.
_GRADE_ORDINAL = {
    # Strong sell (1)
    "strong sell": 1, "sell": 1, "underweight": 1, "underperform": 1,
    "reduce": 1, "reduce in price": 1,
    # Bearish (2)
    "negative": 2, "below average": 2, "below market": 2,
    "market underperform": 2, "market underperformer": 2,
    # Hold / neutral (3)
    "hold": 3, "neutral": 3, "sector weight": 3, "sector perform": 3,
    "market perform": 3, "in-line": 3, "peer perform": 3,
    "equal-weight": 3, "equal weight": 3, "market weight": 3,
    "fair value": 3, "average": 3, "maintain": 3,
    # Accumulate / mild buy (4)
    "accumulate": 4, "add": 4, "overweight": 4, "outperform": 4,
    "market outperform": 4, "trading buy": 4, "positive": 4,
    "buy": 4, "sector outperform": 4,
    # Strong buy (5)
    "strong buy": 5, "strong outperform": 5, "top pick": 5,
    "action list buy": 5, "conviction buy": 5,
}


def _grade_to_ordinal(grade: str | None) -> int | None:
    """Map a grade label string to its ordinal value (1-5).
    Returns None if the label is unrecognized or null."""
    if grade is None or not isinstance(grade, str):
        return None
    g = grade.strip().lower()
    if not g:
        return None
    # Direct lookup
    if g in _GRADE_ORDINAL:
        return _GRADE_ORDINAL[g]
    # Fuzzy: try partial matches (e.g. "Long-Term Buy" contains "buy")
    for key, val in _GRADE_ORDINAL.items():
        if key in g:
            return val
    return None


def load_grades(permaTicker: str) -> pd.DataFrame | None:
    """Load /analyst/grades/{permaTicker} if it exists. Returns None if the
    node doesn't exist (permaTicker has no analyst coverage)."""
    key = f"/{GRADES_GROUP_KEY}/{permaTicker}"
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if key not in store.keys():
            return None
        return store.get(key)


def compute_revision_momentum(
    grades_df: pd.DataFrame | None,
    report_date: pd.Timestamp,
) -> dict:
    """Compute analyst revision momentum features for a given report_date.

    Two families of features:
    1. COUNT-based (original, coarse): net upgrades minus downgrades.
    2. ORDINAL-magnitude-based (new, richer): sum of ordinal grade changes,
       capturing HOW MUCH each revision moved on the rating scale.

    A "Sell -> Strong Buy" (ordinal +4) contributes +4 to ordinal momentum
    but only +1 to count momentum. A "Buy -> Outperform" (ordinal +0)
    contributes 0 to ordinal momentum but +1 to count momentum. The ordinal
    version preserves the magnitude information that the literature shows
    is the #1 PEAD predictor.

    All features are **Sunday-safe** — they use only data BEFORE report_date.

    Args:
        grades_df: DataFrame with columns date, grading_company, action,
                   previous_grade, new_grade (sorted by date ascending),
                   or None if no grades node.
        report_date: the earnings announcement date T.

    Returns dict with 8 features:
        revision_momentum_30d            : int   -- count: net upgrades - downgrades in [T-30d, T)
        revision_momentum_60d            : int   -- same, 60-day window
        revision_momentum_90d            : int   -- same, 90-day window
        revision_ordinal_momentum_90d    : float -- sum of ordinal grade changes in [T-90d, T)
        revision_intensity_90d          : int   -- total upgrades + downgrades in [T-90d, T) (analyst attention)
        grade_dispersion_90d            : int   -- distinct ordinal levels among covering analysts (uncertainty)
        n_analysts_covering              : int   -- unique grading firms in [T-90d, T)
        last_action_days_before_earnings: float -- days from last action to T
    """
    nan_result = {
        "revision_momentum_30d": np.nan,
        "revision_momentum_60d": np.nan,
        "revision_momentum_90d": np.nan,
        "revision_ordinal_momentum_90d": np.nan,
        "revision_intensity_90d": np.nan,
        "grade_dispersion_90d": np.nan,
        "n_analysts_covering": np.nan,
        "last_action_days_before_earnings": np.nan,
    }

    if grades_df is None or grades_df.empty:
        return nan_result

    # Filter to actions STRICTLY BEFORE report_date (exclusive).
    rd = pd.Timestamp(report_date)
    pre = grades_df[grades_df["date"] < rd].copy()

    if pre.empty:
        return {
            "revision_momentum_30d": 0,
            "revision_momentum_60d": 0,
            "revision_momentum_90d": 0,
            "revision_ordinal_momentum_90d": 0.0,
            "revision_intensity_90d": 0,
            "grade_dispersion_90d": 0,
            "n_analysts_covering": 0,
            "last_action_days_before_earnings": np.nan,
        }

    # Compute ordinal values for previous and new grades.
    pre["prev_ordinal"] = pre["previous_grade"].apply(_grade_to_ordinal)
    pre["new_ordinal"] = pre["new_grade"].apply(_grade_to_ordinal)
    # Ordinal change = new - previous (positive = bullish shift, negative = bearish)
    pre["ordinal_delta"] = pre.apply(
        lambda r: (r["new_ordinal"] - r["prev_ordinal"])
        if r["prev_ordinal"] is not None and r["new_ordinal"] is not None
        else None,
        axis=1,
    )

    # Define window start dates.
    d30 = rd - pd.Timedelta(days=30)
    d60 = rd - pd.Timedelta(days=60)
    d90 = rd - pd.Timedelta(days=90)

    def _net_count(df: pd.DataFrame) -> int:
        """Count-based: net upgrades - downgrades."""
        if df.empty:
            return 0
        ups = int((df["action"] == "upgrade").sum())
        downs = int((df["action"] == "downgrade").sum())
        return ups - downs

    def _ordinal_momentum(df: pd.DataFrame) -> float:
        """Magnitude-based: sum of ordinal grade changes."""
        if df.empty:
            return 0.0
        deltas = df["ordinal_delta"].dropna()
        if deltas.empty:
            return 0.0
        return float(deltas.sum())

    def _intensity(df: pd.DataFrame) -> int:
        """Total revision activity (upgrades + downgrades, excludes maintains)."""
        if df.empty:
            return 0
        ups = int((df["action"] == "upgrade").sum())
        downs = int((df["action"] == "downgrade").sum())
        return ups + downs

    def _dispersion(df: pd.DataFrame) -> int:
        """Distinct ordinal levels among current grades (uncertainty measure)."""
        if df.empty:
            return 0
        ordinals = df["new_ordinal"].dropna()
        if ordinals.empty:
            return 0
        return int(ordinals.nunique())

    w30 = pre[pre["date"] >= d30]
    w60 = pre[pre["date"] >= d60]
    w90 = pre[pre["date"] >= d90]

    last_date = pre["date"].max()
    last_days = float((rd - last_date).days) if pd.notna(last_date) else np.nan

    return {
        "revision_momentum_30d": _net_count(w30),
        "revision_momentum_60d": _net_count(w60),
        "revision_momentum_90d": _net_count(w90),
        "revision_ordinal_momentum_90d": _ordinal_momentum(w90),
        "revision_intensity_90d": _intensity(w90),
        "grade_dispersion_90d": _dispersion(w90),
        "n_analysts_covering": int(w90["grading_company"].nunique()) if not w90.empty else 0,
        "last_action_days_before_earnings": last_days,
    }


def compute_block1_features_per_perm_id(
    permaticker_earnings: pd.DataFrame,      # FULL per-permaTicker earnings
                                              # slice, sorted by report_date asc.
                                              # Columns: report_date, fiscal_period_end,
                                              # actual, estimate, difference, percent,
                                              # before_after_market.
    car_60d_pass1_full: pd.Series,           # per-event 60d CAR aligned to
                                              # permaticker_earnings.index (NaN for
                                              # events that were never computed by
                                              # Pass A -- such events shouldn't
                                              # exist now because Pass A computes
                                              # car_60d_pass1 for ALL per-permaTicker
                                              # earnings rows, but defensive NaN
                                              # allowed).
) -> pd.DataFrame:
    """Compute all 7 Block-1 features for a permaTicker over its FULL earnings
    timeline.

    Returns a DataFrame indexed by permaticker_earnings.index with 7 columns:
        sue_score, eps_surprise_pct, consecutive_surprises,
        sue_acceleration, sue_lag_1, sue_lag_2,
        car_drift_historical_q1
    """
    sue = _compute_sue_score(permaticker_earnings["difference"])
    eps_surprise_pct = permaticker_earnings["percent"]
    # Hard cap (option b per inspection 2026-07-15): EODHD's percent =
    # (actual - estimate) / |estimate| * 100 explodes when estimate is
    # tiny (e.g. -320M% for SUNE 2015-Q1). Clip at +/-EPS_SURPRISE_PCT_CAP.
    # NaN rows stay NaN. This avoids ranker split distortion from nonsense
    # outlier values while preserving true large surprises.
    eps_surprise_pct = eps_surprise_pct.clip(
        lower=-EPS_SURPRISE_PCT_CAP, upper=EPS_SURPRISE_PCT_CAP
    )
    consecutive = _compute_consecutive_surprises(permaticker_earnings["actual"], permaticker_earnings["estimate"])
    sue_acc = sue.diff()
    sue_lag_1 = sue.shift(1)
    sue_lag_2 = sue.shift(2)
    car_drift = car_60d_pass1_full.shift(1)
    out = pd.DataFrame({
        "sue_score": sue,
        "eps_surprise_pct": eps_surprise_pct,
        "consecutive_surprises": consecutive,
        "sue_acceleration": sue_acc,
        "sue_lag_1": sue_lag_1,
        "sue_lag_2": sue_lag_2,
        "car_drift_historical_q1": car_drift,
    }, index=permaticker_earnings.index)
    return out


# ==============================================================================
# PART 5 -- PER-PERMATICKER ORCHESTRATOR (Phase E v2)
# ==============================================================================
def process_permaticker(
    permaTicker: str,
    canonical: str,
    index_ref: str,
    gated_for_pt: pd.DataFrame,            # gated events for THIS permaTicker
                                            # (already filtered by permaTicker).
    permaticker_earnings_full: pd.DataFrame,  # FULL per-permaTicker earnings slice
                                              # (sorted by report_date asc) used for
                                              # Block 1 Rolling 12Q over all priors.
    ijh_df: pd.DataFrame,
    benchmark_prices_cache: dict[str, pd.DataFrame],
) -> tuple[list[dict], list[tuple[str, pd.Timestamp]]]:
    """Process one permaTicker's gated events end to end.

    v2 changes vs v1 process_canonical:
      * No outer canonical-grouped loop. Each permaTicker is its own iteration
        (Phase B v2 stores each permaTicker at /sp400/{permaTicker}).
      * One sector ETF per permaTicker (not per-perm_id under a canonical).
      * Removed the entire §7.7 disambiguation machinery (no collisions possible).

    Returns:
        row_dicts : list of dict rows (this permaTicker's chunk of train_matrix)
        t_failures: list of (permaTicker, report_date) tuples where T-match
                    failed (these rows are dropped per features.md §4).
    """
    # Load stock prices once for this permaTicker.
    stock_df = load_stock_prices(permaTicker)
    stock_dates_np = stock_df["Date"].values  # sorted ascending (load_stock_prices)

    # Load analyst grades for this permaTicker (Phase H, FMP data).
    # Returns None if no grades node exists (no analyst coverage).
    grades_df = load_grades(permaTicker)

    # Resolve sector ETF.
    if not isinstance(index_ref, str) or not index_ref:
        index_ref = DEFAULT_SECTOR_INDEX
    sector_key = _sector_key(index_ref)
    if sector_key not in benchmark_prices_cache:
        benchmark_prices_cache[sector_key] = load_benchmark_prices(sector_key)
    sector_df = benchmark_prices_cache[sector_key]

    # Build the aligned frame (LEFT join stock.Date -> benchmarks).
    aligned = build_aligned_price_frame(stock_df, ijh_df, sector_df)
    stock_ret_full = _log_ret(pd.Series(aligned["stock_Adj_Close"].values, index=aligned.index))
    ijh_ret_full = _log_ret(pd.Series(aligned["ijh_Close"].values, index=aligned.index))

    # Earnings for this permaTicker (FULL timeline -- used for Block 1).
    perm_earnings_sorted = (permaticker_earnings_full
                            .sort_values("report_date").reset_index(drop=True))
    if perm_earnings_sorted.empty:
        # Defensive -- shouldn't happen: gated events exist iff /earnings/raw
        # has rows for this permaTicker.
        return [], []

    # Gated events for this permaTicker, sorted by report_date.
    gated_for_pt_sorted = gated_for_pt.sort_values("report_date").reset_index(drop=True)

    # Pass A: compute car_60d_pass1 for ALL earnings rows of this permaTicker
    # (gated or not -- for car_drift_historical_q1 shift accuracy).
    per_event_car_60d: dict[pd.Timestamp, float] = {}
    for _, erow in perm_earnings_sorted.iterrows():
        rdate = pd.Timestamp(erow["report_date"])
        t_pos = match_T(stock_dates_np, rdate)
        if t_pos is None:
            per_event_car_60d[rdate] = np.nan
            continue
        car60 = compute_car_window(stock_ret_full, ijh_ret_full, t_pos, +1, CAR_60D_END_OFFSET)
        per_event_car_60d[rdate] = car60
        # (car_10d is computed per gated event below -- we don't double-compute)

    # Build a per-report_date lookup for before_after_market.
    bam_lookup = pd.Series(
        perm_earnings_sorted["before_after_market"].values,
        index=pd.to_datetime(perm_earnings_sorted["report_date"]),
    )

    # Compute per-event Block 2/3 features + car_10d for GATED events.
    gated_payloads: list[dict] = []
    t_failures: list[tuple[str, pd.Timestamp]] = []
    for _, grow in gated_for_pt_sorted.iterrows():
        rdate = pd.Timestamp(grow["report_date"])
        t_pos = match_T(stock_dates_np, rdate)
        if t_pos is None:
            # T-match failure: drop row + log (features.md §4 explicit exception).
            t_failures.append((permaTicker, rdate))
            continue
        car10 = compute_car_window(stock_ret_full, ijh_ret_full, t_pos, +1, CAR_10D_END_OFFSET)
        car60 = per_event_car_60d.get(rdate, np.nan)
        bam = bam_lookup.get(rdate, None)

        block2 = compute_block2_features(aligned, stock_ret_full, ijh_ret_full, t_pos, bam)
        block3 = compute_block3_features(aligned, t_pos)

        # Phase H: analyst revision momentum (Sunday-safe, from FMP /stable/grades)
        rev_mom = compute_revision_momentum(grades_df, rdate)

        row_dict = {
            "permaTicker": permaTicker,
            "canonical_ticker": canonical,
            "cik": grow.get("cik", None),
            "report_date": rdate,
            "T": aligned.index[t_pos],
            "calendar_week_group": grow["calendar_week_group"],
            "added": pd.Timestamp(grow["added"]) if pd.notna(grow.get("added")) else pd.NaT,
            "car_10d": car10,
            "car_60d_pass1": car60,
            **block2,
            **block3,
            **rev_mom,
        }
        gated_payloads.append(row_dict)

    if not gated_payloads:
        return [], t_failures

    # Pass B: Block 1 on the FULL earnings timeline (rolling 12Q std).
    car_60d_full = pd.Series(
        [per_event_car_60d.get(pd.Timestamp(rd), np.nan)
         for rd in perm_earnings_sorted["report_date"]],
        index=perm_earnings_sorted.index,
    )
    block1_full = compute_block1_features_per_perm_id(perm_earnings_sorted, car_60d_full)
    block1_indexed_by_date = block1_full.set_index(
        pd.to_datetime(perm_earnings_sorted["report_date"])
    )

    # Merge Block 1 + Block 4 interaction into each gated payload.
    block1_cols = ["sue_score", "eps_surprise_pct", "consecutive_surprises",
                   "sue_acceleration", "sue_lag_1", "sue_lag_2",
                   "car_drift_historical_q1"]
    for row_dict in gated_payloads:
        rdate = row_dict["report_date"]
        # rdate is pd.Timestamp above; block1 indexed by report_date (also TS).
        if rdate in block1_indexed_by_date.index:
            blk1 = block1_indexed_by_date.loc[rdate]
            # Defensive: if duplicate report_date in earnings, blk1 may be a
            # multi-row DataFrame; coerce to a single Series by taking the
            # first row (SUE rolling 12Q is order-sensitive but duplicates
            # with the SAME report_date are rare calendar collisions).
            if isinstance(blk1, pd.DataFrame):
                blk1 = blk1.iloc[0]
            for col in block1_cols:
                row_dict[col] = blk1[col]
        else:
            for col in block1_cols:
                row_dict[col] = np.nan

        # Block 4 interaction.
        sue_score_val = row_dict.get("sue_score")
        if isinstance(sue_score_val, pd.Series):
            sue_score_val = sue_score_val.iloc[0]
        idio_v = row_dict.get("pre_event_idiosyncratic_vol")
        if isinstance(idio_v, pd.Series):
            idio_v = idio_v.iloc[0]
        if (sue_score_val is None or pd.isna(sue_score_val)
            or idio_v is None or pd.isna(idio_v) or idio_v == 0):
            row_dict["sue_abs_x_inverse_vol"] = np.nan
        else:
            row_dict["sue_abs_x_inverse_vol"] = float(abs(sue_score_val) / float(idio_v))

    return gated_payloads, t_failures


# ==============================================================================
# PART 6 — STORAGE
# ==============================================================================
def write_train_matrix(df: pd.DataFrame, key: str = TRAIN_MATRIX_KEY) -> None:
    """Persist /features/train_matrix to DB_FILE.

    Safety: HDFStore(mode='a'); store.remove(key) first if exists. NEVER mode='w'.
    data_columns for Stage 3 (training) walk-forward query speed.
    """
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if key in store.keys():
            store.remove(key)
        store.put(
            key,
            df,
            format="table",
            data_columns=["calendar_week_group", "permaTicker",
                           "canonical_ticker", "report_date"],
        )


# ==============================================================================
# PART 7 — BUILD REPORT
# ==============================================================================
def print_build_report(
    df: pd.DataFrame,
    t_match_failures: list[tuple[str, pd.Timestamp]],
    pass_total_seconds: float,
) -> None:
    bar = "=" * 70
    print(bar)
    print("STAGE 2 — FEATURE MATRIX BUILD REPORT (Phase E)")
    print(bar)
    print(f"  Rows in train_matrix:        {len(df):,}")
    print(f"  Columns:                      {df.shape[1]}")
    print(f"  Distinct permaTickers:        {df['permaTicker'].nunique() if len(df) else 0}")
    print(f"  Distinct canonical_tickers:   {df['canonical_ticker'].nunique() if len(df) else 0}")
    print(f"  Distinct calendar weeks:      {df['calendar_week_group'].nunique() if len(df) else 0}")
    print(f"  Date range:                   {df['report_date'].min()} -> {df['report_date'].max()}")
    print(f"  T-match failures (drops):     {len(t_match_failures)}")
    if t_match_failures:
        print(f"  First 10 T-match failures (permaTicker, report_date):")
        for pt_, rd in t_match_failures[:10]:
            print(f"    {pt_}  {rd.date()}")
    print(f"  Build wall-clock elapsed:     {pass_total_seconds:.1f}s")

    if len(df):
        print()
        print("Per-feature non-NaN coverage (NaNs allowed per features.md sec 4; just audit):")
        feature_cols = [
            "sue_score", "eps_surprise_pct", "consecutive_surprises",
            "sue_acceleration", "sue_lag_1", "sue_lag_2",
            "car_drift_historical_q1",
            "is_bmo",
            "volume_vma20_ratio_pre_event", "suv_day_1",
            "pre_event_idiosyncratic_vol", "opening_gap_t1",
            "intraday_range_t", "pre_event_volume_trend",
            "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d",
            "rel_ret_30d", "sector_adjusted_ret_20d",
            "sue_abs_x_inverse_vol",
            "car_10d", "car_60d_pass1",
            # Phase H: analyst revision momentum (Sunday-safe, FMP data)
            "revision_momentum_30d", "revision_momentum_60d",
            "revision_momentum_90d", "revision_ordinal_momentum_90d",
            "revision_intensity_90d", "grade_dispersion_90d",
            "n_analysts_covering", "last_action_days_before_earnings",
        ]
        for col in feature_cols:
            if col not in df.columns:
                continue
            non_null = df[col].notna().sum()
            frac = non_null / len(df) * 100
            flag = " <-- LOW!" if frac < 70 and col != "car_drift_historical_q1" else ""
            print(f"  {col:35s} {non_null:>6d} / {len(df):<6d}  ({frac:5.1f}%){flag}")
    print(bar)


# ==============================================================================
# MAIN
# ==============================================================================
def main(dry_run: bool = False, limit: int | None = None) -> int:
    bar = "=" * 70
    print(bar)
    print("STAGE 2 -- Feature Matrix Builder  [Phase E v2: permaTicker-keyed]")
    print(f"DB file: {DB_FILE}")
    print(f"Output key: {TRAIN_MATRIX_KEY}  {'(dry-run, no write)' if dry_run else ''}")
    if limit is not None:
        print(f"Limit: processing first {limit} permaTickers only.")
    print(bar)

    t0 = time.time()

    print("[1/6] Loading inputs ...")
    gated_df = load_gated_events()
    permatickers_meta = load_permatickers_map()
    earnings_full = load_earnings_full()
    print(
        f"  gated: {len(gated_df):,} rows | permatickers: {len(permatickers_meta):,} "
        f"| earnings full: {len(earnings_full):,}"
    )

    # Pre-process permatickers meta: filter to permaTickers that appear in
    # gated events (skip permaTickers with no gated events -- safe optimization).
    gated_pt_set = set(gated_df["permaTicker"].unique())
    permatickers_meta = permatickers_meta[permatickers_meta["permaTicker"].isin(gated_pt_set)]
    # Index by permaTicker for O(1) metadata lookup.
    permatickers_meta = permatickers_meta.set_index("permaTicker", drop=False)

    print("[2/6] Loading per-permaTicker earnings slices (full timeline) ...")
    # Group /earnings/raw by permaTicker ONCE.
    earnings_full = earnings_full.sort_values(["permaTicker", "report_date"]).reset_index(drop=True)
    earnings_by_pt: dict[str, pd.DataFrame] = {
        pt_: g.reset_index(drop=True)
        for pt_, g in earnings_full.groupby("permaTicker")
    }

    print("[3/6] Pre-loading benchmarks (IJH + sector ETFs) ...")
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if IJH_KEY not in store.keys():
            raise FileNotFoundError(f"Missing {IJH_KEY}. Run 04_index_data_gathering.py first.")
    ijh_df = load_benchmark_prices(IJH_KEY)
    benchmark_prices_cache: dict[str, pd.DataFrame] = {IJH_KEY: ijh_df}
    # Pre-load all sector ETFs that will be used (cache hit).
    sector_tickers_used = set(permatickers_meta["index_ref"].dropna().unique().tolist())
    sector_tickers_used.add(DEFAULT_SECTOR_INDEX)
    print(f"  Sector ETFs in use: {sorted(sector_tickers_used)}")
    for st in sector_tickers_used:
        sk = _sector_key(st)
        if sk not in benchmark_prices_cache:
            benchmark_prices_cache[sk] = load_benchmark_prices(sk)
    print(f"  Benchmark cache loaded: {sorted(benchmark_prices_cache.keys())}")

    # We process per permaTicker, sorted alphabetically for reproducibility.
    permatickers_to_process = sorted(permatickers_meta["permaTicker"].unique().tolist())
    if limit is not None:
        permatickers_to_process = permatickers_to_process[:limit]
    print(f"[4/6] Processing {len(permatickers_to_process)} permaTickers ...")

    all_rows: list[dict] = []
    all_t_failures: list[tuple[str, pd.Timestamp]] = []
    for i, pt_ in enumerate(permatickers_to_process, 1):
        rec = permatickers_meta.loc[pt_]
        canonical = rec["canonical_ticker"]
        index_ref = rec["index_ref"] if isinstance(rec.get("index_ref"), str) else DEFAULT_SECTOR_INDEX
        gated_for_pt = gated_df[gated_df["permaTicker"] == pt_]
        pt_earnings = earnings_by_pt.get(pt_, pd.DataFrame())
        if pt_earnings.empty:
            continue  # permaTicker had gated events but earnings_by_pt missing (defensive)

        chunk_rows, chunk_t_failures = process_permaticker(
            pt_,
            canonical,
            index_ref,
            gated_for_pt,
            pt_earnings,
            ijh_df,
            benchmark_prices_cache,
        )
        all_rows.extend(chunk_rows)
        all_t_failures.extend(chunk_t_failures)
        if i % 50 == 0 or i == len(permatickers_to_process):
            elapsed_so_far = time.time() - t0
            avg_per_pt = elapsed_so_far / i if i else 0
            eta = avg_per_pt * (len(permatickers_to_process) - i)
            print(
                f"  progress: {i}/{len(permatickers_to_process)} "
                f"(rows={len(all_rows):,}, t_fail={len(all_t_failures)}, "
                f"elapsed={elapsed_so_far:.0f}s, eta={eta:.0f}s)"
            )

    print("[5/6] Assembling train_matrix DataFrame ...")
    if not all_rows:
        print("  (empty matrix -- nothing to do)")
        print(bar)
        return 2
    matrix_df = pd.DataFrame(all_rows)
    # Sort by [calendar_week_group, permaTicker, report_date] for ranker contiguity.
    matrix_df = matrix_df.sort_values(
        ["calendar_week_group", "permaTicker", "report_date"]
    ).reset_index(drop=True)

    print(f"  Assembled: {len(matrix_df):,} rows, {matrix_df.shape[1]} columns")

    print("[6/6] Writing /features/train_matrix ...")
    if dry_run:
        print("  (--dry-run: NOT writing to db.h5)")
    else:
        write_train_matrix(matrix_df)
        print(f"  Wrote {len(matrix_df):,} rows to {TRAIN_MATRIX_KEY}")

    elapsed = time.time() - t0
    print_build_report(matrix_df, all_t_failures, elapsed)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 (v2): build the listwise-ranker training matrix "
            "(/features/train_matrix) from the gated events pool. "
            "(Phase E v2: permaTicker-keyed, per-permaTicker price processing.)"
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build in memory and print the report; do NOT write to db.h5.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N permaTickers (smoke testing).",
    )
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run, limit=args.limit))

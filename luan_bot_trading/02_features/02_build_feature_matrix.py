#!/usr/bin/env python3
"""
Stage 2: Feature Matrix Builder  (Phase E implementation)
==========================================================

PURPOSE
-------
Expand the Stage-1 gated events table (`/features/gated_events`, 21,269 rows,
7 columns including perm_id) into the full listwise-ranker training matrix
at `/features/train_matrix`:

  * 21 active features in `X` (Block 1-4 per `features.md` §1).
  * 7 mandatory non-feature metadata columns (Phase E adds `perm_id`).
  * 1 target label `car_10d` (continuous 10-day LOG CAR; NDCG target).
  * 1 Pass-1 audit column `car_60d_pass1` (used by Pass B for
    `car_drift_historical_q1`; also live-inference audit).

Total: 7 + 1 + 1 + 21 = 30 columns phase E (matched v1's count + perm_id
expansion).

INPUTS (from `01_data/db.h5`; READ-ONLY; zero external API calls)
-----------------------------------------------------------------
  * `/features/gated_events`        — Stage-1 output (Phase E schema):
        perm_id, canonical_ticker, cik, report_date, added, removed,
        calendar_week_group                                 (7 cols)
  * `/earnings/raw`                    — required for Block 1 (12Q rolling
        std over ALL prior quarters per perm_id). Phase D schema includes
        perm_id column so we group correctly.
  * `/sp400/{canonical_ticker}`        — 850+ individual price-series nodes
        (Date + Adj_Close/Adj_Volume etc, 11 cols). Loaded once per
        canonical in the per-canonical processing loop. The 12 canonical-
        collision pairs share the load (Phase B v2 stores union alias
        history here).
  * `/metadata/sp400_perm_ids`         — required for:
        perm_id -> canonical_ticker, cik, index_ref, combined_intervals
        (Phase A schema).
  * `/macros/IJH`                      — mid-cap benchmark for
        pre_event_idiosyncratic_vol and all rel_ret_*d features. Loaded ONCE
        (constant across all canonicals).
  * `/macros/{index_ref}`              — sector ETF nodes (per-perm_id's
        index_ref field, defaults to IJH). Loaded on first access; 8 distinct
        ETFs total: IJK, IJJ, XLF, XLB, IJS, XLU, XLRE, IJH (default).

OUTPUTS
-------
  * `/features/train_matrix` in `01_data/db.h5` (HDF5 table).
  * stdout build report including per-feature non-NaN coverage, T-match
    failure count, distinct group count, pass timings.

PASS STRATEGY (Phase E)
-----------------------
  Pass A — Per-canonical price-frame alignment + per-perm_id Pass-A CAR
           windows (car_10d for gated events; car_60d_pass1 for ALL
           per-perm_id earnings rows so car_drift_historical_q1 shift
           stays accurate for gated events whose prior event was non-gated).
           Also per-event Block 2/3 features computed here.

  Pass B — Per-perm_id Block 1 (rolling 12Q std over full earnings
           timeline) + Block 4 interaction (needs Pass A vol + Pass B sue).

  Pass C — Assemble row-major dataframe. Sort by
           ['calendar_week_group', 'perm_id', 'report_date']. Write.

§7.7 DISAMBIGUATION
-------------------
Stage 1 already pruned loser-perm_id events in their overlap zone (105
events, ~0.23% of raw). Stage 2 trusts gated_events as input; no re-filter
needed.

LOG UNITS (LOCKED — Design.md §17.4, features.md §1)
------------------------------------------------------
`car_10d` and `car_60d_pass1` (therefore `car_drift_historical_q1`) are
stored in LOG units. The arithmetic-percentage conversion happens at the
Stage-3 isotonic calibration bridge (`np.expm1` on the calibrator fit
target). NO conversion in Stage 2.

§12 PRIMING CUTOFF (features.md §0)
------------------------------------
Stage 2 stores ALL gated events (2012-03-31 onward). The §12 train-time
cutoff (report_date >= 2015-01-01) is the training script's
responsibility (Stage 3), not here. Same for the sparse-week cutoff
(<3 events/week).

CLI
---
    python luan_bot_trading/02_features/02_build_feature_matrix.py
    python luan_bot_trading/02_features/02_build_feature_matrix.py --dry-run
    python luan_bot_trading/02_features/02_build_feature_matrix.py --limit N
        # Process only first N canonicals (smoke testing). Default = all.

HDF5 WRITE SAFETY
-----------------
NEVER `mode='w'` on existing DB. Use `HDFStore(mode='a')` + `store.remove()`
if the target key already exists (per STOP_DOING_EXTRA_SHIT.md).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURATION
# ==============================================================================
DB_FILE = Path(__file__).resolve().parent.parent / "01_data" / "db.h5"

GATED_EVENTS_KEY = "/features/gated_events"
TRAIN_MATRIX_KEY = "/features/train_matrix"
EARNINGS_KEY = "/earnings/raw"
PERM_IDS_KEY = "/metadata/sp400_perm_ids"
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
SUE_MIN_PERIODS = 12        # see features.md §4


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


def load_perm_ids_map() -> pd.DataFrame:
    """Load /metadata/sp400_perm_ids (Phase A schema). Returns a DataFrame
    indexed by perm_id with columns: canonical_ticker, cik, index_ref,
    combined_intervals (parsed JSON list)."""
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if PERM_IDS_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {PERM_IDS_KEY} not found. Run 02b_build_company_map.py (Phase A) first."
            )
        df = store.get(PERM_IDS_KEY)
    # Use only the columns we need.
    return df[["perm_id", "canonical_ticker", "cik", "index_ref", "combined_intervals"]]


def load_earnings_full() -> pd.DataFrame:
    """Load the FULL /earnings/raw (Phase D schema). Required columns:
        report_date, fiscal_period_end, perm_id, canonical_ticker, cik,
        actual, estimate, difference, percent, before_after_market, currency
    """
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if EARNINGS_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {EARNINGS_KEY} not found. Run 06_earnings_gathering.py first."
            )
        return store.get(EARNINGS_KEY)


def load_stock_prices(canonical: str) -> pd.DataFrame:
    """Load /sp400/{canonical}. Returns columns (11):
        Date, Open, High, Low, Close, Volume,
        Adj_Open, Adj_High, Adj_Low, Adj_Close, Adj_Volume
    Date is tz-naive datetime64[us]; sorted ascending.
    """
    key = f"/sp400/{canonical}"
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if key not in store.keys():
            raise KeyError(
                f"Price node {key} missing. "
                f"This should never happen post Stage 1 (price_unavailable perm_ids "
                f"are excluded at the gate). Investigate."
            )
        df = store.get(key)
    # Defensive: ensure sorted by Date ascending.
    df = df.sort_values("Date").reset_index(drop=True)
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
        pre_event_volume_trend       : OLS slope of Adj_Volume over [T-10:T-1]

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

    # is_bmo
    bam = before_after_market_str
    is_bmo = 1 if (isinstance(bam, str) and bam.strip().lower() == "bmo") else 0

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

    # pre_event_volume_trend: OLS slope over [T-10:T-1] of Adj_Volume.
    pre_trend = _safe_slice(aligned["stock_Adj_Volume"], i - VOLUME_TREND_LOOKBACK, i - 1)
    if pre_trend is not None and pre_trend.notna().sum() >= 2:
        # Drop NaNs; use numeric index for day_index.
        valid = pre_trend.dropna()
        if len(valid) >= 2:
            x = np.arange(len(valid), dtype=float)
            y = valid.values.astype(float)
            # polyfit deg=1 returns [slope, intercept]
            try:
                slope = float(np.polyfit(x, y, 1)[0])
            except (np.linalg.LinAlgError, TypeError):
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


def compute_block1_features_per_perm_id(
    perm_earnings: pd.DataFrame,          # FULL per-perm_id earnings slice
                                          # sorted by report_date ascending.
                                          # Columns: report_date, fiscal_period_end,
                                          # actual, estimate, difference, percent,
                                          # before_after_market.
    car_60d_pass1_full: pd.Series,        # per-event 60d CAR aligned to
                                          # perm_earnings.index (NaN for events
                                          # that were never computed by Pass A —
                                          # such events shouldn't exist now
                                          # because Pass A computes car_60d_pass1
                                          # for ALL per-perm_id earnings rows,
                                          # but defensive NaN allowed).
) -> pd.DataFrame:
    """Compute all 7 Block-1 features for a perm_id over its FULL earnings
    timeline.

    Returns a DataFrame indexed by perm_earnings.index with 7 columns:
        sue_score, eps_surprise_pct, consecutive_surprises,
        sue_acceleration, sue_lag_1, sue_lag_2,
        car_drift_historical_q1
    """
    sue = _compute_sue_score(perm_earnings["difference"])
    eps_surprise_pct = perm_earnings["percent"]
    consecutive = _compute_consecutive_surprises(perm_earnings["actual"], perm_earnings["estimate"])
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
    }, index=perm_earnings.index)
    return out


# ==============================================================================
# PART 5 — PER-CANONICAL ORCHESTRATOR
# ==============================================================================
def process_canonical(
    canonical: str,
    gated_canonical_df: pd.DataFrame,        # gated events for THIS canonical
                                              # (perm_ids sharing it).
    canonical_perm_ids: pd.DataFrame,        # perm_ids metadata rows (with
                                              # index_ref per perm_id) sharing this
                                              # canonical.
    full_earnings_by_perm_id: dict[str, pd.DataFrame],
                                              # FULL per-perm_id earnings slice
                                              # for every perm_id (used for Block1
                                              # Rolling 12Q over all priors).
    ijh_df: pd.DataFrame,
    benchmark_prices_cache: dict[str, pd.DataFrame],
) -> tuple[list[dict], list[tuple[str, pd.Timestamp]]]:
    """Process one canonical's gated events end to end.

    Returns:
        row_dicts : list of dict rows (the canonical's chunk of train_matrix)
        t_failures: list of (perm_id, report_date) tuples where T-match failed
                    (these rows are dropped per NaN policy).
    """
    # Load stock prices once for this canonical.
    stock_df = load_stock_prices(canonical)
    stock_dates_np = stock_df["Date"].values  # sorted ascending (load_stock_prices)

    # Build the aligned frame (LEFT join stock.Date -> benchmarks).
    # The sector ETF is per-perm_id (permids may have different index_ref
    # under the same canonical-ticker if they came from different SIC codes).
    # So we can't bake the sector into the aligned frame; we keep it as a
    # separate dict per-perm_id and join in Block 3.  For simplicity we
    # defer the aligned-frame building to a per-perm_id call to align with
    # the sector that the perm_id uses.
    row_dicts: list[dict] = []
    t_failures: list[tuple[str, pd.Timestamp]] = []

    # We'll opt to build one aligned frame per perm_id with its own sector
    # ETF (cheap; the sector join is fast).
    for perm_id, perm_record in canonical_perm_ids.iterrows():
        # perm_record has: canonical_ticker, cik, index_ref, combined_intervals
        index_ref = perm_record["index_ref"]
        if not isinstance(index_ref, str) or not index_ref:
            index_ref = DEFAULT_SECTOR_INDEX
        sector_key = _sector_key(index_ref)
        if sector_key not in benchmark_prices_cache:
            benchmark_prices_cache[sector_key] = load_benchmark_prices(sector_key)
        sector_df = benchmark_prices_cache[sector_key]

        aligned = build_aligned_price_frame(stock_df, ijh_df, sector_df)
        # Pre-compute full-length log returns for Block 2 idio vol.
        stock_ret_full = _log_ret(pd.Series(aligned["stock_Adj_Close"].values, index=aligned.index))
        ijh_ret_full = _log_ret(pd.Series(aligned["ijh_Close"].values, index=aligned.index))

        # Earnings for this perm_id (FULL timeline -- used for Block 1).
        perm_earnings = full_earnings_by_perm_id.get(perm_id)
        if perm_earnings is None or perm_earnings.empty:
            # Defensive — shouldn't happen because the gated events for this
            # perm_id exist iff /earnings/raw has rows for it.
            continue

        # Pass A per-perm_id: car_60d_pass1 for ALL earnings rows of this
        # perm_id (gated or not — for car_drift_historical_q1 shift accuracy).
        # car_10d only for gated rows (it's the label; non-gated rows have no
        # use for it).
        gated_for_pid = gated_canonical_df[
            gated_canonical_df["perm_id"] == perm_id
        ].sort_values("report_date").reset_index(drop=True)

        # Map report_date -> aligned t_position. Use searchsorted on stock_dates_np.
        per_event_car_60d = {}     # report_date_TS -> float (car_60d_pass1; NaN if fail)

        # Iterate over the FULL perm_earnings timeline (sorted by report_date asc).
        perm_earnings_sorted = perm_earnings.sort_values("report_date").reset_index(drop=True)

        for _, erow in perm_earnings_sorted.iterrows():
            rdate = pd.Timestamp(erow["report_date"])
            t_pos = match_T(stock_dates_np, rdate)
            if t_pos is None:
                # T-match failure; this earnings row has no Pass-A car_60d.
                # For non-gated rows, this is silent (no row to drop). For gated
                # rows, this becomes a drop and is logged below.
                per_event_car_60d[rdate] = np.nan
                continue
            car60 = compute_car_window(stock_ret_full, ijh_ret_full, t_pos, +1, CAR_60D_END_OFFSET)
            per_event_car_60d[rdate] = car60
            # (car_10d is computed per gated event below — we don't double-compute here).

        # Now compute the per-event Block 2/3 features + car_10d for GATED events.
        gated_payloads: list[dict] = []
        for _, grow in gated_for_pid.iterrows():
            rdate = pd.Timestamp(grow["report_date"])
            t_pos = match_T(stock_dates_np, rdate)
            if t_pos is None:
                # T-match failure: drop row + log. NaN policy permits this
                # exact drop case (features.md §4).
                t_failures.append((perm_id, rdate))
                continue
            car10 = compute_car_window(stock_ret_full, ijh_ret_full, t_pos, +1, CAR_10D_END_OFFSET)
            car60 = per_event_car_60d.get(rdate, np.nan)

            # Find the `before_after_market` for this event in the perm_earnings
            # timeline (we need it for is_bmo).
            bam = None
            match_idx = perm_earnings_sorted.index[
                perm_earnings_sorted["report_date"] == rdate
            ]
            if len(match_idx) > 0:
                bam = perm_earnings_sorted.loc[match_idx[0], "before_after_market"]

            block2 = compute_block2_features(
                aligned, stock_ret_full, ijh_ret_full, t_pos, bam
            )
            block3 = compute_block3_features(aligned, t_pos)

            row_dict = {
                "perm_id": perm_id,
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
            }
            gated_payloads.append(row_dict)

        if not gated_payloads:
            continue

        # Pass B per-perm_id: Block 1 features on the FULL earnings timeline.
        # Build aligned per-perm_id car_60d_pass1 full series for shift.
        car_60d_full = pd.Series(
            [per_event_car_60d.get(pd.Timestamp(rd), np.nan)
             for rd in perm_earnings_sorted["report_date"]],
            index=perm_earnings_sorted.index,
        )
        block1_full = compute_block1_features_per_perm_id(
            perm_earnings_sorted, car_60d_full
        )

        # Inner-join: keep only gated rows. Index the block1 result by
        # report_date for lookup.
        block1_indexed_by_date = block1_full.set_index(
            perm_earnings_sorted["report_date"]
        )

        # Merge block1 features into each gated payload (by report_date).
        for row_dict in gated_payloads:
            rdate = row_dict["report_date"]
            if rdate in block1_indexed_by_date.index:
                blk1 = block1_indexed_by_date.loc[rdate]
                # blk1 may be a Series (single row match).
                for col in ["sue_score", "eps_surprise_pct", "consecutive_surprises",
                            "sue_acceleration", "sue_lag_1", "sue_lag_2",
                            "car_drift_historical_q1"]:
                    row_dict[col] = (blk1[col] if isinstance(blk1, pd.Series)
                                      else blk1[col])
            else:
                # Defensive: block1 had no entry for this report_date — fill NaN.
                for col in ["sue_score", "eps_surprise_pct", "consecutive_surprises",
                            "sue_acceleration", "sue_lag_1", "sue_lag_2",
                            "car_drift_historical_q1"]:
                    row_dict[col] = np.nan

            # Block 4 interaction.
            sue_abs = abs(row_dict["sue_score"]) if pd.notna(row_dict.get("sue_score")) else np.nan
            idio_v = row_dict.get("pre_event_idiosyncratic_vol")
            if (sue_abs is None or pd.isna(sue_abs)
                or idio_v is None or pd.isna(idio_v) or idio_v == 0):
                row_dict["sue_abs_x_inverse_vol"] = np.nan
            else:
                row_dict["sue_abs_x_inverse_vol"] = float(sue_abs / idio_v)

            row_dicts.append(row_dict)

    return row_dicts, t_failures


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
            data_columns=["calendar_week_group", "perm_id",
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
    print(f"  Distinct perm_ids:            {df['perm_id'].nunique() if len(df) else 0}")
    print(f"  Distinct canonical_tickers:   {df['canonical_ticker'].nunique() if len(df) else 0}")
    print(f"  Distinct calendar weeks:      {df['calendar_week_group'].nunique() if len(df) else 0}")
    print(f"  Date range:                   {df['report_date'].min()} -> {df['report_date'].max()}")
    print(f"  T-match failures (drops):     {len(t_match_failures)}")
    if t_match_failures:
        print(f"  First 10 T-match failures (perm_id, report_date):")
        for pid, rd in t_match_failures[:10]:
            print(f"    {pid}  {rd.date()}")
    print(f"  Build wall-clock elapsed:     {pass_total_seconds:.1f}s")

    if len(df):
        print()
        print("Per-feature non-NaN coverage (NaNs allowed per features.md §4; just audit):")
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
    print("STAGE 2 — Feature Matrix Builder  [Phase E implementation]")
    print(f"DB file: {DB_FILE}")
    print(f"Output key: {TRAIN_MATRIX_KEY}  {'(dry-run, no write)' if dry_run else ''}")
    if limit is not None:
        print(f"Limit: processing first {limit} canonicals only.")
    print(bar)

    t0 = time.time()

    print("[1/6] Loading inputs ...")
    gated_df = load_gated_events()
    perm_ids_meta = load_perm_ids_map()
    earnings_full = load_earnings_full()
    print(
        f"  gated: {len(gated_df):,} rows | perm_ids: {len(perm_ids_meta):,} "
        f"| earnings full: {len(earnings_full):,}"
    )

    # Pre-process perm_ids meta: index by perm_id.
    perm_ids_meta = perm_ids_meta.set_index("perm_id", drop=False)
    # Filter to perm_ids that appear in gated events (skip perm_ids with no
    # gated events -- safe optimization).
    gated_perm_id_set = set(gated_df["perm_id"].unique())
    perm_ids_meta = perm_ids_meta[perm_ids_meta["perm_id"].isin(gated_perm_id_set)]
    # Group perm_ids by canonical so we process per-canonical.
    canonical_grouped = perm_ids_meta.groupby("canonical_ticker")

    print("[2/6] Loading per-perm_id earnings slices (full timeline) ...")
    # Group /earnings/raw by perm_id ONCE.
    earnings_full = earnings_full.sort_values(["perm_id", "report_date"]).reset_index(drop=True)
    earnings_by_perm_id: dict[str, pd.DataFrame] = {
        pid: g.reset_index(drop=True)
        for pid, g in earnings_full.groupby("perm_id")
    }

    print("[3/6] Pre-loading benchmarks (IJH + sector ETFs) ...")
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if IJH_KEY not in store.keys():
            raise FileNotFoundError(f"Missing {IJH_KEY}. Run 04_index_data_gathering.py first.")
    ijh_df = load_benchmark_prices(IJH_KEY)
    benchmark_prices_cache: dict[str, pd.DataFrame] = {IJH_KEY: ijh_df}
    # Pre-load all 8 sector ETFs (cache hit -- they will be used).
    sector_tickers_used = set(perm_ids_meta["index_ref"].dropna().unique().tolist())
    sector_tickers_used.add(DEFAULT_SECTOR_INDEX)
    print(f"  Sector ETFs in use: {sorted(sector_tickers_used)}")
    for st in sector_tickers_used:
        sk = _sector_key(st)
        if sk not in benchmark_prices_cache:
            benchmark_prices_cache[sk] = load_benchmark_prices(sk)
    print(f"  Benchmark cache loaded: {sorted(benchmark_prices_cache.keys())}")

    # We process per canonical_ticker, sorted alphabetically for reproducibility.
    canonicals_to_process = sorted(canonical_grouped.groups.keys())
    if limit is not None:
        canonicals_to_process = canonicals_to_process[:limit]
    print(f"[4/6] Processing {len(canonicals_to_process)} canonicals ...")

    all_rows: list[dict] = []
    all_t_failures: list[tuple[str, pd.Timestamp]] = []
    elapsed_per_canonical = 0.0
    for i, canon in enumerate(canonicals_to_process, 1):
        canon_perm_ids_df = canonical_grouped.get_group(canon).set_index("perm_id", drop=False)
        gated_canonical = gated_df[gated_df["canonical_ticker"] == canon]
        chunk_rows, chunk_t_failures = process_canonical(
            canon,
            gated_canonical,
            canon_perm_ids_df,
            earnings_by_perm_id,
            ijh_df,
            benchmark_prices_cache,
        )
        all_rows.extend(chunk_rows)
        all_t_failures.extend(chunk_t_failures)
        if i % 50 == 0 or i == len(canonicals_to_process):
            elapsed_so_far = time.time() - t0
            avg_per_canon = elapsed_so_far / i if i else 0
            eta = avg_per_canon * (len(canonicals_to_process) - i)
            print(
                f"  progress: {i}/{len(canonicals_to_process)} "
                f"(rows={len(all_rows):,}, t_fail={len(all_t_failures)}, "
                f"elapsed={elapsed_so_far:.0f}s, eta={eta:.0f}s)"
            )

    print("[5/6] Assembling train_matrix DataFrame ...")
    if not all_rows:
        print("  (empty matrix — nothing to do)")
        print(bar)
        return 2
    matrix_df = pd.DataFrame(all_rows)
    # Sort by [calendar_week_group, perm_id, report_date] for ranker contiguity.
    matrix_df = matrix_df.sort_values(
        ["calendar_week_group", "perm_id", "report_date"]
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
            "Stage 2: build the listwise-ranker training matrix "
            "(/features/train_matrix) from the gated events pool. "
            "(Phase E: perm_id-keyed, per-canonical price processing.)"
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
        help="Process only the first N canonicals (smoke testing).",
    )
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run, limit=args.limit))

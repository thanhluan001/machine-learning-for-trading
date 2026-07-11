#!/usr/bin/env python3
"""
Stage 2: Feature Matrix Builder  (DESIGN / SKELETON ONLY)
=========================================================

STATUS: DESIGN ONLY — implementation deferred until plan is approved.
        This file currently contains the full functional spec, algorithm,
        pass structure, formulas, NaN edge cases, and function signatures
        with docstrings; the bodies are NOT yet implemented. After approval,
        fill in the bodies per the docstrings.

PURPOSE
-------
Expand the Stage-1 gated events table (`/features/gated_events`, 21,853 rows,
6 audit columns) into the **full listwise-ranker training matrix** at
`/features/train_matrix` containing:

  * **20 active features in `X`** (Blocks 1-4 per `features.md` §1; macros
    excluded per §3).
  * **6 mandatory non-feature metadata columns** (`canonical_ticker`, `cik`,
    `report_date`, `T`, `calendar_week_group`, `added`; see NOTES on `removed`
    and the `index_ref` join below).
  * **1 target label** `car_10d` (continuous 10-day **log CAR** — kept
    continuous for NDCG gain; see Design.md §17.1). Stored in log units
    (`Σ_{t=T+1}^{T+11} (log R_stock − log R_IJH)`). NDCG is rank-invariant
    under monotonic transforms, so log-units are mathematically safe for the
    ranker. The arithmetic-UNIT conversion happens at the Stage-3 isotonic
    calibration bridge (§17.4): `y_arith = np.expm1(car_10d_log)` BEFORE
    fitting the calibrator, after which the calibrator's output `mu` is in
    true per-position percentages and feeds Kelly with no further transform.

Total columns in `/features/train_matrix`:
    20 (X) + 6 (metadata) + 1 (car_10d target) + 1 (car_60d_pass1, see Pass A
    note) = 28 stored. The training loop drops `car_60d_pass1` and any
    non-feature metadata from `X` itself; they are stored for audit and
    Stage-2 re-derivation.

The training-time sparse-week cutoff (drop rows whose `calendar_week_group`
has fewer than 3 members) is applied by the TRAINING SCRIPT (`03_model/...`),
NOT here — Stage 2 stores every gated row.

§12 Priming-Runway Training Cutoff (applied at TRAINING time, not here):
    Stage 2 stores ALL gated rows (2012-03-31 onward). The training script
    in `03_model/...` applies the Design.md §12 priming-runway cut BEFORE
    building the DMatrix:
        train_df = train_df[train_df.report_date >= pd.Timestamp('2015-01-01')]
    This drops rows whose `y` (car_10d) date falls in the first 3 years of the
    global data window (2012-01-01 -> 2014-12-31) from `y` training — those
    rows are feature-priming only (they populate the `sue_score` 12Q rolling
    std for later rows in 2015+). The 3-year runway matches the longest
    actual feature lookback (`sue_score` 12Q). Per §12's "Index Membership
    vs. Feature Lookback Separation" clarification, this is a storage-node
    data-availability requirement, NOT an index-membership requirement.
    Stage 2 passes the 2012-2015 priming rows through so the training script's
    TimeSlice operations, rolling-12Q computation, and Block-1 lookback
    can find their prior-quarter context in storage.
    The §12 cut is applied BEFORE the sparse-week (<3 events) cutoff so both
    rules compose cleanly (sparse-week counts should be reprofiled after the
    §12 cut in `03_model`, not here).

INPUTS (from `01_data/db.h5`; READ-ONLY; zero external API calls)
-----------------------------------------------------------------
  * `/features/gated_events`           — Stage-1 output:
        canonical_ticker, cik, report_date, added, removed,
        calendar_week_group
    (21,853 rows).
  * `/earnings/raw`                    — required for Block 1:
        report_date, canonical_ticker, cik, actual, estimate, difference,
        percent, before_after_market
    (44,637 rows — but we filter to the union of canonicals that appear in
    gated_events, ~850 tickers).
  * `/sp400/{canonical_ticker}`        — 850 individual price-series nodes,
    each: Date, Open, High, Low, Close, Volume, Adj_Open, Adj_High, Adj_Low,
    Adj_Close, Adj_Volume
    (~3,768 rows each).
  * `/macros/IJH`                      — mid-cap benchmark for
    pre_event_idiosyncratic_vol and all rel_ret_*d features:
    Date, Open, High, Low, Close, Volume (6 cols, OHLCV raw; no adj needed
    since IJH is a continuous ETF with negligible splits/dividends).
  * `/macros/{index_ref}`              — 8 sector ETF nodes (IJK, IJJ, XLB,
    XLF, XLRE, XLU, IJS, plus IJH for default) — used by
    sector_adjusted_ret_20d.
  * `/metadata/sp400_companies`        — for the SIC-derived `index_ref` join:
    canonical_ticker, cik, index_ref.

OUTPUTS
-------
  * `/features/train_matrix` in `01_data/db.h5` (HDF5 table, `HDFStore(mode='a')`,
    overwrite via `store.remove` if key exists — see STOP_DOING_EXTRA_SHIT.md).
    Columns (exactly 28; components listed in STORAGE SCHEMA below).
  * stdout build report:
      - rows written, columns written
      - unique groups (`calendar_week_group` count)
      - per-feature non-null coverage (% non-NaN, used to verify no silent
        mass-NaN bug)
      - pass A / pass B wall-clock timings
      - T-matching failures (count of gated rows with no T >= report_date
        in the price series — these rows are DROPPED with a logged event_id)

BUILD STRATEGY: TWO-PASS BUILD
-----------------------------
Why two passes:
    `car_drift_historical_q1` (Block 1, #7) needs the prior event's
    POST-event 60-day CAR (window T_prev+1 -> T_prev+60). Computing that
    requires knowing T_prev+60 for every event before we can compute the
    Block-1 shift. So we split:
    Pass A — Compute every event's own post-event CAR windows:
        car_60d_pass1   = CAR(window  T+1 -> T+60, vs IJH)
                          (the 60-day post-event PEAD window — design.md §6)
        car_10d         = CAR(window  T+1 -> T+11, vs IJH)
                          (the ranker training target — design.md §6/§17)
    Pass B — Shift pass-A CAR downward per `canonical_ticker` (sorted by
        report_date ascending) to construct the historical features:
        car_drift_historical_q1 = car_60d_pass1.shift(1) per canonical
            (the previous event's 60-day drift signature, available at
             the time of the current event — no lookahead)

PASS A — PER-EVENT CAR WINDOWS + PRICE-MATCHED FEATURES
-------------------------------------------------------
For each canonical (one block of work; can be parallelized later):

    A1. Load `/sp400/{canonical}` and `/macros/IJH` into a single aligned
        DataFrame indexed by Date. Left-join on Date; missing IJH rows
        (holidays where canonical traded but IJH didn't, or vice versa)
        are forward-filled using the most recent prior value
        (ffill). We use log returns; first-value NaN is acceptable.

    A2. Load the canonical's earnings slice: `report_date, actual, estimate,
        difference, percent, before_after_market` (sorted by report_date asc).

    A3. For each gated event of this canonical:
        T = first date in `/sp400/{canonical}` where Date >= report_date
           (roll FORWARD per features.md §1 Block 2; if no such date, drop
            the event and log it).

        # Pass-A CAR windows (Pass-A output, becomes input to Pass B).
        # BOTH car_10d and car_60d_pass1 are stored in LOG units (sum of daily
        # log excess returns). NDCG is invariant to monotonic transforms of the
        # gain, so log-units preserve ranking training signal without ANY
        # approximation error. The arithmetic-percent conversion happens once,
        # at the Stage-3 isotonic calibration bridge (`np.expm1` on the calibrator
        # fit target); it is NOT done here. Car_drift_historical_q1 (Block 1) is
        # therefore also in log units (it's just shift(1) of car_60d_pass1).
        car_10d       = SUM(log_stock_ret - log_IJH_ret) over T+1..T+11
        car_60d_pass1 = SUM(log_stock_ret - log_IJH_ret) over T+1..T+60

        # Block 2 microstructure (computed here because we already loaded
        # the price slice and have T):
        is_bmo    = 1 if before_after_market == "Bmo" else 0
        volume_vma20_ratio_pre_event = Volume[T] / mean(Volume[T-20:T-1])
        suv_day_1 = Adj_Volume[T] / mean(Adj_Volume[T-20:T-1])
        pre_event_idiosyncratic_vol = std(stock_logret - IJH_logret,
                                          ddof=1) over T-20..T-1
        opening_gap_t1 = (Open[T+1] - Close[T]) / Close[T]
        intraday_range_t = (High[T] - Low[T]) / Close[T]
        pre_event_volume_trend = OLS_slope(Adj_Volume, day_index, T-10..T-1)

        # Block 3 relative returns vs IJH (mid-cap benchmark):
        for h in [3,5,10,20,30]:
            rel_ret_{h}d = log(stock Adj_Close[T-1] / Adj_Close[T-1-h])
                         - log(IJH  Close   [T-1] / Close   [T-1-h])

        # Block 3 relative returns vs sector index_ref (one feature):
        sector_idx = company's index_ref (default IJH if missing)
        sector_adjusted_ret_20d =
            log(stock Adj_Close[T-1] / Adj_Close[T-21])
          - log(sector_idx Close   [T-1] / Close   [T-21])

    Note: ALL log-stock-return formulas use `Adj_Close` (split/dividend-
    adjusted). Returns-vs-IJH and returns-vs-sector use IJH/sector `Close`
    (unadjusted) — IJH/sector ETFs have negligible corporate-action noise.

    Note on benchmark join timing: aligning IJH and sector ETFs to the
    stock's Date index via left-join + ffill guarantees we never forward-
    look (a missing IJH row on a stock trading day is filled from the
    PREVIOUS IJH row, not the next one).

PASS B — SUE FAMILY + HISTORICAL CAR DRIFT (per canonical, sorted asc)
-----------------------------------------------------------------------
For each canonical:

    B1. Sort the canonical's earnings slice by report_date ascending.
        (For Block-1 lookback we use ALL earnings of the canonical — not
        just gated ones — per features.md §1: "rolling 12-quarter std over
        ALL prior quarters of the company, not restricted to the SP400
        membership window". So we keep the full earnings series here, even
        events that didn't survive the gate.)

    B2. Compute `sue_score`:
        diff_series = difference.values
        # Option B per design.md §15: NaN estimates -> EODHD sets diff=0.0,
        # included in rolling std denominator. min_periods=12.
        rolling_std = diff_series.rolling(12, min_periods=12).std(ddof=1)
        sue_score  = diff_series / rolling_std
        (NaN for the first 11 quarters where rolling_std has no value;
         per features.md §4 this is the documented NaN case.)

    B3. Compute the sue family derivatives:
        sue_acceleration   = sue_score.diff()              # NaN at row[0]
        sue_lag_1          = sue_score.shift(1)            # per canonical
        sue_lag_2          = sue_score.shift(2)

    B4. consecutive_surprises (running counter, strict beat only):
        prev_beat_run = []
        run = 0
        for actual, estimate in zip(actual.values, estimate.values):
            if not pd.isna(actual) and not pd.isna(estimate) and actual > estimate:
                run += 1
            else:
                run = 0
            prev_beat_run.append(run)
        consecutive_surprises = Series(prev_beat_run)

    B5. eps_surprise_pct = `percent` from `/earnings/raw` (direct passthrough).

    B6. car_drift_historical_q1 = car_60d_pass1.shift(1)   # Pass-A 60d CAR
                                                            shifted per canonical

    B7. JOIN Pass-B Block-1 features onto the gated events subset:
        The gated events for this canonical are a subset of all earnings
        rows (only events within buffered intervals survive). Block-1
        per-canonical rolling features must be computed on the FULL
        earnings timeline (to roll correctly over ALL prior quarters),
        then indexed-down to just the gated events.

PASS C — INTERACTION TERM + ASSEMBLY
------------------------------------
After Pass A + Pass B, assemble the final row-major frame.

    C1. Compute Block 4 interaction:
        sue_abs_x_inverse_vol = abs(sue_score) /
            pre_event_idiosyncratic_vol
        # NaN-safe: if denominator is NaN or 0 -> NaN.

    C2. Assemble 28-column DataFrame in the canonical column order
        (see STORAGE SCHEMA below).

    C3. Sort by `['calendar_week_group', 'canonical_ticker']` per
        features.md §0 (ranker requires contiguous-group layout).

    C4. Persist at `/features/train_matrix` using HDFStore(mode='a') +
        store.remove() if key exists.

    C5. Print build report (per-feature non-null coverage; rows written;
        T-match failures count; groups count; pass timings).

STORAGE SCHEMA — `/features/train_matrix` (28 columns)
------------------------------------------------------
NON-FEATURE METADATA (6):
  canonical_ticker       : str
  cik                    : str
  report_date            : datetime64[us]
  T                      : datetime64[us]          # matched trading day
  calendar_week_group    : str                      # YYYY-Www LTR anchor
  added                  : datetime64[us]           # audit only

TARGET LABEL (1):
  car_10d                : float                    # T+1..T+11 vs IJH, IN LOG UNITS
                                                      #   (sum of daily log excess
                                                      #    returns).
                                                      # Conversion to arithmetic %
                                                      # happens at Stage-3 isotonic
                                                      # bridge via np.expm1()
                                                      # (Design.md §17.4).

PASS-1 CAR (kept for audit and live inference; NOT in X):
  car_60d_pass1          : float                    # T+1..T+60 vs IJH, IN LOG UNITS
                                                      #   (shifted by 1 in Block 1 to
                                                      #    make car_drift_historical_q1)

BLOCK 1 — CATALYST FUNDAMENTALS (7):
  sue_score              : float
  eps_surprise_pct       : float
  consecutive_surprises  : int
  sue_acceleration       : float
  sue_lag_1              : float
  sue_lag_2              : float
  car_drift_historical_q1 : float                   # = car_60d_pass1.shift(1)

BLOCK 2 — MICROSTRUCTURE (7):
  is_bmo                              : int
  volume_vma20_ratio_pre_event        : float
  suv_day_1                           : float
  pre_event_idiosyncratic_vol         : float
  opening_gap_t1                      : float
  intraday_range_t                    : float
  pre_event_volume_trend              : float

BLOCK 3 — RELATIVE RETURNS (6):
  rel_ret_3d, rel_ret_5d, rel_ret_10d, rel_ret_20d, rel_ret_30d : float
  sector_adjusted_ret_20d             : float

BLOCK 4 — INTERACTION (1):
  sue_abs_x_inverse_vol : float

  TOTAL: 6 + 1 + 1 + 7 + 7 + 6 + 1 = 29 columns

  (Draft above said 28; the audit column `car_60d_pass1` makes it 29. If
  you want to save space, the audit column can be dropped post-Pass-B and
  the script re-derives it from `car_drift_historical_q1.shift(-1)` when
  needed — but storing it is cheap and makes `car_drift_historical_q1`
  auditable. Default: store 29 columns.)

  UNIT NOTE: `car_10d` and `car_60d_pass1` (and therefore
  `car_drift_historical_q1`, which is `car_60d_pass1.shift(1)`) are all in
  **LOG units**. No arithmetic conversion is performed in Stage 2; the
  conversion to per-position percentages is the responsibility of the
  Stage-3 isotonic calibration bridge (`Design.md` §17.4) and is a single
  `np.expm1` call on the calibrator fit target.

T-MATCHING & EDGE CASES (NaN policy per features.md §4)
-------------------------------------------------------
* T = first Date >= report_date in `/sp400/{canonical}`. If NO such date
  exists in the price series (event beyond the last stored trading day),
  DROP the row and log the (canonical, report_date) pair to a T-match
  failure list. This is the ONE case where rows are dropped, because no
  feature could conceivably be computed without T.
* T+1, T+11, T+60, T-20, T-21, T-1-h: all offsets are in TRADING DAYS
  (relative to the canonical's own `Date` index). No calendar math.
* If `T+11`, `T+60`, or `T+1` falls beyond the price-series end,
  `car_10d`, `car_60d_pass1`, or `opening_gap_t1` evaluates to NaN
  (no row drop — features.md §4 policy).
* If a trailing window (T-20..T-1, T-10..T-1) extends before the
  price-series start, window-slice operates on whatever exists; if fewer
  than 2 valid points -> std() -> NaN; if fewer than 2 points -> OLS slope
  -> NaN. No row drop.
* `sue_score` NaN for the first 11 quarters of a canonical's earnings
  history — documented case (features.md §4). Row survives.
* `pre_event_idiosyncratic_vol` NaN if fewer than 2 valid residual-return
  observations in T-20..T-1 — documented case. Row survives.

WHY TWO PASSES (formal proof of need)
-------------------------------------
The training matrix at row (canonical_i, event_t) needs:
  Block 1 feature #7 `car_drift_historical_q1` = the 60-day CAR of event
  (canonical_i, event_{t-1}).
Computing event_{t-1}'s 60-day CAR requires knowing date T_{t-1}+60,
which is itself computed in Pass A from the price series. So the
dependency chain is:

      Pass A: per-event CAR windows (T+1..T+11 and T+1..T+60) need T.
      Pass B: car_drift_historical_q1 = shift(1) of Pass-A 60d CAR.

You cannot compute the Block-1 features in Pass A alone, because the
shift operation needs the full Pass-A 60d CAR series first. Hence: split.

BUILD ORDER (within canonical) — RECOMMENDED PIPELINE
-----------------------------------------------------
For memory efficiency (~850 canonicals × ~3,800 price rows × ~13 cols is
light; we can hold everything in RAM, but cleaner to process per-canonical):

    1. Load gated_events (full frame, 21,853 rows).
    2. Resolve canonical_ticker -> index_ref join (one-shot) from
       /metadata/sp400_companies; default to IJH where missing.
    3. Load ALL of /earnings/raw (or filtered to gated canonicals — we need
       the FULL per-canonical earnings timeline for Block 1 rolling std).
    4. Per-canonical loop (Pass A + Pass B + assemble):
         A. Load /sp400/{canonical} + /macros/IJH + /macros/{index_ref}.
         B. Build aligned price frame (left-join on Date, ffill benchmark).
         C. Gate-loop the canonical's gated events -> compute Pass-A CAR
            windows + Block 2 + Block 3 features.
         D. Compute Block 1 features on the canonical's FULL earnings
            timeline (for the 12Q rolling std), then index-down to gated.
         E. Assemble the canonical's row chunk.
    5. (No need for a separate Pass B — the per-canonical nature of
       Block 1 means we already have car_60d_pass1 to shift.)
Concatenate -> sort -> store.

CLI
---
    python luan_bot_trading/02_features/build_feature_matrix.py
        # Build full train_matrix and write to db.h5.
    python luan_bot_trading/02_features/build_feature_matrix.py --dry-run
        # Build in memory and print the report; do NOT write.
    python luan_bot_trading/02_features/build_feature_matrix.py --limit N
        # Process only first N canonicals (for smoke-testing; default = all).

HDF5 WRITE SAFETY
-----------------
Per STOP_DOING_EXTRA_SHIT.md and the team convention: NEVER open `db.h5`
with `mode='w'`. Always use `HDFStore(mode='a')` and, if the target node
already exists, `store.remove('/features/train_matrix')` before
`store.put(...)`. Use `data_columns=['calendar_week_group',
'canonical_ticker', 'report_date']` for Stage-3 (training) query speed.
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
COMPANIES_KEY = "/metadata/sp400_companies"
EARNINGS_KEY = "/earnings/raw"
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
    """
    Map a sector ticker (e.g. 'IJK') to its HDF5 key under /macros.
    Centralized so we can change the path prefix in one place.
    """
    return f"/macros/{ticker}"


# ==============================================================================
# PART 1 — LOADERS
# ==============================================================================
def load_gated_events() -> pd.DataFrame:
    """
    Load Stage-1 /features/gated_events. Columns:
        canonical_ticker, cik, report_date, added, removed, calendar_week_group
    Sorted by stage 1 as ['calendar_week_group', 'canonical_ticker'];
    we will re-sort at storage time anyway.

    Raises FileNotFoundError if DB_FILE or the key is missing.
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


def load_index_ref_map() -> pd.Series:
    """
    Build a `pd.Series` mapping `canonical_ticker -> index_ref`
    from /metadata/sp400_companies. Missing/None index_ref values map
    to DEFAULT_SECTOR_INDEX ('IJH') per features.md §1 Block 3.

    Returns:  pd.Series(index=canonical_ticker, values=index_ref ticker)

    Raised as a Series (not a dict) so the per-canonical loop can do
        idx = idx_ref_map.get(canonical, DEFAULT_SECTOR_INDEX)
    with one dict-like lookup.
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


def load_earnings_full() -> pd.DataFrame:
    """
    Load the FULL /earnings/raw (all 44,637 rows). Even rows that didn't
    survive Stage-1 gating are needed here — Block-1's rolling 12Q std is
    computed over ALL prior quarters of the company, regardless of SP400
    membership window.

    Required columns:
        report_date, canonical_ticker, actual, estimate, difference,
        percent, before_after_market
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


def load_stock_prices(canonical: str) -> pd.DataFrame:
    """
    Load /sp400/{canonical}. Raises KeyError (gated-event-time error) if
    the node is missing — this should never happen because Stage 1 already
    filtered out price_unavailable=True companies.

    Returns columns (exactly 11):
        Date, Open, High, Low, Close, Volume,
        Adj_Open, Adj_High, Adj_Low, Adj_Close, Adj_Volume

    Index is reset to a RangeIndex; Date is a column.
    Date is timezone-naive datetime64[us].
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


def load_benchmark_prices(macro_key: str) -> pd.DataFrame:
    """
    Generic loader for an index/benchmark OHLCV table under /macros/{name}.
    Used for /macros/IJH (always) and /macros/{index_ref} (per-canonical).

    Returns columns: Date, Open, High, Low, Close, Volume (6-col schema).
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


# ==============================================================================
# PART 2 — ALIGNMENT & RETURN HELPERS
# ==============================================================================
def build_aligned_price_frame(
    stock_df: pd.DataFrame,
    ijh_df: pd.DataFrame,
    sector_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a single indexed frame aligned on the STOCK's `Date` index:

        Date | stock_Open | stock_High | ... | stock_Adj_Close | stock_Adj_Volume
            | ijh_Close  | sector_Close |   # LEFT join stock Date to benchmarks

    Left-join strategy:
        - stock's Date column is the master index.
        - ijh.rename(columns=...) -> left join on Date; ffill missing.
        - sector.rename(columns=...) -> left join on Date; ffill missing.

    Notes:
        - We use `Close` (raw) for IJH/sector because IJH/sector ETFs have
          negligible corporate-action noise — the macros tables store only
          6-col OHLCV (no adj_* schema).
        - We use `Adj_Close` and `Adj_Volume` for the STOCK because PEAD
          requires split/dividend adjustment for log-return continuity.

    Returns:  DataFrame indexed by Date, with these columns (NOT precomputed
    log-returns — those are computed in the per-event functions to keep this
    function unit-testable):
        stock_Open, stock_High, stock_Low, stock_Close, stock_Volume,
        stock_Adj_Close, stock_Adj_Volume,
        ijh_Close, sector_Close

    (Sector and IJH raw OHLC not needed beyond Close — only the stock's full
    OHLCV is used for opening_gap_t1, intraday_range_t.)
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


def _log_ret(series: pd.Series) -> pd.Series:
    """
    log(s[t] / s[t-1])  — natural log return. NaN-safe; returns NaN
    at lead position.

    TODO: np.log(series / series.shift(1))
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


# ==============================================================================
# PART 3 — T-MATCH + PER-EVENT FEATURE COMPUTERS (PASS A)
# ==============================================================================
def match_T(stock_df: pd.DataFrame, report_date: pd.Timestamp) -> pd.Timestamp | None:
    """
    Return the first Date in stock_df where Date >= report_date (roll FORWARD
    per features.md §1 Block 2). Return None if no such date exists (event
    beyond the price-series end); the caller drops the row and logs it.

    Implementation note: pre-sort stock_df by Date ONCE at load time, then
    use searchsorted(side='left', report_date) — O(log N) per lookup.
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


def compute_car_window(
    stock_ret: pd.Series,           # log_ret of stock aligned to its Date index
    benchmark_ret: pd.Series,       # log_ret of IJH aligned to stock's Date index
    t_position: int,                # integer index of T in the aligned frame
    start_offset: int,              # +1 (always; window starts day after T)
    end_offset: int,                # +11 for car_10d, +60 for car_60d_pass1
) -> float:
    """
    CAR = SUM[ log_stock_ret - log_benchmark_ret ]  over trading days
    [t_position + start_offset, t_position + end_offset]. Returns NaN if
    the window extends beyond the price-series end (no row drop).

    UNITS: the returned value is in **LOG units** (a log CAR), NOT plain
    percentages. NDCG is rank-invariant under monotonic transforms so the
    ranker can train directly on this value; the arithmetic-percent
    conversion is deferred to the Stage-3 isotonic calibration bridge via
    `np.expm1(car_log)` (Design.md §17.4). Do NOT convert inside this
    function — keeping CAR in log space here preserves numerical precision
    for tiny-magnitude returns and avoids per-canonical asymmetry.

    start_offset is always +1 (per design.md §6); kept as parameter for
    testability and to keep the CAR-window arithmetic explicit in the caller.
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


def compute_block2_features(
    aligned: pd.DataFrame,
    t_position: int,
    is_bmo_raw: str,
) -> dict:
    """
    Compute the 7 Block-2 features for a single event.

    Args:
        aligned      : output of build_aligned_price_frame()
        t_position   : integer position of T in `aligned`
        is_bmo_raw   : raw `before_after_market` string for this event
                       (e.g. 'Bmo', 'Amc', NaN)

    Returns dict with keys (all features.md §1 Block 2 definitions):
        is_bmo                       : int
        volume_vma20_ratio_pre_event : float | NaN
        suv_day_1                    : float | NaN
        pre_event_idiosyncratic_vol  : float | NaN
        opening_gap_t1               : float | NaN
        intraday_range_t             : float | NaN
        pre_event_volume_trend       : float | NaN

    NaN-safe per features.md §4: never raises; missing lookback data yields
    NaN. Indices used (all positions, i = t_position):
        Volume[T], Volume[T-20:T-1]                 (volume_vma20_ratio_pre_event)
        Adj_Volume[T], mean(Adj_Volume[T-20:T-1])   (suv_day_1)
        residuals = log_stock_ret - log_ijh_ret,
            std(residuals[T-20:T-1], ddof=1)         (pre_event_idiosyncratic_vol)
        Open[T+1], Close[T]                          (opening_gap_t1)
        High[T], Low[T], Close[T]                   (intraday_range_t)
        Adj_Volume[T-10:T-1]                         (pre_event_volume_trend)
            -> OLS slope via np.polyfit(day_idx, adj_vol, 1)[0]
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


def compute_block3_features(
    aligned: pd.DataFrame,
    t_position: int,
) -> dict:
    """
    Compute the 6 Block-3 relative-return features for a single event.

    rel_ret_{3,5,10,20,30}d
        = log(stock_Adj_Close[T-1] / stock_Adj_Close[T-1-h])
        - log(ijh_Close       [T-1] / ijh_Close       [T-1-h])

    sector_adjusted_ret_20d
        = log(stock_Adj_Close[T-1] / stock_Adj_Close[T-21])
        - log(sector_Close      [T-1] / sector_Close      [T-21])

    Returns dict with keys:
        rel_ret_3d, rel_ret_5d, rel_ret_10d, rel_ret_20d, rel_ret_30d,
        sector_adjusted_ret_20d

    NaN-safe. All indices are trading-day positions relative to t_position,
    not calendar days.
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


# ==============================================================================
# PART 4 — BLOCK 1 + HISTORICAL CAR DRIFT (PASS B)
# ==============================================================================
def _compute_sue_score(
    difference: pd.Series,
) -> pd.Series:
    """
    sue_score = difference /
        rolling_12Q_std(difference, min_periods=12, ddof=1)

    Per features.md §1 and design.md §15 Option B:
        - NaN estimates -> EODHD sets difference=0.0 already.
        - Include the 0.0 values in the rolling std denominator.
        - min_periods=12 (first 11 quarters -> NaN).
        - Use ALL prior quarters (no SP400-window restriction) — the caller
          must pass the FULL per-canonical `difference` series here.

    TODO:
        roll = difference.rolling(window=SUE_ROLLING_WINDOW,
                                   min_periods=SUE_MIN_PERIODS).std(ddof=1)
        return difference / roll
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


def _compute_consecutive_surprises(
    actual: pd.Series,
    estimate: pd.Series,
) -> pd.Series:
    """
    Strict-beat running counter per features.md §1:
        +1 if actual > estimate, else reset to 0.
    NaN actual or estimate -> reset to 0 (treated as non-beat).

    Implemented as a small Python loop on the underlying ndarray for
    clarity: pandas cumsum-with-reset tricks obscure the semantics.
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


def compute_block1_features_per_canonical(
    canonical_earnings: pd.DataFrame,    # full per-canonical earnings slice
                                          # sorted by report_date ascending
    car_60d_pass1_canonical: pd.Series,  # the per-event 60d CAR series
                                          # ALIGNED to canonical_earnings rows
                                          # by index (so shift(1) is correct)
) -> pd.DataFrame:
    """
    Compute all 7 Block-1 features for a canonical over its FULL earnings
    timeline.

    Returns a DataFrame indexed by canonical_earnings.index with 7 columns:
        sue_score, eps_surprise_pct, consecutive_surprises,
        sue_acceleration, sue_lag_1, sue_lag_2,
        car_drift_historical_q1

    Algorithm:
        # 1. sue_score (rolling 12Q std over all prior quarters)
        sue_score = _compute_sue_score(canonical_earnings['difference'])

        # 2. eps_surprise_pct = direct passthrough of `percent`
        eps_surprise_pct = canonical_earnings['percent']

        # 3. consecutive_surprises (strict-beat run counter)
        consecutive_surprises = _compute_consecutive_surprises(actual, estimate)

        # 4. sue derivatives
        sue_acceleration = sue_score.diff()             # NaN at row[0]
        sue_lag_1        = sue_score.shift(1)
        sue_lag_2        = sue_score.shift(2)

        # 5. car_drift_historical_q1 = previous event's 60d post CAR (shift(1))
        car_drift_historical_q1 = car_60d_pass1_canonical.shift(1)

    IMPORTANT:
        - `canonical_earnings` here is the FULL per-canonical earnings slice
          (NOT just gated events). The 12Q rolling std must see all prior
          quarters, gated or not — features.md §1.
        - `car_60d_pass1_canonical` is the per-event 60d CAR series aligned
          to the full earnings index. For events that failed gating (and thus
          have no Pass-A car_60d_pass1 computed), this is NaN — but those
          events are non-contributors anyway (they'll be filtered out by the
          gated-events merge at the end of Pass B).
        - The OUTPUT rows are merged into the gated-events row-major frame
          via an inner join on (canonical_ticker, report_date) at the end of
          Pass B per-canonical; the rolling-std neighbors in pure-earnings
          space stay in `canonical_earnings` only for rolling lookback computation.
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


# ==============================================================================
# PART 5 — PER-CANONICAL ORCHESTRATOR
# ==============================================================================
def process_canonical(
    canonical: str,
    gated_canonical_df: pd.DataFrame,   # this canonical's slice of /features/gated_events
    company_idx_ref: str,               # IJK | IJJ | ... | IJH (default)
    all_earnings_canonical: pd.DataFrame, # full per-canonical earnings slice
                                          # sorted by report_date asc
    stock_prices_cache: dict[str, pd.DataFrame],   # canonical -> /sp400/{canonical}
    benchmark_prices_cache: dict[str, pd.DataFrame],  # macro_key -> /macros/{name}
) -> pd.DataFrame:
    """
    Process one canonical's gated events end to end.

    Steps:
        1. stock_df = stock_prices_cache[canonical]                (preloaded)
        2. ijh_df   = benchmark_prices_cache[IJH_KEY]             (preloaded)
        3. sector_df = benchmark_prices_cache[sector_key(company_idx_ref)]
        4. aligned = build_aligned_price_frame(stock_df, ijh_df, sector_df)
        5. stock_ret    = _log_ret(aligned.stock_Adj_Close)
           ijh_log_ret  = _log_ret(aligned.ijh_Close)
           sector_log_ret = _log_ret(aligned.sector_Close)
        6. For each gated event in gated_canonical_df (sorted by report_date):
              T = match_T(stock_df, event.report_date)
              if T is None:
                  log T-match failure; continue  (row drop)
              car_10d       = compute_car_window(stock_ret, ijh_log_ret,
                                                 t_position, +1, +11)
              car_60d_pass1 = compute_car_window(stock_ret, ijh_log_ret,
                                                 t_position, +1, +60)
              block2 = compute_block2_features(aligned, t_position,
                                               event.before_after_market)
              block3 = compute_block3_features(aligned, t_position)
              accumulate (event_report_date, car_10d, car_60d_pass1, block2, block3)
        7. car_60d_pass1_canonical = align per-event car_60d_pass1 values to the
           full per-canonical earnings index by report_date (NaN for non-gated):
                car_60d_pass1_canonical = full_idx.map(per_event_car_map)
           block1 = compute_block1_features_per_canonical(all_earnings_canonical,
                                                          car_60d_pass1_canonical)
        8. inner-join block1 onto the gated per-event frame on report_date
           (so only gated events survive; the full-earnings rolling neighbors
            are not in the output).
        9. compute Block-4 interaction:
              sue_abs_x_inverse_vol = abs(sue_score) / pre_event_idiosyncratic_vol
              (NaN-safe: NaN denom or denom=0 -> NaN)
       10. assemble the canonical's row chunk (29 columns) and return it.

    Returns: pd.DataFrame — this canonical's chunk of /features/train_matrix.

    The caller concatenates all chunks, sorts by
        ['calendar_week_group', 'canonical_ticker'],
    and writes the result to /features/train_matrix.
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


# ==============================================================================
# PART 6 — STORAGE
# ==============================================================================
def write_train_matrix(df: pd.DataFrame, key: str = TRAIN_MATRIX_KEY) -> None:
    """
    Persist /features/train_matrix to DB_FILE.

    Safety: HDFStore(mode='a'); store.remove(key) first if exists. NEVER mode='w'.
    data_columns=['calendar_week_group','canonical_ticker','report_date'] for
    Stage-3 walk-forward query speed.
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


# ==============================================================================
# PART 7 — BUILD REPORT
# ==============================================================================
def print_build_report(
    df: pd.DataFrame,
    t_match_failures: list[tuple[str, pd.Timestamp]],
    pass_a_seconds: float,
    pass_b_seconds: float,
) -> None:
    """
    Human-readable build report.
      - rows written, columns written, distinct groups
      - per-feature non-null coverage (%); flag any feature below 70% non-NaN
      - T-match failure count (and first 10 (canonical, report_date) pairs)
      - pass A / pass B wall-clock timings
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


# ==============================================================================
# MAIN
# ==============================================================================
def main(dry_run: bool = False, limit: int | None = None) -> int:
    """
    Orchestrates Stage 2:

        1. print header (config + DB_FILE + flags)
        2. gated_df    = load_gated_events()
        3. idx_ref_map = load_index_ref_map()
        4. earnings    = load_earnings_full()
        5. Determine canonicals-to-process:
             if limit is None: all canonicals in gated_df.canonical_ticker.unique()
             else:             first `limit` canonicals (for smoke testing)
        6. Preload benchmarks (IJH is needed for every canonical; load it
           ONCE — the IJH node is constant; same for sector ETFs, but those
           are loaded on first access and cached in benchmark_prices_cache):
             benchmark_prices_cache[IJH_KEY] = load_benchmark_prices(IJH_KEY)
        7. For each canonical (in alphabetical order for reproducibility):
             stock_df = load_stock_prices(canonical)  (per-call; discard after)
             chunk = process_canonical(canonical, gated_canonical_df,
                                       idx_ref_map[canonical],
                                       all_earnings_canonical,
                                       stock_df, benchmark_prices_cache)
             chunks.append(chunk); free stock_df memory
             progress print every 100 canonicals (counter, wall-clock ETA)
        8. matrix_df = pd.concat(chunks, ignore_index=True)
        9. matrix_df = matrix_df.sort_values(
            ['calendar_week_group', 'canonical_ticker']).reset_index(drop=True)
       10. if not dry_run: write_train_matrix(matrix_df)
           else:           print "(--dry-run: not writing)"
       11. print_build_report(matrix_df, t_match_failures,
                              pass_a_seconds, pass_b_seconds)

    Exit codes:
        0 — success
        1 — missing DB_FILE or required keys (the load_* raise
            FileNotFoundError with helpful "run 01_features_gate_events.py
            first" message)
        2 — empty matrix (no canonicals processed — caller upstream problem)
    """
    raise NotImplementedError("Stage 2 — body not implemented yet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: build the listwise-ranker training matrix "
            "(/features/train_matrix) from the gated events pool."
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

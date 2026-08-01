#!/usr/bin/env python3
"""Stage 3: Listwise-Ranker Training + Isotonic Calibration  (Phase F v2)
===========================================================================

DEPRECATION NOTE (2024-07-22, post Doc J restructure):
  - This file's main() / training entry-point is OBSOLETE. It trains the
    Phase F v2 `rank:ndcg` ranker targeting `car_10d` -- the LEAKY model
    whose entire OOS edge came from the forward-looking `opening_gap_t1`
    feature (NaN-ing it dropped Sharpe 4.31 -> -0.14). Do NOT run
    main() for production.
  - The DEPLOYABLE Phase G model (Sunday classifier, binary target =
    pead_pass, 17 Sunday-safe features) is trained by:
      `03_model/02_phase_g_sunday_classifier.py`
    (HP selection: `03_model/03_phase_g_sweep.py`).
  - WHY THIS FILE IS KEPT: it exports the shared utility API used by all
    Phase G backtest scripts:
      * `load_train_matrix()`           -- `/features/train_matrix` loader
      * `apply_priming_cutoff(df, ...)` -- §12 priming-runway filter
      * `DB_FILE`, `PRIMING_RUNWAY_START`
      * `split_walk_forward()`, `drop_sparse_weeks()`, `compute_ndcg_at_k_per_group()`
    These helpers stay load-bearing and are imported via:
        importlib.util.spec_from_file_location("tm", HERE / "01_train_model.py")
  - TO ENGINEER A NEXT-GENERATION MODEL ARTIFACT, write a NEW
    `02_*.py`/`03_*.py` script in `03_model/` rather than touching
    this `01_train_model.py` main() pipeline.

REVISION: v2 retrain on Phase E v2 train_matrix (permaTicker-keyed, is_bmo
fixed, eps_surprise_pct capped at +/-300%). v1 was trained on contaminated
EODHD price data (Class-W NSR +363,862% etc.) and is OBSOLETE; do not reuse.

PURPOSE
-------
Train the cross-sectional XGBoost listwise ranker (`rank:ndcg`, `ndcg@3`)
on `/features/train_matrix` plus fit the §17.4 isotonic calibration bridge
that maps raw rank scores -> absolute expected CAR percentages.

The trained artifacts (`ranker.json` + `calibrator.pkl`) are persisted to
`03_model/models/` for downstream Stage 4 (live inference / backtest /
execution bot) consumption.

ARCHITECTURE (Design.md §17)
---------------------------
  * Model class      : `xgboost.XGBRanker` (objective `rank:ndcg`).
  * Eval metric      : `ndcg@3`.
  * Group structure  : rows of one `calendar_week_group` form ONE listwise
                       query; group sizes passed to `.fit()` via `group=...`.
  * Target label     : `car_10d` stored in LOG units (Stage 2). NDCG is
                       rank-invariant so ranker trains directly on log-CAR.
  * Calibration      : `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')`
                       fit on the *validation* split's raw rank scores vs
                       `np.expm1(car_10d_log)` arithmetic targets. Output `mu`
                       is in true per-position percentages (feeds Kelly directly,
                       no further conversion).

WALK-FORWARD SPLIT (Design.md §17.3 implied; §12 priming cut)
-------------------------------------------------------------
  * §12 priming-runway cutoff (LOCKED): `train_df = train_df[
        train_df.report_date >= pd.Timestamp('2015-01-01')]`. Applied BEFORE
    the sparse-week cutoff so the §12 floor is firm and the sparse-week drop
    sees only the eligible pool. (matches features.md §0 §12 note.)
  * Sparse-week cutoff (LOCKED): drop calendar weeks with fewer than
    `min_group_size=3` events (Strategy 1 from the Stage-1 profile; 10.9% of
    weeks dropped, 0.59% of events dropped). Per Design.md §17.3.
  * Walk-forward train/val split: choose a calendar cutoff date `SPLIT_DATE`.
    Rows with `report_date <= SPLIT_DATE` are TRAIN, the rest are VAL. We use
    2024-01-01 as the default SPLIT_DATE (matches the §17.3 walk-forward ideal
    of ~10y train + ~2y val window).

XGBOOST LABEL DISCRETIZATION (XGBoost 3.x constraint adjustment)
--------------------------------------------------------------
Design.md §17.2 says "do NOT convert to discrete ordinal ranks in the dataset.
Maintaining continuous labels is required for the Listwise Gain function." --
that advice applies to a *hypothetical* continuous-NDCG LTR model. In practice
XGBoost 3.x's `rank:ndcg` objective requires INTEGER relevance labels (the
`lambdamart_num_threshold` parameter is unused on `rank:ndcg`; the gain is
literally the integer label).

To honor the spirit of "keep the ranker training on the rich CAR signal" we
discretize `car_10d` (log CAR) at TRAINING TIME into N=10 equal-frequency
quantile buckets (0..9), with bucket 9 = best post-event drift. This is
standard LTR practice (MS MARCO style). The calibrator still re-maps raw
scores to continuous arithmetic CAR via `np.expm1` -- the discretization is
strictly for the ranker's gradient bucket assignment, NOT for downstream
sizing.

An alternative would be `rank:pairwise` (pairwise LambdaMART with continuous y),
which DOES accept continuous labels. We chose `rank:ndcg` to stay close to
the §17.3 spec text and to get the `ndcg@3` eval metric in line (it requires
integer labes too). If a future audit wants to revisit, switching to
`rank:pairwise` is a one-line change.

CLI
---
    python luan_bot_trading/03_model/01_train_model.py
    python luan_bot_trading/03_model/01_train_model.py --dry-run       # no train, no persist
    python luan_bot_trading/03_model/01_train_model.py --split 2024-01-01
    python luan_bot_trading/03_model/01_train_model.py --min-group-size 3
    python luan_bot_trading/03_model/01_train_model.py --out-dir models/baseline_v1/

HDF5 WRITE SAFETY
-----------------
Per STOP_DOING_EXTRA_SHIT.md. We do NOT write to db.h5; only to
`03_model/models/...` files (ranker.json and calibrator.pkl via
`joblib.dump`). Existing files are OVERWRITTEN only on `main()` exit code 0.

OUT-OF-SCOPE (Stage 4+)
-----------------------
  * Backtester / live inference loop -- uses this trained pair.
  * Continuous-time Kelly sizing.
  * Execution bot (entry timing, slippage, etc.).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import io
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd

# HDF5 file with all features
DB_FILE = Path(__file__).resolve().parent.parent / "01_data" / "db.h5"
TRAIN_MATRIX_KEY = "/features/train_matrix"

# Walk-forward defaults
DEFAULT_SPLIT_DATE = "2024-01-01"
DEFAULT_MIN_GROUP_SIZE = 3       # drop weeks with <3 events (ndcg@3 floor)
PRIMING_RUNWAY_START = "2015-01-01"   # Design §12 priming runway cutoff

# Features in X (Phase E confirmed 21 active features; see features.md §5).
FEATURE_COLUMNS = [
    # Block 1 (7)
    "sue_score", "eps_surprise_pct", "consecutive_surprises",
    "sue_acceleration", "sue_lag_1", "sue_lag_2",
    "car_drift_historical_q1",
    # Block 2 (7)
    "is_bmo", "volume_vma20_ratio_pre_event", "suv_day_1",
    "pre_event_idiosyncratic_vol", "opening_gap_t1",
    "intraday_range_t", "pre_event_volume_trend",
    # Block 3 (6)
    "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d",
    "rel_ret_30d", "sector_adjusted_ret_20d",
    # Block 4 (1)
    "sue_abs_x_inverse_vol",
    # Block 6 — FMP analyst revision momentum (8, Phase H)
    "revision_momentum_30d", "revision_momentum_60d",
    "revision_momentum_90d", "revision_ordinal_momentum_90d",
    "revision_intensity_90d", "grade_dispersion_90d",
    "n_analysts_covering", "last_action_days_before_earnings",
]
assert len(FEATURE_COLUMNS) == 29, "features.md: 21 original + 8 revision momentum = 29"

# Label column
LABEL_COLUMN = "car_10d"

# Group column
GROUP_COLUMN = "calendar_week_group"

# Sort key for LTR contiguity (Phase E v2: permaTicker within canonical ordering).
SORT_KEYS = ["calendar_week_group", "permaTicker", "report_date"]

# Hyperparameters (baseline v1 -- not tuned; speed-mode run).
# Per Design.md §17.3, lambdamart_num_threshold=64 is the bin size for
# the NDCG gain discretization; keep the rest at safe defaults for now.
XGB_PARAMS = dict(
    objective="rank:ndcg",
    eval_metric="ndcg@3",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

# Number of relevance buckets for `car_10d` discretization (XGBoost 3.x
# rank:ndcg requires integer labels). 10 buckets (0..9) is standard LTR;
# bucket 9 = best 10-day drift.


# ------------------------------------------------------------------
# Loading and pre-flight
# ------------------------------------------------------------------

def load_train_matrix() -> pd.DataFrame:
    """Load /features/train_matrix from db.h5 and ensure expected columns."""
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"DB file not found: {DB_FILE} — run 02_features/02_build_feature_matrix.py first."
        )
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if TRAIN_MATRIX_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {TRAIN_MATRIX_KEY} not in {DB_FILE} — "
                f"run 02_features/02_build_feature_matrix.py first."
            )
        return store.get(TRAIN_MATRIX_KEY)


def split_walk_forward(
    df: pd.DataFrame,
    split_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-respecting walk-forward split at `split_date` (inclusive train <= split_date,
    exclusive val report_date > split_date). Returns (train_df, val_df), both sorted by SORT_KEYS.
    """
    split_ts = pd.Timestamp(split_date)
    train_mask = pd.to_datetime(df["report_date"]) <= split_ts
    val_mask = pd.to_datetime(df["report_date"]) > split_ts
    train_df = df.loc[train_mask].sort_values(SORT_KEYS).reset_index(drop=True)
    val_df = df.loc[val_mask].sort_values(SORT_KEYS).reset_index(drop=True)
    return train_df, val_df


def apply_priming_cutoff(df: pd.DataFrame, priming_start: str = PRIMING_RUNWAY_START) -> pd.DataFrame:
    """§12 priming-runway cut: drop rows with report_date < PRIMING_RUNWAY_START.
    Applied at TRAINING time only (per Design.md §12, NOT at storage time)."""
    start_ts = pd.Timestamp(priming_start)
    mask = pd.to_datetime(df["report_date"]) >= start_ts
    return df.loc[mask].reset_index(drop=True)


def drop_sparse_weeks(
    df: pd.DataFrame,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> tuple[pd.DataFrame, dict]:
    """Drop calendar_week_group rows with fewer than `min_group_size` events.

    Returns (filtered_df, audit_dict) where audit_dict has:
        total_rows, weeks_total, weeks_dropped, rows_dropped
    """
    if df.empty:
        return df, {"total_rows": 0, "weeks_total": 0,
                    "weeks_dropped": 0, "rows_dropped": 0}
    counts = df.groupby(GROUP_COLUMN).size()
    kept_weeks = counts[counts >= min_group_size].index
    filtered = df[df[GROUP_COLUMN].isin(kept_weeks)]
    n_total = len(df)
    n_kept = len(filtered)
    n_w_total = len(counts)
    n_w_dropped = n_w_total - len(kept_weeks)
    return filtered.reset_index(drop=True), {
        "total_rows": n_total,
        "weeks_total": n_w_total,
        "weeks_dropped": int(n_w_dropped),
        "rows_dropped": int(n_total - n_kept),
    }


# ------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------

def prepare_dmatrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    group_col: str,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Prepare (X, y, groups) for XGBRanker.fit(). group sizes are derived
    from `group_col` after the (already-sorted) sort."""
    if df.empty:
        return df[[]].copy(), pd.Series([], dtype=np.float64), np.array([], dtype=int)
    # Sanity: ensure sorted by group_col so group sizes are contiguous.
    df = df.sort_values([group_col, "permaTicker", "report_date"]).reset_index(drop=True)
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[label_col], errors="coerce")
    groups = df.groupby(group_col, sort=True).size().values.astype(int)
    return X, y, groups


def discretize_label_quantiles(
    y_log: pd.Series,
    n_buckets: int = 10,
    boundaries: np.ndarray | None = None,
) -> tuple[pd.Series, np.ndarray | None]:
    """Quantile-bucket log-CAR labels into integer relevance degrees [0, n_buckets-1].

    If `boundaries` is None, compute fresh quantile cut points from y_log
    (training-time fit). If `boundaries` is provided (e.g., for val/test),
    reuse them so we don't peek at val label distribution. NaN labels are
    passed through as NaN (XGBoost treats them in objective as gain=0; calibrator
    fit later masks NaN out).

    Returns (y_int, boundaries). y_int is pd nullable Int64 (NaN allowed), and
    any out-of-range values are clipped to [0, n_buckets-1]."""
    if y_log.empty:
        return pd.Series([], dtype="Int64"), boundaries
    valid = y_log.notna()
    out = pd.Series(pd.NA, index=y_log.index, dtype="Int64")
    if boundaries is None and valid.sum() >= n_buckets:
        q = np.linspace(0.0, 1.0, num=n_buckets + 1)
        qs = np.quantile(y_log[valid].to_numpy(), q)
        qs = np.unique(qs)
        if len(qs) < 2:
            out.loc[valid] = 0
            return out, qs
        boundaries = qs
    if boundaries is None:
        out.loc[valid] = 0
        return out, None
    y_valid_vals = y_log[valid].to_numpy()
    bucket_idx = np.searchsorted(boundaries, y_valid_vals, side="right") - 1
    bucket_idx = np.clip(bucket_idx, 0, n_buckets - 1)
    out.loc[valid] = bucket_idx
    return out, boundaries


def train_ranker(
    X_train: pd.DataFrame,
    y_train_int: pd.Series,             # integer relevance degrees (0..n-1)
    groups_train: np.ndarray,
    X_val: pd.DataFrame | None,
    y_val_int: pd.Series | None,       # integer relevance degrees for val
    groups_val: np.ndarray | None,
    params: dict,
) -> "xgboost.XGBRanker":
    """Train XGBRanker with optional val eval_set.

    y_train_int / y_val_int are integer relevance degrees (post-discretization).
    XGBoost 3.x `rank:ndcg` requires integer labels; NaN is coalesced to 0
    (neutral gain)."""
    import xgboost as xgb  # local import so Phase F model tests can patch.

    y_train_clean = y_train_int.fillna(0).astype(int)
    ranker = xgb.XGBRanker(**params)
    if X_val is not None and X_val.shape[0] > 0 and groups_val is not None and len(groups_val) > 0:
        y_val_clean = y_val_int.fillna(0).astype(int)
        ranker.fit(
            X_train, y_train_clean, group=groups_train,
            eval_set=[(X_val, y_val_clean)],
            eval_group=[groups_val],
            verbose=False,
        )
    else:
        ranker.fit(
            X_train, y_train_clean, group=groups_train,
            verbose=False,
        )
    return ranker


def fit_calibrator(
    ranker,
    X_val: pd.DataFrame,
    y_val_log: pd.Series,
) -> "IsotonicRegression":
    """§17.4 isotonic bridge. Fit on val raw scores vs np.expm1(y_val_log)
    arithmetic targets. Output `mu` is in true percentage units.

    Note: we fit the calibrator on the CONTINUOUS log-CAR -> arithmetic CAR,
    NOT on the discretized integer relevance degrees. The discretization is a
    ranker-only shortcut; downstream (calibrator, Kelly) still uses the
    original continuous `car_10d` for absolute-return prediction."""
    from sklearn.isotonic import IsotonicRegression
    raw_scores = ranker.predict(X_val)
    # Mask out NaN-label rows -- cannot fit calibrator on NaN targets.
    mask = y_val_log.notna()
    if mask.sum() < 2:
        raise ValueError(
            f"Not enough non-NaN val labels ({mask.sum()}) to fit calibrator."
        )
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_scores[mask], np.expm1(y_val_log[mask].to_numpy()))
    return calibrator


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def persist_artifacts(
    ranker,
    calibrator,
    out_dir: Path,
    meta: dict,
) -> None:
    """Persist ranker (JSON) + calibrator (pickle) + metadata (JSON)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # XGBRanker.save_model prefers .json for human-readable model dump.
    ranker_path = out_dir / "ranker.json"
    ranker.save_model(str(ranker_path))

    # Sklearn isotonic regressor: use joblib-style pickle.
    calib_path = out_dir / "calibrator.pkl"
    with open(calib_path, "wb") as f:
        pickle.dump(calibrator, f)

    # Metadata (feature_columns, split_date, etc.) for downstream consumers.
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"  Wrote: {ranker_path}")
    print(f"  Wrote: {calib_path}")
    print(f"  Wrote: {meta_path}")


# ------------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------------

def compute_ndcg_at_k_per_group(
    scores: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    k: int = 3,
) -> float:
    """Compute the mean NDCG@k across query groups. NaN-safe in y (rows with
    NaN y are excluded from the numerator gain after sorting). Used for a
    simple out-of-sample reporting number (XGBoost's internal ndcg metric
    may use slightly different smoothing; this is a quick连贯y sanity check).
    """
    # Standard NDCG@k formula with ideal DCG (IDCG) normalization.
    # Treat y as the gain directly (identity gain -- standard NDCG convention).
    out = []
    start = 0
    for g in groups:
        end = start + int(g)
        s = scores[start:end]
        ys = y[start:end]
        # Sort by score desc; keep aligned y.
        order = np.argsort(-s)
        ys_sorted = ys[order]
        # DCG@k: gain_t / log2(t+1) for t in 1..k
        gains = ys_sorted[:k]
        # Replace NaN in gain with 0 (NDCG: no gain means no contribution).
        gains = np.nan_to_num(gains, nan=0.0)
        discounts = 1.0 / np.log2(np.arange(1, min(k, len(gains)) + 1) + 1)
        dcg = float(np.sum(gains[:k] * discounts))
        # IDCG@k: sort by y desc (perfect ranking) then take top-k.
        y_sorted = np.sort(-ys)
        # gains for the perfect ordering (negate twice: sort -ys asc => ys desc).
        ideal_gains = (-y_sorted)[:k]
        ideal_gains = np.nan_to_num(ideal_gains, nan=0.0)
        idcg = float(np.sum(ideal_gains[:k] * discounts))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        out.append(ndcg)
        start = end
    if not out:
        return float("nan")
    return float(np.mean(out))


def evaluate_ranker(
    ranker,
    X: pd.DataFrame,
    y_log: pd.Series,
    groups: np.ndarray,
    label: str,
    k: int = 3,
) -> dict:
    raw_scores = ranker.predict(X)
    ndcg_k = compute_ndcg_at_k_per_group(
        raw_scores, y_log.to_numpy(), groups, k=k
    )
    # Also compute a repeated ndcg@1 as "is the top scorer the actual top-y?".
    return {
        f"ndcg_at_{k}": ndcg_k,
        f"n_groups": int(len(groups)),
        f"n_rows": int(len(X)),
    }


def evaluate_calibrator(
    calibrator,
    ranker,
    X: pd.DataFrame,
    y_log: pd.Series,
) -> dict:
    """Brute-force calibration audit: is calibrator's `mu` monotonic with
    actual realized arithmetic CAR? Pearson + mean/median mu."""
    raw_scores = ranker.predict(X)
    mu = calibrator.predict(raw_scores)
    y_arith = np.expm1(y_log.to_numpy())
    mask = ~np.isnan(y_arith)
    out = {
        "n_valid": int(mask.sum()),
        "mu_mean": float(np.mean(mu[mask])) if mask.sum() else float("nan"),
        "mu_median": float(np.median(mu[mask])) if mask.sum() else float("nan"),
        "y_arith_mean": float(np.mean(y_arith[mask])) if mask.sum() else float("nan"),
    }
    if mask.sum() >= 2:
        # Pearson between raw_scores and y_arith (a weak signal check).
        raw_v = raw_scores[mask]
        ya_v = y_arith[mask]
        corr = float(np.corrcoef(raw_v, ya_v)[0, 1])
        out["pearson_raw_vs_actual"] = corr
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(
    dry_run: bool = False,
    split_date: str = DEFAULT_SPLIT_DATE,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    out_dir: Path | str | None = None,
) -> int:
    bar = "=" * 70
    print(bar)
    print("STAGE 3 — XGBRanker Training + Isotonic Calibration  [Phase F v2]")
    print(f"DB file: {DB_FILE}")
    print(f"Source key: {TRAIN_MATRIX_KEY}")
    print(f"Split date: {split_date}  (train <= split_date, val > split_date)")
    print(f"§12 priming cutoff: {PRIMING_RUNWAY_START}")
    print(f"Min group size: {min_group_size}")
    if dry_run:
        print("(--dry-run: training runs, no artifacts persisted)")
    print(bar)

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "models" / "phase_f_baseline_v2"
    elif not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        # Resolve relative out-dirs under 03_model/models/, not 03_model/ directly,
        # so e.g. `--out-dir phase_f_baseline_v2_md4` lands in the canonical
        # models/ tree instead of polluting the script's parent folder.
        out_dir = (Path(__file__).resolve().parent / "models" / out_dir).resolve()
    print(f"Out dir: {out_dir}")
    print()

    # 1. Load
    print("[1/7] Loading /features/train_matrix ...")
    df = load_train_matrix()
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols")
    print(f"  Date range: {pd.to_datetime(df['report_date']).min()} -> "
          f"{pd.to_datetime(df['report_date']).max()}")

    # 2. §12 priming cut
    print(f"[2/7] Applying §12 priming cutoff (report_date >= {PRIMING_RUNWAY_START}) ...")
    df = apply_priming_cutoff(df, PRIMING_RUNWAY_START)
    print(f"  After cutoff: {len(df):,} rows")

    # 3. Walk-forward split (BEFORE sparse-week cutoff so train and val cut
    # independently — keeps group-buffer sizes internally consistent).
    print(f"[3/7] Walk-forward split @ {split_date} ...")
    train_df, val_df = split_walk_forward(df, split_date)
    print(f"  TRAIN: {len(train_df):,} rows ({train_df['calendar_week_group'].nunique()} weeks, "
          f"{train_df['permaTicker'].nunique()} permaTickers)")
    print(f"  VAL:   {len(val_df):,} rows ({val_df['calendar_week_group'].nunique()} weeks, "
          f"{val_df['permaTicker'].nunique()} permaTickers)")

    # 4. Sparse-week cutoff on each split
    print(f"[4/7] Sparse-week cutoff (min_group_size={min_group_size}) per split ...")
    train_df, train_cut = drop_sparse_weeks(train_df, min_group_size)
    val_df, val_cut = drop_sparse_weeks(val_df, min_group_size)
    print(f"  TRAIN: dropped {train_cut['weeks_dropped']} weeks / "
          f"{train_cut['rows_dropped']} rows "
          f"-> {len(train_df):,} rows ({train_df['calendar_week_group'].nunique()} weeks)")
    print(f"  VAL:   dropped {val_cut['weeks_dropped']} weeks / "
          f"{val_cut['rows_dropped']} rows "
          f"-> {len(val_df):,} rows ({val_df['calendar_week_group'].nunique()} weeks)")

    if len(train_df) == 0 or len(val_df) == 0:
        print("ERROR: One of (train, val) is empty after cutoffs; nothing to do.")
        print(bar)
        return 2

    # 5. Prepare DMatrix-compatible objects
    print("[5/7] Preparing (X, y, groups) for each split ...")
    X_train, y_train_log, g_train = prepare_dmatrix(
        train_df, FEATURE_COLUMNS, LABEL_COLUMN, GROUP_COLUMN
    )
    X_val, y_val_log, g_val = prepare_dmatrix(
        val_df, FEATURE_COLUMNS, LABEL_COLUMN, GROUP_COLUMN
    )
    print(f"  X_train: {X_train.shape}  groups_train: {g_train.shape}")
    print(f"  X_val:   {X_val.shape}  groups_val:   {g_val.shape}")
    print(f"  y_train mean (log CAR): {y_train_log.mean():.5f}  "
          f"std: {y_train_log.std():.5f}")
    print(f"  y_val   mean (log CAR): {y_val_log.mean():.5f}  "
          f"std: {y_val_log.std():.5f}")
    n_train_nan_y = int(y_train_log.isna().sum())
    n_val_nan_y = int(y_val_log.isna().sum())
    print(f"  y_train NaN rows (filled with 0 for ranking loss): {n_train_nan_y}")
    print(f"  y_val   NaN rows (excluded from calibrator fit): {n_val_nan_y}")

    # 5b. Discretize y to integer relevance degrees for XGBoost rank:ndcg.
    N_BUCKETS = 10
    print(f"  Discretizing y_train into {N_BUCKETS} quantile buckets (val reuses train boundaries) ...")
    y_train_int, boundaries = discretize_label_quantiles(y_train_log, n_buckets=N_BUCKETS)
    y_val_int, _ = discretize_label_quantiles(y_val_log, n_buckets=N_BUCKETS, boundaries=boundaries)
    if boundaries is not None and len(boundaries) > 1:
        print(f"  Discretization boundaries (quantile cut points): {np.round(boundaries, 4).tolist()}")
    else:
        print(f"  Single-bucket case: no discretization")
    print(f"  y_train_int bucket counts:")
    bc = pd.Series(y_train_int).fillna(-1).astype(int).value_counts().sort_index()
    for b, c in bc.items():
        print(f"    bucket {b}: {c} rows")

    # 6. Train ranker
    print("[6/7] Training XGBRanker (this is the long step) ...")
    if dry_run:
        print("  (--dry-run): using fewer estimators for speed.")
        params = dict(XGB_PARAMS)
        params["n_estimators"] = 30
    else:
        params = XGB_PARAMS
    print(f"  hyperparams: {params}")
    t0 = time.time()
    ranker = train_ranker(
        X_train, y_train_int, g_train,
        X_val, y_val_int, g_val,
        params,
    )
    train_seconds = time.time() - t0
    print(f"  Trained in {train_seconds:.1f}s")

    # 6b. Train calibrator (§17.4 isotonic bridge).
    raw_scores_val = ranker.predict(X_val)
    print(f"  Raw val scores: mean={raw_scores_val.mean():.5f} std={raw_scores_val.std():.5f}")
    try:
        calibrator = fit_calibrator(ranker, X_val, y_val_log)
        calib_ok = True
        print(f"  Calibrator fit OK on {y_val_log.notna().sum()} non-NaN val labels")
    except ValueError as e:
        calib_ok = False
        calibrator = None
        print(f"  Calibrator fit SKIPPED: {e}")

    # 7. Evaluate + persist
    print("[7/7] Evaluating + persisting ...")
    train_eval = evaluate_ranker(ranker, X_train, y_train_log, g_train, "TRAIN")
    val_eval = evaluate_ranker(ranker, X_val, y_val_log, g_val, "VAL")
    print(f"  TRAIN: {train_eval}")
    print(f"  VAL:   {val_eval}")
    if calib_ok:
        calib_eval = evaluate_calibrator(calibrator, ranker, X_val, y_val_log)
        print(f"  Calibrator audit (val set): {calib_eval}")

    # Feature importance track (XGBoost's "gain" importance).
    try:
        fi = pd.Series(ranker.feature_importances_, index=FEATURE_COLUMNS)
        fi_top = fi.sort_values(ascending=False).head(10)
        print("  Top 10 feature importances (gain):")
        for f, v in fi_top.items():
            print(f"    {f:35s} {v:.5f}")
    except Exception as e:
        print(f"  (feature_importances_ unavailable: {e})")

    if dry_run:
        print("\n(--dry-run: NOT persisting artifacts)")
    elif calib_ok:
        meta = {
            "split_date": split_date,
            "priming_cutoff": PRIMING_RUNWAY_START,
            "min_group_size": min_group_size,
            "feature_columns": FEATURE_COLUMNS,
            "label_column": LABEL_COLUMN,
            "group_column": GROUP_COLUMN,
            "xgb_params": params,
            "n_buckets": N_BUCKETS,
            "label_discretization_boundaries": (boundaries.tolist() if boundaries is not None else None),
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "train_groups": int(len(g_train)),
            "val_groups": int(len(g_val)),
            "train_seconds": float(train_seconds),
            "train_eval": train_eval,
            "val_eval": val_eval,
            "calib_audit": calib_eval,
            "created_at": pd.Timestamp.now().isoformat(),
        }
        persist_artifacts(ranker, calibrator, out_dir, meta)
    else:
        print("\n  (calibrator unfit -- not persisting artifacts for safety)")
        print(bar)
        return 3

    print(bar)
    print("STAGE 3 - DONE")
    print(bar)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: train listwise XGBRanker + fit isotonic calibration "
            "on /features/train_matrix. (Phase F v2 retrain on clean Phase B+E data.)"
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run with a small n_estimators and do NOT persist artifacts.",
    )
    parser.add_argument(
        "--split", type=str, default=DEFAULT_SPLIT_DATE,
        help=f"Walk-forward split date (default: {DEFAULT_SPLIT_DATE}). "
             "Train rows have report_date <= split; val rows have report_date > split.",
    )
    parser.add_argument(
        "--min-group-size", type=int, default=DEFAULT_MIN_GROUP_SIZE,
        help=f"Drop weeks with fewer than this many events (default: {DEFAULT_MIN_GROUP_SIZE}).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for artifacts. Default: 03_model/models/phase_f_baseline_v2/",
    )
    args = parser.parse_args()
    sys.exit(main(
        dry_run=args.dry_run,
        split_date=args.split,
        min_group_size=args.min_group_size,
        out_dir=args.out_dir,
    ))

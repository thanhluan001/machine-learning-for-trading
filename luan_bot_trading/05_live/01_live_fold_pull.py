#!/usr/bin/env python3
"""
Live Fold #5 — Incremental Pull + Frozen Binary Model Inference
=================================================================

PURPOSE
-------
After the deployable binary classifier was frozen on
2026-07-30 (trained on all available data, 24 Sunday-safe features,
binary pead_pass target), the NEXT forward-looking OOS data point is
the live fold. This script:

  1. INCREMENTALLY fetches new Tiingo prices (only dates after last stored)
  2. RE-RUNS FMP earnings fetch (06b) + FMP grades fetch (07)
  3. RE-RUNS Stage 1 (gate events) + Stage 2 (build feature matrix)
  4. LOADS the FROZEN binary classifier from
     `03_model/models/phase_g_v2_binary/classifier.json`
  5. PREDICTS P(PEAD) on live-fold events
  6. APPLIES the deployable rule: P(PEAD) >= 0.20
  7. COMPUTES realized PnL using PRE-GAP entry:
       BMO: entry=Close[T-1], exit=Close[T+5]
       AMC: entry=Close[T],   exit=Close[T+5]
     -10% delayed stop-loss (skip gap day, check days 1+). Statistically
     neutral but caps worst-case tail risk.
     EXCLUDE XLF (Financials) sector at inference — model still trains on
     XLF. Financials have 13% PEAD precision vs 41% for rest. Structural:
     financial earnings are more macro-driven, less surprise-driven.
  8. REPORTS trade-level stats vs baseline expectations

The classifier is NEVER retrained in this script. The entire point of
the live fold is to test the FROZEN artifact on truly forward-looking data.

USAGE
-----
    python luan_bot_trading/05_live/01_live_fold_pull.py
    python luan_bot_trading/05_live/01_live_fold_pull.py --skip-fetch   # inference only
    python luan_bot_trading/05_live/01_live_fold_pull.py --dry-run      # show what would be fetched

PREREQUISITES
-------------
- Frozen classifier at `03_model/models/phase_g_v2_binary/`
- db.h5 with Phase A/B/D/E/F data (run 02b, 03, 06b, 07, 01, 02 first)
- .env with TIINGO_API_KEY and FMP_API_KEY

HDF5 WRITE SAFETY
-----------------
NEVER `mode='w'` on db.h5. Uses `HDFStore(mode='a')` + `store.remove()`.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

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
DATA_DIR = PROJECT_ROOT / "01_data"
FEATURES_DIR = PROJECT_ROOT / "02_features"
MODEL_DIR = PROJECT_ROOT / "03_model"

# Frozen classifier artifact
FROZEN_CLASSIFIER_PATH = MODEL_DIR / "models" / "phase_g_v2_binary" / "classifier.json"
FROZEN_META_PATH = MODEL_DIR / "models" / "phase_g_v2_binary" / "meta.json"

# Deployable operating point (binary, pre-gap, -10% delayed stop, exclude XLF)
THETA = 0.20           # P(PEAD) >= 0.20
EXIT_DAYS = 5             # exit at Close[T+5] (5 trading days from report date)
N_SLOTS = 4               # max simultaneous positions (informational)
STOP_LOSS = 0.10          # -10% delayed stop (skip gap day, check days 1+)
                           # Statistically neutral but caps worst case.
EXCLUDE_SECTORS = ["XLF"] # Exclude Financials at inference only (model still
                           # trains on XLF). Financials have 13% PEAD precision
                           # vs 41% for rest. Structural: financial earnings are
                           # more macro-driven, less surprise-driven.

# H5 keys
SP400_GROUP = "sp400"
PERMATICKERS_KEY = "/metadata/sp400_permatickers"
EARNINGS_KEY = "/earnings/fmp"
TRAIN_MATRIX_KEY = "/features/train_matrix"

# Tiingo config
TIINGO_TIMEOUT = 60
OUTPUT_COLUMNS = [
    "Date", "Open", "High", "Low", "Close", "Volume",
    "Adj_Open", "Adj_High", "Adj_Low", "Adj_Close", "Adj_Volume",
]

# 24 Sunday-safe features (must match 03_freeze_3class_model.py)
SUNDAY_SAFE_FEATURES = [
    "sue_score", "eps_surprise_pct", "consecutive_surprises",
    "sue_acceleration", "sue_lag_1", "sue_lag_2",
    "car_drift_historical_q1",
    "pre_event_idiosyncratic_vol", "pre_event_volume_trend",
    "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d",
    "rel_ret_30d", "sector_adjusted_ret_20d",
    "sue_abs_x_inverse_vol",
    "revision_momentum_30d", "revision_momentum_60d",
    "revision_momentum_90d", "revision_ordinal_momentum_90d",
    "revision_intensity_90d", "grade_dispersion_90d",
    "n_analysts_covering", "last_action_days_before_earnings",
]

# Baseline expectations (from 4-fold nested CV, binary theta=0.20, exclude XLF, -10% delayed stop)
BASELINE_STATS = {
    "n_trades": 101,
    "win_rate": 0.752,
    "avg_pnl": 6.66,
    "avg_win": 12.36,
    "avg_loss": -6.30,
    "payoff": 1.36,
    "total_pnl": 672.4,
    "pead_precision": 0.386,
    "large_pead_win_rate": 0.905,
    "label": "Binary P(PEAD)>=0.20, pre-gap, 5-day hold, exclude XLF, -10% delayed stop (101 trades, 4-fold CV)",
}


# ==============================================================================
# PART 1: INCREMENTAL TIINGO PRICE FETCH
# ==============================================================================
_TIINGO_HEADERS = {"Content-Type": "application/json"}


def fetch_tiingo_incremental(permaTicker: str, start: str, end: str) -> pd.DataFrame:
    url = f"https://api.tiingo.com/tiingo/daily/{requests.utils.quote(permaTicker)}/prices"
    params = {"token": TIINGO_API_KEY, "startDate": start, "endDate": end}
    try:
        resp = requests.get(url, params=params, headers=_TIINGO_HEADERS, timeout=TIINGO_TIMEOUT)
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
    out = out[OUTPUT_COLUMNS]
    out = out.dropna(subset=["Adj_Close", "Adj_Volume"]).reset_index(drop=True)
    out = out.sort_values("Date", kind="mergesort").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    return out


def get_latest_stored_date(permaTicker: str) -> pd.Timestamp | None:
    h5_path = f"/{SP400_GROUP}/{permaTicker}"
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if h5_path not in store.keys():
            return None
        df = store[h5_path]
        if df.empty:
            return None
        return pd.Timestamp(df["Date"].max())


def append_price_data(permaTicker: str, new_data: pd.DataFrame):
    if new_data.empty:
        return
    h5_path = f"/{SP400_GROUP}/{permaTicker}"
    with pd.HDFStore(DB_FILE, mode="a") as store:
        existing = store[h5_path] if h5_path in store.keys() else pd.DataFrame()
        if not existing.empty:
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.sort_values("Date", kind="mergesort").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
        else:
            combined = new_data
        if h5_path in store.keys():
            store.remove(h5_path)
        store.put(h5_path, combined, format="table", data_columns=["Date"])


def incremental_tiingo_fetch(dry_run: bool = False) -> dict:
    if not TIINGO_API_KEY:
        raise ValueError("TIINGO_API_KEY not in .env")
    print("\n" + "=" * 70)
    print("  [1] INCREMENTAL TIINGO PRICE FETCH")
    print("=" * 70)
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if PERMATICKERS_KEY not in store.keys():
            raise FileNotFoundError(f"{PERMATICKERS_KEY} not in db.h5")
        pt_df = store[PERMATICKERS_KEY]
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    history_start = (datetime.now() - timedelta(days=15 * 365)).strftime("%Y-%m-%d")
    fetchable = pt_df[~pt_df["price_unavailable"]].copy()
    n_total = len(fetchable)
    print(f"  Universe: {n_total} permaTickers (price_unavailable=False)")
    stats = {"skipped_uptodate": 0, "incremental": 0, "full_fetch": 0,
             "empty": 0, "failed": 0, "new_rows": 0}
    t0 = time.time()
    progress_every = max(1, n_total // 20)
    for i, row in fetchable.iterrows():
        pt = str(row["permaTicker"])
        try:
            latest = get_latest_stored_date(pt)
            if latest is not None:
                if latest >= pd.Timestamp(yesterday):
                    stats["skipped_uptodate"] += 1
                else:
                    start = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
                    if dry_run:
                        stats["incremental"] += 1
                    else:
                        new_data = fetch_tiingo_incremental(pt, start, yesterday)
                        if not new_data.empty:
                            append_price_data(pt, new_data)
                            stats["new_rows"] += len(new_data)
                            stats["incremental"] += 1
                        else:
                            stats["empty"] += 1
            else:
                if dry_run:
                    stats["full_fetch"] += 1
                else:
                    new_data = fetch_tiingo_incremental(pt, history_start, yesterday)
                    if not new_data.empty:
                        append_price_data(pt, new_data)
                        stats["new_rows"] += len(new_data)
                        stats["full_fetch"] += 1
                    else:
                        stats["empty"] += 1
        except Exception as e:
            stats["failed"] += 1
            print(f"  [ERROR] {pt}: {e}")
        if (i + 1) % progress_every == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            print(f"  [PROGRESS] {i+1}/{n_total}  incr={stats['incremental']} "
                  f"full={stats['full_fetch']} skip={stats['skipped_uptodate']} "
                  f"empty={stats['empty']} failed={stats['failed']} "
                  f"new_rows={stats['new_rows']}  elapsed={elapsed:.1f}s")
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  New rows: {stats['new_rows']}")
    return stats


# ==============================================================================
# PART 2: FMP EARNINGS + GRADES RE-FETCH
# ==============================================================================
def rerun_script(script_path: Path, dry_run: bool = False, name: str = "") -> bool:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    if dry_run:
        print(f"  [DRY] Would run: python {script_path.name}")
        return True
    cmd = [sys.executable, str(script_path)]
    print(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(script_path.parent), capture_output=False)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)  exit={result.returncode}")
    return result.returncode == 0


def rerun_fmp_earnings(dry_run: bool = False) -> bool:
    if not FMP_API_KEY:
        raise ValueError("FMP_API_KEY not in .env")
    print("\n" + "=" * 70)
    print("  [2] FMP EARNINGS RE-FETCH (06b_fmp_earnings_gathering.py)")
    print("=" * 70)
    return rerun_script(DATA_DIR / "06b_fmp_earnings_gathering.py", dry_run, "FMP earnings")


def rerun_fmp_grades(dry_run: bool = False) -> bool:
    print("\n" + "=" * 70)
    print("  [2b] FMP GRADES RE-FETCH (07_fmp_grades_gathering.py)")
    print("=" * 70)
    return rerun_script(DATA_DIR / "07_fmp_grades_gathering.py", dry_run, "FMP grades")


def rerun_stage1(dry_run: bool = False) -> bool:
    print("\n" + "=" * 70)
    print("  [3] STAGE 1: GATE EVENTS (01_features_gate_events.py)")
    print("=" * 70)
    return rerun_script(FEATURES_DIR / "01_features_gate_events.py", dry_run, "Stage 1")


def rerun_stage2(dry_run: bool = False) -> bool:
    print("\n" + "=" * 70)
    print("  [4] STAGE 2: BUILD FEATURE MATRIX (02_build_feature_matrix.py)")
    print("=" * 70)
    return rerun_script(FEATURES_DIR / "02_build_feature_matrix.py", dry_run, "Stage 2")


# ==============================================================================
# PART 3: LIVE-FOLD INFERENCE (FROZEN 3-CLASS CLASSIFIER)
# ==============================================================================
def get_live_fold_start() -> pd.Timestamp:
    """Live fold = events AFTER the classifier was frozen."""
    if FROZEN_META_PATH.exists():
        with open(FROZEN_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        created = pd.Timestamp(meta.get("created_at", ""))
        return created.normalize()
    return pd.Timestamp("2026-07-29")


def load_frozen_classifier() -> xgb.XGBClassifier:
    if not FROZEN_CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Frozen classifier not found at {FROZEN_CLASSIFIER_PATH}\n"
            "Run 03_model/03_freeze_3class_model.py first."
        )
    clf = xgb.XGBClassifier()
    clf.load_model(str(FROZEN_CLASSIFIER_PATH))
    print(f"  Loaded frozen binary classifier from {FROZEN_CLASSIFIER_PATH}")
    return clf


def compute_pregap_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pre-gap PnL for each row with optional -10% delayed stop:
       BMO: entry=Close[T-1], exit=Close[T+5]
       AMC: entry=Close[T],   exit=Close[T+5]
    Stop-loss: -10% delayed (skip gap day, check days 1+). If Low[k] <=
    entry*(1-stop), exit at stop price. If Open[k] <= stop (gap-down),
    exit at Open. Statistically neutral but caps worst-case tail."""
    df = df.copy()
    df["ret_arith"] = np.nan
    df["hit"] = np.nan
    df["stop_hit"] = False
    df["is_bmo"] = df["is_bmo"].fillna(0).astype(int) == 1

    with pd.HDFStore(DB_FILE, mode="r") as s:
        pts = df["permaTicker"].unique()
        n_done = 0
        for pt in pts:
            key = f"/{SP400_GROUP}/{pt}"
            if key not in s:
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_close = p["Adj_Close"].values
            p_open = p["Adj_Open"].values
            p_low = p["Adj_Low"].values
            sub = df[df["permaTicker"] == pt]
            for idx, row in sub.iterrows():
                rdate = pd.to_datetime(row["report_date"]).to_datetime64()
                t_mask = p_index >= rdate
                if not t_mask.any():
                    continue
                t_idx = int(np.argmax(t_mask))
                if t_idx + EXIT_DAYS >= len(p_close):
                    continue  # exit not yet available
                is_bmo = row["is_bmo"]
                entry_t = t_idx - 1 if is_bmo else t_idx
                gap_day = t_idx if is_bmo else t_idx + 1
                if entry_t < 0:
                    continue
                entry_price = p_close[entry_t]
                exit_price = p_close[t_idx + EXIT_DAYS]
                if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                    continue

                # Default: no stop hit, exit at Close[T+5]
                arith_ret = float(exit_price / entry_price - 1.0)

                # Check delayed stop (skip gap day, check days gap_day+1 .. exit)
                stop_price = entry_price * (1.0 - STOP_LOSS)
                for k in range(gap_day + 1, t_idx + EXIT_DAYS + 1):
                    if k >= len(p_open):
                        break
                    o_k = p_open[k]
                    lo_k = p_low[k]
                    if pd.notna(o_k) and o_k <= stop_price:
                        # Gap-down below stop: exit at open
                        arith_ret = float(o_k / entry_price - 1.0)
                        df.loc[idx, "stop_hit"] = True
                        break
                    elif pd.notna(lo_k) and lo_k <= stop_price:
                        # Intraday stop hit: exit at stop price
                        arith_ret = float(stop_price / entry_price - 1.0)
                        df.loc[idx, "stop_hit"] = True
                        break

                df.loc[idx, "ret_arith"] = arith_ret
                df.loc[idx, "hit"] = 1.0 if arith_ret > 0 else 0.0
            n_done += 1
            if n_done % 100 == 0:
                print(f"    [pnl] {n_done}/{len(pts)} permaTickers")
    return df


def run_inference(dry_run: bool = False):
    print("\n" + "=" * 70)
    print("  [5] LIVE-FOLD INFERENCE (FROZEN 3-CLASS CLASSIFIER)")
    print("=" * 70)

    live_fold_start = get_live_fold_start()
    print(f"  Live fold start (T >= ): {live_fold_start.date()}")

    if dry_run:
        print("  [DRY] Would load train_matrix, filter to live-fold events,")
        print("        predict 3-class probabilities, apply deployable rule,")
        print("        compute pre-gap PnL.")
        return

    # Load rebuilt train_matrix
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if TRAIN_MATRIX_KEY not in store.keys():
            raise FileNotFoundError(f"{TRAIN_MATRIX_KEY} not in db.h5 -- run Stage 2 first")
        tm = store[TRAIN_MATRIX_KEY]
    print(f"  Train matrix: {len(tm):,} rows")

    # Filter to live-fold events
    tm["report_date"] = pd.to_datetime(tm["report_date"])
    live = tm[tm["report_date"] >= live_fold_start].copy()
    print(f"  Live-fold events (T >= {live_fold_start.date()}): {len(live)}")

    if len(live) == 0:
        print(f"\n  *** NO live-fold events yet. Wait for more earnings. ***")
        print(f"  (Last T in train_matrix: {tm['report_date'].max().date()})")
        return

    # Load frozen binary classifier
    clf = load_frozen_classifier()

    # Predict P(PEAD) using Sunday-safe features
    X_live = live[SUNDAY_SAFE_FEATURES].copy()
    proba = clf.predict_proba(X_live)[:, 1]
    live["P_PEAD"] = proba

    print(f"\n  P(PEAD) distribution:")
    print(f"    min={proba.min():.4f}  p25={np.percentile(proba, 25):.4f}  "
          f"median={np.median(proba):.4f}  p75={np.percentile(proba, 75):.4f}  "
          f"max={proba.max():.4f}")
    print(f"    events with P(PEAD) >= {THETA}: {(proba >= THETA).sum()}")

    # Apply deployable rule: P(PEAD) >= theta AND exclude XLF sector
    accepted = live[live["P_PEAD"] >= THETA].copy()
    print(f"\n  Deployable rule: P(PEAD) >= {THETA}")
    print(f"    Accepted (pre-sector-filter): {len(accepted)} of {len(live)} live events")

    # Exclude XLF (Financials) — inference-only filter
    if EXCLUDE_SECTORS:
        with pd.HDFStore(DB_FILE, mode="r") as s:
            pt_meta = s[PERMATICKERS_KEY]
        sector_lookup = pt_meta[["permaTicker", "index_ref"]].drop_duplicates("permaTicker")
        accepted = accepted.merge(sector_lookup, on="permaTicker", how="left")
        n_before = len(accepted)
        accepted = accepted[~accepted["index_ref"].isin(EXCLUDE_SECTORS)].copy()
        print(f"    Excluded {EXCLUDE_SECTORS}: {n_before} -> {len(accepted)} trades")

    if len(accepted) == 0:
        print("\n  *** NO trades accepted. Wait for more high-conviction events. ***")
        return

    # Compute pre-gap PnL
    print(f"\n  Computing pre-gap PnL for {len(accepted)} accepted trades ...")
    print(f"    BMO: entry=Close[T-1], exit=Close[T+{EXIT_DAYS}]")
    print(f"    AMC: entry=Close[T],   exit=Close[T+{EXIT_DAYS}]")
    print(f"    No stop-loss.")
    accepted = compute_pregap_pnl(accepted)

    realized = accepted[accepted["ret_arith"].notna()].copy()
    pending = accepted[accepted["ret_arith"].isna()].copy()

    print(f"\n  Realized trades (T+{EXIT_DAYS} passed): {len(realized)}")
    print(f"  Pending trades (T+{EXIT_DAYS} not yet passed): {len(pending)}")

    if len(realized) > 0:
        n = len(realized)
        wins = realized[realized["hit"] == 1]
        losses = realized[realized["hit"] == 0]
        win_rate = len(wins) / n if n > 0 else 0
        avg_win = wins["ret_arith"].mean() * 100 if len(wins) > 0 else 0
        avg_loss = losses["ret_arith"].mean() * 100 if len(losses) > 0 else 0
        expectancy = realized["ret_arith"].mean() * 100
        total = realized["ret_arith"].sum() * 100

        print("\n" + "-" * 70)
        print("  LIVE FOLD #5 -- TRADE-LEVEL STATS (REALIZED ONLY)")
        print("-" * 70)
        print(f"  Metric              | Live Fold      | Baseline")
        print(f"  ---------------------|----------------|----------------")
        print(f"  N trades            | {n:>14}  | {BASELINE_STATS['n_trades']:>14}")
        print(f"  Win rate            | {win_rate*100:>13.1f}%  | {BASELINE_STATS['win_rate']*100:>13.1f}%")
        print(f"  Avg win             | {avg_win:>+13.2f}%  | {BASELINE_STATS['avg_win']:>+13.2f}%")
        print(f"  Avg loss            | {avg_loss:>+13.2f}%  | {BASELINE_STATS['avg_loss']:>+13.2f}%")
        print(f"  Expectancy/trade    | {expectancy:>+13.2f}%  | {BASELINE_STATS['avg_pnl']:>+13.2f}%")
        print(f"  Total PnL (sum)     | {total:>+13.1f}%  | {BASELINE_STATS['total_pnl']:>+13.1f}%")
        print("-" * 70)

        # Per-trade detail
        print(f"\n  Per-trade detail:")
        for _, t in realized.sort_values("report_date").iterrows():
            pt = t["permaTicker"]
            rd = pd.Timestamp(t["report_date"]).date()
            ticker = t.get("canonical_ticker", pt)
            p_pead = t["P_PEAD"]
            is_bmo = "BMO" if t.get("is_bmo", False) else "AMC"
            r = t["ret_arith"]
            h = "WIN " if t["hit"] == 1 else "LOSS"
            print(f"    {h}  {ticker:10s}  report={rd}  P(PEAD)={p_pead:.3f}  "
                  f"{is_bmo}  ret={r*100:+.2f}%")

        # Directional verdict
        print("\n  DIRECTIONAL VERDICT:")
        if n < 5:
            print(f"    N={n} is too small for any conclusion. Wait for more trades.")
        elif win_rate >= 0.50 and expectancy > 0:
            print(f"    POSITIVE: win rate {win_rate*100:.0f}% >= 50%, expectancy {expectancy:+.2f}% > 0")
            print(f"    Model appears to generalize to live data.")
        else:
            print(f"    WARNING: win rate {win_rate*100:.0f}%, expectancy {expectancy:+.2f}%")
            print(f"    Model may NOT generalize -- investigate regime change.")
    else:
        print(f"\n  No realized trades yet -- all accepted trades are pending (T+{EXIT_DAYS} not passed).")
        print("  Wait ~1 week, then re-run:")
        print("    python 05_live/01_live_fold_pull.py --skip-fetch")

    # Pending trades detail
    if len(pending) > 0:
        print(f"\n  PENDING TRADES (T+{EXIT_DAYS} not yet passed -- check back later):")
        for _, t in pending.sort_values("report_date").iterrows():
            ticker = t.get("canonical_ticker", t["permaTicker"])
            rd = pd.Timestamp(t["report_date"]).date()
            p_pead = t["P_PEAD"]
            is_bmo = "BMO" if t.get("is_bmo", False) else "AMC"
            print(f"    PENDING  {ticker:10s}  report={rd}  P(PEAD)={p_pead:.3f}  "
                  f"{is_bmo}")

    print("\n" + "=" * 70)
    print("  LIVE FOLD #5 COMPLETE")
    print("=" * 70)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live Fold #5: incremental pull + frozen 3-class model inference")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetching -- inference only on existing db.h5 data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched, no writes")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  LIVE FOLD #5 -- Incremental Pull + Frozen Binary Model Inference")
    print("=" * 70)
    print(f"  DB file:      {DB_FILE}")
    print(f"  Classifier:   {FROZEN_CLASSIFIER_PATH}")
    print(f"  Deployable:   P(PEAD)>={THETA}, exclude {EXCLUDE_SECTORS}, pre-gap entry,")
    print(f"                exit=Close[T+{EXIT_DAYS}], -{STOP_LOSS*100:.0f}% delayed stop, n_slots={N_SLOTS}")
    print(f"  Dry run:      {args.dry_run}")
    print("=" * 70)

    if not FROZEN_CLASSIFIER_PATH.exists():
        print(f"\n  [ERROR] Frozen classifier not found at {FROZEN_CLASSIFIER_PATH}")
        print(f"  Run 03_model/03_freeze_3class_model.py first.")
        return 1

    if not args.skip_fetch:
        incremental_tiingo_fetch(dry_run=args.dry_run)
        rerun_fmp_earnings(dry_run=args.dry_run)
        rerun_fmp_grades(dry_run=args.dry_run)
        rerun_stage1(dry_run=args.dry_run)
        rerun_stage2(dry_run=args.dry_run)
    else:
        print("\n  [--skip-fetch] Skipping data fetch -- using existing db.h5 data")

    run_inference(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Freeze v4: timing-correct 23-feature PEAD model.

Information-set contract
-------------------------
For an earnings event on trading day T:
  * AMC: executable entry is Close[T]; use daily features through T-1.
  * BMO: executable entry is Close[T-1]; use daily features through T-2.

The existing v3 train matrix computes selected price features through T-1 for
both timings. This script creates a separate corrected matrix in memory (and
writes it to /features/train_matrix_v4_timing_correct), evaluates the same
walk-forward strategy, and saves a separate classifier artifact.

No current-quarter earnings result or future price is used as an input feature.
Future prices are used only for PEAD labels / backtest PnL.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Shared loaders, labels, feature helpers, and identical CV implementation.
tm = _load("tm_v4", HERE / "01_train_model.py")
s2 = _load("s2_v4", ROOT / "02_features" / "02_build_feature_matrix.py")
v3 = _load("v3_v4", ROOT / "04_backtest" / "_pead_target_retrain.py")
bt = _load("bt_v4", ROOT / "04_backtest" / "51_hp_theta_sweep_23feat.py")

DB_FILE = tm.DB_FILE
V4_MATRIX_KEY = "/features/train_matrix_v4_timing_correct"
MODEL_DIR = HERE / "models" / "phase_g_v4_timing_correct"

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

# These are the selected daily price/volume features whose endpoint changes.
BMO_SHIFT_FEATURES = [
    "pre_event_idiosyncratic_vol", "pre_event_volume_trend",
    "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d", "rel_ret_30d",
    "sector_adjusted_ret_20d",
]

HP = {
    "gamma": 3,
    "min_child_weight": 100,
    "max_depth": 2,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "reg_lambda": 1.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "random_state": 42,
    "n_jobs": -1,
}
THETA = 0.20


def _recompute_bmo_tminus2(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Replace selected BMO price features with values ending at T-2.

    Existing train_matrix rows use t_position=T and the selected features end
    at T-1. For BMO, the order is placed at Close[T-1], so the latest complete
    daily bar available at decision time is T-2. Calling the feature helpers
    with i=T-1 makes their selected windows end at i-1=T-2.
    """
    out = df.copy()
    audit = {"bmo_rows": int(df["is_bmo"].astype(bool).sum()), "recomputed": 0,
             "changed_by_feature": {c: 0 for c in BMO_SHIFT_FEATURES}}

    with pd.HDFStore(DB_FILE, mode="r") as store:
        meta = store["/metadata/sp400_permatickers"].drop_duplicates("permaTicker")
        meta = meta.set_index("permaTicker")
        ijh = store["/macros/IJH"].sort_values("Date").reset_index(drop=True)
        benchmark_cache = {"/macros/IJH": ijh}

        bmo_groups = df[df["is_bmo"].astype(bool)].groupby("permaTicker")
        for perma_ticker, rows in bmo_groups:
            stock_key = f"/sp400/{perma_ticker}"
            if stock_key not in store:
                continue

            stock = store[stock_key].sort_values("Date").reset_index(drop=True)
            if perma_ticker in meta.index:
                index_ref = meta.loc[perma_ticker].get("index_ref", "IJH")
            else:
                index_ref = "IJH"
            if not isinstance(index_ref, str) or not index_ref:
                index_ref = "IJH"

            sector_key = f"/macros/{index_ref}"
            if sector_key not in benchmark_cache:
                benchmark_cache[sector_key] = (
                    store[sector_key].sort_values("Date").reset_index(drop=True)
                    if sector_key in store.keys() else ijh
                )

            aligned = s2.build_aligned_price_frame(
                stock, ijh, benchmark_cache[sector_key]
            )
            stock_ret = s2._log_ret(
                pd.Series(aligned["stock_Adj_Close"].values, index=aligned.index)
            )
            ijh_ret = s2._log_ret(
                pd.Series(aligned["ijh_Close"].values, index=aligned.index)
            )
            stock_dates = stock["Date"].values

            for row_index, row in rows.iterrows():
                report_date = pd.Timestamp(row["report_date"])
                t_idx = int(np.searchsorted(
                    stock_dates, np.datetime64(report_date), side="left"
                ))
                # i=T-1. Selected feature formulas end at i-1=T-2.
                feature_position = t_idx - 1
                if t_idx >= len(stock_dates) or feature_position < 21:
                    continue

                block2 = s2.compute_block2_features(
                    aligned, stock_ret, ijh_ret, feature_position, "BeforeMarket"
                )
                block3 = s2.compute_block3_features(aligned, feature_position)
                values = {
                    "pre_event_idiosyncratic_vol": block2["pre_event_idiosyncratic_vol"],
                    "pre_event_volume_trend": block2["pre_event_volume_trend"],
                    "rel_ret_3d": block3["rel_ret_3d"],
                    "rel_ret_5d": block3["rel_ret_5d"],
                    "rel_ret_10d": block3["rel_ret_10d"],
                    "rel_ret_20d": block3["rel_ret_20d"],
                    "rel_ret_30d": block3["rel_ret_30d"],
                    "sector_adjusted_ret_20d": block3["sector_adjusted_ret_20d"],
                }
                for column, value in values.items():
                    old_value = out.at[row_index, column]
                    if pd.notna(old_value) and pd.notna(value) and abs(float(old_value) - float(value)) > 1e-12:
                        audit["changed_by_feature"][column] += 1
                    out.at[row_index, column] = value
                audit["recomputed"] += 1

    return out, audit


def _apply_timing_correct_information(df: pd.DataFrame) -> pd.DataFrame:
    """Use all point-in-time data available at the actual daily cutoff.

    This replaces both macro and analyst-revision fields. Prior reported
    earnings are already fixed historical facts; the revision and macro fields
    can change between T-2/T-1 and the report date and therefore need the same
    cutoff as price features.
    """
    out = df.copy()
    out["_feature_date"] = pd.NaT
    revision_cols = [
        "revision_momentum_30d", "revision_momentum_60d",
        "revision_momentum_90d", "revision_ordinal_momentum_90d",
        "revision_intensity_90d", "grade_dispersion_90d",
        "n_analysts_covering", "last_action_days_before_earnings",
    ]
    with pd.HDFStore(DB_FILE, mode="r") as store:
        grouped = out.groupby("permaTicker")
        for pt, rows in grouped:
            key = f"/sp400/{pt}"
            if key not in store:
                continue
            dates = pd.to_datetime(store[key]["Date"]).sort_values().values
            for idx, row in rows.iterrows():
                rd = pd.Timestamp(row["report_date"])
                t_idx = int(np.searchsorted(dates, np.datetime64(rd), side="left"))
                offset = 2 if bool(row["is_bmo"]) else 1
                cutoff_idx = t_idx - offset
                if 0 <= cutoff_idx < len(dates):
                    out.at[idx, "_feature_date"] = pd.Timestamp(dates[cutoff_idx])

        # Analyst grades are point-in-time data too. Recompute them against
        # the executable cutoff, not report_date.
        for pt, rows in out.groupby("permaTicker"):
            grade_key = f"/analyst/grades/{pt}"
            grades = store[grade_key] if grade_key in store.keys() else None
            for idx, row in rows.iterrows():
                cutoff = row["_feature_date"]
                if pd.isna(cutoff):
                    continue
                rev = s2.compute_revision_momentum(grades, pd.Timestamp(cutoff))
                for col in revision_cols:
                    out.at[idx, col] = rev[col]

        macro_specs = {
            "vix": ("/macros/fred_vix_close", "vix"),
            "fed_funds": ("/macros/fred_fed_funds_rate", "fed_funds"),
            "unemployment": ("/macros/fred_unemployment_rate", "unemployment_roc21"),
        }
        left = out.sort_values("_feature_date").copy()
        for name, (key, target) in macro_specs.items():
            if key not in store.keys():
                continue
            macro = store[key].copy()
            macro["Date"] = pd.to_datetime(macro["Date"])
            macro = macro.sort_values("Date")
            value_col = macro.columns[1]
            macro[name] = pd.to_numeric(macro[value_col], errors="coerce")
            if name == "unemployment":
                macro[target] = macro[name].pct_change(21).replace([np.inf, -np.inf], np.nan)
            right = macro[["Date", target]].rename(columns={"Date": "_feature_date"})
            # HDF5 can return microsecond-resolution timestamps while the
            # constructed feature dates are nanosecond-resolution. Normalize
            # both merge keys before merge_asof.
            left["_feature_date"] = pd.to_datetime(left["_feature_date"]).astype("datetime64[ns]")
            right["_feature_date"] = pd.to_datetime(right["_feature_date"]).astype("datetime64[ns]")
            valid_left = left["_feature_date"].notna()
            merged = pd.merge_asof(
                left.loc[valid_left, ["_feature_date"]],
                right.sort_values("_feature_date"),
                on="_feature_date", direction="backward"
            )
            # Only overwrite rows with a valid executable cutoff. Rows without
            # enough price history retain the existing NaN/value behavior and
            # are not allowed to create a forward-looking macro observation.
            left.loc[valid_left, target] = merged[target].to_numpy()
        out = left.sort_index()
    return out.drop(columns=["_feature_date"])


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add labels/PnL/sectors and timing-correct derived features."""
    out = bt.compute_pregap(df, DB_FILE, bt.EXIT_SNAP, bt.STOP_LOSS)
    with pd.HDFStore(DB_FILE, mode="r") as store:
        meta = store["/metadata/sp400_permatickers"]
    out = out.merge(
        meta[["permaTicker", "index_ref"]].drop_duplicates("permaTicker"),
        on="permaTicker", how="left"
    )
    out["sector"] = out["index_ref"]
    out = bt.add_consecutive_pre(out)
    # Start with the existing derivation for schema/NaN behavior, then replace
    # the three daily macro fields with values available at the v4 cutoff.
    out = bt.add_macro_features(out, DB_FILE)
    out = _apply_timing_correct_information(out)
    return out


def _write_matrix(df: pd.DataFrame) -> None:
    """Write only the new v4 node; never overwrite the original matrix."""
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if V4_MATRIX_KEY in store.keys():
            store.remove(V4_MATRIX_KEY)
        store.put(V4_MATRIX_KEY, df, format="table", data_columns=["report_date", "is_bmo"])


def main() -> None:
    import xgboost as xgb

    print("=" * 88)
    print("FREEZING phase_g_v4_timing_correct")
    print("=" * 88)
    print("  AMC price cutoff: T-1")
    print("  BMO price cutoff: T-2")
    print(f"  Features: {len(DEPLOY_FEATURES)} | theta={THETA}")
    print(f"  DB: {DB_FILE}")

    print("\n[1] Loading train matrix and computing labels ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    print(f"  rows={len(df):,} | BMO={int(df['is_bmo'].sum()):,} | AMC={int((~df['is_bmo'].astype(bool)).sum()):,}")

    print("\n[2] Recomputing BMO price features with T-2 cutoff ...")
    df, timing_audit = _recompute_bmo_tminus2(df)
    print(json.dumps(timing_audit, indent=2))

    print("\n[3] Preparing strategy data and running four-fold walk-forward CV ...")
    prepared = _prepare(df)
    stats = bt.run_cv(prepared, DEPLOY_FEATURES, HP, THETA)
    if stats is None:
        raise RuntimeError("v4 walk-forward CV produced no results")
    print("  v4 OOS statistics:")
    for key in ["n", "wr", "avg", "nav", "prec_pead", "rec_pead", "f1_pead", "min_fold", "fold_range"]:
        print(f"    {key}: {stats[key]}")
    print(f"    fold_navs: {stats['fold_navs']}")

    print(f"\n[4] Writing separate matrix: {V4_MATRIX_KEY}")
    # Write the fully prepared v4 matrix, including the v4-only derived
    # features used by the classifier (prior beat streak + macros).
    _write_matrix(prepared)

    print("\n[5] Training final v4 classifier on all timing-correct rows ...")
    missing = [feature for feature in DEPLOY_FEATURES if feature not in prepared.columns]
    if missing:
        raise RuntimeError(f"Missing deploy features: {missing}")
    X = prepared[DEPLOY_FEATURES]
    y = prepared["pead_pass"].astype(int).values
    clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric=["logloss", "auc"], **HP)
    clf.fit(X, y, eval_set=[(X, y)], verbose=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    classifier_path = MODEL_DIR / "classifier.json"
    clf.save_model(str(classifier_path))

    meta = {
        "model_class": "XGBClassifier",
        "model_version": "phase_g_v4_timing_correct",
        "objective": "binary:logistic",
        "features": DEPLOY_FEATURES,
        "n_features": len(DEPLOY_FEATURES),
        "hyperparameters": HP,
        "theta": THETA,
        "training_matrix": V4_MATRIX_KEY,
        "information_set": {
            "AMC": "price/volume features through T-1 close; entry at Close[T]",
            "BMO": "price/volume features through T-2 close; entry at Close[T-1]",
            "prior_earnings_features": "historical reported quarters; unchanged and pre-event",
            "analyst_revisions": "strictly before the same T-2/T-1 execution cutoff",
            "macros": "last observation available on the same T-2/T-1 execution cutoff",
        },
        "timing_corrected_features": BMO_SHIFT_FEATURES + [
            "revision_momentum_30d", "revision_momentum_60d",
            "revision_momentum_90d", "revision_ordinal_momentum_90d",
            "revision_intensity_90d", "grade_dispersion_90d",
            "n_analysts_covering", "last_action_days_before_earnings",
            "unemployment_roc21", "fed_funds", "vix",
        ],
        "oos_stats_4fold": {
            "n_trades": int(stats["n"]),
            "win_rate_pct": float(stats["wr"]),
            "avg_pnl_per_trade_pct": float(stats["avg"]),
            "nav_compounded_pct": float(stats["nav"]),
            "pead_precision_pct": float(stats["prec_pead"]),
            "pead_recall_pct": float(stats["rec_pead"]),
            "pead_f1_pct": float(stats["f1_pead"]),
            "min_fold_nav_pct": float(stats["min_fold"]),
            "fold_range_pp": float(stats["fold_range"]),
            "fold_navs_pct": [float(x) for x in stats["fold_navs"]],
        },
        "deployment": {
            "entry": "pre-gap MOC: Close[T-1] BMO / Close[T] AMC",
            "exit": "Close[T+5]",
            "stop_loss": "-10% delayed; skip gap day",
            "exclude_sectors": ["XLF"],
            "max_slots": 4,
            "daily_decision_rule": "Use latest fully completed daily close; do not use partial intraday bars.",
        },
        "timing_audit": timing_audit,
        "source_v3_model": "phase_g_v3_honest",
    }
    with open(MODEL_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  Classifier: {classifier_path}")
    print(f"  Metadata:   {MODEL_DIR / 'meta.json'}")
    print("  v3 artifact was not modified.")
    print("=" * 88)


if __name__ == "__main__":
    main()

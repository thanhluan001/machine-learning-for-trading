#!/usr/bin/env python3
"""
Option B: Retrain on 19 TRULY pre-event features (drop 5 SUE look-ahead features).

The 24-feature model had look-ahead bias: 5 features (sue_score,
eps_surprise_pct, consecutive_surprises, sue_acceleration, sue_abs_x_inverse_vol)
use the CURRENT earnings result, which isn't available at pre-gap entry time.

This script retrains on 19 features available strictly BEFORE earnings,
with the SAME operating point: theta=0.20, pre-gap entry, 5-day hold,
-10% delayed stop, exclude XLF, 4-slot weekly batch selection.

If the edge survives, this becomes the new deployable model.
If it collapses, the pre-gap alpha was mostly look-ahead.
"""
import sys, io, importlib.util
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tm", HERE.parent / "03_model" / "01_train_model.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
pg_spec = importlib.util.spec_from_file_location("pg", HERE.parent / "03_model" / "02_phase_g_sunday_classifier.py")
pg = importlib.util.module_from_spec(pg_spec); pg_spec.loader.exec_module(pg)
v3_spec = importlib.util.spec_from_file_location("v3", HERE / "_pead_target_retrain.py")
v3 = importlib.util.module_from_spec(v3_spec); v3_spec.loader.exec_module(v3)

DB = tm.DB_FILE
N_SLOTS = 4
EXIT_SNAP = 5
THETA = 0.20
STOP_LOSS = 0.10
EXCLUDE_SECTORS = ["XLF"]

# ======================================================================
# THE 19 TRULY PRE-EVENT FEATURES (available BEFORE earnings)
# ======================================================================
# DROPPED (look-ahead — need current earnings result):
#   sue_score, eps_surprise_pct, consecutive_surprises,
#   sue_acceleration, sue_abs_x_inverse_vol
#
# KEPT: prior SUE history + price/volume momentum + analyst revisions
PRE_EVENT_FEATURES = [
    # Prior earnings history (available before current earnings)
    "sue_lag_1", "sue_lag_2", "car_drift_historical_q1",
    # Price/volume momentum (available up to T-1 close)
    "pre_event_idiosyncratic_vol", "pre_event_volume_trend",
    "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d", "rel_ret_30d",
    "sector_adjusted_ret_20d",
    # Analyst revision momentum (available before earnings)
    "revision_momentum_30d", "revision_momentum_60d", "revision_momentum_90d",
    "revision_ordinal_momentum_90d", "revision_intensity_90d",
    "grade_dispersion_90d", "n_analysts_covering", "last_action_days_before_earnings",
]

DROPPED_FEATURES = [
    "sue_score", "eps_surprise_pct", "consecutive_surprises",
    "sue_acceleration", "sue_abs_x_inverse_vol",
]

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]


def fit_clf(X_tr, y_tr, X_val, y_val, hp):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=hp["n_estimators"], learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def compute_pregap_with_path(df, db_path, hold_days=5, stop_loss=0.10):
    df = df.copy()
    df["pregap_return"] = np.nan
    df["pregap_entry_date"] = pd.NaT
    df["pregap_exit_date"] = pd.NaT
    with pd.HDFStore(db_path, mode="r") as s:
        for idx, row in df.iterrows():
            pt = row["permaTicker"]
            key = f"/sp400/{pt}"
            if key not in s:
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_close = p["Adj_Close"].values
            rdate = pd.to_datetime(row["report_date"]).to_datetime64()
            t_mask = p_index >= rdate
            if not t_mask.any():
                continue
            t_idx = int(np.argmax(t_mask))
            is_bmo = bool(row.get("is_bmo", False))
            entry_t = t_idx - 1 if is_bmo else t_idx
            exit_t = t_idx + hold_days
            if entry_t < 0 or exit_t >= len(p_close):
                continue
            entry_price = p_close[entry_t]
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            exit_price = p_close[exit_t]
            if pd.isna(exit_price):
                continue
            hold_ret = float(exit_price / entry_price - 1.0)
            stop_price = entry_price * (1.0 - stop_loss)
            path_start = t_idx
            path_end = exit_t + 1
            path_prices = p_close[path_start:path_end]
            path_valid = path_prices[~np.isnan(path_prices)]
            stop_exit = 0
            final_ret = hold_ret
            for sp in path_valid[1:]:  # skip gap day
                if not np.isnan(sp) and sp <= stop_price:
                    final_ret = float(sp / entry_price - 1.0)
                    stop_exit = 1
                    break
            df.at[idx, "pregap_return"] = final_ret
            df.at[idx, "pregap_entry_date"] = pd.Timestamp(p_index[entry_t])
            df.at[idx, "pregap_exit_date"] = pd.Timestamp(p_index[exit_t])
    return df


def select_weekly_top_n(picks, n_slots=4, sort_col="p"):
    if picks.empty:
        return picks
    pk = picks.copy()
    pk["entry_date"] = pd.to_datetime(pk["entry_date"])
    pk["exit_date"] = pd.to_datetime(pk["exit_date"])
    iso = pk["entry_date"].dt.isocalendar()
    pk["_week_key"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    selected_rows = []
    active_positions = []
    for week_key, week_df in pk.groupby("_week_key", sort=True):
        week_start = week_df["entry_date"].min()
        active_positions = [ex for ex in active_positions if ex >= week_start]
        free_slots = n_slots - len(active_positions)
        if free_slots <= 0:
            continue
        week_sorted = week_df.sort_values(sort_col, ascending=False)
        taken = week_sorted.head(min(free_slots, len(week_sorted)))
        selected_rows.append(taken)
        for _, row in taken.iterrows():
            active_positions.append(row["exit_date"])
    if selected_rows:
        return pd.concat(selected_rows).sort_values("entry_date").reset_index(drop=True)
    return pd.DataFrame(columns=picks.columns)


def main():
    print("=" * 100)
    print("OPTION B: 19-FEATURE PRE-EVENT MODEL (NO SUE LOOK-AHEAD)")
    print("Dropped:", DROPPED_FEATURES)
    print("Kept:   ", len(PRE_EVENT_FEATURES), "features")
    print("=" * 100)

    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = compute_pregap_with_path(df, DB, EXIT_SNAP, STOP_LOSS)

    with pd.HDFStore(DB, mode="r") as s:
        pt_meta = s["/metadata/sp400_permatickers"]
    sector_lookup = pt_meta[["permaTicker", "index_ref"]].drop_duplicates("permaTicker")
    df = df.merge(sector_lookup, on="permaTicker", how="left")
    df["sector"] = df["index_ref"]

    # Verify features exist
    missing = [f for f in PRE_EVENT_FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")

    hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
    fold_data = {}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_ts = pd.concat([train_df[PRE_EVENT_FEATURES], sweep_df[PRE_EVENT_FEATURES]])
        y_ts = pd.concat([train_df, sweep_df])["pead_pass"].astype(int).values
        y_te = test_df["pead_pass"].astype(int).values
        clf = fit_clf(X_ts, y_ts, test_df[PRE_EVENT_FEATURES], y_te, hp)
        test_df = test_df.copy()
        test_df["p"] = clf.predict_proba(test_df[PRE_EVENT_FEATURES])[:, 1]
        fold_data[fi] = {"test_df": test_df}

    # Execute trades
    all_exec = []
    for fi in range(1, 5):
        td = fold_data[fi]["test_df"]
        mask = (td["p"] >= THETA) & (td["pregap_return"].notna()) & (~td["sector"].isin(EXCLUDE_SECTORS))
        picks = td[mask].copy()
        if len(picks) == 0:
            continue
        picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
        picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
        picks["fold"] = fi
        sel = select_weekly_top_n(picks, N_SLOTS, sort_col="p")
        if len(sel) > 0:
            all_exec.append(sel)
    exec_df = pd.concat(all_exec).reset_index(drop=True)

    pnls = exec_df["pregap_return"].dropna()
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n_pead = int(exec_df["pead_pass"].sum())

    # ===== HEADLINE COMPARISON =====
    print(f"\n{'='*100}")
    print("HEADLINE: 19-feature (no SUE) vs 24-feature (with SUE look-ahead)")
    print(f"{'='*100}")
    print(f"\n  {'Metric':<35} {'19-feat (honest)':>18} {'24-feat (look-ahead)':>20}")
    print("  " + "-" * 75)
    print(f"  {'Trades':<35} {n:>18} {'101':>20}")
    print(f"  {'Win rate':<35} {len(wins)/n*100:>17.1f}% {'75.2%':>20}")
    print(f"  {'PEAD precision':<35} {n_pead/n*100:>17.1f}% {'38.6%':>20}")
    print(f"  {'Expectancy/trade':<35} {pnls.mean()*100:>+17.2f}% {'6.72%':>20}")
    print(f"  {'Median return':<35} {pnls.median()*100:>+17.2f}% {'5.99%':>20}")
    print(f"  {'Avg win':<35} {wins.mean()*100:>+17.2f}% {'11.66%':>20}")
    print(f"  {'Avg loss':<35} {losses.mean()*100:>+17.2f}% {'-8.28%':>20}")
    print(f"  {'Payoff ratio':<35} {wins.mean()/abs(losses.mean()):>18.2f} {'1.41':>20}")
    print(f"  {'Total PnL (raw sum)':<35} {pnls.sum()*100:>+17.1f}% {'672.4%':>20}")

    # NAV-compounded
    nav = 1.0
    iso_all = exec_df["entry_date"].dt.isocalendar()
    exec_df["week_key"] = iso_all["year"].astype(str) + "-W" + iso_all["week"].astype(str).str.zfill(2)
    for week_key, week_df in exec_df.groupby("week_key", sort=True):
        trades = week_df["pregap_return"].dropna()
        if len(trades) == 0:
            continue
        week_ret = (trades * (1.0 / N_SLOTS)).sum()
        nav *= (1 + week_ret)
    print(f"  {'Total PnL (NAV-compounded)':<35} {(nav-1)*100:>+17.1f}% {'391.3%':>20}")
    print(f"  {'NAV multiplier':<35} {nav:>18.2f}x {'4.91x':>20}")

    # ===== PER-FOLD BREAKDOWN =====
    print(f"\n{'='*100}")
    print("PER-FOLD BREAKDOWN (19-feature)")
    print(f"{'='*100}")
    print(f"\n  {'Fold':>6} {'Period':<18} {'N':>4} {'Win%':>6} {'Avg':>8} {'Raw sum':>9} {'NAV-comp':>9} {'PEAD':>6}")
    print("  " + "-" * 70)
    for fi in range(1, 5):
        sub = exec_df[exec_df["fold"] == fi]
        if len(sub) == 0:
            continue
        r = sub["pregap_return"].dropna()
        wr = (r > 0).mean() * 100
        avg = r.mean() * 100
        raw = r.sum() * 100
        # NAV-comp per fold
        nav_f = 1.0
        iso_f = sub["entry_date"].dt.isocalendar()
        sub_wk = iso_f["year"].astype(str) + "-W" + iso_f["week"].astype(str).str.zfill(2)
        sub2 = sub.copy()
        sub2["wk"] = sub_wk
        for wk, wdf in sub2.groupby("wk", sort=True):
            tr = wdf["pregap_return"].dropna()
            if len(tr) > 0:
                nav_f *= (1 + (tr * (1.0/N_SLOTS)).sum())
        nav_ret = (nav_f - 1) * 100
        n_p = int(sub["pead_pass"].sum())
        period = DEFAULT_FOLDS[fi-1][1][:7] + " -> " + DEFAULT_FOLDS[fi-1][2][:7]
        print(f"  {fi:>6} {period:<18} {len(r):>4} {wr:>5.0f}% {avg:>+7.2f}% {raw:>+8.1f}% {nav_ret:>+8.1f}% {n_p:>5}")

    # ===== PEAD vs NON-PEAD =====
    print(f"\n{'='*100}")
    print("PEAD vs NON-PEAD (19-feature)")
    print(f"{'='*100}")
    pead = exec_df[exec_df["pead_pass"] == 1]
    fp = exec_df[exec_df["pead_pass"] == 0]
    print(f"\n  {'Group':<20} {'N':>4} {'Win%':>6} {'Avg':>8} {'Total':>10}")
    print("  " + "-" * 50)
    for label, sub in [("True PEAD", pead), ("False positive", fp), ("ALL", exec_df)]:
        r = sub["pregap_return"].dropna()
        if len(r) == 0:
            continue
        wr = (r > 0).mean() * 100
        avg = r.mean() * 100
        total = r.sum() * 100
        print(f"  {label:<20} {len(r):>4} {wr:>5.0f}% {avg:>+7.2f}% {total:>+9.1f}%")

    # ===== FEATURE IMPORTANCE =====
    print(f"\n{'='*100}")
    print("FEATURE IMPORTANCE (19-feature model, fold 4)")
    print(f"{'='*100}")
    # Retrain on all data for importance
    X_all = df[PRE_EVENT_FEATURES]
    y_all = df["pead_pass"].astype(int).values
    clf_full = fit_clf(X_all, y_all, X_all, y_all, hp)
    imp = clf_full.feature_importances_
    idx_sorted = np.argsort(imp)[::-1]
    print(f"\n  {'Rank':>4} {'Feature':<35} {'Importance':>10}")
    print("  " + "-" * 50)
    for rank, idx in enumerate(idx_sorted, 1):
        print(f"  {rank:>4} {PRE_EVENT_FEATURES[idx]:<35} {imp[idx]:>9.4f}")

    # ===== RETURN DISTRIBUTION =====
    print(f"\n{'='*100}")
    print("RETURN DISTRIBUTION BUCKETS (19-feature)")
    print(f"{'='*100}")
    buckets = [
        ("<= -20%", pnls <= -0.20),
        ("-20% to -10%", (pnls > -0.20) & (pnls <= -0.10)),
        ("-10% to -5%", (pnls > -0.10) & (pnls <= -0.05)),
        ("-5% to 0%", (pnls > -0.05) & (pnls <= 0)),
        ("0% to +5%", (pnls > 0) & (pnls <= 0.05)),
        ("+5% to +10%", (pnls > 0.05) & (pnls <= 0.10)),
        ("+10% to +20%", (pnls > 0.10) & (pnls <= 0.20)),
        ("+20% to +30%", (pnls > 0.20) & (pnls <= 0.30)),
        ("> +30%", pnls > 0.30),
    ]
    print(f"\n  {'Bucket':<20} {'Count':>6} {'%':>6}")
    print("  " + "-" * 35)
    for label, mask in buckets:
        cnt = int(mask.sum())
        pct = cnt / n * 100
        bar = "#" * int(pct)
        print(f"  {label:<20} {cnt:>6} {pct:>5.1f}% {bar}")

    # ===== VERDICT =====
    print(f"\n{'='*100}")
    print("VERDICT")
    print(f"{'='*100}")
    avg_19 = pnls.mean() * 100
    nav_19 = (nav - 1) * 100
    if avg_19 > 3.0:
        print(f"\n  EDGE SURVIVES without SUE. Pre-gap alpha is real (not look-ahead).")
        print(f"  Expectancy: {avg_19:+.2f}%/trade (vs +6.72% with look-ahead)")
        print(f"  NAV-compounded: {nav_19:+.1f}% (vs +391.3% with look-ahead)")
        print(f"  => This becomes the new deployable model.")
    elif avg_19 > 1.0:
        print(f"\n  EDGE WEAKENED but survives. Some pre-gap alpha was look-ahead,")
        print(f"  some is real. Expectancy: {avg_19:+.2f}%/trade.")
        print(f"  => Usable but lower. May need HP re-tuning.")
    elif avg_19 > 0:
        print(f"\n  EDGE BARELY POSITIVE. Most pre-gap alpha was look-ahead.")
        print(f"  Expectancy: {avg_19:+.2f}%/trade. Marginal at best.")
    else:
        print(f"\n  EDGE DEAD. Pre-gap alpha was entirely look-ahead bias.")
        print(f"  Expectancy: {avg_19:+.2f}%/trade. Must rethink strategy.")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

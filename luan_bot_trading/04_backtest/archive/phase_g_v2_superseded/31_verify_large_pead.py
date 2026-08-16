#!/usr/bin/env python3
"""
Verify the 19 "Large PEAD" tickers (instructor feedback #3).

Pulls the actual executed trades labeled Large PEAD (class 2) from the
3-class P(any)>=0.20 backtest and analyzes:
  1. Per-trade detail (ticker, date, return, P(any), P(large), sector)
  2. Sector concentration
  3. Market regime (VIX at time of trade)
  4. Market cap / size proxy (avg volume)
  5. Time concentration (are they clustered in specific months?)
  6. BMO vs AMC split
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
SUNDAY_SAFE = pg.SUNDAY_SAFE_FEATURES
N_SLOTS = 4
EXIT_SNAP = 5
CAR_LARGE_THRESH = 10.0
THETA = 0.20

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]


def fit_clf_3class(X_tr, y_tr, X_val, y_val, hp):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss",
        n_estimators=hp["n_estimators"], learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


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


def compute_pregap_returns(df, db_path, hold_days=5):
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
            exit_price = p_close[exit_t]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue
            df.at[idx, "pregap_return"] = float(exit_price / entry_price - 1.0)
            df.at[idx, "pregap_entry_date"] = pd.Timestamp(p_index[entry_t])
            df.at[idx, "pregap_exit_date"] = pd.Timestamp(p_index[exit_t])
    return df


def get_vix_at_date(date, vix_index, vix_values):
    """Get VIX value closest to (but before) the given date."""
    idx = np.searchsorted(vix_index, np.datetime64(date), side="right") - 1
    if idx < 0 or idx >= len(vix_values):
        return np.nan
    return float(vix_values[idx])


def get_avg_volume(permaTicker, report_date, db_path, lookback=20):
    """Get avg Adj_Volume in the 20 days before report_date (size proxy)."""
    with pd.HDFStore(db_path, mode="r") as s:
        key = f"/sp400/{permaTicker}"
        if key not in s:
            return np.nan
        p = s[key]
        p_index = pd.to_datetime(p["Date"]).values
        p_vol = p["Adj_Volume"].values
        rdate = np.datetime64(pd.Timestamp(report_date))
        mask = p_index < rdate
        if mask.sum() < 1:
            return np.nan
        recent = p_vol[mask][-lookback:]
        return float(np.nanmean(recent))


def main():
    print("=" * 100)
    print("VERIFY THE 19 LARGE PEAD TICKERS (instructor feedback #3)")
    print("=" * 100)

    print("\n[1] Loading + priming + gates ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)

    df["car_10d_pct"] = np.expm1(df["car_10d"]) * 100
    df["label_3class"] = 0
    df.loc[(df["pead_pass"] == 1) & (df["car_10d_pct"] < CAR_LARGE_THRESH), "label_3class"] = 1
    df.loc[(df["pead_pass"] == 1) & (df["car_10d_pct"] >= CAR_LARGE_THRESH), "label_3class"] = 2

    print("[2] Computing pre-gap returns ...")
    df = compute_pregap_returns(df, DB, EXIT_SNAP)

    # Load VIX for regime analysis
    print("[3] Loading VIX + sector data ...")
    with pd.HDFStore(DB, mode="r") as s:
        if "/macros/fred_vix_close" in s.keys():
            vix_df = s["/macros/fred_vix_close"]
            vix_index = pd.to_datetime(vix_df["Date"]).values
            vix_values = vix_df["vix_close"].values
            print(f"    VIX: {len(vix_df)} rows")
        else:
            vix_index = None
            vix_values = None
            print("    VIX not found")
        # Sector from permatickers metadata (index_ref = sector ETF)
        if "/metadata/sp400_permatickers" in s.keys():
            pt_meta = s["/metadata/sp400_permatickers"]
            print(f"    Permatickers: {len(pt_meta)} rows, index_ref values: {pt_meta['index_ref'].value_counts().to_dict()}")

    print("[4] Training 3-class classifiers per fold ...")
    fold_data = {}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_tr = train_df[SUNDAY_SAFE]; X_sv = sweep_df[SUNDAY_SAFE]
        X_te = test_df[SUNDAY_SAFE]
        X_ts = pd.concat([X_tr, X_sv])
        y_tr_3 = train_df["label_3class"].values
        y_sv_3 = sweep_df["label_3class"].values
        y_te_3 = test_df["label_3class"].values
        y_ts_3 = np.concatenate([y_tr_3, y_sv_3])
        hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
        clf_3 = fit_clf_3class(X_ts, y_ts_3, X_te, y_te_3, hp)
        proba_3 = clf_3.predict_proba(X_te)
        test_df = test_df.copy()
        test_df["p_any_pead"] = proba_3[:, 1] + proba_3[:, 2]
        test_df["p_large"] = proba_3[:, 2]
        test_df["p_small"] = proba_3[:, 1]
        fold_data[fi] = {"test_df": test_df}

    print("[5] Running weekly batch selection ...")
    all_exec_list = []
    for fi in range(1, 5):
        test_df = fold_data[fi]["test_df"]
        mask = (test_df["p_any_pead"] >= THETA) & (test_df["pregap_return"].notna())
        picks = test_df[mask].copy()
        if len(picks) == 0:
            continue
        picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
        picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
        picks["fold"] = fi
        selected = select_weekly_top_n(picks, N_SLOTS, sort_col="p_any_pead")
        if len(selected) > 0:
            all_exec_list.append(selected)

    exec_df = pd.concat(all_exec_list).reset_index(drop=True)
    print(f"  Total executed: {len(exec_df)}")
    large_df = exec_df[exec_df["label_3class"] == 2].copy()
    print(f"  Large PEAD: {len(large_df)}")

    # Enrich with VIX, volume, sector
    print("\n[6] Enriching with VIX, volume, sector ...")
    if vix_index is not None:
        large_df["vix_at_entry"] = large_df["pregap_entry_date"].apply(
            lambda d: get_vix_at_date(d, vix_index, vix_values))
    large_df["avg_volume_20d"] = large_df.apply(
        lambda r: get_avg_volume(r["permaTicker"], r["report_date"], DB), axis=1)

    # Sector lookup (index_ref = sector ETF)
    if "pt_meta" in dir():
        sector_lookup = pt_meta[["permaTicker", "index_ref"]].drop_duplicates("permaTicker")
        large_df = large_df.merge(sector_lookup, on="permaTicker", how="left")
        large_df.rename(columns={"index_ref": "sector"}, inplace=True)

    # ===== PER-TRADE DETAIL =====
    print(f"\n{'='*100}")
    print(f"1. THE 19 LARGE PEAD TRADES (per-trade detail)")
    print(f"{'='*100}")

    print(f"\n  {'#':>3} {'Ticker':<8} {'Date':<12} {'Ret':>8} {'P(any)':>8} {'P(large)':>9} "
          f"{'BMO/AMC':>8} {'VIX':>6} {'AvgVol':>12} {'Sector':<20} {'Fold':>5}")
    print("  " + "-" * 105)
    for i, (_, t) in enumerate(large_df.sort_values("report_date").iterrows(), 1):
        ticker = t.get("canonical_ticker", t["permaTicker"])
        rd = str(pd.Timestamp(t["report_date"]).date())
        ret = t["pregap_return"] * 100
        p_any = t["p_any_pead"]
        p_large = t["p_large"]
        bam = "BMO" if t.get("is_bmo", False) else "AMC"
        vix = t.get("vix_at_entry", np.nan)
        vol = t.get("avg_volume_20d", np.nan)
        sector = t.get("sector", "?")
        fold = int(t["fold"])
        vol_str = f"{vol/1e6:.1f}M" if pd.notna(vol) else "?"
        vix_str = f"{vix:.1f}" if pd.notna(vix) else "?"
        print(f"  {i:>3} {ticker:<8} {rd:<12} {ret:>+7.2f}% {p_any:>7.3f} {p_large:>8.3f} "
              f"{bam:>8} {vix_str:>6} {vol_str:>12} {str(sector):<20} {fold:>5}")

    # ===== CONCENTRATION ANALYSIS =====
    print(f"\n{'='*100}")
    print("2. CONCENTRATION ANALYSIS")
    print(f"{'='*100}")

    # By sector
    print(f"\n  A. SECTOR CONCENTRATION")
    if "sector" in large_df.columns:
        sec_counts = large_df["sector"].value_counts()
        for sec, n in sec_counts.items():
            sub = large_df[large_df["sector"] == sec]
            avg_ret = sub["pregap_return"].mean() * 100
            print(f"    {str(sec):<25} {n:>3} trades  avg={avg_ret:+.2f}%")
    else:
        print("    (no sector data)")

    # By BMO/AMC
    print(f"\n  B. BMO vs AMC")
    large_df["is_bmo"] = large_df["is_bmo"].fillna(0).astype(int) == 1
    for label, mask in [("BMO", large_df["is_bmo"]), ("AMC", ~large_df["is_bmo"])]:
        sub = large_df[mask]
        if len(sub) > 0:
            print(f"    {label}: {len(sub)} trades  avg={sub['pregap_return'].mean()*100:+.2f}%")

    # By VIX regime
    print(f"\n  C. VIX REGIME (at entry)")
    if "vix_at_entry" in large_df.columns:
        vix_vals = large_df["vix_at_entry"].dropna()
        print(f"    VIX range: {vix_vals.min():.1f} - {vix_vals.max():.1f}, median={vix_vals.median():.1f}")
        for lo, hi, label in [(0, 15, "low (<15)"), (15, 20, "normal (15-20)"),
                               (20, 30, "elevated (20-30)"), (30, 100, "high (>30)")]:
            sub = large_df[(large_df["vix_at_entry"] >= lo) & (large_df["vix_at_entry"] < hi)]
            if len(sub) > 0:
                print(f"    {label:<20} {len(sub):>3} trades  avg={sub['pregap_return'].mean()*100:+.2f}%")

    # By market cap proxy (avg volume)
    print(f"\n  D. SIZE PROXY (avg 20-day volume, pre-event)")
    if "avg_volume_20d" in large_df.columns:
        vol_vals = large_df["avg_volume_20d"].dropna()
        print(f"    Volume range: {vol_vals.min()/1e6:.1f}M - {vol_vals.max()/1e6:.1f}M, "
              f"median={vol_vals.median()/1e6:.1f}M")
        med = vol_vals.median()
        small = large_df[large_df["avg_volume_20d"] < med]
        big = large_df[large_df["avg_volume_20d"] >= med]
        print(f"    Below median vol: {len(small)} trades  avg={small['pregap_return'].mean()*100:+.2f}%")
        print(f"    Above median vol: {len(big)} trades  avg={big['pregap_return'].mean()*100:+.2f}%")

    # By time (monthly)
    print(f"\n  E. TIME CONCENTRATION (by month)")
    large_df["year_month"] = pd.to_datetime(large_df["report_date"]).dt.strftime("%Y-%m")
    month_counts = large_df["year_month"].value_counts().sort_index()
    for ym, n in month_counts.items():
        sub = large_df[large_df["year_month"] == ym]
        avg_ret = sub["pregap_return"].mean() * 100
        print(f"    {ym}: {n} trades  avg={avg_ret:+.2f}%")

    # By fold
    print(f"\n  F. BY FOLD")
    for fi in range(1, 5):
        sub = large_df[large_df["fold"] == fi]
        if len(sub) > 0:
            print(f"    Fold {fi}: {len(sub)} trades  avg={sub['pregap_return'].mean()*100:+.2f}%")

    # ===== KEY QUESTIONS =====
    print(f"\n{'='*100}")
    print("3. KEY QUESTIONS")
    print(f"{'='*100}")

    # Are they concentrated in specific tickers? (same ticker multiple times)
    print(f"\n  A. Same ticker multiple times?")
    dup = large_df["permaTicker"].value_counts()
    dups = dup[dup > 1]
    if len(dups) > 0:
        print(f"    YES - {len(dups)} tickers appear multiple times:")
        for pt, n in dups.items():
            ticker = large_df[large_df["permaTicker"] == pt]["canonical_ticker"].iloc[0]
            print(f"      {ticker}: {n} times")
    else:
        print(f"    NO - all {len(large_df)} trades are distinct tickers")

    # Are returns driven by 1-2 outliers?
    print(f"\n  B. Return distribution (outlier check)")
    rets = large_df["pregap_return"].values * 100
    print(f"    min={rets.min():+.2f}%, median={np.median(rets):+.2f}%, max={rets.max():+.2f}%")
    print(f"    Top 3: {np.sort(rets)[-3:]}")
    print(f"    Bottom 3: {np.sort(rets)[:3]}")
    print(f"    If we remove top 1 ({rets.max():+.2f}%), avg drops from {rets.mean():+.2f}% to "
          f"{(rets.sum()-rets.max())/(len(rets)-1):+.2f}%")
    print(f"    If we remove top 3, avg drops to {(rets.sum()-np.sort(rets)[-3:].sum())/(len(rets)-3):+.2f}%")

    # EPS surprise magnitude
    print(f"\n  C. EPS surprise magnitude")
    if "eps_surprise_pct" in large_df.columns:
        eps = large_df["eps_surprise_pct"].dropna()
        print(f"    EPS surprise: min={eps.min():+.1f}%, median={eps.median():+.1f}%, max={eps.max():+.1f}%")
        print(f"    (Are these all massive earnings beats?)")

    # SUE score
    print(f"\n  D. SUE score (standardized unexpected earnings)")
    if "sue_score" in large_df.columns:
        sue = large_df["sue_score"].dropna()
        print(f"    SUE: min={sue.min():+.2f}, median={sue.median():+.2f}, max={sue.max():+.2f}")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

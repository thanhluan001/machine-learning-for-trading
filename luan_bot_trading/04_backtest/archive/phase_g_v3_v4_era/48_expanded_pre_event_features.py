#!/usr/bin/env python3
"""
Option B+: Expand pre-event features using PRIOR-QUARTER earnings history.

The 5 dropped features (sue_score, eps_surprise_pct, consecutive_surprises,
sue_acceleration, sue_abs_x_inverse_vol) all used the CURRENT earnings result.
But the same DIMENSIONS can be captured from PRIOR quarters:

  sue_score          → sue_mean4q_pre (4Q avg SUE), sue_std4q_pre (consistency)
  eps_surprise_pct   → eps_surprise_pct_lag1, eps_surprise_pct_ma4
  consecutive_surp   → consecutive_surprises_pre (streak BEFORE current Q)
  sue_acceleration   → sue_accel_lag1 (sue_lag_1 - sue_lag_2)
  sue_abs_x_inv_vol  → sue_abs_lag1_x_vol (|prior SUE| / vol)

All use .shift(1) to exclude current quarter. Tests whether richer prior-quarter
patterns recover the alpha lost by dropping current SUE.
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

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]


def add_prior_earnings_features(df):
    """Add pre-event features derived from PRIOR quarters' earnings data."""
    df = df.sort_values(["permaTicker", "report_date"]).copy()

    # Group by permaTicker to compute per-company earnings history
    for pt, grp in df.groupby("permaTicker"):
        idx = grp.index

        # 1. consecutive_surprises_pre: beat streak BEFORE current quarter
        #    = consecutive_surprises.shift(1)
        if "consecutive_surprises" in grp.columns:
            df.loc[idx, "consecutive_surprises_pre"] = grp["consecutive_surprises"].shift(1)

        # 2. eps_surprise_pct_lag1: last quarter's surprise %
        if "eps_surprise_pct" in grp.columns:
            df.loc[idx, "eps_surprise_pct_lag1"] = grp["eps_surprise_pct"].shift(1)
            # 3. eps_surprise_pct_ma4: 4Q rolling mean (shifted to exclude current)
            df.loc[idx, "eps_surprise_pct_ma4"] = grp["eps_surprise_pct"].shift(1).rolling(4, min_periods=2).mean()

        # 4. sue_mean4q_pre: 4Q rolling mean of SUE (shifted)
        if "sue_score" in grp.columns:
            df.loc[idx, "sue_mean4q_pre"] = grp["sue_score"].shift(1).rolling(4, min_periods=2).mean()
            # 5. sue_std4q_pre: SUE consistency (rolling std)
            df.loc[idx, "sue_std4q_pre"] = grp["sue_score"].shift(1).rolling(4, min_periods=2).std()

        # 6. sue_accel_lag1: change in SUE from Q-2 to Q-1
        if "sue_lag_1" in grp.columns and "sue_lag_2" in grp.columns:
            df.loc[idx, "sue_accel_lag1"] = grp["sue_lag_1"] - grp["sue_lag_2"]

        # 7. sue_abs_lag1_x_vol: |prior SUE| / inverse vol (proxy for sue_abs_x_inverse_vol)
        if "sue_lag_1" in grp.columns and "pre_event_idiosyncratic_vol" in grp.columns:
            vol = grp["pre_event_idiosyncratic_vol"]
            df.loc[idx, "sue_abs_lag1_x_vol"] = grp["sue_lag_1"].abs() / vol.replace(0, np.nan)

    return df


# Original 19 pre-event features
PRE_EVENT_19 = [
    "sue_lag_1", "sue_lag_2", "car_drift_historical_q1",
    "pre_event_idiosyncratic_vol", "pre_event_volume_trend",
    "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d", "rel_ret_30d",
    "sector_adjusted_ret_20d",
    "revision_momentum_30d", "revision_momentum_60d", "revision_momentum_90d",
    "revision_ordinal_momentum_90d", "revision_intensity_90d",
    "grade_dispersion_90d", "n_analysts_covering", "last_action_days_before_earnings",
]

# 7 NEW prior-quarter features
NEW_PRIOR_FEATURES = [
    "consecutive_surprises_pre",
    "eps_surprise_pct_lag1",
    "eps_surprise_pct_ma4",
    "sue_mean4q_pre",
    "sue_std4q_pre",
    "sue_accel_lag1",
    "sue_abs_lag1_x_vol",
]

# Combined: 26 features
EXPANDED_FEATURES = PRE_EVENT_19 + NEW_PRIOR_FEATURES


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


def compute_pregap(df, db_path, hold_days=5, stop_loss=0.10):
    df = df.copy()
    df["pregap_return"] = np.nan
    df["pregap_entry_date"] = pd.NaT
    df["pregap_exit_date"] = pd.NaT
    with pd.HDFStore(db_path, mode="r") as s:
        for idx, row in df.iterrows():
            pt = row["permaTicker"]; key = f"/sp400/{pt}"
            if key not in s: continue
            p = s[key]; p_index = pd.to_datetime(p["Date"]).values; p_close = p["Adj_Close"].values
            rdate = pd.to_datetime(row["report_date"]).to_datetime64()
            t_mask = p_index >= rdate
            if not t_mask.any(): continue
            t_idx = int(np.argmax(t_mask))
            is_bmo = bool(row.get("is_bmo", False))
            entry_t = t_idx - 1 if is_bmo else t_idx; exit_t = t_idx + hold_days
            if entry_t < 0 or exit_t >= len(p_close): continue
            ep = p_close[entry_t]
            if pd.isna(ep) or ep <= 0: continue
            xp = p_close[exit_t]
            if pd.isna(xp): continue
            hold_ret = float(xp / ep - 1.0)
            stop_price = ep * (1.0 - stop_loss)
            path_prices = p_close[t_idx:exit_t+1]
            path_valid = path_prices[~np.isnan(path_prices)]
            final_ret = hold_ret
            for sp in path_valid[1:]:
                if not np.isnan(sp) and sp <= stop_price:
                    final_ret = float(sp / ep - 1.0); break
            df.at[idx, "pregap_return"] = final_ret
            df.at[idx, "pregap_entry_date"] = pd.Timestamp(p_index[entry_t])
            df.at[idx, "pregap_exit_date"] = pd.Timestamp(p_index[exit_t])
    return df


def select_weekly(picks, n_slots=4, sort_col="p"):
    if picks.empty: return picks
    pk = picks.copy()
    pk["entry_date"] = pd.to_datetime(pk["entry_date"]); pk["exit_date"] = pd.to_datetime(pk["exit_date"])
    iso = pk["entry_date"].dt.isocalendar()
    pk["_week_key"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    sel = []; active = []
    for wk, wdf in pk.groupby("_week_key", sort=True):
        ws = wdf["entry_date"].min()
        active = [ex for ex in active if ex >= ws]
        free = n_slots - len(active)
        if free <= 0: continue
        taken = wdf.sort_values(sort_col, ascending=False).head(min(free, len(wdf)))
        sel.append(taken)
        for _, r in taken.iterrows(): active.append(r["exit_date"])
    return pd.concat(sel).sort_values("entry_date").reset_index(drop=True) if sel else pd.DataFrame()


def run_model(df, features, label, hp):
    """Run 4-fold nested CV with given feature set, return exec_df."""
    fold_data = {}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_ts = pd.concat([train_df[features], sweep_df[features]])
        y_ts = pd.concat([train_df, sweep_df])["pead_pass"].astype(int).values
        y_te = test_df["pead_pass"].astype(int).values
        clf = fit_clf(X_ts, y_ts, test_df[features], y_te, hp)
        test_df = test_df.copy()
        test_df["p"] = clf.predict_proba(test_df[features])[:, 1]
        fold_data[fi] = test_df

    all_exec = []
    for fi in range(1, 5):
        td = fold_data[fi]
        mask = (td["p"] >= THETA) & (td["pregap_return"].notna()) & (~td["sector"].isin(EXCLUDE_SECTORS))
        picks = td[mask].copy()
        if len(picks) == 0: continue
        picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
        picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
        picks["fold"] = fi
        sel = select_weekly(picks, N_SLOTS)
        if len(sel) > 0: all_exec.append(sel)
    return pd.concat(all_exec).reset_index(drop=True) if all_exec else pd.DataFrame()


def report(exec_df, label):
    if len(exec_df) == 0:
        print(f"\n  {label}: NO TRADES")
        return
    pnls = exec_df["pregap_return"].dropna()
    n = len(pnls)
    wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
    n_pead = int(exec_df["pead_pass"].sum())

    # NAV-compounded
    iso = exec_df["entry_date"].dt.isocalendar()
    ed2 = exec_df.copy(); ed2["wk"] = iso["year"].astype(str)+"-W"+iso["week"].astype(str).str.zfill(2)
    nav = 1.0; wr_list = []
    for wk, wdf in ed2.groupby("wk", sort=True):
        tr = wdf["pregap_return"].dropna()
        if len(tr) > 0:
            wr_ = (tr * (1.0/N_SLOTS)).sum()
            nav *= (1+wr_); wr_list.append(wr_)
    cum = np.cumsum(wr_list) if wr_list else [0]
    max_dd = (cum - np.maximum.accumulate(cum)).min()*100 if wr_list else 0

    print(f"\n  {label}:")
    print(f"    N={n}, Win={len(wins)/n*100:.0f}%, Avg={pnls.mean()*100:+.2f}%, Med={pnls.median()*100:+.2f}%")
    print(f"    AvgW={wins.mean()*100:+.2f}%, AvgL={losses.mean()*100:+.2f}%, Payoff={wins.mean()/abs(losses.mean()):.2f}")
    print(f"    PEAD prec={n_pead/n*100:.0f}%, Raw sum={pnls.sum()*100:+.1f}%, NAV-comp={(nav-1)*100:+.1f}% ({nav:.2f}x), MaxDD={max_dd:+.1f}%")

    # Per-fold
    print(f"    Per-fold NAV-comp: ", end="")
    for fi in range(1, 5):
        sub = exec_df[exec_df["fold"] == fi]
        if len(sub) == 0: continue
        r = sub["pregap_return"].dropna()
        iso_f = sub["entry_date"].dt.isocalendar()
        sub2 = sub.copy(); sub2["wk"] = iso_f["year"].astype(str)+"-W"+iso_f["week"].astype(str).str.zfill(2)
        nav_f = 1.0
        for wk, wdf in sub2.groupby("wk", sort=True):
            tr = wdf["pregap_return"].dropna()
            if len(tr) > 0: nav_f *= (1 + (tr * (1.0/N_SLOTS)).sum())
        wr_pct = (r > 0).mean() * 100
        print(f"F{fi}:{(nav_f-1)*100:+.1f}%(W{wr_pct:.0f}%) ", end="")
    print()


def main():
    print("=" * 100)
    print("OPTION B+: EXPANDED PRE-EVENT FEATURES (prior-quarter earnings patterns)")
    print(f"  Base: 19 features")
    print(f"  New:  {len(NEW_PRIOR_FEATURES)} prior-quarter features: {NEW_PRIOR_FEATURES}")
    print(f"  Total: {len(EXPANDED_FEATURES)} features")
    print("=" * 100)

    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = compute_pregap(df, DB, EXIT_SNAP, STOP_LOSS)

    with pd.HDFStore(DB, mode="r") as s:
        pt_meta = s["/metadata/sp400_permatickers"]
    df = df.merge(pt_meta[["permaTicker","index_ref"]].drop_duplicates("permaTicker"), on="permaTicker", how="left")
    df["sector"] = df["index_ref"]

    # Add new features
    print("\n  Adding prior-quarter features ...")
    df = add_prior_earnings_features(df)

    # Verify new features exist and have coverage
    print(f"\n  New feature coverage:")
    for feat in NEW_PRIOR_FEATURES:
        if feat in df.columns:
            cov = df[feat].notna().sum()
            print(f"    {feat:<30} {cov:>6} non-null ({cov/len(df)*100:.1f}%)")
        else:
            print(f"    {feat:<30} MISSING!")

    hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}

    # ===== A/B/C COMPARISON =====
    print(f"\n{'='*100}")
    print("A/B/C COMPARISON")
    print(f"{'='*100}")

    # A: 19 features (baseline, no SUE proxies)
    exec_19 = run_model(df, PRE_EVENT_19, "19-feat baseline", hp)
    report(exec_19, "A: 19 features (baseline)")

    # B: 26 features (19 + 7 new prior-quarter)
    exec_26 = run_model(df, EXPANDED_FEATURES, "26-feat expanded", hp)
    report(exec_26, "B: 26 features (19 + 7 prior-Q)")

    # C: 7 new features only (ablation — do the new features add value alone?)
    exec_7 = run_model(df, NEW_PRIOR_FEATURES, "7 new only", hp)
    report(exec_7, "C: 7 new features only (ablation)")

    # ===== FEATURE IMPORTANCE FOR 26-FEATURE MODEL =====
    print(f"\n{'='*100}")
    print("FEATURE IMPORTANCE (26-feature model)")
    print(f"{'='*100}")
    X_all = df[EXPANDED_FEATURES]
    y_all = df["pead_pass"].astype(int).values
    clf_full = fit_clf(X_all, y_all, X_all, y_all, hp)
    imp = clf_full.feature_importances_
    idx_sorted = np.argsort(imp)[::-1]
    print(f"\n  {'Rank':>4} {'Feature':<35} {'Importance':>10} {'Type':<15}")
    print("  " + "-" * 65)
    for rank, idx in enumerate(idx_sorted, 1):
        feat = EXPANDED_FEATURES[idx]
        ftype = "NEW" if feat in NEW_PRIOR_FEATURES else "base"
        marker = " <<<" if feat in NEW_PRIOR_FEATURES else ""
        print(f"  {rank:>4} {feat:<35} {imp[idx]:>9.4f} {ftype:<15}{marker}")

    # ===== HEADLINE COMPARISON TABLE =====
    print(f"\n{'='*100}")
    print("HEADLINE COMPARISON")
    print(f"{'='*100}")
    print(f"\n  {'Metric':<25} {'19-feat':>10} {'26-feat':>10} {'24-feat*':>10}")
    print("  " + "-" * 55)
    print(f"  {'(*24-feat has look-ahead)':<25}")
    print("  " + "-" * 55)

    for exec_d, lbl in [(exec_19, "19"), (exec_26, "26")]:
        if len(exec_d) == 0: continue
        r = exec_d["pregap_return"].dropna()
        wins = r[r>0]; losses = r[r<=0]
        n_p = int(exec_d["pead_pass"].sum())
        iso = exec_d["entry_date"].dt.isocalendar()
        ed2 = exec_d.copy(); ed2["wk"] = iso["year"].astype(str)+"-W"+iso["week"].astype(str).str.zfill(2)
        nav = 1.0
        for wk, wdf in ed2.groupby("wk", sort=True):
            tr = wdf["pregap_return"].dropna()
            if len(tr) > 0: nav *= (1 + (tr*(1.0/N_SLOTS)).sum())
        if lbl == "19":
            n19, wr19, avg19, nav19, prec19 = len(r), len(wins)/len(r)*100, r.mean()*100, (nav-1)*100, n_p/len(r)*100
        else:
            n26, wr26, avg26, nav26, prec26 = len(r), len(wins)/len(r)*100, r.mean()*100, (nav-1)*100, n_p/len(r)*100

    print(f"  {'Trades':<25} {n19:>10} {n26:>10} {'101':>10}")
    print(f"  {'Win rate':<25} {wr19:>9.0f}% {wr26:>9.0f}% {'75%':>10}")
    print(f"  {'PEAD precision':<25} {prec19:>9.0f}% {prec26:>9.0f}% {'39%':>10}")
    print(f"  {'Expectancy/trade':<25} {avg19:>+9.2f}% {avg26:>+9.2f}% {'+6.72%':>10}")
    print(f"  {'NAV-compounded':<25} {nav19:>+9.1f}% {nav26:>+9.1f}% {'+391%':>10}")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

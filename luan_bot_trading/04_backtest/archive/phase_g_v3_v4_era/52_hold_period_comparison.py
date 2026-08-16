#!/usr/bin/env python3
"""
Hold period comparison: 5-day vs 10-day for the 23-feature honest model.

We inherited the 5-day hold from the 24-feature look-ahead model. Now that
the model is different (honest features, different HP), we should re-verify.

Tests:
  - 5-day hold (current default)
  - 7-day hold
  - 10-day hold (original Phase G default)
  - 15-day hold
  - 21-day hold (1 month)

For each: NAV-compounded, win rate, avg/trade, per-fold stability.
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
THETA = 0.20
STOP_LOSS = 0.10
EXCLUDE_SECTORS = ["XLF"]

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]

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
    "yield_spread": "/macros/fred_yield_curve_spread",
    "fed_funds": "/macros/fred_fed_funds_rate",
    "cpi": "/macros/fred_cpi",
    "unemployment": "/macros/fred_unemployment_rate",
    "oil": "/macros/fred_wti_oil",
}


def add_consecutive_pre(df):
    df = df.sort_values(["permaTicker", "report_date"]).copy()
    for pt, grp in df.groupby("permaTicker"):
        idx = grp.index
        if "consecutive_surprises" in grp.columns:
            df.loc[idx, "consecutive_surprises_pre"] = grp["consecutive_surprises"].shift(1)
    return df


def add_macro_features(df, db_path):
    with pd.HDFStore(db_path, mode="r") as s:
        for name, key in MACRO_KEYS.items():
            if key not in s: continue
            m = s[key].copy()
            m["Date"] = pd.to_datetime(m["Date"])
            m = m.sort_values("Date").rename(columns={"Date": "report_date"})
            close_col = "Close" if "Close" in m.columns else m.columns[1]
            m = m[["report_date", close_col]].rename(columns={close_col: name})
            m[name] = pd.to_numeric(m[name], errors="coerce")
            m = m.sort_values("report_date")
            m[f"{name}_roc21"] = m[name].pct_change(21).replace([np.inf, -np.inf], np.nan)
            m[f"{name}_roc63"] = m[name].pct_change(63).replace([np.inf, -np.inf], np.nan)
            df = df.sort_values("report_date").copy()
            df["report_date"] = pd.to_datetime(df["report_date"])
            df = pd.merge_asof(df, m[["report_date", name, f"{name}_roc21", f"{name}_roc63"]],
                             on="report_date", direction="backward")
    return df


def fit_clf(X_tr, y_tr, X_val, y_val, hp):
    import xgboost as xgb
    clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric=["logloss","auc"],
        n_estimators=hp["n_estimators"], learning_rate=0.05, max_depth=hp["max_depth"],
        min_child_weight=hp["min_child_weight"], gamma=hp["gamma"], reg_lambda=1.0,
        subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def compute_pregap_for_hold(df, db_path, hold_days, stop_loss=0.10):
    """Compute pre-gap PnL for a specific hold period."""
    col_ret = f"ret_h{hold_days}"
    col_entry = f"entry_h{hold_days}"
    col_exit = f"exit_h{hold_days}"
    df[col_ret] = np.nan
    df[col_entry] = pd.NaT
    df[col_exit] = pd.NaT
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
            df.at[idx, col_ret] = final_ret
            df.at[idx, col_entry] = pd.Timestamp(p_index[entry_t])
            df.at[idx, col_exit] = pd.Timestamp(p_index[exit_t])
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


def run_hold_period(df, features, hp, hold_days):
    """Run full CV for a specific hold period."""
    ret_col = f"ret_h{hold_days}"
    entry_col = f"entry_h{hold_days}"
    exit_col = f"exit_h{hold_days}"

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
        mask = (td["p"] >= THETA) & (td[ret_col].notna()) & (~td["sector"].isin(EXCLUDE_SECTORS))
        picks = td[mask].copy()
        if len(picks) == 0: continue
        picks["entry_date"] = pd.to_datetime(picks[entry_col])
        picks["exit_date"] = pd.to_datetime(picks[exit_col])
        picks["fold"] = fi
        picks["pregap_return"] = picks[ret_col]
        sel = select_weekly(picks, N_SLOTS)
        if len(sel) > 0: all_exec.append(sel)

    if not all_exec:
        return None
    exec_df = pd.concat(all_exec).reset_index(drop=True)
    pnls = exec_df["pregap_return"].dropna()
    n = len(pnls)
    if n == 0: return None
    wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
    n_pead = int(exec_df["pead_pass"].sum())

    # NAV
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

    fold_navs = []
    for fi in range(1, 5):
        sub = exec_df[exec_df["fold"] == fi]
        if len(sub) == 0: fold_navs.append(0); continue
        iso_f = sub["entry_date"].dt.isocalendar()
        sub2 = sub.copy(); sub2["wk"] = iso_f["year"].astype(str)+"-W"+iso_f["week"].astype(str).str.zfill(2)
        nav_f = 1.0
        for wk, wdf in sub2.groupby("wk", sort=True):
            tr = wdf["pregap_return"].dropna()
            if len(tr) > 0: nav_f *= (1 + (tr * (1.0/N_SLOTS)).sum())
        fold_navs.append((nav_f-1)*100)

    return {
        "n": n, "wr": len(wins)/n*100, "avg": pnls.mean()*100, "med": pnls.median()*100,
        "aw": wins.mean()*100 if len(wins)>0 else 0, "al": losses.mean()*100 if len(losses)>0 else 0,
        "payoff": wins.mean()/abs(losses.mean()) if len(losses)>0 and losses.mean()!=0 else 0,
        "nav": (nav-1)*100, "navx": nav, "maxdd": max_dd,
        "n_pead": n_pead, "prec": n_pead/n*100,
        "fold_navs": fold_navs, "min_fold": min(fold_navs),
        "fold_range": max(fold_navs) - min(fold_navs),
    }


def main():
    print("=" * 110)
    print("HOLD PERIOD COMPARISON: 23-feature honest model")
    print("=" * 110)

    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)

    with pd.HDFStore(DB, mode="r") as s:
        pt_meta = s["/metadata/sp400_permatickers"]
    df = df.merge(pt_meta[["permaTicker","index_ref"]].drop_duplicates("permaTicker"), on="permaTicker", how="left")
    df["sector"] = df["index_ref"]
    df = add_consecutive_pre(df)
    df = add_macro_features(df, DB)

    # Pre-compute PnL for all hold periods
    hold_periods = [3, 5, 7, 10, 15, 21]
    print("\n  Computing pre-gap PnL for each hold period ...")
    for hold in hold_periods:
        df = compute_pregap_for_hold(df, DB, hold, STOP_LOSS)
        col = f"ret_h{hold}"
        valid = df[col].notna().sum()
        print(f"    {hold:>2}-day hold: {valid} events with valid PnL")

    hp = {"gamma": 3, "min_child_weight": 100, "max_depth": 2, "n_estimators": 300}

    # ===== MAIN COMPARISON =====
    print(f"\n{'='*110}")
    print("HOLD PERIOD SWEEP")
    print(f"{'='*110}")

    results = {}
    for hold in hold_periods:
        print(f"\n  Running {hold}-day hold ...")
        stats = run_hold_period(df, DEPLOY_FEATURES, hp, hold)
        if stats:
            results[hold] = stats

    # ===== RESULTS TABLE =====
    print(f"\n{'='*110}")
    print("RESULTS")
    print(f"{'='*110}")
    print(f"\n  {'Hold':>5} {'N':>4} {'Win%':>5} {'Avg':>7} {'Med':>7} {'AvgW':>7} {'AvgL':>7} {'Payoff':>7} {'NAV-comp':>9} {'NAVx':>6} {'MaxDD':>7} {'F1':>7} {'F2':>7} {'F3':>7} {'F4':>7} {'MinF':>7} {'Range':>7}")
    print("  " + "-" * 120)
    for hold in hold_periods:
        if hold not in results: continue
        r = results[hold]
        print(f"  {hold:>4}d {r['n']:>4} {r['wr']:>4.0f}% {r['avg']:>+6.2f}% {r['med']:>+6.2f}% "
              f"{r['aw']:>+6.2f}% {r['al']:>+6.2f}% {r['payoff']:>6.2f} {r['nav']:>+8.1f}% "
              f"{r['navx']:>5.2f}x {r['maxdd']:>+6.1f}% "
              f"{r['fold_navs'][0]:>+6.1f}% {r['fold_navs'][1]:>+6.1f}% {r['fold_navs'][2]:>+6.1f}% {r['fold_navs'][3]:>+6.1f}% "
              f"{r['min_fold']:>+6.1f}% {r['fold_range']:>6.1f}%")

    # ===== ANALYSIS =====
    print(f"\n{'='*110}")
    print("ANALYSIS")
    print(f"{'='*110}")

    best_nav = max(results.items(), key=lambda x: x[1]["nav"])
    best_wr = max(results.items(), key=lambda x: x[1]["wr"])
    best_avg = max(results.items(), key=lambda x: x[1]["avg"])
    best_stab = max(results.items(), key=lambda x: x[1]["min_fold"])

    print(f"\n  Best by NAV:          {best_nav[0]}-day hold (NAV={best_nav[1]['nav']:+.1f}%)")
    print(f"  Best by win rate:     {best_wr[0]}-day hold (Win={best_wr[1]['wr']:.0f}%)")
    print(f"  Best by avg/trade:    {best_avg[0]}-day hold (Avg={best_avg[1]['avg']:+.2f}%)")
    print(f"  Best by stability:    {best_stab[0]}-day hold (MinFold={best_stab[1]['min_fold']:+.1f}%)")

    # 5-day vs 10-day head-to-head
    if 5 in results and 10 in results:
        r5 = results[5]; r10 = results[10]
        print(f"\n  5-DAY vs 10-DAY HEAD-TO-HEAD:")
        print(f"  {'Metric':<25} {'5-day':>10} {'10-day':>10} {'Delta':>10}")
        print("  " + "-" * 55)
        for key, label in [("n","Trades"), ("wr","Win %"), ("avg","Avg/trade %"), ("med","Median %"),
                            ("aw","Avg win %"), ("al","Avg loss %"), ("payoff","Payoff"),
                            ("nav","NAV-comp %"), ("maxdd","Max DD %"), ("min_fold","Min fold %"),
                            ("fold_range","Fold range pp")]:
            v5 = r5[key]; v10 = r10[key]; delta = v5 - v10
            if isinstance(v5, float):
                print(f"  {label:<25} {v5:>+9.2f} {v10:>+9.2f} {delta:>+9.2f}")
            else:
                print(f"  {label:<25} {v5:>10} {v10:>10} {delta:>+10}")

    print(f"\n{'='*110}")


if __name__ == "__main__":
    main()

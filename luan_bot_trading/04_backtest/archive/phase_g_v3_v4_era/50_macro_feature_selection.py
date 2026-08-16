#!/usr/bin/env python3
"""
Option B++ macro feature selection: test which macro subsets help vs hurt.

From script 49, all 18 macro features (6 series x 3 variants: level, roc21, roc63)
gave NAV +149% but fold 3 collapsed (+4%, 39% win). The model is macro-dominated
which risks overfitting.

This script tests progressively smaller macro subsets to find the sweet spot:
  - All 18 macros
  - Top 6 (one per series, best variant)
  - Top 3 (unemployment_roc21, fed_funds, vix)
  - Level only (6 series, no ROC)
  - ROC only (12 ROC features)
  - Individual series

Goal: find the minimal macro set that helps without fold instability.
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

PRE_EVENT_19 = [
    "sue_lag_1", "sue_lag_2", "car_drift_historical_q1",
    "pre_event_idiosyncratic_vol", "pre_event_volume_trend",
    "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d", "rel_ret_30d",
    "sector_adjusted_ret_20d",
    "revision_momentum_30d", "revision_momentum_60d", "revision_momentum_90d",
    "revision_ordinal_momentum_90d", "revision_intensity_90d",
    "grade_dispersion_90d", "n_analysts_covering", "last_action_days_before_earnings",
]

MACRO_KEYS = {
    "vix": "/macros/fred_vix_close",
    "yield_spread": "/macros/fred_yield_curve_spread",
    "fed_funds": "/macros/fred_fed_funds_rate",
    "cpi": "/macros/fred_cpi",
    "unemployment": "/macros/fred_unemployment_rate",
    "oil": "/macros/fred_wti_oil",
}

# Macro subsets to test
MACRO_SUBSETS = {
    "none": [],
    "top3": ["unemployment_roc21", "fed_funds", "vix"],
    "top4": ["unemployment_roc21", "fed_funds", "unemployment", "vix"],
    "levels_only": ["vix", "yield_spread", "fed_funds", "cpi", "unemployment", "oil"],
    "roc_only": ["vix_roc21", "vix_roc63", "yield_spread_roc21", "yield_spread_roc63",
                 "fed_funds_roc21", "fed_funds_roc63", "cpi_roc21", "cpi_roc63",
                 "unemployment_roc21", "unemployment_roc63", "oil_roc21", "oil_roc63"],
    "all18": None,  # all available macros
    "vix_only": ["vix", "vix_roc21", "vix_roc63"],
    "unemployment_only": ["unemployment", "unemployment_roc21", "unemployment_roc63"],
    "fed_funds_only": ["fed_funds", "fed_funds_roc21", "fed_funds_roc63"],
    "yield_only": ["yield_spread", "yield_spread_roc21", "yield_spread_roc63"],
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
            if key not in s:
                continue
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


def run_model(df, features, hp):
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


def calc_stats(exec_df):
    if len(exec_df) == 0:
        return {"n": 0, "wr": 0, "avg": 0, "nav": 0, "folds": []}
    pnls = exec_df["pregap_return"].dropna()
    n = len(pnls)
    wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
    n_pead = int(exec_df["pead_pass"].sum())

    iso = exec_df["entry_date"].dt.isocalendar()
    ed2 = exec_df.copy(); ed2["wk"] = iso["year"].astype(str)+"-W"+iso["week"].astype(str).str.zfill(2)
    nav = 1.0
    for wk, wdf in ed2.groupby("wk", sort=True):
        tr = wdf["pregap_return"].dropna()
        if len(tr) > 0: nav *= (1 + (tr * (1.0/N_SLOTS)).sum())

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
        "n": n,
        "wr": len(wins)/n*100,
        "avg": pnls.mean()*100,
        "med": pnls.median()*100,
        "nav": (nav-1)*100,
        "prec": n_pead/n*100,
        "folds": fold_navs,
    }


def main():
    print("=" * 100)
    print("MACRO FEATURE SELECTION: finding the minimal macro subset")
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

    df = add_consecutive_pre(df)
    df = add_macro_features(df, DB)

    all_macros = [c for c in df.columns if any(c.startswith(m) for m in MACRO_KEYS)]
    base_features = PRE_EVENT_19 + ["consecutive_surprises_pre"]

    hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}

    # ===== RUN ALL SUBSETS =====
    print(f"\n  Base: {len(base_features)} features (19 + consecutive_surprises_pre)")
    print(f"  Testing {len(MACRO_SUBSETS)} macro subsets\n")

    results = {}
    for label, macro_list in MACRO_SUBSETS.items():
        if macro_list is None:
            macros = all_macros
        else:
            macros = [m for m in macro_list if m in df.columns]
        features = base_features + macros
        exec_df = run_model(df, features, hp)
        stats = calc_stats(exec_df)
        results[label] = {"stats": stats, "n_features": len(features), "n_macros": len(macros), "macros": macros}

    # ===== RESULTS TABLE =====
    print(f"{'='*100}")
    print("RESULTS")
    print(f"{'='*100}")
    print(f"\n  {'Subset':<20} {'#Feat':>5} {'#Mac':>4} {'N':>4} {'Win%':>5} {'Avg':>7} {'Med':>7} {'NAV':>8} {'F1':>7} {'F2':>7} {'F3':>7} {'F4':>7} {'MinFold':>8}")
    print("  " + "-" * 105)

    for label in ["none", "top3", "top4", "levels_only", "roc_only", "all18",
                   "vix_only", "unemployment_only", "fed_funds_only", "yield_only"]:
        r = results[label]
        s = r["stats"]
        if s["n"] == 0:
            print(f"  {label:<20} {r['n_features']:>5} {r['n_macros']:>4}  NO TRADES")
            continue
        min_fold = min(s["folds"])
        print(f"  {label:<20} {r['n_features']:>5} {r['n_macros']:>4} {s['n']:>4} {s['wr']:>4.0f}% "
              f"{s['avg']:>+6.2f}% {s['med']:>+6.2f}% {s['nav']:>+7.1f}% "
              f"{s['folds'][0]:>+6.1f}% {s['folds'][1]:>+6.1f}% {s['folds'][2]:>+6.1f}% {s['folds'][3]:>+6.1f}% "
              f"{min_fold:>+7.1f}%")

    # ===== ANALYSIS =====
    print(f"\n{'='*100}")
    print("ANALYSIS")
    print(f"{'='*100}")

    # Find best by different criteria
    best_nav = max(results.items(), key=lambda x: x[1]["stats"]["nav"] if x[1]["stats"]["n"] > 0 else -999)
    best_wr = max(results.items(), key=lambda x: x[1]["stats"]["wr"] if x[1]["stats"]["n"] > 0 else -999)
    best_minfold = max(results.items(), key=lambda x: min(x[1]["stats"]["folds"]) if x[1]["stats"]["n"] > 0 else -999)
    best_avg = max(results.items(), key=lambda x: x[1]["stats"]["avg"] if x[1]["stats"]["n"] > 0 else -999)

    print(f"\n  Best by NAV-compounded:     {best_nav[0]:<20} NAV={best_nav[1]['stats']['nav']:+.1f}%")
    print(f"  Best by win rate:           {best_wr[0]:<20} Win={best_wr[1]['stats']['wr']:.0f}%")
    print(f"  Best by min-fold (stable):  {best_minfold[0]:<20} MinFold={min(best_minfold[1]['stats']['folds']):+.1f}%")
    print(f"  Best by avg/trade:          {best_avg[0]:<20} Avg={best_avg[1]['stats']['avg']:+.2f}%")

    # Stability score = NAV / (max_fold - min_fold)
    print(f"\n  Stability ranking (NAV / fold range):")
    scored = []
    for label, r in results.items():
        s = r["stats"]
        if s["n"] == 0: continue
        fold_range = max(s["folds"]) - min(s["folds"])
        stability = s["nav"] / fold_range if fold_range > 0 else s["nav"]
        scored.append((label, s["nav"], min(s["folds"]), fold_range, stability))
    scored.sort(key=lambda x: x[4], reverse=True)
    print(f"    {'Subset':<20} {'NAV':>8} {'MinFold':>8} {'FoldRange':>10} {'Stability':>10}")
    print("    " + "-" * 60)
    for label, nav, min_f, fr, stab in scored:
        print(f"    {label:<20} {nav:>+7.1f}% {min_f:>+7.1f}% {fr:>9.1f}% {stab:>9.1f}")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

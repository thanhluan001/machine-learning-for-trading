#!/usr/bin/env python3
"""
Option B++: 19 features + consecutive_surprises_pre + macro regime features.

Two changes from the 19-feature baseline:
1. Add ONLY consecutive_surprises_pre (the #3 importance new feature)
2. Add macro regime features (FRED data) — may help now that SUE look-ahead is gone

The hypothesis: without current-quarter SUE, the model can't distinguish
earnings-season regimes. Macros (VIX, yield curve, fed funds) provide
regime context that helps filter bad-environment picks. Fold 2 (2025 H1)
had a mini-recession — macros might have caught this.
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

# Macro series keys in db.h5
MACRO_KEYS = {
    "vix": "/macros/fred_vix_close",            # VIX (risk regime)
    "yield_spread": "/macros/fred_yield_curve_spread", # Yield curve (recession indicator)
    "fed_funds": "/macros/fred_fed_funds_rate",  # Fed funds (monetary policy)
    "cpi": "/macros/fred_cpi",                   # CPI (inflation)
    "unemployment": "/macros/fred_unemployment_rate", # Unemployment (economic health)
    "oil": "/macros/fred_wti_oil",               # WTI oil (commodity cycle)
}


def add_consecutive_pre(df):
    """Add consecutive_surprises_pre (beat streak BEFORE current quarter)."""
    df = df.sort_values(["permaTicker", "report_date"]).copy()
    for pt, grp in df.groupby("permaTicker"):
        idx = grp.index
        if "consecutive_surprises" in grp.columns:
            df.loc[idx, "consecutive_surprises_pre"] = grp["consecutive_surprises"].shift(1)
    return df


def add_macro_features(df, db_path):
    """Add macro level + ROC features, forward-filled to earnings dates."""
    macros = {}
    with pd.HDFStore(db_path, mode="r") as s:
        for name, key in MACRO_KEYS.items():
            if key in s:
                m = s[key].copy()
                m["Date"] = pd.to_datetime(m["Date"])
                m = m.sort_values("Date").rename(columns={"Date": "report_date"})
                close_col = "Close" if "Close" in m.columns else ([c for c in m.columns if "close" in c.lower() or c in ("Value", "value")][0] if any("close" in c.lower() or c in ("Value", "value") for c in m.columns) else m.columns[1])
                m = m[["report_date", close_col]].rename(columns={close_col: name})
                m[name] = pd.to_numeric(m[name], errors="coerce")
                macros[name] = m

    if not macros:
        print("  WARNING: No macro data found in db.h5")
        return df

    # Merge each macro series by nearest date (merge_asof)
    df = df.sort_values("report_date").copy()
    df["report_date"] = pd.to_datetime(df["report_date"])

    for name, m in macros.items():
        m = m.sort_values("report_date")
        df = pd.merge_asof(df, m, on="report_date", direction="backward")

        # ROC (month-over-month = ~21 trading days)
        # Compute 21-day and 63-day ROC on the macro series itself
        m_sorted = m.sort_values("report_date").copy()
        m_sorted[f"{name}_roc21"] = m_sorted[name].pct_change(21).replace([np.inf, -np.inf], np.nan)
        m_sorted[f"{name}_roc63"] = m_sorted[name].pct_change(63).replace([np.inf, -np.inf], np.nan)
        df = pd.merge_asof(df, m_sorted[["report_date", f"{name}_roc21", f"{name}_roc63"]],
                          on="report_date", direction="backward")

    return df


def get_macro_feature_names():
    """Return list of all macro feature column names."""
    names = []
    for name in MACRO_KEYS:
        names.append(name)
        names.append(f"{name}_roc21")
        names.append(f"{name}_roc63")
    return names


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


def report(exec_df, label):
    if len(exec_df) == 0:
        print(f"\n  {label}: NO TRADES")
        return None
    pnls = exec_df["pregap_return"].dropna()
    n = len(pnls)
    wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
    n_pead = int(exec_df["pead_pass"].sum())

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
    print(f"    Per-fold: ", end="")
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
    return {"n": n, "wr": len(wins)/n*100, "avg": pnls.mean()*100, "nav": (nav-1)*100, "prec": n_pead/n*100}


def main():
    print("=" * 100)
    print("OPTION B++: 19 + consecutive_surprises_pre + macro regime features")
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

    # Add consecutive_surprises_pre
    df = add_consecutive_pre(df)
    print(f"  consecutive_surprises_pre coverage: {df['consecutive_surprises_pre'].notna().sum()}/{len(df)}")

    # Add macro features
    print("  Adding macro features ...")
    df = add_macro_features(df, DB)
    macro_feats = get_macro_feature_names()
    available_macros = [f for f in macro_feats if f in df.columns]
    print(f"  Macro features available: {len(available_macros)}/{len(macro_feats)}")
    for f in available_macros:
        cov = df[f].notna().sum()
        print(f"    {f:<20} {cov:>6} non-null ({cov/len(df)*100:.1f}%)")

    # Feature sets to test
    FEAT_19 = PRE_EVENT_19
    FEAT_20 = PRE_EVENT_19 + ["consecutive_surprises_pre"]
    FEAT_20_MACRO = FEAT_20 + available_macros

    hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}

    # ===== A/B/C COMPARISON =====
    print(f"\n{'='*100}")
    print("A/B/C COMPARISON")
    print(f"{'='*100}")

    results = {}
    results["A"] = report(run_model(df, FEAT_19, hp), f"A: 19 features (baseline)")
    results["B"] = report(run_model(df, FEAT_20, hp), f"B: 20 features (+consecutive_surprises_pre)")
    results["C"] = report(run_model(df, FEAT_20_MACRO, hp), f"C: {len(FEAT_20_MACRO)} features (+consecutive_pre +macros)")

    # ===== FEATURE IMPORTANCE FOR MODEL C =====
    print(f"\n{'='*100}")
    print(f"FEATURE IMPORTANCE ({len(FEAT_20_MACRO)}-feature model with macros)")
    print(f"{'='*100}")
    X_all = df[FEAT_20_MACRO]
    y_all = df["pead_pass"].astype(int).values
    clf_full = fit_clf(X_all, y_all, X_all, y_all, hp)
    imp = clf_full.feature_importances_
    idx_sorted = np.argsort(imp)[::-1]
    print(f"\n  {'Rank':>4} {'Feature':<35} {'Importance':>10} {'Type':<10}")
    print("  " + "-" * 60)
    for rank, idx in enumerate(idx_sorted, 1):
        feat = FEAT_20_MACRO[idx]
        if feat == "consecutive_surprises_pre": ftype = "PRE"
        elif feat in available_macros: ftype = "MACRO"
        else: ftype = "base"
        marker = " <<<" if ftype != "base" else ""
        print(f"  {rank:>4} {feat:<35} {imp[idx]:>9.4f} {ftype:<10}{marker}")

    # ===== SUMMARY =====
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"\n  {'Model':<45} {'N':>4} {'Win%':>5} {'Avg':>7} {'NAV-comp':>9}")
    print("  " + "-" * 70)
    for key, label in [("A", "19 features"), ("B", "+ consecutive_surprises_pre"), ("C", "+ consecutive_pre + macros")]:
        r = results[key]
        if r:
            print(f"  {label:<45} {r['n']:>4} {r['wr']:>4.0f}% {r['avg']:>+6.2f}% {r['nav']:>+8.1f}%")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

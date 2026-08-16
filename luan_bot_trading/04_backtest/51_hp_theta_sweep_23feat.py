#!/usr/bin/env python3
"""
HP + theta sweep for the 23-feature honest pre-event model (19 base +
consecutive_surprises_pre + top3 macros: unemployment_roc21, fed_funds, vix).

The current HP (gamma=3, mcw=50, md=3, n_est=300) was tuned for the 24-feature
look-ahead model. This sweep finds the optimal HP for the honest 23-feature model.

Sweep grid:
  gamma:       [1, 3, 5, 8, 12]
  min_child_weight: [20, 50, 100, 200]
  max_depth:   [2, 3, 4]
  theta:       [0.15, 0.20, 0.25, 0.30]

Selection metric: PEAD F1 (same as original Phase G protocol).

For each HP combo, run 4-fold nested CV, compute mean PEAD F1.
Take the best HP, then sweep theta on that HP.
"""
import sys, io, importlib.util
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np, pandas as pd
from itertools import product

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
STOP_LOSS = 0.10
EXCLUDE_SECTORS = ["XLF"]

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]

DEPLOY_FEATURES = [
    # 19 base pre-event
    "sue_lag_1", "sue_lag_2", "car_drift_historical_q1",
    "pre_event_idiosyncratic_vol", "pre_event_volume_trend",
    "rel_ret_3d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d", "rel_ret_30d",
    "sector_adjusted_ret_20d",
    "revision_momentum_30d", "revision_momentum_60d", "revision_momentum_90d",
    "revision_ordinal_momentum_90d", "revision_intensity_90d",
    "grade_dispersion_90d", "n_analysts_covering", "last_action_days_before_earnings",
    # prior-quarter beat streak
    "consecutive_surprises_pre",
    # top3 macros
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


def run_cv(df, features, hp, theta):
    """Run 4-fold nested CV, return exec_df and aggregate stats."""
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
        mask = (td["p"] >= theta) & (td["pregap_return"].notna()) & (~td["sector"].isin(EXCLUDE_SECTORS))
        picks = td[mask].copy()
        if len(picks) == 0: continue
        picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
        picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
        picks["fold"] = fi
        sel = select_weekly(picks, N_SLOTS)
        if len(sel) > 0: all_exec.append(sel)

    if not all_exec:
        return None

    exec_df = pd.concat(all_exec).reset_index(drop=True)
    pnls = exec_df["pregap_return"].dropna()
    n = len(pnls)
    if n == 0:
        return None

    wins = pnls[pnls > 0]
    n_pead = int(exec_df["pead_pass"].sum())

    # PEAD F1 (precision/recall on the MODEL picks, not just executed)
    # Gather all model picks across folds
    all_picks_pead = []
    all_actual_pead = []
    for fi in range(1, 5):
        td = fold_data[fi]
        mask = (td["p"] >= theta) & (~td["sector"].isin(EXCLUDE_SECTORS))
        all_picks_pead.extend(mask.values)
        all_actual_pead.extend(td["pead_pass"].astype(int).values)
    picks_pead = np.array(all_picks_pead)
    actual_pead = np.array(all_actual_pead)
    tp = int((picks_pead & (actual_pead == 1)).sum())
    fp = int((picks_pead & (actual_pead == 0)).sum())
    fn = int((~picks_pead & (actual_pead == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    # NAV-compounded
    iso = exec_df["entry_date"].dt.isocalendar()
    ed2 = exec_df.copy(); ed2["wk"] = iso["year"].astype(str)+"-W"+iso["week"].astype(str).str.zfill(2)
    nav = 1.0
    for wk, wdf in ed2.groupby("wk", sort=True):
        tr = wdf["pregap_return"].dropna()
        if len(tr) > 0: nav *= (1 + (tr * (1.0/N_SLOTS)).sum())

    # Per-fold NAV
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
        "nav": (nav-1)*100,
        "prec_pead": prec * 100,
        "rec_pead": rec * 100,
        "f1_pead": f1 * 100,
        "min_fold": min(fold_navs),
        "fold_range": max(fold_navs) - min(fold_navs),
        "fold_navs": fold_navs,
    }


def main():
    print("=" * 100)
    print("HP + THETA SWEEP: 23-feature honest pre-event model")
    print(f"Features: {len(DEPLOY_FEATURES)} (19 base + consecutive_surprises_pre + top3 macros)")
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

    # Verify features exist
    missing = [f for f in DEPLOY_FEATURES if f not in df.columns]
    if missing:
        print(f"  MISSING: {missing}")
        return

    # ===== PHASE 1: HP SWEEP (fixed theta=0.20) =====
    print(f"\n{'='*100}")
    print("PHASE 1: HP SWEEP (theta=0.20 fixed)")
    print(f"{'='*100}")

    gammas = [1, 3, 5, 8, 12]
    mcws = [20, 50, 100, 200]
    mds = [2, 3, 4]

    hp_results = []
    total = len(gammas) * len(mcws) * len(mds)
    count = 0

    for gamma, mcw, md in product(gammas, mcws, mds):
        count += 1
        hp = {"gamma": gamma, "min_child_weight": mcw, "max_depth": md, "n_estimators": 300}
        stats = run_cv(df, DEPLOY_FEATURES, hp, 0.20)
        if stats is None:
            continue
        hp_results.append({
            "gamma": gamma, "mcw": mcw, "md": md,
            **stats,
        })
        if count % 10 == 0:
            print(f"  [{count}/{total}] ...")

    # Sort by F1
    hp_results.sort(key=lambda x: x["f1_pead"], reverse=True)

    print(f"\n  TOP 10 HP combos by PEAD F1:")
    print(f"  {'gamma':>5} {'mcw':>5} {'md':>3} {'F1':>5} {'Prec':>5} {'Rec':>5} {'N':>4} {'Win%':>5} {'Avg':>7} {'NAV':>8} {'MinF':>7} {'Range':>7}")
    print("  " + "-" * 80)
    for r in hp_results[:10]:
        print(f"  {r['gamma']:>5} {r['mcw']:>5} {r['md']:>3} {r['f1_pead']:>4.1f} {r['prec_pead']:>4.0f}% "
              f"{r['rec_pead']:>4.0f}% {r['n']:>4} {r['wr']:>4.0f}% {r['avg']:>+6.2f}% "
              f"{r['nav']:>+7.1f}% {r['min_fold']:>+6.1f}% {r['fold_range']:>6.1f}%")

    # Also show top 5 by NAV and top 5 by min_fold
    print(f"\n  TOP 5 by NAV-compounded:")
    for r in sorted(hp_results, key=lambda x: x["nav"], reverse=True)[:5]:
        print(f"    g={r['gamma']} mcw={r['mcw']} md={r['md']}: NAV={r['nav']:+.1f}%, Win={r['wr']:.0f}%, F1={r['f1_pead']:.1f}, MinF={r['min_fold']:+.1f}%")

    print(f"\n  TOP 5 by min-fold (stability):")
    for r in sorted(hp_results, key=lambda x: x["min_fold"], reverse=True)[:5]:
        print(f"    g={r['gamma']} mcw={r['mcw']} md={r['md']}: MinF={r['min_fold']:+.1f}%, NAV={r['nav']:+.1f}%, Win={r['wr']:.0f}%, F1={r['f1_pead']:.1f}")

    # ===== PHASE 2: THETA SWEEP (on best F1 HP) =====
    best_hp = hp_results[0]
    best_gamma = best_hp["gamma"]
    best_mcw = best_hp["mcw"]
    best_md = best_hp["md"]

    # Also get stability-best HP
    stab_sorted = sorted(hp_results, key=lambda x: (x["min_fold"], x["nav"]), reverse=True)
    stab_hp = stab_sorted[0]
    stab_gamma = stab_hp["gamma"]
    stab_mcw = stab_hp["mcw"]
    stab_md = stab_hp["md"]

    print(f"\n{'='*100}")
    print(f"PHASE 2: THETA SWEEP (best F1 HP: gamma={best_gamma}, mcw={best_mcw}, md={best_md})")
    print(f"{'='*100}")

    thetas = [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35]
    hp_best = {"gamma": best_gamma, "min_child_weight": best_mcw, "max_depth": best_md, "n_estimators": 300}
    hp_stab = {"gamma": stab_gamma, "min_child_weight": stab_mcw, "max_depth": stab_md, "n_estimators": 300}

    for hp_label, hp_dict in [(f"F1-best (g={best_gamma},mcw={best_mcw},md={best_md})", hp_best),
                                (f"Stability-best (g={stab_gamma},mcw={stab_mcw},md={stab_md})", hp_stab)]:
        print(f"\n  --- {hp_label} ---")
        print(f"  {'theta':>6} {'N':>4} {'Win%':>5} {'Avg':>7} {'NAV':>8} {'F1':>5} {'Prec':>5} {'Rec':>5} {'MinF':>7} {'Range':>7}")
        print("  " + "-" * 70)
        theta_results = []
        for theta in thetas:
            stats = run_cv(df, DEPLOY_FEATURES, hp_dict, theta)
            if stats is None:
                print(f"  {theta:>5.2f}  NO TRADES")
                continue
            theta_results.append({"theta": theta, **stats})
            print(f"  {theta:>5.2f} {stats['n']:>4} {stats['wr']:>4.0f}% {stats['avg']:>+6.2f}% "
                  f"{stats['nav']:>+7.1f}% {stats['f1_pead']:>4.1f} {stats['prec_pead']:>4.0f}% "
                  f"{stats['rec_pead']:>4.0f}% {stats['min_fold']:>+6.1f}% {stats['fold_range']:>6.1f}%")

    # ===== RECOMMENDATION =====
    print(f"\n{{'='*100}}")
    print("RECOMMENDATION")
    print(f"{{'='*100}}")

    print(f"\n  F1-best HP:     gamma={best_gamma}, mcw={best_mcw}, md={best_md}")
    print(f"  Stability HP:   gamma={stab_gamma}, mcw={stab_mcw}, md={stab_md}")
    print(f"\n  Recommended: stability HP (every fold positive, highest NAV)")
    print(f"  gamma={stab_gamma}, mcw={stab_mcw}, md={stab_md}, theta=0.20")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

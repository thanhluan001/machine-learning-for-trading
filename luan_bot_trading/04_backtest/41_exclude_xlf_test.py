#!/usr/bin/env python3
"""
Test excluding XLF (Financials) sector — the precision killer.

From the false positive analysis (40_false_positive_analysis.py):
  XLF: 15 trades, only 2 PEAD (13% precision), -1.9% avg FP return
  Rest: 94 trades, 37 PEAD (39% precision)

Tests:
  A. Baseline (all sectors)
  B. Exclude XLF only
  C. Exclude XLF + IJH (both low-precision sectors)
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
THETA = 0.20

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


def compute_pregap(df, db_path, hold_days=5):
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
            ep = p_close[entry_t]; xp = p_close[exit_t]
            if pd.isna(ep) or pd.isna(xp) or ep <= 0:
                continue
            df.at[idx, "pregap_return"] = float(xp / ep - 1.0)
            df.at[idx, "pregap_entry_date"] = pd.Timestamp(p_index[entry_t])
            df.at[idx, "pregap_exit_date"] = pd.Timestamp(p_index[exit_t])
    return df


def run_scenario(fold_data, theta, exclude_sectors=None):
    all_exec = []
    for fi in range(1, 5):
        td = fold_data[fi]["test_df"]
        mask = (td["p"] >= theta) & (td["pregap_return"].notna())
        if exclude_sectors:
            mask = mask & (~td["sector"].isin(exclude_sectors))
        picks = td[mask].copy()
        if len(picks) == 0:
            continue
        picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
        picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
        picks["fold"] = fi
        sel = select_weekly_top_n(picks, N_SLOTS, sort_col="p")
        if len(sel) > 0:
            all_exec.append(sel)
    return pd.concat(all_exec) if all_exec else pd.DataFrame()


def stats_row(exec_df, label):
    if len(exec_df) == 0:
        print(f"  {label:<40} N=0")
        return
    pnls = exec_df["pregap_return"].dropna()
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / n * 100
    avg = pnls.mean() * 100
    aw = wins.mean() * 100 if len(wins) > 0 else 0
    al = losses.mean() * 100 if len(losses) > 0 else 0
    payoff = aw / abs(al) if al != 0 else float('inf')
    total = pnls.sum() * 100
    n_pead = int(exec_df["pead_pass"].sum())
    prec = n_pead / n * 100
    print(f"  {label}")
    print(f"    N={n}, Win={wr:.1f}%, Avg={avg:+.2f}%, Total={total:+.1f}%, "
          f"Payoff={payoff:.2f}, PEAD={n_pead}({prec:.0f}%)")


def main():
    print("=" * 100)
    print("SECTOR EXCLUSION TEST: XLF (Financials) precision killer")
    print("=" * 100)

    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = compute_pregap(df, DB, EXIT_SNAP)

    # Join sector
    with pd.HDFStore(DB, mode="r") as s:
        pt_meta = s["/metadata/sp400_permatickers"]
    sector_lookup = pt_meta[["permaTicker", "index_ref"]].drop_duplicates("permaTicker")
    df = df.merge(sector_lookup, on="permaTicker", how="left")
    df["sector"] = df["index_ref"]

    # Train + select
    fold_data = {}
    hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_ts = pd.concat([train_df[SUNDAY_SAFE], sweep_df[SUNDAY_SAFE]])
        y_ts = pd.concat([train_df, sweep_df])["pead_pass"].astype(int).values
        y_te = test_df["pead_pass"].astype(int).values
        clf = fit_clf(X_ts, y_ts, test_df[SUNDAY_SAFE], y_te, hp)
        test_df = test_df.copy()
        test_df["p"] = clf.predict_proba(test_df[SUNDAY_SAFE])[:, 1]
        fold_data[fi] = {"test_df": test_df}

    # ===== MAIN COMPARISON =====
    print(f"\n{'='*100}")
    print("RESULTS: Sector exclusion sweep")
    print(f"{'='*100}")

    print()
    stats_row(run_scenario(fold_data, THETA), "A. Baseline (all sectors)")
    print()
    stats_row(run_scenario(fold_data, THETA, exclude_sectors=["XLF"]),
              "B. Exclude XLF (Financials)")
    print()
    stats_row(run_scenario(fold_data, THETA, exclude_sectors=["XLF", "IJH"]),
              "C. Exclude XLF + IJH")
    print()
    stats_row(run_scenario(fold_data, THETA, exclude_sectors=["XLF", "IJH", "XLB"]),
              "D. Exclude XLF + IJH + XLB")

    # ===== PER-FOLD FOR BASELINE vs XLF-EXCLUDED =====
    print(f"\n{'='*100}")
    print("PER-FOLD BREAKDOWN: Baseline vs Exclude XLF")
    print(f"{'='*100}")

    for label, excl in [("Baseline", None), ("No XLF", ["XLF"])]:
        exec_df = run_scenario(fold_data, THETA, exclude_sectors=excl)
        print(f"\n  {label}:")
        for fi in range(1, 5):
            sub = exec_df[exec_df["fold"] == fi]
            if len(sub) == 0:
                continue
            pnls = sub["pregap_return"].dropna()
            n = len(pnls)
            wr = (pnls > 0).mean() * 100
            avg = pnls.mean() * 100
            total = pnls.sum() * 100
            n_pead = int(sub["pead_pass"].sum())
            prec = n_pead / n * 100 if n > 0 else 0
            print(f"    Fold {fi}: N={n:>3}, Win={wr:>5.1f}%, Avg={avg:>+6.2f}%, "
                  f"Total={total:>+7.1f}%, PEAD={n_pead}({prec:.0f}%)")

    # ===== PRECISION vs TOTAL RETURN TRADEOFF =====
    print(f"\n{'='*100}")
    print("PRECISION vs TOTAL RETURN TRADEOFF")
    print(f"{'='*100}")

    print(f"\n  {'Scenario':<35} {'N':>4} {'Precision':>10} {'Total PnL':>10} {'Trades/yr':>10}")
    print("  " + "-" * 75)
    for label, excl in [("A. Baseline (all sectors)", None),
                         ("B. Exclude XLF", ["XLF"]),
                         ("C. Exclude XLF + IJH", ["XLF", "IJH"]),
                         ("D. Exclude XLF + IJH + XLB", ["XLF", "IJH", "XLB"])]:
        exec_df = run_scenario(fold_data, THETA, exclude_sectors=excl)
        if len(exec_df) == 0:
            continue
        pnls = exec_df["pregap_return"].dropna()
        n = len(pnls)
        n_pead = int(exec_df["pead_pass"].sum())
        prec = n_pead / n * 100
        total = pnls.sum() * 100
        trades_yr = n / 2
        print(f"  {label:<35} {n:>4} {prec:>9.1f}% {total:>+9.1f}% {trades_yr:>9.1f}")

    # ===== XLF DETAIL: what do the XLF trades look like? =====
    print(f"\n{'='*100}")
    print("XLF TRADE DETAIL (why are they bad?)")
    print(f"{'='*100}")

    exec_all = run_scenario(fold_data, THETA)
    xlf = exec_all[exec_all["sector"] == "XLF"].sort_values("report_date")
    print(f"\n  All {len(xlf)} XLF trades:")
    print(f"  {'Ticker':<10} {'Date':<12} {'P(PEAD)':>8} {'Ret%':>7} {'PEAD':>5} {'CAR%':>7} {'EPS%':>7}")
    for _, t in xlf.iterrows():
        ticker = t.get("canonical_ticker", t["permaTicker"])
        rd = str(pd.Timestamp(t["report_date"]).date())
        p = t["p"]
        ret = t["pregap_return"] * 100
        pead = int(t["pead_pass"])
        car = np.expm1(t.get("car_10d", 0)) * 100 if pd.notna(t.get("car_10d", np.nan)) else float('nan')
        eps = t.get("eps_surprise_pct", float('nan'))
        print(f"  {ticker:<10} {rd:<12} {p:>7.3f} {ret:>+6.1f}% {pead:>5} {car:>+6.1f}% {eps:>+6.1f}%")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

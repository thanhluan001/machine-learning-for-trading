#!/usr/bin/env python3
"""
Hybrid test: 3-class model with lower theta to recover trade count.

The head-to-head showed 3-class P(any)>=0.25 gets better precision
(+6.7pp) but fewer trades (-16), losing on total return (-43.5%).
This tests lower thetas on P(any PEAD) to find the sweet spot where
3-class matches/exceeds binary's total return while keeping better
precision.

Sweeps: 0.15, 0.18, 0.20, 0.22, 0.25
Compares against binary theta=0.25 baseline.
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
ps_spec = importlib.util.spec_from_file_location("ps", HERE / "04_phase_g_portfolio.py")
ps = importlib.util.module_from_spec(ps_spec); ps_spec.loader.exec_module(ps)
rb_spec = importlib.util.spec_from_file_location("rb", HERE / "_phase_g_random_baseline.py")
rb = importlib.util.module_from_spec(rb_spec); rb_spec.loader.exec_module(rb)

DB = tm.DB_FILE
SUNDAY_SAFE = pg.SUNDAY_SAFE_FEATURES
N_SLOTS = 4
EXIT_SNAP = 5
CAR_LARGE_THRESH = 10.0

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]

THETAS_3CLASS = [0.15, 0.18, 0.20, 0.22, 0.25]


def fit_clf_binary(X_tr, y_tr, X_val, y_val, hp):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=hp["n_estimators"], learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


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


def run_scenario(fold_data, theta, prob_col):
    all_exec_list = []
    for fi in range(1, 5):
        test_df = fold_data[fi]["test_df"]
        mask = (test_df[prob_col] >= theta) & (test_df["pregap_return"].notna())
        picks = test_df[mask].copy()
        if len(picks) == 0:
            continue
        picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
        picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])
        picks["fold"] = fi
        selected = select_weekly_top_n(picks, N_SLOTS, sort_col=prob_col)
        if len(selected) > 0:
            all_exec_list.append(selected)
    return pd.concat(all_exec_list) if all_exec_list else pd.DataFrame()


def stats_row(exec_df):
    if len(exec_df) == 0:
        return {"n": 0, "pead": 0, "prec": 0, "win": 0, "avg": 0, "total": 0, "payoff": 0}
    pnls = exec_df["pregap_return"].dropna()
    n = len(pnls)
    n_pead = int((exec_df["label_3class"] >= 1).sum())
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    aw = wins.mean() * 100 if len(wins) > 0 else 0
    al = losses.mean() * 100 if len(losses) > 0 else 0
    return {
        "n": n,
        "pead": n_pead,
        "prec": n_pead / n * 100 if n > 0 else 0,
        "win": len(wins) / n * 100 if n > 0 else 0,
        "avg": pnls.mean() * 100,
        "total": pnls.sum() * 100,
        "payoff": aw / abs(al) if al != 0 else float('inf'),
        "avg_win": aw,
        "avg_loss": al,
    }


def main():
    print("=" * 105)
    print("HYBRID TEST: 3-class with lower theta to recover trade count")
    print(f"  Pre-gap entry, 5-day hold, {N_SLOTS} slots, weekly batch")
    print(f"  Baseline: Binary theta=0.25")
    print("=" * 105)

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

    print("[3] Training classifiers per fold ...")
    fold_data = {}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_tr = train_df[SUNDAY_SAFE]; X_sv = sweep_df[SUNDAY_SAFE]
        X_te = test_df[SUNDAY_SAFE]
        X_ts = pd.concat([X_tr, X_sv])
        y_tr_b = train_df["pead_pass"].astype(int).values
        y_sv_b = sweep_df["pead_pass"].astype(int).values
        y_te_b = test_df["pead_pass"].astype(int).values
        y_ts_b = np.concatenate([y_tr_b, y_sv_b])
        y_tr_3 = train_df["label_3class"].values
        y_sv_3 = sweep_df["label_3class"].values
        y_te_3 = test_df["label_3class"].values
        y_ts_3 = np.concatenate([y_tr_3, y_sv_3])
        hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
        clf_b = fit_clf_binary(X_ts, y_ts_b, X_te, y_te_b, hp)
        clf_3 = fit_clf_3class(X_ts, y_ts_3, X_te, y_te_3, hp)
        proba_3 = clf_3.predict_proba(X_te)
        test_df = test_df.copy()
        test_df["p_binary"] = clf_b.predict_proba(X_te)[:, 1]
        test_df["p_any_pead"] = proba_3[:, 1] + proba_3[:, 2]
        fold_data[fi] = {"test_df": test_df}

    # Run baseline
    print("\n[4] Running scenarios ...")
    binary_exec = run_scenario(fold_data, 0.25, "p_binary")
    b_stats = stats_row(binary_exec)

    # Run 3-class sweep
    results = {}
    for theta in THETAS_3CLASS:
        exec_df = run_scenario(fold_data, theta, "p_any_pead")
        results[theta] = {"exec": exec_df, "stats": stats_row(exec_df)}

    # ===== COMPARISON TABLE =====
    print(f"\n{'='*105}")
    print("COMPARISON: Binary baseline vs 3-class lower-theta sweep")
    print(f"{'='*105}")

    print(f"\n{'Scenario':<28} {'N':>4} {'PEAD':>5} {'Prec':>6} {'Win%':>6} {'Avg':>8} {'Total':>8} {'Payoff':>7} {'AvgWin':>8} {'AvgLoss':>8}")
    print("-" * 105)

    # Baseline
    s = b_stats
    print(f"{'Binary theta=0.25':<28} {s['n']:>4} {s['pead']:>5} {s['prec']:>5.1f}% {s['win']:>5.1f}% "
          f"{s['avg']:>+7.2f}% {s['total']:>+7.1f}% {s['payoff']:>6.2f} {s['avg_win']:>+7.2f}% {s['avg_loss']:>+7.2f}%")

    # 3-class sweep
    for theta in THETAS_3CLASS:
        s = results[theta]["stats"]
        print(f"{'3-class P(any)>='+str(theta):<28} {s['n']:>4} {s['pead']:>5} {s['prec']:>5.1f}% {s['win']:>5.1f}% "
              f"{s['avg']:>+7.2f}% {s['total']:>+7.1f}% {s['payoff']:>6.2f} {s['avg_win']:>+7.2f}% {s['avg_loss']:>+7.2f}%")

    # Per-fold breakdown for each
    print(f"\n{'='*105}")
    print("PER-FOLD BREAKDOWN")
    print(f"{'='*105}")

    scenarios = [("Binary 0.25", binary_exec)] + \
                [(f"3class {t:.2f}", results[t]["exec"]) for t in THETAS_3CLASS]

    print(f"\n{'Scenario':<15} {'Fold':>5} {'N':>4} {'PEAD':>5} {'Prec':>6} {'Win%':>6} {'Avg':>8} {'Total':>8}")
    print("-" * 70)
    for label, exec_df in scenarios:
        for fi in range(1, 5):
            sub = exec_df[exec_df["fold"] == fi]
            if len(sub) == 0:
                print(f"  {label:<13} {fi:>5} {0:>4}")
                continue
            pnls = sub["pregap_return"].dropna()
            n = len(pnls)
            n_pead = int((sub["label_3class"] >= 1).sum())
            prec = n_pead / n * 100 if n > 0 else 0
            wr = (pnls > 0).mean() * 100
            avg = pnls.mean() * 100
            total = pnls.sum() * 100
            print(f"  {label:<13} {fi:>5} {n:>4} {n_pead:>5} {prec:>5.1f}% {wr:>5.1f}% {avg:>+7.2f}% {total:>+7.1f}%")
        print()

    # Summary: which 3-class theta beats binary on total return?
    print(f"{'='*105}")
    print("SUMMARY: Does any 3-class theta beat binary on total return?")
    print(f"{'='*105}")

    b_total = b_stats["total"]
    b_n = b_stats["n"]
    b_prec = b_stats["prec"]

    print(f"\n  Binary theta=0.25 baseline: {b_n} trades, {b_prec:.1f}% prec, "
          f"+{b_total:.1f}% total\n")

    for theta in THETAS_3CLASS:
        s = results[theta]["stats"]
        delta_total = s["total"] - b_total
        delta_prec = s["prec"] - b_prec
        delta_n = s["n"] - b_n
        verdict = "BETTER" if s["total"] > b_total else "WORSE"
        ratio = s["total"] / b_total if b_total != 0 else 0
        print(f"  3-class P(any)>={theta:.2f}: {s['n']} trades, {s['prec']:.1f}% prec, "
              f"+{s['total']:.1f}% total | "
              f"delta: {delta_n:+d} trades, {delta_prec:+.1f}pp prec, {delta_total:+.1f}% total "
              f"({ratio:.2f}x) => {verdict}")

    # Find the winner
    best_theta = max(THETAS_3CLASS, key=lambda t: results[t]["stats"]["total"])
    best_s = results[best_theta]["stats"]
    print(f"\n  Best 3-class by total return: P(any)>={best_theta:.2f} "
          f"({best_s['n']} trades, {best_s['prec']:.1f}% prec, +{best_s['total']:.1f}% total)")
    if best_s["total"] > b_total:
        print(f"  => 3-class BEATS binary by {best_s['total']-b_total:.1f}% total "
              f"({best_s['total']/b_total:.2f}x)")
    else:
        print(f"  => Binary still wins by {b_total-best_s['total']:.1f}% total "
              f"({b_total/best_s['total']:.2f}x)")

    print(f"\n{'='*105}")


if __name__ == "__main__":
    main()

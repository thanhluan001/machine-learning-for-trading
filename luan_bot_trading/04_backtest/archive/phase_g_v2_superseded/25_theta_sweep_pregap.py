#!/usr/bin/env python3
"""
Theta re-sweep with PRE-GAP entry.

The old theta sweep (17_theta_sweep.py) used post-gap Open[T+1] entry.
With pre-gap entry (Close[T-1] BMO / Close[T] AMC), the entire return
distribution shifted massively. The precision-recall-PnL tradeoff has
changed. This re-sweeps theta to find the new sweet spot targeting
40-50% precision.

For each theta:
  - 4-fold nested CV (gamma=3 fixed)
  - Weekly batch selection (4 slots, 5-day hold)
  - Pre-gap returns (Close[T-1]/Close[T] entry -> Close[T+5] exit)
  - Reports: picks, executed, PEAD precision/recall/F1, win rate,
    avg PnL, avg win, avg loss, payoff
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
EXIT_SNAP = 5  # Close[T+5] exit

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]

THETAS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


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


def select_weekly_top_n(picks, n_slots=4):
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
        week_sorted = week_df.sort_values("p", ascending=False)
        n_take = min(free_slots, len(week_sorted))
        taken = week_sorted.head(n_take)
        selected_rows.append(taken)
        for _, row in taken.iterrows():
            active_positions.append(row["exit_date"])
    if selected_rows:
        return pd.concat(selected_rows).sort_values("entry_date").reset_index(drop=True)
    return pd.DataFrame(columns=picks.columns)


def compute_pregap_entry_exit(df, db_path, calendar, cal_idx, hold_days=5):
    """Compute pre-gap entry/exit dates and returns for all rows.
    BMO: entry=Close[T-1], exit=Close[T+5]
    AMC: entry=Close[T], exit=Close[T+5]
    Adds: pregap_entry_date, pregap_exit_date, pregap_return, pregap_entry_idx
    """
    df = df.copy()
    df["pregap_entry_idx"] = np.nan
    df["pregap_exit_idx"] = np.nan
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

            if is_bmo:
                entry_t = t_idx - 1  # Close[T-1]
            else:
                entry_t = t_idx      # Close[T]

            exit_t = t_idx + hold_days  # Close[T+5]

            if entry_t < 0 or exit_t >= len(p_close):
                continue

            entry_price = p_close[entry_t]
            exit_price = p_close[exit_t]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue

            ret = float(exit_price / entry_price - 1.0)
            df.at[idx, "pregap_return"] = ret
            df.at[idx, "pregap_entry_date"] = pd.Timestamp(p_index[entry_t])
            df.at[idx, "pregap_exit_date"] = pd.Timestamp(p_index[exit_t])

    return df


def main():
    print("=" * 100)
    print("THETA RE-SWEEP WITH PRE-GAP ENTRY")
    print(f"  Thetas: {THETAS}")
    print(f"  Features: {len(SUNDAY_SAFE)} Sunday-safe, gamma=3 fixed")
    print(f"  Entry: Close[T-1] (BMO) / Close[T] (AMC), pre-gap")
    print(f"  Exit: Close[T+5], 5-day hold")
    print(f"  Selection: weekly batch, {N_SLOTS} slots")
    print("=" * 100)

    # Load + prime + gates + paths
    print("\n[1] Loading + priming + gates + paths ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)
    n_total = len(df)
    n_pead = int(df["pead_pass"].sum())
    base_rate = n_pead / n_total
    print(f"    rows: {n_total}, pead: {n_pead} ({base_rate*100:.1f}%)")

    # Cache calendar
    print("[2] Caching trading calendar ...")
    with pd.HDFStore(DB, mode="r") as sstore:
        all_dates = set()
        for key in sstore.keys():
            if not key.startswith("/sp400/"):
                continue
            d = sstore[key]
            all_dates.update(pd.to_datetime(d["Date"]).tolist())
    calendar = sorted(all_dates)
    cal_idx = {d: i for i, d in enumerate(calendar)}
    print(f"    {len(calendar)} trading days")

    # Compute pre-gap returns for ALL rows (once)
    print("[3] Computing pre-gap returns for all rows ...")
    df = compute_pregap_entry_exit(df, DB, calendar, cal_idx, EXIT_SNAP)
    n_valid = df["pregap_return"].notna().sum()
    print(f"    {n_valid}/{len(df)} rows have valid pre-gap returns")

    # Pre-train classifiers per fold
    print("\n[4] Pre-training 4 fold classifiers (gamma=3) ...")
    fold_data = {}
    for fi, (te, sve, tse) in enumerate(DEFAULT_FOLDS, 1):
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= pd.Timestamp(te)].copy()
        sweep_df = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))].copy()
        test_df = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        X_tr = train_df[SUNDAY_SAFE]
        y_tr = train_df["pead_pass"].astype(int).values
        X_sv = sweep_df[SUNDAY_SAFE]
        y_sv = sweep_df["pead_pass"].astype(int).values
        X_te = test_df[SUNDAY_SAFE]
        y_te = test_df["pead_pass"].astype(int).values
        X_ts = pd.concat([X_tr, X_sv])
        y_ts = np.concatenate([y_tr, y_sv])
        hp = {"gamma": 3, "min_child_weight": 50, "max_depth": 3, "n_estimators": 300}
        clf = fit_clf(X_ts, y_ts, X_te, y_te, hp)
        proba = clf.predict_proba(test_df[SUNDAY_SAFE])[:, 1]
        test_df = test_df.copy()
        test_df["p"] = proba
        fold_data[fi] = {"test_df": test_df}
        print(f"    Fold {fi}: TEST={len(test_df)}, pead={int(test_df['pead_pass'].sum())}, "
              f"valid_pregap={test_df['pregap_return'].notna().sum()}")

    # Sweep theta
    print(f"\n[5] Sweeping {len(THETAS)} theta values ...")
    print(f"\n{'theta':>6s}  {'Picks':>5s} {'Exec':>5s} {'PEAD':>5s} {'Prec':>6s} {'Rec':>6s} "
          f"{'F1':>6s} {'AvgPnL':>8s} {'Win%':>6s} {'AvgWin':>8s} {'AvgLoss':>8s} {'Payoff':>7s}")
    print("-" * 100)

    results = []
    for theta in THETAS:
        all_picks_list = []
        all_exec_list = []
        total_pead_in_test = 0

        for fi in range(1, 5):
            test_df = fold_data[fi]["test_df"]
            total_pead_in_test += int(test_df["pead_pass"].sum())

            # Select picks above theta with valid pre-gap return
            mask = (test_df["p"] >= theta) & (test_df["pregap_return"].notna())
            picks = test_df[mask].copy()
            if len(picks) == 0:
                continue

            # Prepare for weekly batch selection
            picks["entry_date"] = pd.to_datetime(picks["pregap_entry_date"])
            picks["exit_date"] = pd.to_datetime(picks["pregap_exit_date"])

            all_picks_list.append(picks[["permaTicker", "pead_pass", "p", "pregap_return",
                                         "entry_date", "exit_date", "report_date",
                                         "canonical_ticker", "is_bmo"]].copy())

            # Weekly batch selection
            selected = select_weekly_top_n(picks, N_SLOTS)
            if len(selected) > 0:
                all_exec_list.append(selected[["permaTicker", "pead_pass", "p", "pregap_return",
                                                "entry_date", "exit_date", "report_date",
                                                "canonical_ticker", "is_bmo"]].copy())

        # Aggregate
        all_picks = pd.concat(all_picks_list) if all_picks_list else pd.DataFrame()
        all_exec = pd.concat(all_exec_list) if all_exec_list else pd.DataFrame()

        n_picks = len(all_picks)
        n_exec = len(all_exec)
        n_pead_picks = int(all_picks["pead_pass"].sum()) if n_picks > 0 else 0
        n_pead_exec = int(all_exec["pead_pass"].sum()) if n_exec > 0 else 0

        precision = n_pead_picks / n_picks if n_picks > 0 else 0
        recall = n_pead_picks / total_pead_in_test if total_pead_in_test > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Trade-level stats on executed trades
        if n_exec > 0:
            pnls = all_exec["pregap_return"].dropna()
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            win_rate = len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0
            avg_pnl = pnls.mean() * 100
            avg_win = wins.mean() * 100 if len(wins) > 0 else 0
            avg_loss = losses.mean() * 100 if len(losses) > 0 else 0
            payoff = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
            prec_exec = n_pead_exec / n_exec * 100
        else:
            win_rate = avg_pnl = avg_win = avg_loss = payoff = prec_exec = 0

        results.append({
            "theta": theta, "n_picks": n_picks, "n_exec": n_exec,
            "n_pead_picks": n_pead_picks, "n_pead_exec": n_pead_exec,
            "precision": precision * 100, "recall": recall * 100, "f1": f1 * 100,
            "avg_pnl": avg_pnl, "win_rate": win_rate,
            "avg_win": avg_win, "avg_loss": avg_loss, "payoff": payoff,
            "prec_exec": prec_exec,
        })

        print(f"{theta:>6.2f}  {n_picks:>5d} {n_exec:>5d} {n_pead_picks:>5d} "
              f"{precision*100:>5.1f}% {recall*100:>5.1f}% {f1*100:>5.1f}% "
              f"{avg_pnl:>+7.2f}% {win_rate:>5.1f}% {avg_win:>+7.2f}% "
              f"{avg_loss:>+7.2f}% {payoff:>6.2f}")

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY: theta sweep with pre-gap entry")
    print(f"{'='*100}")

    # Find best theta by different criteria
    print(f"\n  Best by PEAD precision:   theta={max(results, key=lambda r: r['precision'])['theta']:.2f} "
          f"(prec={max(r['precision'] for r in results):.1f}%)")
    print(f"  Best by F1:              theta={max(results, key=lambda r: r['f1'])['theta']:.2f} "
          f"(F1={max(r['f1'] for r in results):.1f}%)")
    print(f"  Best by expectancy:      theta={max(results, key=lambda r: r['avg_pnl'])['theta']:.2f} "
          f"(avg={max(r['avg_pnl'] for r in results):+.2f}%)")
    print(f"  Best by win rate:        theta={max(results, key=lambda r: r['win_rate'])['theta']:.2f} "
          f"(win={max(r['win_rate'] for r in results):.1f}%)")

    # Find theta that hits 40-50% precision
    print(f"\n  Theta values hitting 40%+ precision:")
    for r in results:
        if r["precision"] >= 40:
            print(f"    theta={r['theta']:.2f}: prec={r['precision']:.1f}%, "
                  f"exec={r['n_exec']}, pnl={r['avg_pnl']:+.2f}%, "
                  f"win={r['win_rate']:.1f}%, payoff={r['payoff']:.2f}")

    print(f"\n  Theta values hitting 45%+ precision:")
    for r in results:
        if r["precision"] >= 45:
            print(f"    theta={r['theta']:.2f}: prec={r['precision']:.1f}%, "
                  f"exec={r['n_exec']}, pnl={r['avg_pnl']:+.2f}%, "
                  f"win={r['win_rate']:.1f}%, payoff={r['payoff']:.2f}")

    # Per-fold breakdown for the top candidates
    print(f"\n  PER-FOLD BREAKDOWN for top theta candidates:")
    for target_theta in [0.25, 0.35, 0.45, 0.55]:
        r = next((x for x in results if x["theta"] == target_theta), None)
        if r is None:
            continue
        print(f"\n    theta={target_theta:.2f}: "
              f"picks={r['n_picks']}, exec={r['n_exec']}, "
              f"pead_picks={r['n_pead_picks']}, pead_exec={r['n_pead_exec']}")
        print(f"      prec={r['precision']:.1f}%, recall={r['recall']:.1f}%, F1={r['f1']:.1f}%")
        print(f"      pnl={r['avg_pnl']:+.2f}%, win={r['win_rate']:.1f}%, "
              f"win={r['avg_win']:+.2f}%, loss={r['avg_loss']:+.2f}%, payoff={r['payoff']:.2f}")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

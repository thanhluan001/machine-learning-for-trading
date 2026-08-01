#!/usr/bin/env python3
"""
Investigate the 65% false positives: what's common among them?

The user's insight: 35% precision means 65% of model picks are NOT true
PEAD events. What do these 65% have in common? Are they:
  - Events that ALMOST pass the PEAD gates (e.g., CAR=2.5%, just under 3%)?
  - Events with specific characteristics the model overweights?
  - Events in specific sectors, size bands, or time periods?
  - Events where the model is "right" about positive drift but the strict
    3-gate PEAD definition rejects them?

Key question: Are these "false positives" actually BAD, or are they
false-NEGATIVES of the PEAD label (i.e., the label is too strict)?
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


def main():
    print("=" * 100)
    print("FALSE POSITIVE INVESTIGATION: What's common among the 65%?")
    print("=" * 100)

    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    df = v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df["car_10d_pct"] = np.expm1(df["car_10d"]) * 100
    df = compute_pregap(df, DB, EXIT_SNAP)

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

    all_exec = []
    for fi in range(1, 5):
        td = fold_data[fi]["test_df"]
        mask = (td["p"] >= THETA) & (td["pregap_return"].notna())
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

    pead = exec_df[exec_df["pead_pass"] == 1].copy()
    fp = exec_df[exec_df["pead_pass"] == 0].copy()
    n_total = len(exec_df)
    n_pead = len(pead)
    n_fp = len(fp)

    print(f"\n  Total executed: {n_total}")
    print(f"  True PEAD (label=1): {n_pead} ({n_pead/n_total*100:.1f}%)")
    print(f"  False positives (label=0): {n_fp} ({n_fp/n_total*100:.1f}%)")

    # ===== 1. ARE THE FALSE POSITIVES ACTUALLY BAD TRADES? =====
    print(f"\n{'='*100}")
    print("1. ARE THE FALSE POSITIVES ACTUALLY BAD TRADES?")
    print(f"{'='*100}")

    print(f"\n  Return comparison (pre-gap entry, 5-day hold):")
    for label, sub in [("True PEAD", pead), ("False positives", fp)]:
        r = sub["pregap_return"]
        wins = r[r > 0]
        losses = r[r <= 0]
        wr = len(wins) / len(r) * 100
        print(f"    {label:<18} N={len(r):>3}  Win={wr:>5.1f}%  Avg={r.mean()*100:>+6.2f}%  "
              f"AvgWin={wins.mean()*100:>+6.2f}%  AvgLoss={losses.mean()*100:>+6.2f}%  "
              f"Total={r.sum()*100:>+7.1f}%")

    # ===== 2. WHICH GATE DO FALSE POSITIVES FAIL, AND BY HOW MUCH? =====
    print(f"\n{'='*100}")
    print("2. GATE FAILURE ANALYSIS (which gate, by how much?)")
    print(f"{'='*100}")

    # For each false positive, find the "closest" gate to passing
    fp_gates = []
    for _, row in fp.iterrows():
        gates = {"g1_CAR": row.get("pass_g1", 0), "g2_vol": row.get("pass_g2", 0),
                  "g3_maxdd": row.get("pass_g3", 0)}
        n_pass = sum(gates.values())
        car = row.get("car_10d_pct", float('nan'))
        vol = row.get("inst_vol_ratio", float('nan'))
        maxdd = row.get("maxdd_ma", float('nan'))

        # Which gate is closest to passing?
        closest = []
        if gates["g1_CAR"] == 0 and pd.notna(car):
            closest.append(("g1_CAR", car, 3.0, car - 3.0))  # how far from 3%
        if gates["g2_vol"] == 0 and pd.notna(vol):
            closest.append(("g2_vol", vol, 2.0, vol - 2.0))  # how far from 2.0x
        if gates["g3_maxdd"] == 0 and pd.notna(maxdd):
            closest.append(("g3_maxdd", maxdd * 100, -1.5, maxdd * 100 + 1.5))

        fp_gates.append({
            "ticker": row.get("canonical_ticker", row["permaTicker"]),
            "n_gates_pass": n_pass,
            "car_pct": car,
            "vol_ratio": vol,
            "maxdd_ma_pct": maxdd * 100 if pd.notna(maxdd) else float('nan'),
            "pregap_return": row["pregap_return"] * 100,
            "p_pead": row["p"],
            "closest_gate": closest[0][0] if closest else "none",
            "closest_margin": closest[0][3] if closest else float('nan'),
        })

    fp_gates_df = pd.DataFrame(fp_gates)

    print(f"\n  Gate failure breakdown:")
    print(f"    Fail ALL 3 gates: {int((fp_gates_df['n_gates_pass'] == 0).sum())} ({(fp_gates_df['n_gates_pass'] == 0).mean()*100:.1f}%)")
    print(f"    Pass 1, fail 2:   {int((fp_gates_df['n_gates_pass'] == 1).sum())} ({(fp_gates_df['n_gates_pass'] == 1).mean()*100:.1f}%)")
    print(f"    Pass 2, fail 1:   {int((fp_gates_df['n_gates_pass'] == 2).sum())} ({(fp_gates_df['n_gates_pass'] == 2).mean()*100:.1f}%)")

    # Which single gate is the most common blocker?
    print(f"\n  Which gate is the SOLE blocker (pass 2, fail 1)?")
    fp_reset = fp.reset_index(drop=True).copy()
    fp_reset["n_gates_pass"] = fp_gates_df["n_gates_pass"].values
    fail_1 = fp_reset[fp_reset["n_gates_pass"] == 2]
    for gate_name, gate_col in [("g1 (CAR)", "pass_g1"), ("g2 (vol)", "pass_g2"), ("g3 (MaxDD)", "pass_g3")]:
        if gate_col in fail_1.columns:
            n = int((fail_1[gate_col] == 0).sum())
            print(f"    {gate_name}: {n}")

    # ===== 3. THE "NEAR-PEAD" PHENOMENON =====
    print(f"\n{'='*100}")
    print("3. THE 'NEAR-PEAD' PHENOMENON (CAR close to 3% threshold)")
    print(f"{'='*100}")

    car_fp = fp_gates_df["car_pct"].dropna()
    print(f"\n  CAR distribution of false positives:")
    print(f"    min={car_fp.min():.1f}%, p10={np.percentile(car_fp,10):.1f}%, "
          f"p25={np.percentile(car_fp,25):.1f}%, p50={np.percentile(car_fp,50):.1f}%, "
          f"p75={np.percentile(car_fp,75):.1f}%, p90={np.percentile(car_fp,90):.1f}%, "
          f"max={car_fp.max():.1f}%")

    # Categorize false positives by CAR
    print(f"\n  False positive categories by CAR:")
    categories = [
        ("Negative CAR (< 0%)", car_fp < 0),
        ("Near-zero CAR (0-1%)", (car_fp >= 0) & (car_fp < 1)),
        ("Low CAR (1-3%, near threshold)", (car_fp >= 1) & (car_fp < 3)),
        ("Moderate CAR (3-5%, passes g1)", (car_fp >= 3) & (car_fp < 5)),
        ("Strong CAR (5-10%, passes g1)", (car_fp >= 5) & (car_fp < 10)),
        ("Very strong CAR (>= 10%)", car_fp >= 10),
    ]
    for label, mask in categories:
        sub = fp_gates_df[mask.values]
        n = len(sub)
        avg_ret = sub["pregap_return"].mean() if n > 0 else 0
        wr = (sub["pregap_return"] > 0).mean() * 100 if n > 0 else 0
        print(f"    {label:<35} N={n:>3} ({n/n_fp*100:>4.1f}%)  Ret={avg_ret:>+6.1f}%  Win={wr:.0f}%")

    # ===== 4. SECTOR CONCENTRATION OF FALSE POSITIVES =====
    print(f"\n{'='*100}")
    print("4. SECTOR CONCENTRATION")
    print(f"{'='*100}")

    with pd.HDFStore(DB, mode="r") as s:
        pt_meta = s["/metadata/sp400_permatickers"]
    sector_lookup = pt_meta[["permaTicker", "index_ref"]].drop_duplicates("permaTicker")
    exec_sector = exec_df.merge(sector_lookup, on="permaTicker", how="left")

    print(f"\n  {'Sector':<8} {'Total':>6} {'PEAD':>5} {'FP':>4} {'Prec':>6} {'FP avg ret':>11}")
    for sec in sorted(exec_sector["index_ref"].dropna().unique()):
        sub = exec_sector[exec_sector["index_ref"] == sec]
        sub_pead = sub[sub["pead_pass"] == 1]
        sub_fp = sub[sub["pead_pass"] == 0]
        prec = len(sub_pead) / len(sub) * 100 if len(sub) > 0 else 0
        fp_ret = sub_fp["pregap_return"].mean() * 100 if len(sub_fp) > 0 else 0
        print(f"  {sec:<8} {len(sub):>6} {len(sub_pead):>5} {len(sub_fp):>4} {prec:>5.0f}% {fp_ret:>+10.1f}%")

    # ===== 5. FEATURE COMPARISON: FP vs TRUE PEAD =====
    print(f"\n{'='*100}")
    print("5. FEATURE COMPARISON: FALSE POSITIVES vs TRUE PEAD")
    print(f"{'='*100}")

    print(f"\n  {'Feature':<35} {'PEAD mean':>10} {'FP mean':>10} {'Delta':>8} {'Cohen d':>8}")
    print("  " + "-" * 75)
    for feat in SUNDAY_SAFE:
        if feat not in exec_df.columns:
            continue
        p_vals = pead[feat].dropna()
        f_vals = fp[feat].dropna()
        if len(p_vals) == 0 or len(f_vals) == 0:
            continue
        pm = p_vals.mean()
        fm = f_vals.mean()
        delta = pm - fm
        pooled_std = np.sqrt((p_vals.std()**2 + f_vals.std()**2) / 2)
        cohens_d = delta / pooled_std if pooled_std > 0 else 0
        marker = "***" if abs(cohens_d) > 0.5 else ("**" if abs(cohens_d) > 0.3 else ("*" if abs(cohens_d) > 0.15 else ""))
        print(f"  {feat:<35} {pm:>+9.3f} {fm:>+9.3f} {delta:>+7.3f} {cohens_d:>+7.3f} {marker}")

    # ===== 6. KEY INSIGHT: ARE THE FALSE POSITIVES FALSE NEGATIVES? =====
    print(f"\n{'='*100}")
    print("6. KEY INSIGHT: ARE FALSE POSITIVES ACTUALLY FALSE NEGATIVES OF THE LABEL?")
    print(f"{'='*100}")

    print(f"\n  The PEAD label requires ALL 3 gates to pass:")
    print(f"    Gate 1: CAR > 3% (abnormal return vs market)")
    print(f"    Gate 2: Volume ratio > 2.0x (institutional interest)")
    print(f"    Gate 3: MaxDD_MA > -1.5% (abnormal return never drops below -1.5%)")
    print(f"\n  If a false positive has CAR=2.5% (just under gate 1), the model")
    print(f"  correctly identified a positive-drift event, but the strict label")
    print(f"  rejected it. These are 'false negatives of the label', not model errors.")

    # Count "near-PEAD" (pass 2 gates, marginally fail 1)
    near_pead = fp_gates_df[fp_gates_df["n_gates_pass"] == 2].copy()
    print(f"\n  Near-PEAD (pass 2 gates, fail 1): {len(near_pead)} of {n_fp} false positives")

    # Of the false positives with positive CAR and positive return
    pos_car_fp = fp_gates_df[fp_gates_df["car_pct"] > 0]
    pos_ret_fp = fp_gates_df[fp_gates_df["pregap_return"] > 0]
    print(f"\n  False positives with POSITIVE CAR (> 0%): {len(pos_car_fp)} ({len(pos_car_fp)/n_fp*100:.0f}%)")
    print(f"  False positives with POSITIVE return (> 0%): {len(pos_ret_fp)} ({len(pos_ret_fp)/n_fp*100:.0f}%)")
    print(f"  => {len(pos_ret_fp)/n_fp*100:.0f}% of 'false positives' are actually profitable trades!")

    # What if we relaxed the PEAD definition? (e.g., CAR > 1% instead of 3%)
    print(f"\n  What if we relaxed the CAR gate threshold?")
    for relaxed_car in [1.0, 2.0, 3.0]:
        relaxed_pead = (exec_df["car_10d_pct"] >= relaxed_car) & \
                       (exec_df.get("pass_g2", 0) == 1) & \
                       (exec_df.get("pass_g3", 0) == 1)
        n_relaxed = int(relaxed_pead.sum())
        prec = n_relaxed / n_total * 100
        print(f"    CAR >= {relaxed_car:.0f}%: {n_relaxed} PEAD ({prec:.1f}% precision)")

    # ===== 7. INDIVIDUAL FALSE POSITIVE DETAIL =====
    print(f"\n{'='*100}")
    print("7. ALL 70 FALSE POSITIVES (sorted by CAR)")
    print(f"{'='*100}")

    print(f"\n  {'Ticker':<8} {'Date':<12} {'P(PEAD)':>8} {'CAR%':>7} {'Ret%':>7} {'g1':>3} {'g2':>3} {'g3':>3} {'VolR':>6} {'EPS%':>7} {'Category':<15}")
    for _, t in fp_gates_df.sort_values("car_pct", ascending=False).iterrows():
        ticker = t["ticker"]
        # Find original row
        orig = fp[fp.get("canonical_ticker", fp["permaTicker"]) == ticker]
        if len(orig) == 0:
            continue
        orig = orig.iloc[0]
        rd = str(pd.Timestamp(orig["report_date"]).date())
        p = t["p_pead"]
        car = t["car_pct"]
        ret = t["pregap_return"]
        g1 = int(orig.get("pass_g1", 0))
        g2 = int(orig.get("pass_g2", 0))
        g3 = int(orig.get("pass_g3", 0))
        vol = orig.get("inst_vol_ratio", float('nan'))
        eps = orig.get("eps_surprise_pct", float('nan'))

        # Category
        if car >= 5:
            cat = "Strong CAR"
        elif car >= 3:
            cat = "Passes g1"
        elif car >= 1:
            cat = "Near thresh"
        elif car >= 0:
            cat = "Low CAR"
        else:
            cat = "Negative CAR"

        print(f"  {ticker:<8} {rd:<12} {p:>7.3f} {car:>+6.1f}% {ret:>+6.1f}% "
              f"{g1:>3} {g2:>3} {g3:>3} {vol:>5.1f}x {eps:>+6.1f}% {cat:<15}")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()

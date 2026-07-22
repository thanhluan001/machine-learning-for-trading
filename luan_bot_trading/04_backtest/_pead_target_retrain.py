"""
PEAD-Target Retrain Experiment:
  Train two new v2 rankers on the SAME train period (2015-01-01 -> 2024-01-01):
    - model A: target = binary PEAD-gate-pass (did the 3 verification gates pass?) 
  Then evaluate top-N selection over VAL (2024-2026) by:
    a. Recall on actual PEAD-gate-pass events
    b. Realized arithmetic PnL = mean(expm1(car_10d)) of selected events
    c. Realized 9-day REMAINING drift (Close[T+1] -> Close[T+11]) 
  And compare to the existing car_10d-label ranker.

Uses ranking architecture (XGBRanker ndcg@3) for parity with v2 baseline.

To save iteration time, ONLY 1 model variant is tried (the user is asking
for a comparison experiment, not a hyperparameter sweep). Use the v2 reg
hyperparams that won the sweep: max_depth=3, min_child_weight=50, gamma=5,
subsample=0.7, colsample_bytree=0.7, n_estimators=300.

NO DB WRITES. Reads: train_matrix + permaTicker price nodes + IJH prices.
"""
from __future__ import annotations
import sys, os, importlib.util, json, pickle
from pathlib import Path
import numpy as np, pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "tm", HERE.parent / "03_model" / "01_train_model.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)

# Load exploration helpers for gate computation
e_spec = importlib.util.spec_from_file_location(
    "explor", HERE / "_pead_exploration.py")
em = importlib.util.module_from_spec(e_spec); e_spec.loader.exec_module(em)

DB = tm.DB_FILE

GATE1_CAR_MIN = em.GATE1_CAR_MIN
GATE2_VOL_RATIO_MIN = em.GATE2_VOL_RATIO_MIN
GATE3_MAXDD_MIN = em.GATE3_MAXDD_MIN


def compute_pead_gates_full(train_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute the 3 PEAD gates for every event in train_matrix (TRAIN + VAL
    rows together). Returns train_matrix with added cols:
       pass_g1 (CAR), pass_g2 (volume), pass_g3 (MaxDD_MA),
       inst_vol_ratio, maxdd_ma, pead_pass (int 0/1).
    """
    print(f"\n[*] Computing 3 PEAD gates over {len(train_matrix)} rows ...")
    needed = set(train_matrix["permaTicker"].unique())
    print(f"  distinct permaTickers: {len(needed)}")
    out_arr = []
    with pd.HDFStore(DB, mode="r") as s:
        ijh_df = s["/macros/IJH"]
        ijh_index = pd.to_datetime(ijh_df["Date"]).values
        ijh_close = ijh_df["Close"].values
        for i, pt in enumerate(needed):
            key = f"/sp400/{pt}"
            if key not in s:
                continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_close = p["Adj_Close"].values
            p_vol = p["Adj_Volume"].values
            sub = train_matrix[train_matrix["permaTicker"] == pt]
            for idx, row in sub.iterrows():
                rdate = pd.to_datetime(row["report_date"]).to_datetime64()
                t_mask = p_index >= rdate
                if not t_mask.any():
                    continue
                t_idx = int(np.argmax(t_mask))
                if t_idx < 20:
                    continue
                vma20 = float(np.mean(p_vol[t_idx-20:t_idx]))
                if vma20 <= 0:
                    continue
                if t_idx + 12 >= len(p_close):
                    continue
                # Vol ratio
                inst_vol_avg = float(np.mean(p_vol[t_idx:t_idx+3]))
                vol_ratio = inst_vol_avg / vma20
                # MaxDD_MA over T+1..T+11
                stock_path = p_close[t_idx+1:t_idx+12] / p_close[t_idx] - 1.0
                ijh_t_idx = int(np.searchsorted(ijh_index, p_index[t_idx]))
                if ijh_t_idx + 12 >= len(ijh_close):
                    continue
                ijh_path = ijh_close[ijh_t_idx+1:ijh_t_idx+12] / ijh_close[ijh_t_idx] - 1.0
                n = min(len(stock_path), len(ijh_path))
                stock_path = stock_path[:n]
                ijh_path = ijh_path[:n]
                ma_dd = stock_path - ijh_path
                if len(ma_dd) == 0:
                    continue
                maxdd_ma = float(np.min(ma_dd))
                out_arr.append({
                    "row_idx": idx,
                    "inst_vol_ratio": vol_ratio,
                    "maxdd_ma": maxdd_ma,
                })
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(needed)} permaTickers processed")
    print(f"  computed gate signals for {len(out_arr)} events "
          f"of {len(train_matrix)} total")
    g_df = pd.DataFrame(out_arr).set_index("row_idx")
    train_matrix = train_matrix.join(g_df, how="left")
    train_matrix["pass_g1"] = (train_matrix["car_10d"].fillna(-9) > GATE1_CAR_MIN).astype(int)
    train_matrix["pass_g2"] = (train_matrix["inst_vol_ratio"] > GATE2_VOL_RATIO_MIN).astype(int)
    train_matrix["pass_g3"] = (train_matrix["maxdd_ma"] > GATE3_MAXDD_MIN).astype(int)
    # The 3 gates combined label:
    train_matrix["pead_pass"] = (
        (train_matrix["pass_g1"] == 1)
        & (train_matrix["pass_g2"] == 1)
        & (train_matrix["pass_g3"] == 1)
    ).astype(int)
    # Single per-gate masks too for diagnostics
    return train_matrix


def main():
    # Load full train_matrix + apply 12 priming cutoff
    print("[1] Loading train_matrix + applying §12 priming cutoff ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    print(f"  rows after §12 cut: {len(df)}")

    # Compute the 3 gates on ALL rows (TRAIN + VAL together).
    df = compute_pead_gates_full(df)

    # Quick stats
    print(f"\n[*] BASE RATES (over {len(df)} primed rows):")
    for col, name in [
        ("pass_g1", "Gate1 (CAR > +3%)"),
        ("pass_g2", "Gate2 (Vol > 2x vma20)"),
        ("pass_g3", "Gate3 (MaxDD_MA > -1.5%)"),
        ("pead_pass", "All 3 PEAD gates combined"),
    ]:
        n_pos = int(df[col].sum())
        print(f"  {name:25s} positives: {n_pos:5d} ({n_pos/len(df)*100:.2f}%)")

    # Split into TRAIN / VAL
    train_df, val_df = tm.split_walk_forward(df, tm.DEFAULT_SPLIT_DATE)
    train_df, _ = tm.drop_sparse_weeks(train_df, tm.DEFAULT_MIN_GROUP_SIZE)
    val_df, _ = tm.drop_sparse_weeks(val_df, tm.DEFAULT_MIN_GROUP_SIZE)
    print(f"  TRAIN rows: {len(train_df)}  weeks: {train_df['calendar_week_group'].nunique()}  "
          f"PEAD-pos: {int(train_df['pead_pass'].sum())} "
          f"({train_df['pead_pass'].mean()*100:.2f}%)")
    print(f"  VAL   rows: {len(val_df)}  weeks: {val_df['calendar_week_group'].nunique()}  "
          f"PEAD-pos: {int(val_df['pead_pass'].sum())} "
          f"({val_df['pead_pass'].mean()*100:.2f}%)")

    # Sort for listwise ranker
    train_df = train_df.sort_values(
        [tm.GROUP_COLUMN, "permaTicker", "report_date"]
    ).reset_index(drop=True)
    val_df = val_df.sort_values(
        [tm.GROUP_COLUMN, "permaTicker", "report_date"]
    ).reset_index(drop=True)

    X_train = train_df[tm.FEATURE_COLUMNS].copy()
    X_val = val_df[tm.FEATURE_COLUMNS].copy()
    g_train = train_df.groupby(tm.GROUP_COLUMN, sort=True).size().values.astype(int)
    g_val = val_df.groupby(tm.GROUP_COLUMN, sort=True).size().values.astype(int)
    # Target: pead_pass (0/1) -- already int.
    y_pead_train = train_df["pead_pass"].astype(int)
    y_pead_val = val_df["pead_pass"].astype(int)
    y_car_log_train = pd.to_numeric(train_df[tm.LABEL_COLUMN], errors="coerce")
    y_car_log_val = pd.to_numeric(val_df[tm.LABEL_COLUMN], errors="coerce")

    # ----- Train PEAD-target ranker -----
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    params = dict(
        objective="rank:ndcg",
        eval_metric="ndcg@3",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=50,
        gamma=5.0,
        reg_lambda=1.0,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
    )
    print(f"\n[2] Training PEAD-target ranker ...")
    print(f"  params: {params}")
    print(f"  target: pead_pass (binary 0/1, mean={y_pead_train.mean():.4f})")
    t0 = time.time() if False else __import__('time').time()
    ranker_pead = xgb.XGBRanker(**params)
    ranker_pead.fit(
        X_train, y_pead_train, group=g_train,
        eval_set=[(X_val, y_pead_val)],
        eval_group=[g_val],
        verbose=False,
    )
    train_s = __import__('time').time() - t0
    print(f"  Trained in {train_s:.1f}s")
    # Evaluate NDCG@3 on val (ranker's native metric)
    val_eval_pead = tm.evaluate_ranker(ranker_pead, X_val, y_pead_val.astype(float), g_val, "VAL")
    train_eval_pead = tm.evaluate_ranker(ranker_pead, X_train, y_pead_train.astype(float), g_train, "TRAIN")
    print(f"  TRAIN NDCG@3 (pead target): {train_eval_pead['ndcg_at_3']:.4f}")
    print(f"  VAL   NDCG@3 (pead target): {val_eval_pead['ndcg_at_3']:.4f}")

    # ----- For comparison, normalize: train the SAME params on old car_10d target -----
    print(f"\n[3] Re-training CAR-target ranker (sanity baseline, same hyperparams) ...")
    # Discretize car_10d to 10 buckets (same as v2 reg model) for fair ndcg objective.
    N_BUCKETS = 10
    y_train_int, boundaries = tm.discretize_label_quantiles(y_car_log_train, n_buckets=N_BUCKETS)
    y_val_int, _ = tm.discretize_label_quantiles(y_car_log_val, n_buckets=N_BUCKETS, boundaries=boundaries)
    ranker_car = xgb.XGBRanker(**params)
    ranker_car.fit(
        X_train, y_train_int.fillna(0).astype(int), group=g_train,
        eval_set=[(X_val, y_val_int.fillna(0).astype(int))],
        eval_group=[g_val],
        verbose=False,
    )
    train_eval_car = tm.evaluate_ranker(ranker_car, X_train, y_car_log_train, g_train, "TRAIN")
    val_eval_car = tm.evaluate_ranker(ranker_car, X_val, y_car_log_val, g_val, "VAL")
    print(f"  TRAIN NDCG@3 (car target): {train_eval_car['ndcg_at_3']:.4f}")
    print(f"  VAL   NDCG@3 (car target): {val_eval_car['ndcg_at_3']:.4f}")

    # ----- Top-N SELECTION EVALUATION -----
    # For each model, predict scores on val and select top-N per week.
    # Compute (a) top-N recall on PEAD-pass events, (b) realized arith PnL
    # from expm1(car_10d), (c) realized arith 9-day REMAINING drift PnL.
    print(f"\n[4] Top-N selection evaluation ...")
    val_df_with_gates = val_df.copy()
    # We need the realized 9-day remaining drift to fairly evaluate the entry-at-T+1
    # close strategy. Compute it via the same function used in _pead_gap_strategy.py
    s_spec = importlib.util.spec_from_file_location(
        "pead_gap", HERE / "_pead_gap_strategy.py")
    pg = importlib.util.module_from_spec(s_spec); s_spec.loader.exec_module(pg)
    val_with_rem = pg.compute_remaining_drift(val_df_with_gates)

    # Drop rows where car_remaining_9d_log is NaN (incomplete data near series end)
    eval_df = val_with_rem.dropna(subset=["car_remaining_9d_log"])

    # Predict raw scores with each model
    raw_pead = ranker_pead.predict(eval_df[tm.FEATURE_COLUMNS])
    raw_car = ranker_car.predict(eval_df[tm.FEATURE_COLUMNS])
    eval_df["pead_score"] = raw_pead
    eval_df["car_score"] = raw_car

    print(f"\n[*] Per-week top-N selection comparison")
    print(f"\n  Two ways of measuring:")
    print(f"  (A) Realized full 10-day arith CAR = expm1(car_10d)")
    print(f"  (B) Realized REMAINING 9-day arith drift = expm1(car_remaining_9d_log)")
    print(f"      (excludes T+1 open-to-close, which is in (A) by definition)")
    print(f"  (C) Top-N selects an event that is a PEAD? (recall on gate)")
    print(f"{'model':>32s} {'n':>4s} {'full10d%':>10s} "
          f"{'rem9d%':>10s} {'PEAD-recall%':>13s} {'hit%':>7s}")
    print("-" * 84)
    for top_n in [1, 3, 5, 10]:
        for model_name, score_col in [("PEAD-target", "pead_score"),
                                      ("CAR-target", "car_score")]:
            picks_list = []
            for week, g in eval_df.groupby("calendar_week_group", sort=True):
                g_ok = g.dropna(subset=[score_col])
                if g_ok.empty:
                    continue
                n_pick = min(top_n, len(g_ok))
                sel = g_ok.sort_values(score_col, ascending=False).head(n_pick)
                picks_list.append(sel)
            if not picks_list:
                continue
            picks = pd.concat(picks_list)
            # Recall on PEAD-passed events among top-N picks:
            ## top-N picks had {'pead_pass': 1} of all selected events.
            n_picks = len(picks)
            n_pead_picked = int(picks["pead_pass"].sum())
            ## Total PEAD-passed events in val:
            n_pead_total = int(eval_df["pead_pass"].sum())
            recall = n_pead_picked / n_pead_total if n_pead_total else float("nan")
            # Realized returns
            full_arith = float(np.mean(np.expm1(picks["car_10d"].fillna(0))))
            rem_arith = float(np.mean(np.expm1(picks["car_remaining_9d_log"])))
            hit_rate = float((picks["car_remaining_9d_log"] > 0).mean())
            print(f"  {model_name:>28s}  n={top_n:>3d}  {full_arith*100:+8.3f}%  "
                  f"{rem_arith*100:+8.3f}%  {recall*100:11.2f}%  "
                  f"{hit_rate*100:5.1f}%")

    # ----- Persist - target PEAD model -----
    out_dir = HERE.parent / "03_model" / "models" / "phase_f_v2_pead_target"
    out_dir.mkdir(parents=True, exist_ok=True)
    ranker_pead.save_model(str(out_dir / "ranker.json"))
    # Calibrator fit on val raw scores -> expm1(car_10d)  (the mu-unit calibrator
    # we already have in Stage 3 design; for the new pead target the calibrator
    # still maps ranker raw scores -- they're just learned under a different
    # objective -- to arithmetic CAR via isotonic bridge).
    # Calibrator fit on val raw scores -> expm1(car_10d). Align by the same
    # rows the raw_pead prediction was made on.
    raw_pead_scores = ranker_pead.predict(X_val)
    target_arith = np.expm1(y_car_log_val.fillna(0).values)
    calib = IsotonicRegression(out_of_bounds="clip")
    calib.fit(raw_pead_scores, target_arith)
    with open(out_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(calib, f)
    meta = {
        "name": "phase_f_v2_pead_target",
        "objective": "rank:ndcg",
        "target_label": "pead_pass (binary 0/1 from 3 PEAD verification gates)",
        "xgb_params": params,
        "train_rows_pead_positives": int(train_df["pead_pass"].sum()),
        "val_rows_pead_positives": int(val_df["pead_pass"].sum()),
        "train_eval_ndcg_at_3": train_eval_pead["ndcg_at_3"],
        "val_eval_ndcg_at_3": val_eval_pead["ndcg_at_3"],
        "gate_thresholds": {"GATE1_CAR_MIN": GATE1_CAR_MIN,
                            "GATE2_VOL_RATIO_MIN": GATE2_VOL_RATIO_MIN,
                            "GATE3_MAXDD_MIN": GATE3_MAXDD_MIN},
        "feature_columns": tm.FEATURE_COLUMNS,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n[*] Saved PEAD-target model to {out_dir}")
    print(f"  ranker.json, calibrator.pkl, meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
PEAD-target binary classifier experiment.

Two new models to compare:
  1. XGBClassifier (binary: pead_pass @ 0/1). Trained on the same hyperparams 
     that worked for the ranker (though with a binary objective instead of ndcg).
  2. Thresholded trading rule: enter an event if predicted P(PEAD) > threshold,
     hold 10 days (Open[T+1] -> Close[T+11]). Compare PnL envelope across 
     different thresholds and across the ranker baseline.

Hypothesis: a calibrated classifier with a high-precision threshold can 
select a subset of true PEADs and capture meaningful per-event alpha.
"""
from __future__ import annotations
import sys, os, importlib.util, json, pickle, time
from pathlib import Path
import numpy as np, pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "tm", HERE.parent / "03_model" / "01_train_model.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)

# Re-use compute_pead_gates_full from the previous retrain script
r_spec = importlib.util.spec_from_file_location(
    "retrain", HERE / "_pead_target_retrain.py")
m = importlib.util.module_from_spec(r_spec); r_spec.loader.exec_module(m)

DB = tm.DB_FILE


def main():
    # ----- Load + gate -----
    print("[1] Loading train_matrix + applying priming cutoff ...")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    print(f"  rows after §12: {len(df)}")
    df = m.compute_pead_gates_full(df)
    # Base rates
    print(f"  pead_pass positives: {int(df['pead_pass'].sum())} ({df['pead_pass'].mean()*100:.2f}%)")

    train_df, val_df = tm.split_walk_forward(df, tm.DEFAULT_SPLIT_DATE)
    train_df, _ = tm.drop_sparse_weeks(train_df, tm.DEFAULT_MIN_GROUP_SIZE)
    val_df, _ = tm.drop_sparse_weeks(val_df, tm.DEFAULT_MIN_GROUP_SIZE)
    train_df = train_df.sort_values(
        ["calendar_week_group","permaTicker", "report_date"]).reset_index(drop=True)
    val_df = val_df.sort_values(
        ["calendar_week_group","permaTicker", "report_date"]).reset_index(drop=True)
    print(f"  TRAIN rows: {len(train_df)}  pead_pos: {int(train_df['pead_pass'].sum())} "
          f"({train_df['pead_pass'].mean()*100:.2f}%)")
    print(f"  VAL   rows: {len(val_df)}  pead_pos: {int(val_df['pead_pass'].sum())} "
          f"({val_df['pead_pass'].mean()*100:.2f}%)")

    X_train = train_df[tm.FEATURE_COLUMNS].copy()
    y_train = train_df["pead_pass"].astype(int).values
    X_val = val_df[tm.FEATURE_COLUMNS].copy()
    y_val = val_df["pead_pass"].astype(int).values

    # ----- Train classifier -----
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    params = dict(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
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
    print(f"\n[2] Training XGBClassifier (binary logistic, target=pead_pass) ...")
    print(f"  params: {params}")
    clf = xgb.XGBClassifier(**params)
    t0 = time.time()
    clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False)
    train_s = time.time() - t0
    print(f"  Trained in {train_s:.1f}s")
    # Predicted probs on val
    val_proba = clf.predict_proba(X_val)[:, 1]
    train_proba = clf.predict_proba(X_train)[:, 1]
    from sklearn.metrics import roc_auc_score, average_precision_score, \
        precision_recall_curve, brier_score_loss
    auc_train = roc_auc_score(y_train, train_proba)
    auc_val = roc_auc_score(y_val, val_proba)
    ap_train = average_precision_score(y_train, train_proba)
    ap_val = average_precision_score(y_val, val_proba)
    print(f"  TRAIN AUC: {auc_train:.4f}  AP: {ap_train:.4f}")
    print(f"  VAL   AUC: {auc_val:.4f}  AP: {ap_val:.4f}")

    # ----- Add probs to val_df -----
    val_df["pead_proba"] = val_proba

    # ----- Compute realized entry-PnL = Open[T+1] -> Close[T+11] -----
    print(f"\n[3] Computing realized entry PnL Open[T+1]->Close[T+11] ...")
    val_df["ret_open_t1_close_t11"] = np.nan
    with pd.HDFStore(DB, mode="r") as s:
        pts = val_df["permaTicker"].unique()
        for pt in pts:
            key = f"/sp400/{pt}"
            if key not in s: continue
            p = s[key]
            p_index = pd.to_datetime(p["Date"]).values
            p_open = p["Adj_Open"].values
            p_close = p["Adj_Close"].values
            sub = val_df[val_df["permaTicker"]==pt]
            for idx, row in sub.iterrows():
                rdate = pd.to_datetime(row["report_date"]).to_datetime64()
                t_mask = p_index >= rdate
                if not t_mask.any(): continue
                t_idx = int(np.argmax(t_mask))
                if t_idx + 11 >= len(p_close): continue
                o_t1 = p_open[t_idx + 1]; c_t11 = p_close[t_idx + 11]
                if pd.isna(o_t1) or pd.isna(c_t11) or o_t1 <= 0: continue
                val_df.loc[idx, "ret_open_t1_close_t11"] = float(np.log(c_t11 / o_t1))
    print(f"  coverage: {val_df['ret_open_t1_close_t11'].notna().sum()}/{len(val_df)}")

    # ----- Threshold sweep: enter if predicted P(PEAD) > THRESH -----
    print(f"\n[4] THRESHOLD SWEEP (predicted P(PEAD) > THRESH -> trade all such events):")
    print(f"  Unconditional universe mean PnL/event: {val_df['ret_open_t1_close_t11'].mean()*100:+.3f}% (n={val_df['ret_open_t1_close_t11'].notna().sum()})")
    print(f"  Oracle (pead_pass==1) mean PnL/event: {val_df[val_df['pead_pass']==1]['ret_open_t1_close_t11'].mean()*100:+.3f}% (n={int(val_df['pead_pass'].sum())})")
    print(f"\n  {'thresh':>7s} {'n_trades':>9s} {'recall%':>9s} {'precision%':>11s} {'avg_pnl%':>10s} {'hit%':>7s}")
    print('  ' + '-' * 65)
    n_total_pead = int((val_df["pead_pass"] == 1).sum())
    trade_lists_by_thresh = {}
    for thresh in [0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.03]:
        pick_mask = val_df["pead_proba"] >= thresh
        picks = val_df[pick_mask].dropna(subset=["ret_open_t1_close_t11"])
        if len(picks) < 1:
            continue
        n_picks = len(picks)
        n_pos = int((picks["pead_pass"]==1).sum())
        recall = n_pos / n_total_pead * 100
        precision = n_pos / n_picks * 100
        arith = np.expm1(picks["ret_open_t1_close_t11"])
        avg_pnl = float(arith.mean()) * 100
        hit_rate = float((arith > 0).mean()) * 100
        spread_to_universe = avg_pnl - val_df['ret_open_t1_close_t11'].mean()*100
        print(f"  >={thresh:>5.2f}  {n_picks:>6d}   {recall:>7.2f}%   {precision:>9.2f}%   "
              f"{avg_pnl:>+8.3f}%   {hit_rate:>5.1f}%")
        trade_lists_by_thresh[thresh] = picks
    print()

    # ----- ALSO: stratify by quantiles of predicted probability -----
    print(f"\n[5] QUANTILE-BINNED PnL (look at model's calibration):")
    val_df["proba_bucket"] = pd.qcut(val_df["pead_proba"], q=10, duplicates="drop")
    ag = val_df.groupby("proba_bucket", observed=True).agg(
        n=("ret_open_t1_close_t11","count"),
        actual_pead_rate=("pead_pass","mean"),
        avg_min_proba=("pead_proba","min"),
        avg_max_proba=("pead_proba","max"),
        mean_pnl=("ret_open_t1_close_t11","mean"),
        mean_pnl_arith_pct=("ret_open_t1_close_t11", lambda x: float(np.expm1(x).mean()*100)),
    )
    ag["mid_proba"] = (ag["avg_min_proba"] + ag["avg_max_proba"]) / 2
    print(ag.to_string())

    # Persist the classifier
    out_dir = HERE.parent / "03_model" / "models" / "phase_f_v2_pead_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(out_dir / "classifier.json"))
    # Calibrator (predict_proba -> expected CAR, for absolute Kelly sizing)
    mask = val_df["ret_open_t1_close_t11"].notna()
    calib = IsotonicRegression(out_of_bounds="clip")
    calib.fit(val_proba[mask], np.expm1(val_df.loc[mask, 'ret_open_t1_close_t11']))
    with open(out_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(calib, f)
    meta = {
        "name": "phase_f_v2_pead_classifier",
        "objective": "binary:logistic",
        "target_label": "pead_pass (binary 0/1 from 3 PEAD verification gates)",
        "xgb_params": params,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "train_pead_pos": int(y_train.sum()),
        "val_pead_pos": int(y_val.sum()),
        "auc_train": float(auc_train), "auc_val": float(auc_val),
        "ap_train": float(ap_train), "ap_val": float(ap_val),
        "gate_thresholds": {"GATE1_CAR_MIN": em.GATE1_CAR_MIN if False else m.GATE1_CAR_MIN,
                            "GATE2_VOL_RATIO_MIN": m.GATE2_VOL_RATIO_MIN,
                            "GATE3_MAXDD_MIN": m.GATE3_MAXDD_MIN},
        "feature_columns": list(tm.FEATURE_COLUMNS),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n[*] Saved classifier to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

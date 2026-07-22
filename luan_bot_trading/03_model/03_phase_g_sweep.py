"""
Phase G v1.1 -- Sunday-safe classifier hyperparameter sweep.

Per phase_g_findings.md §6.4 item 1:
  Sweep gamma in {3, 5, 10, 20}, min_child_weight in {20, 50, 100},
  max_depth in {2, 3, 4}, n_estimators in {200, 300, 500}.

For each config:
  - Train XGBClassifier on the 17 Sunday-safe features.
  - Compute TRAIN+VAL AUC + AP (the original metric).
  - Compute VAL per-event PnL at two operating points:
       (a) Sunday-passthrough:     P(PEAD) >= 0.20, no gap filter.
       (b) Two-stage deployable:   P(PEAD) >= 0.20 AND opening_gap_t1
                                    in [+2%, +15%].
       (c) Vigilant high-conf:     P(PEAD) >= 0.25 AND gap [+2%, +15%].
  - Write a leaderboard CSV sorted by per-event PnL at (b).

NO DB WRITES. Imports phase_g helpers (gate computation, entry-PnL
computation, feature set) so we get apples-to-apples comparisons.

Saved artifacts (best config only):
  03_model/models/phase_g_v1_1_sunday_sweep/
    leaderboard.csv             -- all configs + metrics
    classifier.json             -- best VAL-per-event-PnL config
    calibrator.pkl
    meta.json
"""
from __future__ import annotations
import sys, importlib.util, json, pickle, time, itertools
from pathlib import Path
import numpy as np, pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

HERE = Path(__file__).resolve().parent
# Reuse train module + Phase G helpers
spec = importlib.util.spec_from_file_location(
    "tm", HERE / "01_train_model.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
pg_spec = importlib.util.spec_from_file_location(
    "pg", HERE / "02_phase_g_sunday_classifier.py")
pg = importlib.util.module_from_spec(pg_spec); pg_spec.loader.exec_module(pg)

DB = tm.DB_FILE
SUNDAY_SAFE_FEATURES = pg.SUNDAY_SAFE_FEATURES

# Imports for fitting
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------------------------
# Sweep grid (72 configs total)
# ---------------------------------------------------------------------------
SWEEP_GRID = list(itertools.product(
    [3, 5, 10, 20],   # gamma
    [20, 50, 100],    # min_child_weight
    [2, 3, 4],        # max_depth
    [200, 300, 500],  # n_estimators
))


def eval_operating_points(clf, val_df: pd.DataFrame,
                          valid_mask: pd.Series,
                          n_total_pead: int) -> dict:
    """Return per-event PnL stats at three operating points."""
    proba = clf.predict_proba(val_df[SUNDAY_SAFE_FEATURES])[:, 1]
    out = {}
    for name, theta, gap_lo, gap_hi in [
        ("sunday_passthru_020", 0.20, None, None),
        ("twostage_020_gap_2_15", 0.20, 0.02, 0.15),
        ("twostage_025_gap_2_15", 0.25, 0.02, 0.15),
    ]:
        mask = (proba >= theta) & valid_mask.values
        if gap_lo is not None:
            mask = mask & (val_df["opening_gap_t1"] >= gap_lo).values \
                         & (val_df["opening_gap_t1"] <= gap_hi).values
        picks = val_df[mask]
        if len(picks) < 1:
            out[name] = {"n": 0, "avg_pnl_pct": float("nan"),
                         "hit_pct": float("nan"),
                         "sharpe": float("nan"),
                         "recall_pct": float("nan")}
            continue
        arith = np.expm1(picks["ret_open_t1_close_t11"])
        n = len(picks)
        n_pos = int((picks["pead_pass"] == 1).sum())
        out[name] = {
            "n": n,
            "avg_pnl_pct": float(arith.mean()) * 100,
            "hit_pct": float((arith > 0).mean()) * 100,
            "sharpe": (float(arith.mean()) / (float(arith.std()) + 1e-9)
                       * np.sqrt(52)) if float(arith.std()) > 0 else 0.0,
            "recall_pct": n_pos / n_total_pead * 100,
        }
    return out


def main():
    print("=" * 78)
    print("PHASE G v1.1 -- Sunday-safe classifier HYPERPARAMETER SWEEP")
    print("=" * 78)
    print(f"\nSweep grid size: {len(SWEEP_GRID)} configs")

    # ------ Shared data prep (gate computation + entry PnL) -------
    print("\n[1] Loading train_matrix + applying §12 cutoff + walk-forward split")
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    print(f"    rows after §12 cut: {len(df)}")

    print("\n[2] Computing 3 PEAD gates across all primed rows ...")
    df = pg.v3.compute_pead_gates_full(df)
    train_df, val_df = tm.split_walk_forward(df, tm.DEFAULT_SPLIT_DATE)
    train_df, _ = tm.drop_sparse_weeks(train_df, tm.DEFAULT_MIN_GROUP_SIZE)
    val_df, _ = tm.drop_sparse_weeks(val_df, tm.DEFAULT_MIN_GROUP_SIZE)
    train_df = train_df.sort_values(
        ["calendar_week_group", "permaTicker", "report_date"]
    ).reset_index(drop=True)
    val_df = val_df.sort_values(
        ["calendar_week_group", "permaTicker", "report_date"]
    ).reset_index(drop=True)
    n_total_pead_val = int((val_df["pead_pass"] == 1).sum())
    print(f"    TRAIN rows: {len(train_df)}  VAL rows: {len(val_df)}  "
          f"VAL pead_pos: {n_total_pead_val}")

    print("\n[3] Computing realized entry PnL Open[T+1] -> Close[T+11] ...")
    val_df = pg.compute_entry_pnl(val_df)
    valid_mask = val_df["ret_open_t1_close_t11"].notna()
    print(f"    coverage: {int(valid_mask.sum())}/{len(val_df)}")

    X_train = train_df[SUNDAY_SAFE_FEATURES].copy()
    y_train = train_df["pead_pass"].astype(int).values
    X_val = val_df[SUNDAY_SAFE_FEATURES].copy()
    y_val = val_df["pead_pass"].astype(int).values

    # ------ Sweep loop ------------------------------------------------
    print("\n[4] Sweep over 72 configs (training + eval) ...")
    print(f"\n{'gamma':>6s} {'mcw':>5s} {'md':>3s} {'n_est':>5s}  "
          f"{'auc_v':>7s} {'ap_v':>6s}  "
          f"{'a_n':>5s} {'a_pnl%':>8s}  "
          f"{'b_n':>5s} {'b_pnl%':>8s} {'b_sr':>6s}  "
          f"{'c_n':>5s} {'c_pnl%':>8s}  "
          f"{'t_s':>4s}")
    print("-" * 115)

    rows = []
    t_sweep_start = time.time()
    for i, (gamma, mcw, md, n_est) in enumerate(SWEEP_GRID):
        params = dict(
            objective="binary:logistic",
            eval_metric=["logloss", "auc"],
            n_estimators=n_est,
            learning_rate=0.05,
            max_depth=md,
            min_child_weight=mcw,
            gamma=gamma,
            reg_lambda=1.0,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=-1,
        )
        t0 = time.time()
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False)
        train_s = time.time() - t0
        train_proba = clf.predict_proba(X_train)[:, 1]
        val_proba = clf.predict_proba(X_val)[:, 1]
        auc_train = roc_auc_score(y_train, train_proba)
        auc_val = roc_auc_score(y_val, val_proba)
        ap_train = average_precision_score(y_train, train_proba)
        ap_val = average_precision_score(y_val, val_proba)
        # Operating-point evals
        op = eval_operating_points(clf, val_df, valid_mask, n_total_pead_val)
        # stash row
        row = dict(
            gamma=gamma, min_child_weight=mcw, max_depth=md,
            n_estimators=n_est,
            auc_train=auc_train, auc_val=auc_val,
            ap_train=ap_train, ap_val=ap_val,
            train_s=train_s,
            # a: sunday passthru 0.20
            a_n=op["sunday_passthru_020"]["n"],
            a_avg_pnl_pct=op["sunday_passthru_020"]["avg_pnl_pct"],
            a_hit_pct=op["sunday_passthru_020"]["hit_pct"],
            a_sharpe=op["sunday_passthru_020"]["sharpe"],
            a_recall_pct=op["sunday_passthru_020"]["recall_pct"],
            # b: twostage 0.20 + gap [+2, +15]
            b_n=op["twostage_020_gap_2_15"]["n"],
            b_avg_pnl_pct=op["twostage_020_gap_2_15"]["avg_pnl_pct"],
            b_hit_pct=op["twostage_020_gap_2_15"]["hit_pct"],
            b_sharpe=op["twostage_020_gap_2_15"]["sharpe"],
            b_recall_pct=op["twostage_020_gap_2_15"]["recall_pct"],
            # c: twostage 0.25 + gap [+2, +15]
            c_n=op["twostage_025_gap_2_15"]["n"],
            c_avg_pnl_pct=op["twostage_025_gap_2_15"]["avg_pnl_pct"],
            c_hit_pct=op["twostage_025_gap_2_15"]["hit_pct"],
            c_sharpe=op["twostage_025_gap_2_15"]["sharpe"],
            c_recall_pct=op["twostage_025_gap_2_15"]["recall_pct"],
        )
        rows.append(row)
        print(f"{gamma:>6d} {mcw:>5d} {md:>3d} {n_est:>5d}  "
              f"{auc_val:>7.4f} {ap_val:>6.4f}  "
              f"{op['sunday_passthru_020']['n']:>5d} "
              f"{op['sunday_passthru_020']['avg_pnl_pct']:>+7.3f}  "
              f"{op['twostage_020_gap_2_15']['n']:>5d} "
              f"{op['twostage_020_gap_2_15']['avg_pnl_pct']:>+7.3f} "
              f"{op['twostage_020_gap_2_15']['sharpe']:>+5.2f}  "
              f"{op['twostage_025_gap_2_15']['n']:>5d} "
              f"{op['twostage_025_gap_2_15']['avg_pnl_pct']:>+7.3f}  "
              f"{train_s:>4.1f}s")

    sweep_s = time.time() - t_sweep_start
    print(f"\nSweep complete in {sweep_s:.1f}s "
          f"({sweep_s/len(SWEEP_GRID):.2f}s/config)")

    leaderboard = pd.DataFrame(rows)
    out_dir = HERE / "models" / "phase_g_v1_1_sunday_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------ Leaderboard sorted byVAL AUC ------------------------------
    leaderboard_by_auc = leaderboard.sort_values(
        "auc_val", ascending=False).reset_index(drop=True)
    leaderboard_by_auc.to_csv(out_dir / "leaderboard_by_auc.csv", index=False)
    print(f"\n[*] TOP 10 by VAL AUC:")
    print(leaderboard_by_auc[
        ["gamma", "min_child_weight", "max_depth", "n_estimators",
         "auc_train", "auc_val", "ap_train", "ap_val",
         "b_avg_pnl_pct", "b_sharpe", "b_recall_pct"]
    ].head(10).to_string(index=False))

    # ------ Leaderboard sorted by per-event PnL at (b) --------------
    # min-N filter: at least 30 trades at (b) for statistical stability.
    leaderboard_by_b = leaderboard[
        leaderboard["b_n"] >= 30
    ].sort_values("b_avg_pnl_pct", ascending=False).reset_index(drop=True)
    leaderboard_by_b.to_csv(out_dir / "leaderboard_by_b_pnl.csv", index=False)
    print(f"\n[*] TOP 10 by per-event PnL at (b) twostage_020_gap_2_15 "
          f"(filtered: b_n >= 30):")
    print(leaderboard_by_b[
        ["gamma", "min_child_weight", "max_depth", "n_estimators",
         "auc_val", "ap_val",
         "b_n", "b_avg_pnl_pct", "b_hit_pct", "b_sharpe", "b_recall_pct"]
    ].head(10).to_string(index=False))

    # ------ Overall leaderboard (saved unfiltered) ----------------
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)

    # ------ Persist best-by-(b)-PnL config ------------------------
    if len(leaderboard_by_b) > 0:
        best = leaderboard_by_b.iloc[0]
        best_params = dict(
            objective="binary:logistic",
            eval_metric=["logloss", "auc"],
            n_estimators=int(best["n_estimators"]),
            learning_rate=0.05,
            max_depth=int(best["max_depth"]),
            min_child_weight=int(best["min_child_weight"]),
            gamma=float(best["gamma"]),
            reg_lambda=1.0,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=-1,
        )
        print(f"\n[*] BEST config (by per-event PnL at b): "
              f"gamma={best['gamma']}, mcw={best['min_child_weight']}, "
              f"md={best['max_depth']}, n_est={best['n_estimators']}")
        print(f"  AUC val: {best['auc_val']:.4f}  AP val: {best['ap_val']:.4f}")
        print(f"  b_n: {int(best['b_n'])}  b_avg_pnl: "
              f"{best['b_avg_pnl_pct']:+.4f}%  b_hit: {best['b_hit_pct']:.1f}%  "
              f"b_sharpe: {best['b_sharpe']:+.2f}  b_recall: "
              f"{best['b_recall_pct']:.2f}%")
        clf_best = xgb.XGBClassifier(**best_params)
        clf_best.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     verbose=False)
        clf_best.save_model(str(out_dir / "classifier.json"))
        calib_mask = valid_mask
        calib = IsotonicRegression(out_of_bounds="clip")
        val_proba_best = clf_best.predict_proba(X_val)[:, 1]
        calib.fit(val_proba_best[calib_mask.values],
                  np.expm1(val_df.loc[calib_mask,
                                      "ret_open_t1_close_t11"]).values)
        with open(out_dir / "calibrator.pkl", "wb") as f:
            pickle.dump(calib, f)
        meta = {
            "name": "phase_g_v1_1_sunday_sweep_best",
            "objective": "binary:logistic",
            "feature_set": "sunday_safe_17",
            "feature_columns": SUNDAY_SAFE_FEATURES,
            "xgb_params": best_params,
            "selection_criterion": "max b_avg_pnl_pct subject to b_n >= 30",
            "auc_train": float(best["auc_train"]),
            "auc_val": float(best["auc_val"]),
            "ap_train": float(best["ap_train"]),
            "ap_val": float(best["ap_val"]),
            "b_n": int(best["b_n"]),
            "b_avg_pnl_pct": float(best["b_avg_pnl_pct"]),
            "b_hit_pct": float(best["b_hit_pct"]),
            "b_sharpe": float(best["b_sharpe"]),
            "b_recall_pct": float(best["b_recall_pct"]),
            "vs_phase_g_v1_baseline": {
                "v1_gamma": 5, "v1_mcw": 50, "v1_md": 3, "v1_n_est": 300,
                "v1_auc_val": 0.642, "v1_b_n": 98, "v1_b_avg_pnl_pct": 1.716,
                "v1_b_sharpe": 1.55, "v1_b_recall_pct": 9.19,
            },
            "gate_thresholds": {
                "GATE1_CAR_MIN": pg.v3.GATE1_CAR_MIN,
                "GATE2_VOL_RATIO_MIN": pg.v3.GATE2_VOL_RATIO_MIN,
                "GATE3_MAXDD_MIN": pg.v3.GATE3_MAXDD_MIN,
            },
            "sweep_grid_size": len(SWEEP_GRID),
            "created_at": pd.Timestamp.now().isoformat(),
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"\n[*] Saved best config to {out_dir}")
        print(f"    classifier.json, calibrator.pkl, meta.json, "
              f"leaderboard.csv, leaderboard_by_auc.csv, "
              f"leaderboard_by_b_pnl.csv")
    else:
        print("\n[!] No configs passed the b_n >= 30 filter -- "
              "no 'best' artifact saved.")

    print("\n" + "=" * 78)
    print("PHASE G v1.1 SWEEP DONE")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

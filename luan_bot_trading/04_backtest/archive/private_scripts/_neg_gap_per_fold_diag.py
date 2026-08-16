"""Diagnostic: per-fold distribution of NEG_only picks by gap bucket."""
import importlib.util, json
import numpy as np, pandas as pd
from pathlib import Path
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

HERE = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location(
    "tm", HERE / "luan_bot_trading/03_model/01_train_model.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
pg_spec = importlib.util.spec_from_file_location(
    "pg", HERE / "luan_bot_trading" / "03_model" / "02_phase_g_sunday_classifier.py")
pg = importlib.util.module_from_spec(pg_spec); pg_spec.loader.exec_module(pg)
ps_spec = importlib.util.spec_from_file_location(
    "ps", HERE / "luan_bot_trading/04_backtest/04_phase_g_portfolio.py")
ps = importlib.util.module_from_spec(ps_spec); ps_spec.loader.exec_module(ps)

appd = pd.read_csv(HERE / "luan_bot_trading/04_backtest/archive/experiments/phase_g_v1_1_nested_cv_n4"
                   / "fold_results.csv")
fold_hp = []
for fold_idx, row in appd.iterrows():
    fold_hp.append({'gamma': int(row['sel_gamma']),
                    'min_child_weight': int(row['sel_mcw']),
                    'max_depth': int(row['sel_md']),
                    'n_estimators': int(row['sel_n_est'])})

df = tm.load_train_matrix()
df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
df = pg.v3.compute_pead_gates_full(df)
df = pg.compute_entry_pnl(df)
df = ps.compute_trade_paths(df)
import xgboost as xgb
all_picks = []
folds = [
    ('2023-12-31', '2024-06-30', '2024-12-31'),
    ('2024-06-30', '2024-12-31', '2025-06-30'),
    ('2024-12-31', '2025-06-30', '2025-12-31'),
    ('2025-06-30', '2025-12-31', '2026-06-30'),
]
for fold_idx, (te, sve, tse) in enumerate(folds, 1):
    train_ts = pd.Timestamp(te)
    sweep_ts = pd.Timestamp(sve)
    test_ts  = pd.Timestamp(tse)
    rd = pd.to_datetime(df['report_date'])
    tr = df[rd <= train_ts].copy()
    sw = df[(rd > train_ts) & (rd <= sweep_ts)].copy()
    ts_df = df[(rd > sweep_ts) & (rd <= test_ts)].copy().reset_index(drop=True)
    X_tr = tr[pg.SUNDAY_SAFE_FEATURES]; y_tr = tr['pead_pass'].astype(int).values
    X_sw = sw[pg.SUNDAY_SAFE_FEATURES]; y_sw = sw['pead_pass'].astype(int).values
    X_ts = pd.concat([X_tr, X_sw], axis=0).reset_index(drop=True)
    y_ts = np.concatenate([y_tr, y_sw])
    X_te = ts_df[pg.SUNDAY_SAFE_FEATURES]; y_te = ts_df['pead_pass'].astype(int).values
    hp = fold_hp[fold_idx-1]
    params = dict(objective='binary:logistic', eval_metric=['logloss','auc'],
                  n_estimators=hp['n_estimators'], learning_rate=0.05,
                  max_depth=hp['max_depth'], min_child_weight=hp['min_child_weight'],
                  gamma=hp['gamma'], reg_lambda=1.0, subsample=0.7,
                  colsample_bytree=0.7, random_state=42, n_jobs=-1)
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_ts, y_ts, eval_set=[(X_te, y_te)], verbose=False)
    proba = clf.predict_proba(X_te)[:, 1]
    ts_df2 = ts_df.copy(); ts_df2['pead_proba'] = proba
    mask = (ts_df2['pead_proba'] >= 0.20) & \
           (ts_df2['opening_gap_t1'] >= -0.15) & \
           (ts_df2['opening_gap_t1'] <= -0.02) & \
           (ts_df2['path_pnl_t11_pct'].notna())
    picks = ts_df2[mask].copy()
    picks['ann_fold'] = fold_idx
    picks['arith_pnl_pct'] = np.expm1(picks['path_pnl_t11_pct']) * 100
    all_picks.append(picks)

picks = pd.concat(all_picks, axis=0).reset_index(drop=True)
print("Per-fold distribution of NEG_only picks by gap bucket")
print("Cols: gap-bucket range; per fold: (n / mean_arith_pnl% / hit%)")
print()
print(f"{'bucket':>15} | {'F1 (2024H2)':>20} | {'F2 (2025H1)':>20} | "
      f"{'F3 (2025H2)':>20} | {'F4 (2026H1)':>20}")
print("-" * 110)
buckets = [(-0.20, -0.15), (-0.15, -0.10), (-0.10, -0.05),
           (-0.05, -0.03), (-0.03, -0.02)]
for lo, hi in buckets:
    cells = []
    for fold_idx in range(1, 5):
        m = (picks['ann_fold'] == fold_idx) & \
            (picks['opening_gap_t1'] > lo) & \
            (picks['opening_gap_t1'] <= hi)
        s = picks[m]
        if len(s):
            arith = s['arith_pnl_pct']
            hit = (arith > 0).mean() * 100
            cells.append(f"n={len(s):>2} / {arith.mean():+5.2f}% / hit={hit:.0f}%")
        else:
            cells.append(f"n=0")
    print(f"  ({lo:+.2f}, {hi:+.2f}]  | " + " | ".join(f"{c:>20s}" for c in cells))
print()
print("ROW TOTALS (per fold):")
for fold_idx in range(1, 5):
    n = int((picks['ann_fold'] == fold_idx).sum())
    arith = picks.loc[picks['ann_fold'] == fold_idx, 'arith_pnl_pct']
    print(f"  Fold {fold_idx}: n={n:>3} total arith_pnl = {arith.mean():+6.3f}%  hit = {(arith>0).mean()*100:.0f}%")

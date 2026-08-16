#!/usr/bin/env python3
"""V6 min-gate threshold sensitivity (DIAGNOSTIC; no policy change).

Question: does raising the min-gate threshold above 0.30 improve win rate and
PnL? Sweeps 0.30-0.35 on:
  (a) DEV OOS = nested folds 1-3 (2024 H2 + 2025 H1 + 2025 H2) -- the selection
      surface, ~109 trades at 0.30.
  (b) HOLDOUT = fold 4 = 2026 H1 -- the frozen untouched test, ~49 trades.
Also reports score-band calibration of executed OOS trades to test directly
whether border trades (0.30-0.33) underperform higher-score trades.

Methodology: the threshold was frozen at 0.30 after research. This scan is
diagnostic only. A threshold change would require nested selection (on DEV) +
bootstrap validation before any promotion; the HOLDOUT numbers are informational
(the holdout was intended to be touched once). No db.h5, model, or policy.json
is modified.
"""
from __future__ import annotations
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
os.chdir(Path(__file__).resolve().parents[2])
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
HERE = Path(__file__).resolve().parent
OUT = HERE / 'archive' / 'experiments' / 'gate_decomposition_v6'
POLICY = json.load(open(OUT / 'policy.json', encoding='utf-8'))
THRESHOLDS = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35]
BAND_EDGES = [0.30, 0.31, 0.32, 0.33, 0.35, 1.0]
BAND_LABELS = ['0.30-0.31', '0.31-0.32', '0.32-0.33', '0.33-0.35', '0.35+']


def load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

bt = load('bt', HERE / '51_hp_theta_sweep_23feat.py')
DB = bt.DB; MATRIX = '/features/train_matrix_v4_timing_correct'
FEATURES = bt.DEPLOY_FEATURES; N_SLOTS = bt.N_SLOTS
GATES = ['pass_g1', 'pass_g2', 'pass_g3']; XLF = POLICY['ensemble']['sector_exclusion']


def fit(X, y, Xe, ye, hp):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective='binary:logistic', eval_metric=['logloss', 'auc'],
        n_estimators=hp['n_estimators'], learning_rate=hp['learning_rate'],
        max_depth=hp['max_depth'], min_child_weight=hp['min_child_weight'],
        gamma=hp['gamma'], reg_lambda=hp['reg_lambda'], subsample=hp['subsample'],
        colsample_bytree=hp['colsample_bytree'], random_state=hp['random_state'],
        n_jobs=-1).fit(X, y, eval_set=[(Xe, ye)], verbose=False)


def predict_gates(fit_df, eval_df):
    out = eval_df.copy()
    for g in GATES:
        hp = POLICY['gate_models'][g]
        m = fit(fit_df[FEATURES], fit_df[g].astype(int),
                eval_df[FEATURES], eval_df[g].astype(int), hp)
        out['p_' + g] = m.predict_proba(eval_df[FEATURES])[:, 1]
    out['score'] = out[['p_pass_g1', 'p_pass_g2', 'p_pass_g3']].min(axis=1)
    return out


def select(raw):
    if raw.empty: return raw
    raw = raw.copy()
    raw['entry_date'] = pd.to_datetime(raw.pregap_entry_date)
    raw['exit_date'] = pd.to_datetime(raw.pregap_exit_date)
    return bt.select_weekly(raw, N_SLOTS)


def summary(ex, raw=None):
    r = np.asarray(ex.pregap_return if len(ex) else [], float)
    out = {'executed': int(len(r)),
           'win_rate_pct': float((r > 0).mean() * 100) if len(r) else 0.,
           'avg_trade_pct': float(r.mean() * 100) if len(r) else 0.,
           'median_trade_pct': float(np.median(r) * 100) if len(r) else 0.,
           'raw_picks': int(len(raw)) if raw is not None else 0}
    if len(ex):
        z = ex.copy(); iso = pd.to_datetime(z.entry_date).dt.isocalendar()
        z['week'] = iso.year.astype(str) + '-W' + iso.week.astype(str).str.zfill(2)
        nav = 1.0
        for _, w in z.groupby('week', sort=True):
            nav *= 1 + float((w.pregap_return / N_SLOTS).sum())
        out['nav_pct'] = float((nav - 1) * 100); out['weeks'] = int(z.week.nunique())
    else:
        out.update({'nav_pct': 0., 'weeks': 0})
    return out


def sweep(df, label):
    rows = []
    for t in THRESHOLDS:
        mask = (df.score >= t) & (~df.sector.isin(XLF)) & df.pregap_return.notna()
        raw = df[mask].copy()
        if not raw.empty: raw['p'] = raw.score
        ex = select(raw)
        s = summary(ex, raw=raw); s['threshold'] = t; rows.append(s)
    print(f"\n=== {label} threshold sweep ===")
    print(f"{'thresh':>7} {'trades':>7} {'win%':>6} {'avg%':>7} {'med%':>7} {'NAV%':>9} {'raw':>5}")
    for s in rows:
        print(f"{s['threshold']:>7.2f} {s['executed']:>7} {s['win_rate_pct']:>6.1f} "
              f"{s['avg_trade_pct']:>7.2f} {s['median_trade_pct']:>7.2f} "
              f"{s['nav_pct']:>9.1f} {s['raw_picks']:>5}")
    return rows


def calibration(df, label):
    mask = (df.score >= 0.30) & (~df.sector.isin(XLF)) & df.pregap_return.notna()
    raw = df[mask].copy()
    if raw.empty:
        print(f"\n=== {label} calibration: no executed trades ==="); return
    raw['p'] = raw.score; ex = select(raw)
    if ex.empty:
        print(f"\n=== {label} calibration: no executed trades ==="); return
    ex = ex.copy()
    ex['band'] = pd.cut(ex.score, bins=BAND_EDGES, labels=BAND_LABELS,
                        include_lowest=True, right=False)
    print(f"\n=== {label} executed-trade calibration by score band (thresh 0.30) ===")
    print(f"{'band':>11} {'n':>4} {'win%':>6} {'avg%':>7} {'med%':>7} {'sum%':>8}")
    for band in BAND_LABELS:
        b = ex[ex.band == band]; r = b.pregap_return
        if len(b):
            print(f"{band:>11} {len(b):>4} {(r > 0).mean() * 100:>6.1f} "
                  f"{r.mean() * 100:>7.2f} {r.median() * 100:>7.2f} {r.sum() * 100:>8.1f}")
        else:
            print(f"{band:>11} {0:>4}")


def main():
    print('=' * 100)
    print('V6 MIN-GATE THRESHOLD SENSITIVITY (diagnostic; no policy change)')
    print('=' * 100)
    print('Frozen threshold:', POLICY['ensemble']['threshold'], '| sweep:', THRESHOLDS)
    df = pd.read_hdf(DB, MATRIX); rd = pd.to_datetime(df.report_date)
    folds_pred = {}
    for i, (te, sw, tt) in enumerate(bt.DEFAULT_FOLDS, 1):
        fit_df = df[rd <= pd.Timestamp(sw)].copy()
        test = df[(rd > pd.Timestamp(sw)) & (rd <= pd.Timestamp(tt))].copy()
        if test.empty: continue
        folds_pred[i] = predict_gates(fit_df, test)
        print(f"fold {i}: fit={len(fit_df):,} test={len(test):,} ({sw} -> {tt})")
    dev = pd.concat([folds_pred[i] for i in (1, 2, 3) if i in folds_pred], ignore_index=True)
    holdout = folds_pred.get(4, pd.DataFrame())
    dev_rows = sweep(dev, 'DEV OOS folds 1-3 (selection surface)')
    calibration(dev, 'DEV OOS folds 1-3')
    ho_rows = sweep(holdout, 'HOLDOUT fold 4 = 2026 H1 (untouched)')
    calibration(holdout, 'HOLDOUT 2026 H1')
    result = {'model_version': 'phase_g_v6_gate_decomposition',
              'status': 'diagnostic_no_policy_change',
              'frozen_threshold': POLICY['ensemble']['threshold'],
              'thresholds': THRESHOLDS,
              'dev_oos_folds_1_3': dev_rows,
              'holdout_2026_h1': ho_rows}
    with open(OUT / 'threshold_sensitivity.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=float)
    print('\nSaved', OUT / 'threshold_sensitivity.json')
    print('No db.h5, production model, or policy.json modified. Diagnostic only.')


if __name__ == '__main__':
    main()

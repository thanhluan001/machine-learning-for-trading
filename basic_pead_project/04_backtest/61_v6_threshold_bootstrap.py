#!/usr/bin/env python3
"""V6 threshold bootstrap validation (DIAGNOSTIC; no policy change).

Validates raising the min-gate threshold from 0.30 to 0.33 / 0.35 on the DEV
OOS surface (nested folds 1-3, the selection surface). Two tests:

  1. Per-threshold trade + week-block bootstrap CIs (avg trade, win rate, NAV)
     at 0.30 (baseline), 0.33, 0.35.
  2. Border-band reliability: are the executed trades scored in [0.30, 0.33)
     and [0.30, 0.35) reliably negative? If their avg-return CI excludes zero,
     removing them is reliably beneficial.

Holdout (fold 4 = 2026 H1) is reported for reference but the bootstrap is
primarily on DEV (larger sample). No db.h5, model, or policy.json modified.
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
TEST_THRESH = [0.30, 0.33, 0.35]


def load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

bt = load('bt', HERE / '51_hp_theta_sweep_23feat.py')
DB = bt.DB; MATRIX = '/features/train_matrix_v4_timing_correct'
FEATURES = bt.DEPLOY_FEATURES; N_SLOTS = bt.N_SLOTS
GATES = ['pass_g1', 'pass_g2', 'pass_g3']; XLF = POLICY['ensemble']['sector_exclusion']
RNG = np.random.default_rng(20260813)


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


def executed_for(df, t):
    mask = (df.score >= t) & (~df.sector.isin(XLF)) & df.pregap_return.notna()
    raw = df[mask].copy()
    if not raw.empty: raw['p'] = raw.score
    ex = select(raw)
    if not ex.empty:
        iso = pd.to_datetime(ex.entry_date).dt.isocalendar()
        ex['week'] = iso.year.astype(str) + '-W' + iso.week.astype(str).str.zfill(2)
    return ex


def weekly_returns(ex):
    if ex.empty: return np.array([])
    return ex.groupby('week', sort=True).pregap_return.sum().div(N_SLOTS).to_numpy(float)


def boot_trade(r, n=10000):
    r = np.asarray(r, float)
    if len(r) < 2: return {'avg_ci95_pct': [0., 0.], 'win_ci95_pct': [0., 0.], 'n': int(len(r))}
    idx = RNG.integers(0, len(r), size=(n, len(r)))
    m = r[idx].mean(axis=1); w = (r[idx] > 0).mean(axis=1)
    return {'avg_ci95_pct': [float(np.percentile(m, 2.5) * 100), float(np.percentile(m, 97.5) * 100)],
            'win_ci95_pct': [float(np.percentile(w, 2.5) * 100), float(np.percentile(w, 97.5) * 100)],
            'n': int(len(r))}


def boot_week(weekly, n=10000, block=4):
    a = np.asarray(weekly, float); L = len(a)
    if L < 2: return {'ci95_nav_pct': [0., 0.], 'weeks': int(L)}
    blocks = [a[i:min(i + block, L)] for i in range(L)]
    vals = []
    for _ in range(n):
        q = []
        while len(q) < L: q.extend(blocks[int(RNG.integers(0, len(blocks)))])
        vals.append(np.prod(1 + np.asarray(q[:L])) - 1)
    return {'ci95_nav_pct': [float(np.percentile(vals, 2.5) * 100), float(np.percentile(vals, 97.5) * 100)],
            'weeks': int(L)}


def main():
    print('=' * 100)
    print('V6 THRESHOLD BOOTSTRAP VALIDATION (diagnostic; no policy change)')
    print('=' * 100)
    df = pd.read_hdf(DB, MATRIX); rd = pd.to_datetime(df.report_date)
    folds_pred = {}
    for i, (te, sw, tt) in enumerate(bt.DEFAULT_FOLDS, 1):
        fit_df = df[rd <= pd.Timestamp(sw)].copy()
        test = df[(rd > pd.Timestamp(sw)) & (rd <= pd.Timestamp(tt))].copy()
        if test.empty: continue
        folds_pred[i] = predict_gates(fit_df, test)
        print(f"fold {i}: fit={len(fit_df):,} test={len(test):,}")
    dev = pd.concat([folds_pred[i] for i in (1, 2, 3) if i in folds_pred], ignore_index=True)
    holdout = folds_pred.get(4, pd.DataFrame())

    def run_surface(frame, label):
        print(f"\n{'=' * 100}\n{label}\n{'=' * 100}")
        ex30 = executed_for(frame, 0.30)
        per_thresh = {}
        print(f"\n--- Per-threshold bootstrap ---")
        print(f"{'thresh':>7} {'n':>4} {'avg%':>6} {'avgCI95%':>16} {'win%':>5} {'NAV%':>7} {'NAVCI95%':>16}")
        for t in TEST_THRESH:
            ex = executed_for(frame, t)
            r = ex.pregap_return.to_numpy(float) if len(ex) else np.array([])
            bt_ci = boot_trade(r)
            bw = boot_week(weekly_returns(ex))
            avg = float(r.mean() * 100) if len(r) else 0.
            win = float((r > 0).mean() * 100) if len(r) else 0.
            nav = bw['ci95_nav_pct']
            nav_mid = float((weekly_returns(ex).sum()) ) if len(ex) else 0.  # raw sum, not compounded
            per_thresh[t] = {'executed': int(len(r)), 'avg_trade_pct': avg,
                             'avg_ci95': bt_ci['avg_ci95_pct'],
                             'win_rate_pct': win, 'win_ci95': bt_ci['win_ci95_pct'],
                             'nav_ci95': nav, 'weeks': bw['weeks']}
            print(f"{t:>7.2f} {len(r):>4} {avg:>6.2f} "
                  f"[{bt_ci['avg_ci95_pct'][0]:>5.2f},{bt_ci['avg_ci95_pct'][1]:>5.2f}] "
                  f"{win:>5.1f} {nav_mid:>7.1f} [{nav[0]:>6.1f},{nav[1]:>6.1f}]")

        # Border-band reliability (executed at 0.30, scored below the candidate)
        print(f"\n--- Border-band reliability (executed @0.30, then banded) ---")
        print(f"{'band':>12} {'n':>4} {'avg%':>7} {'avgCI95%':>16} {'reliably_neg?':>14}")
        border = {}
        for lo, hi, name in [(0.30, 0.33, '0.30-0.33'), (0.30, 0.35, '0.30-0.35'),
                             (0.33, 0.35, '0.33-0.35'), (0.35, 1.0, '0.35+')]:
            if ex30.empty: break
            b = ex30[(ex30.score >= lo) & (ex30.score < hi)]
            r = b.pregap_return.to_numpy(float) if len(b) else np.array([])
            ci = boot_trade(r)['avg_ci95_pct']
            avg = float(r.mean() * 100) if len(r) else 0.
            relneg = 'YES' if (len(r) >= 2 and ci[1] < 0) else ('no' if len(r) >= 2 else 'n/a')
            border[name] = {'n': int(len(r)), 'avg_pct': avg, 'ci95': ci, 'reliably_negative': relneg}
            print(f"{name:>12} {len(r):>4} {avg:>7.2f} [{ci[0]:>6.2f},{ci[1]:>6.2f}] {relneg:>14}")
        return {'per_threshold': per_thresh, 'border_bands': border}

    dev_res = run_surface(dev, 'DEV OOS folds 1-3 (selection surface)')
    ho_res = run_surface(holdout, 'HOLDOUT fold 4 = 2026 H1 (reference, small sample)')

    result = {'model_version': 'phase_g_v6_gate_decomposition',
              'status': 'diagnostic_no_policy_change',
              'seed': 20260813, 'test_thresholds': TEST_THRESH,
              'dev_oos_folds_1_3': dev_res, 'holdout_2026_h1': ho_res}
    with open(OUT / 'threshold_bootstrap.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=float)
    print('\nSaved', OUT / 'threshold_bootstrap.json')
    print('No db.h5, production model, or policy.json modified. Diagnostic only.')


if __name__ == '__main__':
    main()

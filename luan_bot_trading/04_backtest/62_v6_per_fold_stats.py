#!/usr/bin/env python3
"""V6 per-fold execution statistics at threshold 0.33 (current) and 0.30 (prior).

For each nested fold (and dev-aggregate), reports: executed trades, win rate,
avg win, avg loss, payoff ratio, profit factor, max NAV drawdown, compounded
NAV. Drawdown is the peak-to-trough drop of the weekly-compounded NAV curve
(4 slots, each trade 1/4 NAV). Diagnostic only; no db.h5/model/policy change.
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


def executed_for(df, t):
    mask = (df.score >= t) & (~df.sector.isin(XLF)) & df.pregap_return.notna()
    raw = df[mask].copy()
    if not raw.empty: raw['p'] = raw.score
    ex = select(raw)
    if not ex.empty:
        iso = pd.to_datetime(ex.entry_date).dt.isocalendar()
        ex['week'] = iso.year.astype(str) + '-W' + iso.week.astype(str).str.zfill(2)
    return ex


def stats(ex):
    if ex.empty:
        return dict(trades=0, win_rate_pct=0., avg_win_pct=0., avg_loss_pct=0.,
                    payoff=0., profit_factor=0., max_dd_pct=0., nav_pct=0., weeks=0)
    r = ex.pregap_return.to_numpy(float)
    wins = r[r > 0]; losses = r[r <= 0]
    weekly = ex.groupby('week', sort=True).pregap_return.sum().div(N_SLOTS)
    nav = (1 + weekly).cumprod()
    dd_series = (nav - nav.cummax()) / nav.cummax()
    aw = float(wins.mean() * 100) if len(wins) else 0.
    al = float(losses.mean() * 100) if len(losses) else 0.
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else (float('inf') if len(wins) else 0.)
    po = float(wins.mean() / abs(losses.mean())) if len(losses) and losses.mean() != 0 else (float('inf') if len(wins) else 0.)
    return dict(trades=int(len(r)), win_rate_pct=float((r > 0).mean() * 100),
                avg_win_pct=aw, avg_loss_pct=al, payoff=po, profit_factor=pf,
                max_dd_pct=float(dd_series.min() * 100),
                nav_pct=float((nav.iloc[-1] - 1) * 100), weeks=int(ex.week.nunique()))


def fmt(s):
    pf = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else "inf"
    po = f"{s['payoff']:.2f}" if s['payoff'] != float('inf') else "inf"
    return (f"{s['trades']:>4} {s['win_rate_pct']:>6.1f} {s['avg_win_pct']:>8.2f} "
            f"{s['avg_loss_pct']:>8.2f} {po:>6} {pf:>7} {s['max_dd_pct']:>8.1f} {s['nav_pct']:>8.1f}")


def main():
    print('=' * 108)
    print('V6 PER-FOLD EXECUTION STATISTICS')
    print('=' * 108)
    df = pd.read_hdf(DB, MATRIX); rd = pd.to_datetime(df.report_date)
    folds = {}
    labels = {1: '2024 H2', 2: '2025 H1', 3: '2025 H2', 4: '2026 H1 (holdout)'}
    for i, (te, sw, tt) in enumerate(bt.DEFAULT_FOLDS, 1):
        fit_df = df[rd <= pd.Timestamp(sw)].copy()
        test = df[(rd > pd.Timestamp(sw)) & (rd <= pd.Timestamp(tt))].copy()
        if test.empty: continue
        folds[i] = predict_gates(fit_df, test)
        print(f"fold {i}: fit={len(fit_df):,} test={len(test):,} ({labels[i]})")
    dev = pd.concat([folds[i] for i in (1, 2, 3) if i in folds], ignore_index=True)

    for t in (0.33, 0.30):
        tag = 'CURRENT' if t == 0.33 else 'PRIOR'
        print(f"\n=== threshold {t} ({tag}) ===")
        print(f"{'fold':<22} {'trades':>5} {'win%':>6} {'avgWin%':>8} {'avgLoss%':>9} "
              f"{'payoff':>6} {'profFac':>7} {'maxDD%':>8} {'NAV%':>8}")
        for i in (1, 2, 3, 4):
            if i not in folds: continue
            print(f"fold {i} {labels[i]:<14} " + fmt(stats(executed_for(folds[i], t))))
        print(f"{'DEV folds 1-3':<22} " + fmt(stats(executed_for(dev, t))))
    # Save
    result = {f"thresh_{t}": {f"fold_{i}": stats(executed_for(folds[i], t))
                              for i in folds} | {"dev_1_3": stats(executed_for(dev, t))}
              for t in (0.33, 0.30)}
    with open(OUT / 'per_fold_stats.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=float)
    print('\nSaved', OUT / 'per_fold_stats.json')


if __name__ == '__main__':
    main()

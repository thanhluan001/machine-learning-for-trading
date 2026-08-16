#!/usr/bin/env python3
"""Force-refresh min-hold guard sweep + bootstrap (DIAGNOSTIC).

Sweeps min-hold guards (force-sell only positions held >= G trading days) for
G in {0,3,4}, vs conviction(skip). Bootstraps trade-avg CI, win CI, and
week-block NAV CI for each variant on DEV (folds 1-3) and the holdout.
Reuses the validated slot simulator from 63. No db.h5/model/policy/code change.
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


def load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

fr = load('fr', HERE / '63_force_refresh_backtest.py')
bt = fr.bt; DB = fr.DB; MATRIX = fr.MATRIX; N_SLOTS = fr.N_SLOTS
SEED = 20260814; GUARDS = [0, 3, 4]


def boot_trade(r, n=10000):
    r = np.asarray(r, float)
    if len(r) < 2: return {'avg_ci': [0., 0.], 'win_ci': [0., 0.]}
    rng = np.random.default_rng(SEED); idx = rng.integers(0, len(r), size=(n, len(r)))
    m = r[idx].mean(axis=1); w = (r[idx] > 0).mean(axis=1)
    return {'avg_ci': [float(np.percentile(m, 2.5) * 100), float(np.percentile(m, 97.5) * 100)],
            'win_ci': [float(np.percentile(w, 2.5) * 100), float(np.percentile(w, 97.5) * 100)]}


def boot_nav(weekly, n=10000, block=4):
    a = np.asarray(weekly, float); L = len(a)
    if L < 2: return [0., 0.]
    rng = np.random.default_rng(SEED); blocks = [a[i:min(i + block, L)] for i in range(L)]; vals = []
    for _ in range(n):
        q = []
        while len(q) < L: q.extend(blocks[int(rng.integers(0, len(blocks)))])
        vals.append(np.prod(1 + np.asarray(q[:L])) - 1)
    return [float(np.percentile(vals, 2.5) * 100), float(np.percentile(vals, 97.5) * 100)]


def rets_of(trades):
    if not trades: return np.array([]), np.array([])
    d = pd.DataFrame(trades); d['wk'] = d.entry_date.map(fr.iso_week)
    r = d['return'].to_numpy(float)
    wk = d.groupby('wk', sort=True)['return'].sum().div(N_SLOTS).to_numpy(float)
    return r, wk


def main():
    print('=' * 116); print('FORCE-REFRESH GUARD SWEEP + BOOTSTRAP'); print('=' * 116)
    df = pd.read_hdf(DB, MATRIX); rd = pd.to_datetime(df.report_date)
    folds = {}; labels = {1: '2024 H2', 2: '2025 H1', 3: '2025 H2', 4: '2026 H1 (holdout)'}
    with pd.HDFStore(DB, mode='r') as store:
        keys = set(store.keys())
        for i, (te, sw, tt) in enumerate(bt.DEFAULT_FOLDS, 1):
            fit_df = df[rd <= pd.Timestamp(sw)].copy()
            test = df[(rd > pd.Timestamp(sw)) & (rd <= pd.Timestamp(tt))].copy()
            if test.empty: continue
            folds[i] = fr.predict_gates(fit_df, test)
        dev = pd.concat([folds[i] for i in (1, 2, 3) if i in folds], ignore_index=True)

        def run(name, pred):
            cands = fr.build_candidates(pred, store, keys); slate = fr.weekly_slate(cands)
            variants = {'conviction(skip)': fr.simulate(slate, 'skip')}
            for g in GUARDS:
                variants[f'force_refresh(mh={g})'] = fr.simulate(slate, 'force_refresh', min_hold=g)
            print(f"\n=== {name} (slate={len(slate)}) ===")
            print(f"{'variant':<24}{'trades':>7}{'win%':>7}{'NAV%':>9}{'maxDD%':>8}{'frcSold':>8}{'avgHold':>8}")
            for v, tr in variants.items():
                s = fr.stats(tr)
                print(f"{v:<24}{s['trades']:>7}{s['win_rate_pct']:>7.1f}{s['nav_pct']:>9.1f}"
                      f"{s['max_dd_pct']:>8.1f}{s['force_sold']:>8}{s['avg_force_hold']:>8.1f}")
            print(f"  -- bootstrap: avgCI / winCI / week-block-NAVCI --")
            for v in ['conviction(skip)', 'force_refresh(mh=0)', 'force_refresh(mh=3)', 'force_refresh(mh=4)']:
                r, wk = rets_of(variants[v]); tc = boot_trade(r); nc = boot_nav(wk)
                print(f"  {v:<24} avg[{tc['avg_ci'][0]:>5.2f},{tc['avg_ci'][1]:>5.2f}] "
                      f"win[{tc['win_ci'][0]:>4.1f},{tc['win_ci'][1]:>4.1f}] nav[{nc[0]:>6.1f},{nc[1]:>6.1f}]")
            return {v: fr.stats(tr) for v, tr in variants.items()}

        res = {}
        res['dev_1_3'] = run('DEV folds 1-3', dev)
        if 4 in folds: res['holdout_4'] = run('HOLDOUT 2026 H1', folds[4])
    with open(OUT / 'force_refresh_guard_bootstrap.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, default=float)
    print('\nSaved', OUT / 'force_refresh_guard_bootstrap.json')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""SHAP attribution for a V6 pick's three gate probabilities (diagnostic).

Uses XGBoost's native TreeSHAP (booster.predict(pred_contribs=True)) — no `shap`
dependency. Reads a pick's feature vector from plan.json, loads the frozen V6
gate classifiers, and reports each feature's log-odds contribution to every
gate. Use it to sanity-check WHY a pick passes (or narrowly fails) a gate, and
to see which features are for vs. against it.

Usage:
    python 65_shap_pick.py             # analyze the top V6 pick in plan.json
    python 65_shap_pick.py BILL        # analyze a specific ticker (must be a pick)
    python 65_shap_pick.py --all       # analyze every V6 pick

Diagnostic only: reads frozen models + plan.json, writes nothing. Gate
classifiers, threshold, and db.h5 are untouched.
"""
from __future__ import annotations
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
os.chdir(Path(__file__).resolve().parents[2])
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / '05b_alpaca_live' / 'plan.json'
MODEL_DIR = HERE.parent / '03_model' / 'models' / 'phase_g_v6_gate_decomposition'
GATES = ['pass_g1', 'pass_g2', 'pass_g3']
TOPN = 8


def _load_bt():
    s = importlib.util.spec_from_file_location('bt', HERE / '51_hp_theta_sweep_23feat.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


bt = _load_bt()
FEATURES = bt.DEPLOY_FEATURES
_CLFS: dict = {}


def get_clf(gate):
    if gate not in _CLFS:
        c = xgb.XGBClassifier(); c.load_model(str(MODEL_DIR / gate / 'classifier.json'))
        _CLFS[gate] = c
    return _CLFS[gate]


def shap_pick(pick):
    fv = pick['features']
    X = pd.DataFrame([{f: fv.get(f) for f in FEATURES}]).astype(float)
    dmat = xgb.DMatrix(X)
    binding = min(GATES, key=lambda g: pick[g])
    t = pick['canonical_ticker']
    print('=' * 94)
    print(f"{t}  {pick['report_date']} {pick['time'].upper()}  "
          f"min-gate={pick['p_v6_min']:.3f}  "
          f"g1={pick['pass_g1']:.3f} g2={pick['pass_g2']:.3f} g3={pick['pass_g3']:.3f}")
    print('=' * 94)
    for gate in GATES:
        clf = get_clf(gate)
        p = float(clf.predict_proba(X)[0, 1])
        contribs = clf.get_booster().predict(dmat, pred_contribs=True)[0]
        bias = contribs[-1]; sh = contribs[:-1]
        tag = '   <-- BINDING (lowest gate)' if gate == binding else ''
        print(f"\n--- {gate}  p={p:.3f}  (logit={sh.sum() + bias:.3f}, base={bias:.3f}){tag} ---")
        order = np.argsort(-np.abs(sh))
        print(f"  {'feature':34s} {'value':>10}  {'SHAP(logodds)':>13}")
        for i in order[:TOPN]:
            v = X.iloc[0, i]
            vs = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {FEATURES[i]:34s} {vs:>10}  {sh[i]:>+13.3f}")
    print()


def main():
    plan = json.load(open(PLAN, encoding='utf-8'))
    picks = plan.get('picks', [])
    if not picks:
        print('No picks in plan.json'); return
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == '--all':
        for p in picks:
            shap_pick(p)
    elif arg:
        m = [p for p in picks if p['canonical_ticker'] == arg.upper()]
        if not m:
            print(f'{arg} not in plan.json picks: '
                  f"{[p['canonical_ticker'] for p in picks]}"); return
        shap_pick(m[0])
    else:
        shap_pick(picks[0])  # top pick by default


if __name__ == '__main__':
    main()

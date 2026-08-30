"""92_rc1_phase3_ab.py — RC-1 Phase 3: nested walk-forward A/B vs frozen V6.

Pre-registration: 04_backtest/rc1_pre_registration.md (2026-08-30).

Arm A: frozen V6 (23 features, policy.json HPs verbatim, theta=0.33).
Arm B: 27 features (23 + 4 insider), gates RETRAINED with the SAME HPs
       (no HP fishing — pre-registered single config), theta=0.33.

Same folds, same weekly slate/slots/stops/force-refresh (mh=4) simulator.
Per-trade PAIRED bootstrap (10k, seed 7) on common trades; 2026 H1
holdout (fold 4) evaluated LAST.

PROMOTION BAR (pre-registered): DEV pooled paired-diff CI excludes 0 in
Arm B's favor AND no fold directionally negative AND holdout not negative.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


fr = load('fr', HERE / '63_force_refresh_backtest.py')
bt = fr.bt

POLICY = fr.POLICY
THRESH = fr.THRESH
STOP = fr.STOP
N_SLOTS = fr.N_SLOTS
DB = fr.DB
GATES = ('pass_g1', 'pass_g2', 'pass_g3')
FEAT_A = list(bt.DEPLOY_FEATURES)
INSIDER = ["insider_net_buy_90d", "insider_cluster_90d",
           "insider_sell_pressure_30d", "insider_days_since_last_buy"]
FEAT_B = FEAT_A + INSIDER

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def predict_gates(fit_df, eval_df, feats):
    out = eval_df.copy()
    for g in GATES:
        hp = POLICY['gate_models'][g]
        m = fr.fit(fit_df[feats], fit_df[g].astype(int),
                   eval_df[feats], eval_df[g].astype(int), hp)
        out['p_' + g] = m.predict_proba(eval_df[feats])[:, 1]
    out['score'] = out[['p_pass_g1', 'p_pass_g2', 'p_pass_g3']].min(axis=1)
    return out


def build_cands(frame, store, keys):
    mask = (frame.score >= THRESH) & (~frame.sector.isin(fr.XLF)) & frame.pregap_return.notna()
    c = frame[mask].copy()
    c['entry_date'] = pd.to_datetime(c.pregap_entry_date)
    rows = []
    for _, r in c.iterrows():
        pl = fr.get_prices(store, keys, r.permaTicker)
        if pl is None:
            continue
        dates, closes = pl
        eidx = int(np.searchsorted(dates, np.datetime64(r.entry_date), side='left'))
        x = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(r.pregap_exit_date)), side='left'))
        if eidx >= len(closes) or x >= len(closes) or eidx < 0 or x <= eidx:
            continue
        rows.append({**r.to_dict(), 'entry_idx': eidx, 'exit_idx': x,
                     'exit_date': pd.Timestamp(dates[x]),
                     'ret5': fr.ret_with_stop(closes, eidx, x, stop=STOP)})
    out = pd.DataFrame(rows)
    return out[out.ret5.notna()].sort_values(
        ['entry_date', 'score'], ascending=[True, False]).reset_index(drop=True)


def simulate(slate):
    slots, trades = [], []
    for ev in slate.itertuples(index=False):
        ed = ev.entry_date
        evd = ev._asdict()
        kept = []
        for s in slots:
            if s['exit_date'] <= ed:
                trades.append({**s, 'return': s['ret5'], 'exit_reason': 'natural'})
            else:
                kept.append(s)
        slots = kept
        if len(slots) < N_SLOTS:
            slots.append(evd)
        else:
            scored = []
            for s in slots:
                if fr.iso_week(s['entry_date']) >= fr.iso_week(ed):
                    continue
                pl = fr._PCACHE.get(s['permaTicker'])
                if pl is None:
                    continue
                vdates, vcloses = pl
                sidx = int(np.searchsorted(vdates, np.datetime64(ed), side='right')) - 1
                if sidx - s['entry_idx'] < 4:
                    continue
                scored.append((s, sidx, vcloses))
            if scored:
                victim, sidx, vcloses = min(scored, key=lambda x: x[0]['entry_date'])
                part = fr.ret_with_stop(vcloses, victim['entry_idx'], sidx, stop=STOP)
                trades.append({**victim, 'return': part if part is not None else victim['ret5'],
                               'exit_reason': 'force_refresh'})
                slots.remove(victim)
                slots.append(evd)
    for s in slots:
        trades.append({**s, 'return': s['ret5'], 'exit_reason': 'end'})
    return trades


def nav(trades):
    df = pd.DataFrame(trades)
    if df.empty:
        return float('nan')
    return float(np.prod(1 + df.sort_values('entry_date')['return'].astype(float) / N_SLOTS))


def paired(a, b, label, results):
    A = pd.DataFrame(a); B = pd.DataFrame(b)
    m = A.merge(B, on=['permaTicker', 'entry_date'], suffixes=('_A', '_B'))
    d = (m['return_B'] - m['return_A']).astype(float)
    rng = np.random.default_rng(7)
    boots = [float(rng.choice(d.values, len(d), replace=True).mean()) for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    excl0 = (lo > 0) or (hi < 0)
    rA, wA = m['return_A'].mean() * 100, (m['return_A'] > 0).mean()
    rB, wB = m['return_B'].mean() * 100, (m['return_B'] > 0).mean()
    nA_only, nB_only = len(A) - len(m), len(B) - len(m)
    print(f"\n[{label}] common n={len(m)}  (A-only {nA_only}, B-only {nB_only})")
    print(f"  A (23f): mean {rA:+.3f}%  win {wA:.0%}  NAV {nav(a):.3f}x  (all {len(A)})")
    print(f"  B (27f): mean {rB:+.3f}%  win {wB:.0%}  NAV {nav(b):.3f}x  (all {len(B)})")
    print(f"  paired B−A: {d.mean()*100:+.3f}%  median {d.median()*100:+.3f}%  "
          f"win {(d>0).mean():.0%}  CI95 [{lo*100:+.3f}, {hi*100:+.3f}]  "
          f"{'EXCLUDES 0' if excl0 else 'includes 0'}")
    results[label] = {'n_common': len(m), 'mean_diff': float(d.mean()),
                      'ci': [float(lo), float(hi)], 'navA': nav(a), 'navB': nav(b),
                      'nA': len(A), 'nB': len(B)}
    return d.mean(), excl0


def main() -> None:
    print("=" * 100)
    print("RC-1 PHASE 3 — A/B: frozen V6 (23f) vs insider-augmented (27f), same HPs, theta=0.33")
    print("=" * 100)

    mx = pd.read_hdf(DB, fr.MATRIX)
    rc1 = pd.read_hdf(DB, '/features/rc1_insider_features')
    rc1['report_date'] = pd.to_datetime(rc1['report_date'])
    mx['report_date'] = pd.to_datetime(mx.report_date)
    d = mx.merge(rc1, on=['permaTicker', 'report_date'], how='left')
    print(f"matrix {len(d):,} rows | insider cols present: {all(c in d.columns for c in INSIDER)}")
    rd = d.report_date

    results = {}
    dev = {'A': [], 'B': []}
    fold_diffs = {}
    with pd.HDFStore(DB, mode='r') as store:
        keys = set(store.keys())
        foldsA, foldsB = {}, {}
        for i, (te, sw, tt) in enumerate(bt.DEFAULT_FOLDS, 1):
            fit_df = d[rd <= pd.Timestamp(sw)]
            test = d[(rd > pd.Timestamp(sw)) & (rd <= pd.Timestamp(tt))]
            if test.empty:
                continue
            foldsA[i] = predict_gates(fit_df, test, FEAT_A)
            foldsB[i] = predict_gates(fit_df, test, FEAT_B)
            print(f"fold {i}: fit={len(fit_df):,} test={len(test):,} trained both arms")

        def run(label, predA, predB, holdout=False):
            cA = build_cands(predA, store, keys); cB = build_cands(predB, store, keys)
            sA = fr.weekly_slate(cA); sB = fr.weekly_slate(cB)
            tA = simulate(sA); tB = simulate(sB)
            mean_diff, excl0 = paired(tA, tB, label, results)
            results[label]['directionally_negative'] = bool(mean_diff < 0)
            if not holdout:
                dev['A'].extend(tA); dev['B'].extend(tB)
                fold_diffs[label] = mean_diff
            return mean_diff

        for i in (1, 2, 3):
            run(f"fold {i}", foldsA[i], foldsB[i])
        run("DEV 1-3", pd.concat([foldsA[i] for i in (1, 2, 3)]),
            pd.concat([foldsB[i] for i in (1, 2, 3)]))
        # holdout LAST
        run("fold 4 holdout", foldsA[4], foldsB[4], holdout=True)

    dev_diff, dev_excl = results['DEV 1-3']['mean_diff'], (
        results['DEV 1-3']['ci'][0] > 0 or results['DEV 1-3']['ci'][1] < 0)
    no_neg_fold = not any(results[f'fold {i}']['directionally_negative'] for i in (1, 2, 3))
    ho_ok = results['fold 4 holdout']['mean_diff'] >= 0
    promote = dev_excl and (results['DEV 1-3']['ci'][0] > 0) and no_neg_fold and ho_ok
    print("\n" + "=" * 100)
    print("PROMOTION BAR:")
    print(f"  DEV CI excludes 0 in B's favor : {'PASS' if dev_excl and results['DEV 1-3']['ci'][0] > 0 else 'FAIL'}")
    print(f"  no fold directionally negative : {'PASS' if no_neg_fold else 'FAIL'}")
    print(f"  holdout non-negative           : {'PASS' if ho_ok else 'FAIL'}")
    print(f"  ==> {'PROMOTE Arm B' if promote else 'REJECT — insider features do not clear the bar'}")
    print("=" * 100)

    out = HERE / 'archive' / 'experiments' / 'gate_decomposition_v6' / 'rc1_phase3_ab.json'
    out.write_text(json.dumps(results, indent=2, default=float), encoding='utf-8')
    print('saved', out)


if __name__ == '__main__':
    main()

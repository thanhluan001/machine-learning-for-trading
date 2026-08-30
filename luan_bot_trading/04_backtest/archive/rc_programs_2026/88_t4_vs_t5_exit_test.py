"""88_t4_vs_t5_exit_test.py — T+4 vs T+5 exit head-to-head on gated picks.

User observation (paper trading): profits concentrate early; T+4/T+5 feel
like losses. The ungated day-slice scan found d4->d5 = -0.06..-0.09%
(t~-2). This script is the decisive gated test: identical V6 gates,
slate, slots, force-refresh policy and stops; ONLY the natural exit moves
one trading day earlier (T+5 close -> T+4 close).

Per the freeze contract, a promotion requires the paired difference to
be positive and reliably non-zero on DEV folds (bootstrap CI excluding
zero), and not negative on the 2026 H1 holdout.

Reuses script 63's machinery (folds, gates, simulator skeleton).
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

def load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

# NOTE: load 63 only; it loads 51 internally as `fr.bt`. Loading 51 twice
# would double-wrap sys.stdout (51 replaces it with a new TextIOWrapper) and
# the GC'd first wrapper closes the shared buffer -> 'I/O operation on closed file'.
fr = load('fr', HERE / '63_force_refresh_backtest.py')
bt = fr.bt

THRESH = fr.THRESH
STOP = fr.STOP
N_SLOTS = fr.N_SLOTS
DB = fr.DB

_PCACHE = fr._PCACHE


def build_candidates2(frame, store, keys):
    """Threshold passers with BOTH T+5 and T+4 exits/returns from prices."""
    mask = (frame.score >= THRESH) & (~frame.sector.isin(fr.XLF)) & frame.pregap_return.notna()
    cands = frame[mask].copy()
    cands['entry_date'] = pd.to_datetime(cands.pregap_entry_date)
    rows = []
    for _, r in cands.iterrows():
        pt = r.permaTicker
        pl = fr.get_prices(store, keys, pt)
        if pl is None:
            continue
        dates, closes = pl
        eidx = int(np.searchsorted(dates, np.datetime64(r.entry_date), side='left'))
        x5 = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(r.pregap_exit_date)), side='left'))
        if eidx >= len(closes) or x5 >= len(closes) or eidx < 0 or x5 - 1 <= eidx:
            continue
        x4 = x5 - 1  # one trading day earlier (price arrays are trading days)
        ret5 = fr.ret_with_stop(closes, eidx, x5, stop=STOP)
        ret4 = fr.ret_with_stop(closes, eidx, x4, stop=STOP)
        if ret5 is None or ret4 is None:
            continue
        rows.append({**r.to_dict(), 'entry_idx': eidx, 'x4': x4, 'x5': x5,
                     'ret5': ret5, 'ret4': ret4,
                     'exit_date5': pd.Timestamp(dates[x5]),
                     'exit_date4': pd.Timestamp(dates[x4])})
    return pd.DataFrame(rows).sort_values(['entry_date', 'score'], ascending=[True, False]).reset_index(drop=True)


def simulate2(slate, t4: bool):
    """Script-63 simulator, exit horizon parameterized (min_hold=4 like ops)."""
    exit_col = 'exit_date4' if t4 else 'exit_date5'
    ret_col = 'ret4' if t4 else 'ret5'
    slots, trades = [], []
    for ev in slate.itertuples(index=False):
        ed = ev.entry_date
        evd = ev._asdict()
        kept = []
        for s in slots:
            if s[exit_col] <= ed:
                trades.append({**s, 'return': s[ret_col], 'exit_reason': 'natural'})
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
                pl = _PCACHE.get(s['permaTicker'])
                if pl is None:
                    continue
                vdates, vcloses = pl
                sidx = int(np.searchsorted(vdates, np.datetime64(ed), side='right')) - 1
                hold = sidx - s['entry_idx']
                if hold < 4:  # mh=4 operational guard
                    continue
                scored.append((s, sidx, vcloses))
            if scored:
                victim, sidx, vcloses = min(scored, key=lambda x: x[0]['entry_date'])
                part5 = fr.ret_with_stop(vcloses, victim['entry_idx'], sidx, stop=STOP)
                trades.append({**victim, 'return': part5 if part5 is not None else victim['ret5'],
                               'exit_reason': 'force_refresh'})
                slots.remove(victim)
                slots.append(evd)
    for s in slots:
        trades.append({**s, 'return': s[ret_col], 'exit_reason': 'end'})
    return trades


def nav_from(trades, n_slots=N_SLOTS):
    df = pd.DataFrame(trades)
    if df.empty:
        return float('nan')
    df = df.sort_values('entry_date')
    return float(np.prod(1 + df['return'].astype(float) / n_slots))


def paired_compare(t5, t4, label):
    a = pd.DataFrame(t5); b = pd.DataFrame(t4)
    k = ['permaTicker', 'entry_date']
    m = a.merge(b, on=k, suffixes=('_5', '_4'))
    d = (m['return_4'] - m['return_5']).astype(float)
    rng = np.random.default_rng(7)
    boots = [float(rng.choice(d.values, len(d), replace=True).mean()) for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n[{label}] common trades n={len(m)}")
    print(f"  T+5: mean {m['return_5'].mean()*100:+.3f}%  win {(m['return_5']>0).mean():.0%}  NAV {nav_from(t5):.3f}x (all trades n={len(a)})")
    print(f"  T+4: mean {m['return_4'].mean()*100:+.3f}%  win {(m['return_4']>0).mean():.0%}  NAV {nav_from(t4):.3f}x (all trades n={len(b)})")
    print(f"  paired diff (T4 - T5): mean {d.mean()*100:+.3f}%  median {d.median()*100:+.3f}%  "
          f"win {(d>0).mean():.0%}  bootstrap95 [{lo*100:+.3f}%, {hi*100:+.3f}%]  "
          f"{'EXCLUDES 0' if (lo>0 or hi<0) else 'includes 0'}")
    return {'n': len(m), 'mean_diff': float(d.mean()), 'ci': [float(lo), float(hi)],
            'nav5': nav_from(t5), 'nav4': nav_from(t4)}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, RuntimeError):
        pass
    print('=' * 100)
    print(f'T+4 vs T+5 EXIT TEST — gated V6 picks, same folds/gates/stops/force-refresh (mh=4), THRESH={THRESH}')
    print('=' * 100)
    df = pd.read_hdf(DB, fr.MATRIX); rd = pd.to_datetime(df.report_date)
    folds = {}
    labels = {1: '2024 H2', 2: '2025 H1', 3: '2025 H2', 4: '2026 H1 (holdout)'}
    with pd.HDFStore(DB, mode='r') as store:
        keys = set(store.keys())
        for i, (te, sw, tt) in enumerate(bt.DEFAULT_FOLDS, 1):
            fit_df = df[rd <= pd.Timestamp(sw)].copy()
            test = df[(rd > pd.Timestamp(sw)) & (rd <= pd.Timestamp(tt))].copy()
            if test.empty:
                continue
            folds[i] = fr.predict_gates(fit_df, test)
            print(f"fold {i}: fit={len(fit_df):,} test={len(test):,} ({labels[i]})")
        dev_pred = pd.concat([folds[i] for i in (1, 2, 3) if i in folds], ignore_index=True)
        results = {}
        for name, pred in [('DEV folds 1-3', dev_pred)] + [(f'fold {i} {labels[i]}', folds[i]) for i in (1, 2, 3, 4) if i in folds]:
            cands = build_candidates2(pred, store, keys)
            slate = fr.weekly_slate(cands)
            t5 = simulate2(slate, t4=False)
            t4 = simulate2(slate, t4=True)
            print(f"\n=== {name}: candidates={len(cands)} slate={len(slate)} ===")
            results[name] = paired_compare(t5, t4, name)
    out = HERE / 'archive' / 'experiments' / 'gate_decomposition_v6' / 't4_vs_t5_exit.json'
    out.write_text(json.dumps(results, indent=2, default=float), encoding='utf-8')
    print('\nSaved', out)


if __name__ == '__main__':
    main()

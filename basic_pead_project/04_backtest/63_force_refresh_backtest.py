#!/usr/bin/env python3
"""Force-refresh vs conviction-priority backtest (DIAGNOSTIC; no code change).

Question: does force-refreshing all 4 slots each week with that week's top-4
picks (force-selling the oldest last-week position when full) beat
conviction-priority (skip due-today picks when all slots are occupied)?

Both algos share the SAME weekly slate (top-4 threshold-passers by score per
ISO week) and the SAME return definition (raw adj-close pre-gap return with
-10% delayed stop). The ONLY difference is what happens when a slate pick needs
a slot and all 4 are occupied:
  - 'skip' (conviction-priority): skip the pick.
  - 'force_refresh': force-sell the oldest slot occupied since last week
    (entry ISO-week < pick entry ISO-week), assign the new pick.
Force-sold positions earn a PARTIAL return (entry -> force-sell date) computed
from db.h5 prices with the stop rule; natural exits earn the full T+5 return
(matrix pregap_return).

Validation: the 'skip' sim should roughly match the established select_weekly
backtest (script 62). No db.h5/model/policy/code change.
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
THRESH = POLICY['ensemble']['threshold']
STOP = 0.10


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


# ---- price cache + partial returns ----
_PCACHE: dict = {}


def get_prices(store, keys, pt):
    if pt not in _PCACHE:
        key = f"/sp400/{pt}"
        if key in keys:
            p = store[key]
            _PCACHE[pt] = (pd.to_datetime(p["Date"]).values, p["Adj_Close"].to_numpy(float))
        else:
            _PCACHE[pt] = None
    return _PCACHE[pt]


def ret_with_stop(closes, eidx, sidx, stop=STOP):
    if eidx < 0 or sidx >= len(closes) or sidx <= eidx:
        return None
    ep = closes[eidx]
    if not np.isfinite(ep) or ep <= 0:
        return None
    stop_price = ep * (1 - stop)
    path = closes[eidx:sidx + 1]
    path = path[np.isfinite(path)]
    for sp in path[1:]:
        if sp <= stop_price:
            return float(sp / ep - 1.0)
    xp = closes[sidx]
    if not np.isfinite(xp):
        return None
    return float(xp / ep - 1.0)


def iso_week(ts):
    iso = pd.Timestamp(ts).isocalendar()
    return (int(iso.year), int(iso.week))


def build_candidates(frame, store, keys):
    """Filter threshold-passers, attach price-derived entry/exit indices."""
    mask = (frame.score >= THRESH) & (~frame.sector.isin(XLF)) & frame.pregap_return.notna()
    cands = frame[mask].copy()
    cands['entry_date'] = pd.to_datetime(cands.pregap_entry_date)
    cands['exit_date'] = pd.to_datetime(cands.pregap_exit_date)
    rows = []
    for _, r in cands.iterrows():
        pt = r.permaTicker
        pl = get_prices(store, keys, pt)
        if pl is None:
            continue
        dates, closes = pl
        eidx = int(np.searchsorted(dates, np.datetime64(r.entry_date), side='left'))
        xidx = int(np.searchsorted(dates, np.datetime64(r.exit_date), side='left'))
        if eidx >= len(closes) or xidx >= len(closes) or eidx < 0:
            continue
        rows.append({**r.to_dict(),
                     'entry_idx': eidx, 'exit_idx': xidx})
    return pd.DataFrame(rows).sort_values(['entry_date', 'score'], ascending=[True, False]).reset_index(drop=True)


def weekly_slate(cands):
    """Top-N_SLOTS by score per ISO week."""
    keep = []
    for wk, g in cands.groupby(cands.entry_date.map(iso_week), sort=True):
        keep.append(g.sort_values('score', ascending=False).head(N_SLOTS))
    return pd.concat(keep, ignore_index=True).sort_values(['entry_date', 'score'], ascending=[True, False]).reset_index(drop=True) if keep else pd.DataFrame()


def simulate(slate, mode, min_hold=0):
    """Slot simulator. mode in {'skip','force_refresh'}. Returns executed trades."""
    slots = []  # active positions (dicts with entry_date, exit_date, pregap_return, permaTicker, entry_idx, _closes)
    trades = []
    for ev in slate.itertuples(index=False):
        ed = ev.entry_date
        # expire natural T+5 exits
        kept = []
        for s in slots:
            if s['exit_date'] <= ed:
                trades.append({**s, 'return': s['pregap_return'], 'exit_reason': 't5'})
            else:
                kept.append(s)
        slots = kept
        # place ev
        if len(slots) < N_SLOTS:
            slots.append(_slot(ev))
        elif mode == 'force_refresh':
            scored = []
            for s in slots:
                if iso_week(s['entry_date']) >= iso_week(ed):
                    continue  # only force-sell positions from a prior ISO week
                pl = _PCACHE.get(s['permaTicker'])
                if pl is None:
                    continue
                vdates, vcloses = pl
                sidx = int(np.searchsorted(vdates, np.datetime64(ed), side='right')) - 1
                hold = sidx - s['entry_idx']
                if min_hold > 0 and hold < min_hold:
                    continue  # min-hold guard: don't cut fresh positions
                scored.append((s, sidx, hold, vcloses))
            if scored:
                victim, sidx, hold_days, vcloses = min(scored, key=lambda x: x[0]['entry_date'])
                part = ret_with_stop(vcloses, victim['entry_idx'], sidx)
                if part is None:
                    part = victim['pregap_return']  # fallback if dates misalign
                trades.append({**victim, 'return': part, 'exit_reason': 'force_refresh',
                               'force_sell_date': ed, 'hold_days': hold_days})
                slots.remove(victim)
                slots.append(_slot(ev))
            # else: no force-sellable slot -> skip
        # else skip mode: skip
    for s in slots:
        trades.append({**s, 'return': s['pregap_return'], 'exit_reason': 't5_end'})
    return trades


def _slot(ev):
    return {'permaTicker': ev.permaTicker,
            'entry_date': ev.entry_date, 'exit_date': ev.exit_date,
            'pregap_return': ev.pregap_return, 'score': ev.score,
            'entry_idx': ev.entry_idx}


def stats(trades):
    if not trades:
        return dict(trades=0, win_rate_pct=0., avg_win_pct=0., avg_loss_pct=0.,
                    max_dd_pct=0., nav_pct=0., force_sold=0, avg_force_hold=0.)
    df = pd.DataFrame(trades)
    df['week'] = df.entry_date.map(iso_week)
    weekly = df.groupby('week', sort=True)['return'].sum().div(N_SLOTS)
    nav = (1 + weekly).cumprod()
    dd = ((nav - nav.cummax()) / nav.cummax()).min() * 100
    r = df['return'].to_numpy(float)
    wins = r[r > 0]; losses = r[r <= 0]
    fs = df[df.exit_reason == 'force_refresh']
    return dict(trades=int(len(r)),
                win_rate_pct=float((r > 0).mean() * 100),
                avg_win_pct=float(wins.mean() * 100) if len(wins) else 0.,
                avg_loss_pct=float(losses.mean() * 100) if len(losses) else 0.,
                max_dd_pct=float(dd),
                nav_pct=float((nav.iloc[-1] - 1) * 100),
                force_sold=int(len(fs)),
                avg_force_hold=float(fs.hold_days.mean()) if len(fs) else 0.)


def select_weekly_stats(frame):
    """Established backtest baseline (select_weekly, ignores slot carryover)."""
    mask = (frame.score >= THRESH) & (~frame.sector.isin(XLF)) & frame.pregap_return.notna()
    raw = frame[mask].copy()
    if raw.empty:
        return dict(trades=0, win_rate_pct=0., nav_pct=0.)
    raw['p'] = raw.score
    raw['entry_date'] = pd.to_datetime(raw.pregap_entry_date)
    raw['exit_date'] = pd.to_datetime(raw.pregap_exit_date)
    ex = bt.select_weekly(raw, N_SLOTS)
    r = ex.pregap_return.to_numpy(float) if len(ex) else np.array([])
    iso = pd.to_datetime(ex.entry_date).dt.isocalendar()
    ex = ex.copy(); ex['week'] = iso.year.astype(str) + '-W' + iso.week.astype(str).str.zfill(2)
    weekly = ex.groupby('week', sort=True).pregap_return.sum().div(N_SLOTS)
    nav = (1 + weekly).cumprod()
    return dict(trades=int(len(r)), win_rate_pct=float((r > 0).mean() * 100) if len(r) else 0.,
                nav_pct=float((nav.iloc[-1] - 1) * 100) if len(weekly) else 0.)


def main():
    print('=' * 112)
    print(f'FORCE-REFRESH vs CONVICTION-PRIORITY (threshold {THRESH}, {STOP:.0%} stop)')
    print('=' * 112)
    df = pd.read_hdf(DB, MATRIX); rd = pd.to_datetime(df.report_date)
    folds = {}; labels = {1: '2024 H2', 2: '2025 H1', 3: '2025 H2', 4: '2026 H1 (holdout)'}
    with pd.HDFStore(DB, mode='r') as store:
        keys = set(store.keys())
        for i, (te, sw, tt) in enumerate(bt.DEFAULT_FOLDS, 1):
            fit_df = df[rd <= pd.Timestamp(sw)].copy()
            test = df[(rd > pd.Timestamp(sw)) & (rd <= pd.Timestamp(tt))].copy()
            if test.empty: continue
            folds[i] = predict_gates(fit_df, test)
            print(f"fold {i}: fit={len(fit_df):,} test={len(test):,} ({labels[i]})")
        dev_pred = pd.concat([folds[i] for i in (1, 2, 3) if i in folds], ignore_index=True)

        def run(name, pred):
            cands = build_candidates(pred, store, keys)
            slate = weekly_slate(cands)
            sw = select_weekly_stats(pred)
            sk = simulate(slate, 'skip')
            fr = simulate(slate, 'force_refresh')
            print(f"\n=== {name}  (candidates={len(cands)}, slate={len(slate)}) ===")
            print(f"{'mode':<22}{'trades':>7}{'win%':>7}{'avgWin%':>9}{'avgLoss%':>9}"
                  f"{'maxDD%':>8}{'NAV%':>9}{'forceSold':>10}{'avgFrcHold':>10}")
            for label, s in [('select_weekly(ref)', sw | {'avg_win_pct': 0, 'avg_loss_pct': 0, 'max_dd_pct': 0, 'force_sold': 0, 'avg_force_hold': 0}),
                             ('conviction(skip)', stats(sk)), ('force_refresh', stats(fr))]:
                print(f"{label:<22}{s.get('trades',0):>7}{s.get('win_rate_pct',0):>7.1f}"
                      f"{s.get('avg_win_pct',0):>9.2f}{s.get('avg_loss_pct',0):>9.2f}"
                      f"{s.get('max_dd_pct',0):>8.1f}{s.get('nav_pct',0):>9.1f}"
                      f"{s.get('force_sold',0):>10}{s.get('avg_force_hold',0):>10.1f}")
            return {'select_weekly_ref': sw, 'conviction_skip': stats(sk), 'force_refresh': stats(fr)}

        res = {}
        res['dev_1_3'] = run('DEV folds 1-3', dev_pred)
        for i in (1, 2, 3, 4):
            if i in folds:
                res[f'fold_{i}'] = run(f"fold {i} {labels[i]}", folds[i])
    with open(OUT / 'force_refresh_backtest.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, default=float)
    print('\nSaved', OUT / 'force_refresh_backtest.json')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""83_megatrend_partial_exposure_cycle2_robustness.py — RC-4 Cycle 2 gate.

This is a pre-registered robustness/implementation gate for Cycle 1. It does
not search for the best backtest. It reports the complete fixed sensitivity
grid and fixed calendar blocks so parameter fragility is visible.

GRID
----
Theme floor:       5%, 10%, 15%
Theme cap:        50%, 70%, 90%
Monthly step:      5%, 10%, 15%
Rotation:          price-only and price + point-in-time capex
Turnover cost:     0, 25, 50, 100 bps per one-way turnover
Absolute overlay:  fixed 50% exposure after the Cycle-1 Sahm-style trigger,
                   tested only on the fixed price+capex / floor=10 / cap=70 /
                   step=10 configuration.

The target map remains fixed at rank weights 60/30/10. Target weights are
projected onto the feasible floor/cap simplex; no parameter is selected from
results. Weight changes are capped per month and applied to the next month's
returns. The current partial month is excluded.

WALK-FORWARD BLOCKS
-------------------
2014-02–2019-12, 2020-01–2022-12, 2023-01–2026-07.
These are reporting blocks, not tuned training windows. A useful candidate
should not depend on one block or one crisis/recovery episode.

RESEARCH-ONLY. No operational watcher, core allocation, PEAD model, or order
logic is changed.
"""
from __future__ import annotations
import importlib.util
import itertools
import json
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_CSV = HERE / "archive" / "experiments" / "rc4_partial_exposure_cycle2_grid.csv"
OUT_JSON = HERE / "archive" / "experiments" / "rc4_partial_exposure_cycle2_summary.json"
CYCLE1 = HERE / "82_megatrend_partial_exposure_cycle1.py"

spec = importlib.util.spec_from_file_location("cycle1", CYCLE1)
cycle1 = importlib.util.module_from_spec(spec)
sys.modules["cycle1"] = cycle1
assert spec.loader is not None
spec.loader.exec_module(cycle1)

THEMES = cycle1.THEMES
THEME_NAMES = cycle1.THEME_NAMES
RANK_TARGET = np.array([.60, .30, .10])
FLOORS = [.05, .10, .15]
CAPS = [.50, .70, .90]
STEPS = [.05, .10, .15]
COSTS_BPS = [0, 25, 50, 100]
BLOCKS = {
    "2014_2019": ("2014-02-28", "2019-12-31"),
    "2020_2022": ("2020-01-31", "2022-12-31"),
    "2023_2026": ("2023-01-31", "2026-07-31"),
}


def project_simplex(x, floor, cap):
    """Project x onto sum(w)=1 and floor<=w<=cap via scalar water filling."""
    x = np.asarray(x, dtype=float)
    if len(x) * floor > 1 + 1e-12 or len(x) * cap < 1 - 1e-12:
        raise ValueError((x, floor, cap))
    lo, hi = -2.0, 2.0
    for _ in range(100):
        mid = (lo + hi) / 2
        w = np.clip(x + mid, floor, cap)
        if w.sum() > 1:
            hi = mid
        else:
            lo = mid
    w = np.clip(x + (lo + hi) / 2, floor, cap)
    return w / w.sum()


def fixed_price_targets(R, months, floor, cap):
    nav = (1 + R.fillna(0)).cumprod()
    trailing = nav / nav.shift(12) - 1
    out = pd.DataFrame(index=months, columns=THEME_NAMES, dtype=float)
    for d in months:
        scores = trailing.loc[d].dropna()
        base = np.ones(3) / 3
        if len(scores) >= 2:
            order = list(scores.sort_values(ascending=False).index)
            raw = {t: 0.0 for t in THEME_NAMES}
            for t, v in zip(order, RANK_TARGET): raw[t] = v
            base = np.array([raw[t] for t in THEME_NAMES])
            if base.sum() <= 0: base = np.ones(3) / 3
            else: base = base / base.sum()
        out.loc[d] = project_simplex(base, floor, cap)
    return out


def fixed_capex_targets(shares, months, floor, cap):
    out = pd.DataFrame(index=months, columns=THEME_NAMES, dtype=float)
    for d in months:
        x = shares.loc[d].fillna(0).values.astype(float)
        x = x / x.sum() if x.sum() > 0 else np.ones(3) / 3
        out.loc[d] = project_simplex(x, floor, cap)
    return out


def bounded_step(prev, target, step, floor, cap):
    """Move toward target with per-theme step cap, then project feasibly."""
    prev = np.asarray(prev, dtype=float)
    target = np.asarray(target, dtype=float)
    raw = prev + np.clip(target - prev, -step, step)
    # Projection can slightly change an individual move when the simplex must
    # be restored; this is explicit and reported through turnover.
    return project_simplex(raw, floor, cap)


def run_variant(R, spy, shares, recession, rotation, floor, cap, step,
                cost_bps=0, absolute_overlay=False):
    months = R.index
    ptargets = fixed_price_targets(R, months, floor, cap)
    ctargets = fixed_capex_targets(shares, months, floor, cap)
    prev = np.ones(3) / 3
    prev_exp = 1.0
    rows = []
    for d in months:
        if rotation == "price":
            target = ptargets.loc[d].values.astype(float)
        elif rotation == "price_plus_capex":
            target = .5 * ptargets.loc[d].values + .5 * ctargets.loc[d].values
            target = project_simplex(target, floor, cap)
        else:
            raise ValueError(rotation)
        w = bounded_step(prev, target, step, floor, cap)
        turnover = .5 * np.abs(w - prev).sum()
        target_exp = .5 if absolute_overlay and bool(recession.loc[d]) else 1.0
        exp = float(np.clip(target_exp - prev_exp, -.25, .25) + prev_exp) if absolute_overlay else 1.0
        exp_turnover = abs(exp - prev_exp)
        rows.append({"month": d, "w_AI/hyperscale": w[0], "w_clean_energy": w[1], "w_crypto": w[2],
                     "absolute_exposure": exp, "theme_turnover": turnover, "exposure_turnover": exp_turnover,
                     "recession": bool(recession.loc[d])})
        prev, prev_exp = w, exp
    W = pd.DataFrame(rows).set_index("month")
    # Month-end weights and exposure are known for next month. Cost is charged
    # at the same rebalance and therefore shifted with the weights/returns.
    gross = W[[f"w_{t}" for t in THEME_NAMES]].shift(1).to_numpy() * R[THEME_NAMES].to_numpy()
    gross = gross.sum(axis=1) * W.absolute_exposure.shift(1).to_numpy()
    turnover_cost = (W.theme_turnover + W.exposure_turnover) * cost_bps / 10000
    net = pd.Series(gross - turnover_cost.shift(1).fillna(0).to_numpy(), index=months).dropna()
    return W, net


def stats(net, spy, start=None, end=None):
    x = net.copy()
    if start: x = x[x.index >= pd.Timestamp(start)]
    if end: x = x[x.index <= pd.Timestamp(end)]
    x = x.dropna()
    if x.empty: return {"n": 0}
    nav = (1 + x).cumprod()
    dd = nav / nav.cummax() - 1
    by_year = x.groupby(x.index.year).apply(lambda z: float((1 + z).prod() - 1)).to_dict()
    b = spy.reindex(x.index).dropna()
    bnav = (1 + b).cumprod()
    return {"n": int(len(x)), "total": float(nav.iloc[-1] - 1),
            "annualized": float(nav.iloc[-1] ** (12 / len(x)) - 1), "max_dd": float(dd.min()),
            "mean_month": float(x.mean()), "turnover_month": None,
            "spy_total": float(bnav.iloc[-1] - 1) if len(bnav) else np.nan,
            "by_year": {str(k): float(v) for k, v in by_year.items()},
            "2020": float(by_year.get(2020, np.nan)), "2022": float(by_year.get(2022, np.nan))}


def main():
    print("=" * 100)
    print("RC-4 CYCLE 2 — PARTIAL EXPOSURE ROBUSTNESS / IMPLEMENTATION GATE")
    print("=" * 100)
    R, spy = cycle1.monthly_theme_returns()
    months = R.index
    shares = cycle1.point_capex_share(months)
    recession, sahm_gap, macro_audit = cycle1.macro_recession(months)
    rows = []
    summary = {"script": "83_megatrend_partial_exposure_cycle2_robustness.py", "status": "research_only",
               "pre_registered_grid": {"floors": FLOORS, "caps": CAPS, "steps": STEPS,
                                        "cost_bps": COSTS_BPS, "rotations": ["price", "price_plus_capex"]},
               "blocks": BLOCKS, "data": {"first": str(months.min().date()), "last": str(months.max().date()),
                                            "months": len(months), "recession_trigger_months": int(recession.sum()),
                                            "macro": macro_audit}, "records": []}
    for rotation, floor, cap, step, cost in itertools.product(["price", "price_plus_capex"], FLOORS, CAPS, STEPS, COSTS_BPS):
        if 3 * floor > 1 or 3 * cap < 1: continue
        W, net = run_variant(R, spy, shares, recession, rotation, floor, cap, step, cost, False)
        for block, (start, end) in BLOCKS.items():
            s = stats(net, spy, start, end)
            s.update({"rotation": rotation, "floor": floor, "cap": cap, "step": step,
                      "cost_bps": cost, "block": block, "overlay": False,
                      "mean_theme_turnover": float(W.theme_turnover.loc[start:end].mean())})
            summary["records"].append(s); rows.append(s)
    # Fixed absolute overlay: no grid search of recession strength in this gate.
    fixed = {"rotation": "price_plus_capex", "floor": .10, "cap": .70, "step": .10}
    for cost in COSTS_BPS:
        W, net = run_variant(R, spy, shares, recession, **fixed, cost_bps=cost, absolute_overlay=True)
        for block, (start, end) in BLOCKS.items():
            s = stats(net, spy, start, end)
            s.update({**fixed, "cost_bps": cost, "block": block, "overlay": True,
                      "mean_theme_turnover": float(W.theme_turnover.loc[start:end].mean()),
                      "mean_exposure": float(W.absolute_exposure.loc[start:end].mean())})
            summary["records"].append(s); rows.append(s)
    result = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True); result.to_csv(OUT_CSV, index=False)
    # Aggregate only after all fixed records are generated. No winner is chosen.
    summary["n_records"] = len(result)
    summary["aggregate"] = {
        "price_plus_capex_2022": result[(result.rotation == "price_plus_capex") & (result.block == "2020_2022") & (~result.overlay)].groupby(["floor", "cap", "step", "cost_bps"], as_index=False)["2022"].first().to_dict("records"),
        "overlay_fixed": result[result.overlay].to_dict("records"),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1, default=str))
    print(f"[1] data: {months[0].date()} -> {months[-1].date()} ({len(months)} months), recession months={int(recession.sum())}")
    print(f"[2] grid records: {len(result)}")
    for rotation in ["price", "price_plus_capex"]:
        x = result[(result.rotation == rotation) & (result.block == "2020_2022") & (result.cost_bps == 50) & (~result.overlay)]
        print(f"\n{rotation} / 50bps / 2020-2022")
        print(x[["floor", "cap", "step", "total", "max_dd", "2022", "mean_theme_turnover"]].to_string(index=False))
    print("\n[3] fixed recession overlay / 50bps")
    print(result[(result.overlay) & (result.cost_bps == 50)][["block", "total", "max_dd", "2020", "2022", "mean_exposure"]].to_string(index=False))
    print("\n[4] gate status: sensitivity matrix generated; no parameter or operational rule promoted.")
    print(f"[5] artifacts: {OUT_CSV.relative_to(ROOT.parent)}")
    print(f"             {OUT_JSON.relative_to(ROOT.parent)}")

if __name__ == "__main__": main()

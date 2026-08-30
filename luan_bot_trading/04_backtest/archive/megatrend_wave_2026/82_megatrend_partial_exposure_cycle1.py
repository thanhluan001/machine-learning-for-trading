#!/usr/bin/env python3
"""82_megatrend_partial_exposure_cycle1.py — RC-4 partial exposure cycle 1.

RESEARCH QUESTION
-----------------
Can the megatrend overlay remain invested during a non-recessionary,
undecided market while slowly rotating toward the theme with sustained market
support? The failed Phase-3 binary basket treated every broad trend break as a
reason to leave and every bear-rally recovery as a reason to re-enter. This
cycle separates two decisions:

1. THEME MIX: partial, bounded rotation among AI/hyperscale, clean energy, and
   crypto; monthly weight changes are capped.
2. ABSOLUTE EXPOSURE: remains 100% unless a separate, point-in-time Sahm-style
   unemployment recession trigger fires.

This is a research backtest. It does not change the operational watcher.

PRE-REGISTERED VARIANTS
-----------------------
A  equal_theme: 1/3 each, 100% invested.
B  price_rotate: trailing 12-month theme return ranks map to 60/30/10 target
   weights; actual weights move no more than 10 percentage points per month;
   10% theme floor, 70% theme cap; 100% absolute exposure.
C  price_plus_capex: 50% price target + 50% point-in-time capex-share target;
   same floors/caps/10pp monthly transition; 100% absolute exposure.
D  price_plus_capex_recession: C plus absolute exposure reduced to 50% after a
   Sahm-style unemployment trigger; exposure changes are capped at 25pp/month.

The capex target is a slow sponsorship prior, not a timing signal. It is
calculated from FMP cash-flow observations visible by the month-end and is
applied only to the following month. The price target is rank-based to avoid a
single extreme return dictating the allocation.

RECESSION CONTRACT
------------------
Sahm-style trigger = 3-month average unemployment minus its trailing
12-month minimum >= 0.50 percentage points. The signal uses only the latest
FRED observation available by month-end and applies next month. 2022 is not
classified as recession merely because equities are below trend. No absolute
exposure reduction is applied in 2022 unless the fixed macro trigger fires.

INVESTABLE PROXY PANEL
----------------------
AI/hyperscale: SMH
clean_energy:  equal-return blend of ICLN and TAN when available
crypto:        equal-return blend of MSTR and COIN when available

This is a theme-proxy study, not a claim that the proxies represent every
possible constituent. It is intentionally separate from the six-theme
operational reporting panel.

OUTPUTS
-------
archive/experiments/rc4_partial_exposure_cycle1_monthly.csv
archive/experiments/rc4_partial_exposure_cycle1_summary.json

No result is promoted without walk-forward/episode validation, costs, and
explicit review of the 2020 recovery and 2022 undecided regime.
"""
from __future__ import annotations
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
DB_MT = ROOT / "01_data" / "db_megatrend.h5"
DB_CAPEX = ROOT / "01_data" / "db_capex.h5"
DB_MACRO = ROOT / "01_data" / "db.h5"
OUT_CSV = HERE / "archive" / "experiments" / "rc4_partial_exposure_cycle1_monthly.csv"
OUT_JSON = HERE / "archive" / "experiments" / "rc4_partial_exposure_cycle1_summary.json"

THEMES = {
    "AI/hyperscale": {"proxies": ["SMH"], "members": ["MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL"]},
    "clean_energy": {"proxies": ["ICLN", "TAN"], "members": ["FSLR", "ENPH", "SEDG", "NEE", "RUN", "PLUG"]},
    "crypto": {"proxies": ["MSTR", "COIN"], "members": ["MSTR", "COIN", "RIOT", "MARA", "CLSK"]},
}
THEME_NAMES = list(THEMES)
RANK_TARGET = np.array([.60, .30, .10])
FLOOR = .10
CAP = .70
STEP = .10


def monthly_theme_returns():
    """Return theme monthly returns plus SPY benchmark; no synthetic price level."""
    with pd.HDFStore(DB_MT, mode="r") as s:
        daily = {k.rsplit("/", 1)[-1]: s[k].set_index("date")["adjClose"].sort_index()
                 for k in s.keys()}
    max_date = max(x.index.max() for x in daily.values() if len(x))
    cutoff = max_date.to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)
    out = {}
    for theme, cfg in THEMES.items():
        components = []
        for sym in cfg["proxies"]:
            if sym not in daily:
                continue
            p = daily[sym].resample("ME").last()
            r = p.pct_change().rename(sym)
            components.append(r)
        out[theme] = pd.concat(components, axis=1).mean(axis=1, skipna=True) if components else pd.Series(dtype=float)
    R = pd.DataFrame(out).sort_index()
    R = R.loc[R.index <= cutoff]
    spy = daily["SPY"].resample("ME").last().pct_change().reindex(R.index)
    return R, spy


def point_capex_share(months):
    """Point-in-time TTM capex shares, visible at month-end; no future filings."""
    raw = {}
    with pd.HDFStore(DB_CAPEX, mode="r") as s:
        keys = set(s.keys())
        for theme, cfg in THEMES.items():
            by_company = []
            for sym in cfg["members"]:
                k = f"/capex_raw/{sym}"
                if k not in keys:
                    continue
                d = s[k].copy()
                d["period_date"] = pd.to_datetime(d.period_date, errors="coerce").dt.normalize()
                d["available_date"] = pd.to_datetime(d.available_date, errors="coerce").dt.tz_localize(None)
                d["capex"] = pd.to_numeric(d.capex, errors="coerce")
                vals = []
                for asof in months:
                    visible = d[d.available_date <= asof].sort_values("available_date")
                    visible = visible.drop_duplicates("period_date", keep="last")
                    visible = visible[visible.period_date <= asof].sort_values("period_date")
                    vals.append(float(visible.tail(4).capex.sum()) if len(visible) >= 4 else np.nan)
                by_company.append(pd.Series(vals, index=months, name=sym))
            raw[theme] = pd.concat(by_company, axis=1).sum(axis=1, min_count=1) if by_company else pd.Series(np.nan, index=months)
    X = pd.DataFrame(raw, index=months)
    return X.div(X.sum(axis=1), axis=0)


def macro_recession(months):
    """Point-in-time Sahm-style signal from FRED unemployment observations."""
    with pd.HDFStore(DB_MACRO, mode="r") as s:
        key = "/macros/fred_unemployment_rate"
        if key not in s.keys():
            return pd.Series(False, index=months), pd.Series(np.nan, index=months), {"available": False}
        d = s[key].copy()
    d["Date"] = pd.to_datetime(d.Date, errors="coerce").dt.tz_localize(None)
    d["unemployment_rate"] = pd.to_numeric(d.unemployment_rate, errors="coerce")
    d = d.dropna().set_index("Date")["unemployment_rate"].sort_index()
    vals = []
    for asof in months:
        visible = d[d.index <= asof]
        vals.append(visible.iloc[-1] if len(visible) else np.nan)
    u = pd.Series(vals, index=months)
    avg3 = u.rolling(3, min_periods=3).mean()
    min12 = avg3.rolling(12, min_periods=12).min()
    sahm_gap = avg3 - min12
    trigger = sahm_gap >= .50
    return trigger.fillna(False), sahm_gap, {"available": True, "trigger_months": int(trigger.sum())}


def price_targets(R, months):
    """Map 12m theme returns to fixed 60/30/10 rank target."""
    nav = (1 + R.fillna(0)).cumprod()
    trailing = nav / nav.shift(12) - 1
    targets = pd.DataFrame(index=months, columns=THEME_NAMES, dtype=float)
    for d in months:
        x = trailing.reindex(months).loc[d].dropna()
        if len(x) < 2:
            targets.loc[d] = 1 / len(THEME_NAMES)
            continue
        order = list(x.sort_values(ascending=False).index)
        vals = dict(zip(order, RANK_TARGET[:len(order)]))
        # If a proxy lacks history, retain a floor-like equal treatment among
        # available themes; this is research transparency, not data invention.
        for t in THEME_NAMES:
            targets.loc[d, t] = vals.get(t, 1 / len(THEME_NAMES))
        targets.loc[d] = targets.loc[d].fillna(0)
        targets.loc[d] /= targets.loc[d].sum()
    return targets


def capex_targets(shares):
    """Bound capex shares before blending; retains sponsorship information without 95% single-theme exposure."""
    out = shares.copy()
    for d in out.index:
        x = out.loc[d].fillna(0)
        if x.sum() <= 0:
            out.loc[d] = 1 / len(THEME_NAMES)
            continue
        x = x / x.sum()
        x = x.clip(lower=FLOOR, upper=CAP)
        out.loc[d] = x / x.sum()
    return out


def bounded_transition(prev, target, step=STEP):
    prev = np.asarray(prev, dtype=float); target = np.asarray(target, dtype=float)
    delta = np.clip(target - prev, -step, step)
    w = prev + delta
    # Restore the simplex after independent movement caps while keeping the
    # per-theme movement bound as far as possible. Residual is assigned to the
    # themes furthest below target, then clipped defensively.
    for _ in range(5):
        residual = 1 - w.sum()
        if abs(residual) < 1e-10: break
        if residual > 0:
            candidates = np.argsort(-(target - w))
            for i in candidates:
                room = min(step - max(0, w[i] - prev[i]), CAP - w[i])
                add = min(residual, max(0, room)); w[i] += add; residual -= add
                if residual <= 1e-10: break
        else:
            candidates = np.argsort(target - w)
            for i in candidates:
                room = min(step - max(0, prev[i] - w[i]), w[i] - FLOOR)
                sub = min(-residual, max(0, room)); w[i] -= sub; residual += sub
                if residual >= -1e-10: break
    return w / w.sum()


def run_variant(name, R, ptargets, ctargets, recession, recession_exposure=None):
    months = R.index
    weights = []
    exposure = []
    prev = np.ones(len(THEME_NAMES)) / len(THEME_NAMES)
    prev_exp = 1.0
    for d in months:
        if name == "equal_theme":
            target = np.ones(3) / 3
        elif name == "price_rotate":
            target = ptargets.loc[d].values.astype(float)
        elif name in ("price_plus_capex", "price_plus_capex_recession"):
            target = .5 * ptargets.loc[d].values.astype(float) + .5 * ctargets.loc[d].values.astype(float)
            target = target / target.sum()
        else:
            raise ValueError(name)
        w = bounded_transition(prev, target)
        prev = w
        target_exp = recession_exposure if (name == "price_plus_capex_recession" and recession.loc[d]) else 1.0
        if name == "price_plus_capex_recession":
            e = float(np.clip(target_exp - prev_exp, -.25, .25) + prev_exp)
        else:
            e = 1.0
        prev_exp = e
        weights.append(w); exposure.append(e)
    W = pd.DataFrame(weights, index=months, columns=THEME_NAMES)
    E = pd.Series(exposure, index=months, name="absolute_exposure")
    # Weights at d are known after d's close and earn d+1 return.
    port = (W.shift(1).mul(R, axis=0).sum(axis=1) * E.shift(1)).dropna()
    return W, E, port


def stats(port, spy):
    nav = (1 + port.fillna(0)).cumprod()
    bh = (1 + spy.reindex(port.index).fillna(0)).cumprod()
    dd = nav / nav.cummax() - 1
    years = port.index.year
    by_year = port.groupby(years).apply(lambda x: float((1 + x).prod() - 1)).to_dict()
    return {"total": float(nav.iloc[-1] - 1), "annualized": float(nav.iloc[-1] ** (12 / len(port)) - 1),
            "max_dd": float(dd.min()), "by_year": {str(k): v for k, v in by_year.items()},
            "2022": float(by_year.get(2022, np.nan)), "2020": float(by_year.get(2020, np.nan)),
            "spy_total_same_window": float(bh.iloc[-1] - 1), "n_months": int(len(port))}


def main():
    print("=" * 96)
    print("RC-4 CYCLE 1 — PARTIAL THEME EXPOSURE + RECESSION OVERLAY")
    print("=" * 96)
    R, spy = monthly_theme_returns()
    months = R.index
    shares = point_capex_share(months)
    ptargets = price_targets(R, months)
    ctargets = capex_targets(shares)
    recession, sahm_gap, macro_audit = macro_recession(months)
    variants = ["equal_theme", "price_rotate", "price_plus_capex", "price_plus_capex_recession"]
    all_rows = []; summary = {"script": "82_megatrend_partial_exposure_cycle1.py", "status": "research_only",
        "contract": {"theme_floor": FLOOR, "theme_cap": CAP, "monthly_theme_step": STEP,
                      "rank_target": RANK_TARGET.tolist(), "recession_exposure": .50,
                      "recession": "Sahm-style 3m unemployment average minus trailing 12m minimum >= 0.50",
                      "2022_absolute_exposure_reduction": "only if fixed macro trigger fires"},
        "data": {"months": len(months), "first": str(months.min().date()), "last": str(months.max().date()),
                 "theme_proxy_components": {t: v["proxies"] for t, v in THEMES.items()}, "macro": macro_audit}, "variants": {}}
    for name in variants:
        W, E, port = run_variant(name, R, ptargets, ctargets, recession, .50)
        s = stats(port, spy)
        summary["variants"][name] = s
        for d in months:
            row = {"month": d, "variant": name, "ret": port.get(d, np.nan), "absolute_exposure": E.loc[d],
                   "recession_trigger": bool(recession.loc[d]), "sahm_gap": sahm_gap.loc[d]}
            for t in THEME_NAMES:
                row[f"weight_{t}"] = W.loc[d, t]; row[f"return_{t}"] = R.loc[d, t]
                row[f"capex_share_{t}"] = shares.loc[d, t]; row[f"price_target_{t}"] = ptargets.loc[d, t]
            all_rows.append(row)
        print(f"\n{name}")
        print(f"  total={s['total']*100:+.1f}% annualized={s['annualized']*100:+.1f}% maxDD={s['max_dd']*100:.1f}% 2020={s['2020']*100:+.1f}% 2022={s['2022']*100:+.1f}%")
        print(f"  avg weights={W.mean().round(3).to_dict()}  2022 avg={W.loc[W.index.year==2022].mean().round(3).to_dict()}")
    result = pd.DataFrame(all_rows); result["month"] = pd.to_datetime(result.month).dt.strftime("%Y-%m-%d")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True); result.to_csv(OUT_CSV, index=False)
    summary["latest"] = result[result.month == result.month.max()].to_dict("records")
    summary["recession_months"] = [str(d.date()) for d in months[recession.values]]
    OUT_JSON.write_text(json.dumps(summary, indent=1, default=str))
    print("\n[RESULT INTERPRETATION]")
    print("  2022 remains invested unless the fixed unemployment recession trigger fires.")
    print("  Theme weights rotate gradually; capex is a bounded prior, not an exit trigger.")
    print("  These are proxy results and require cost/stability/episode validation before promotion.")
    print(f"\nArtifacts: {OUT_CSV.relative_to(ROOT.parent)}")
    print(f"           {OUT_JSON.relative_to(ROOT.parent)}")

if __name__ == "__main__": main()

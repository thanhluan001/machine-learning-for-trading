#!/usr/bin/env python3
"""81_megatrend_false_reentry_test.py — RC-4 normalized-channel episode test.

QUESTION
--------
Do normalized insider/news observations improve the Phase-3 failure mode?

    equity stress -> breadth recovery -> re-entry -> renewed stress

FIXED, MONTH-END CONTRACT
-------------------------
- Equity stress: <50% of the available members of the category-fixed equity
  panel above their own 10-month means AND SPY trailing 3-month return <0.
  The panel is expanding because several ETFs did not exist before 2023;
  months with fewer than five valid names are excluded as under-covered.
- Stress onset: first stress month after a non-stress month.
- Recovery episode: first later month with equity breadth >=60% after each onset.
  Only one recovery is retained per onset; onsets within 12 months are not
  double-counted.
- Re-entry decision: signal is observed at recovery month-end; forward outcome
  starts with the following completed month-end and covers six observations.
- Only one recovery is retained per non-overlapping stress episode. A recovery
  cannot be reused by two stress onsets.
- Relapse: fixed equity stress occurs during those six forward months.

FILTERS (research only)
-----------------------
A none; B normalized insider candidate; C normalized news candidate;
D either candidate; E both candidates.
A blocked recovery is not assumed to be a good decision. A block is useful only
if it avoids a relapse and does not systematically remove profitable recovery.

The comparison is descriptive/selection-conditioned, not a causal experiment.
No allocation rule is changed.
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
NORM = HERE / "archive" / "experiments" / "rc4_normalized_insider_news_monthly.csv"
OUT_EP = HERE / "archive" / "experiments" / "rc4_false_reentry_episodes.csv"
OUT_SUM = HERE / "archive" / "experiments" / "rc4_false_reentry_summary.json"
EQUITY = ["SPY", "QQQ", "IWM", "IJR", "EFA", "EEM", "XLB", "XLF", "XLI",
          "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "XLC", "XLE"]


def load_monthly_prices():
    with pd.HDFStore(DB_MT, mode="r") as s:
        px = {k.rsplit("/", 1)[-1]: s[k].set_index("date")["adjClose"] for k in s.keys()}
    daily_max = max(x.index.max() for x in px.values() if len(x))
    M = pd.DataFrame({c: x.resample("ME").last() for c, x in px.items() if c in EQUITY})
    # Do not treat the current partial month as a completed observation.
    cutoff = daily_max.to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)
    return M.loc[M.index <= cutoff]


def build_episodes():
    M = load_monthly_prices()
    names = [c for c in EQUITY if c in M.columns]
    p = M[names]
    ma = p.rolling(10).mean()
    valid = p.notna() & ma.notna()
    n_available = valid.sum(axis=1)
    breadth = (p > ma).where(valid).sum(axis=1) / n_available.replace(0, np.nan)
    spy3 = p["SPY"].pct_change(3)
    # Expanding-universe contract: before newer ETFs launched, breadth is
    # computed over names that actually existed and have a valid MA10m.
    covered = n_available >= 5
    stress = ((breadth < .50) & (spy3 < 0) & covered).fillna(False)
    normalized = pd.read_csv(NORM, parse_dates=["month"])
    normalized["month"] = normalized.month.dt.to_period("M").dt.to_timestamp("M")
    sig = normalized.groupby("month").agg(
        insider_candidate=("cluster_warning_candidate", "max"),
        news_candidate=("news_warning_candidate", "max"),
        insider_coverage=("insider_coverage_ok", "min"),
        news_coverage=("news_coverage_ok", "min"),
    )
    idx = breadth.index.intersection(sig.index)
    breadth = breadth.reindex(idx); spy3 = spy3.reindex(idx); stress = stress.reindex(idx); sig = sig.reindex(idx)
    # Get one recovery after each non-overlapping stress episode. The explicit
    # cursor prevents the same bear rally from being counted twice (e.g. two
    # stress onsets before the November 2023 recovery).
    onsets = [d for d in idx if bool(stress.loc[d]) and (idx.get_loc(d) == 0 or not bool(stress.iloc[idx.get_loc(d)-1]))]
    rows = []
    covered_until = None
    for onset in onsets:
        if covered_until is not None and onset <= covered_until:
            continue
        oi = idx.get_loc(onset)
        if oi + 1 >= len(idx): continue
        recovery = None
        for j in range(oi + 1, min(oi + 13, len(idx))):
            if breadth.iloc[j] >= .60:
                recovery = idx[j]; break
        if recovery is None: continue
        covered_until = recovery
        ri = idx.get_loc(recovery)
        future = idx[ri + 1:ri + 7]
        if len(future) == 0: continue
        relapse = bool(stress.reindex(future).fillna(False).any())
        fwd6 = float(p["SPY"].reindex(future).iloc[-1] / p["SPY"].loc[recovery] - 1) if p["SPY"].loc[recovery] and p["SPY"].reindex(future).notna().all() else np.nan
        row = {
            "stress_onset": onset.date().isoformat(), "recovery_month": recovery.date().isoformat(),
            "breadth_at_recovery": float(breadth.loc[recovery]), "spy3_at_recovery": float(spy3.loc[recovery]),
            "available_equity_names_at_recovery": int(n_available.loc[recovery]),
            "relapse_6m": relapse, "fwd_spy_6m": fwd6, "future_months": len(future),
            "insider_candidate": bool(sig.loc[recovery, "insider_candidate"]) if recovery in sig.index else False,
            "news_candidate": bool(sig.loc[recovery, "news_candidate"]) if recovery in sig.index else False,
            "insider_coverage": bool(sig.loc[recovery, "insider_coverage"]) if recovery in sig.index else False,
            "news_coverage": bool(sig.loc[recovery, "news_coverage"]) if recovery in sig.index else False,
            "regime": str(recovery.year),
        }
        rows.append(row)
    return pd.DataFrame(rows), breadth, stress, n_available


def evaluate(episodes: pd.DataFrame, col: str | None):
    if col is None:
        blocked = pd.Series(False, index=episodes.index)
    elif col == "any":
        blocked = episodes.insider_candidate | episodes.news_candidate
    elif col == "both":
        blocked = episodes.insider_candidate & episodes.news_candidate
    else:
        blocked = episodes[col]
    allowed = ~blocked
    relapse_allowed = int(episodes.loc[allowed, "relapse_6m"].sum())
    relapse_blocked = int(episodes.loc[blocked, "relapse_6m"].sum())
    pos = episodes.fwd_spy_6m > 0
    return {
        "filter": col or "none", "n_episodes": int(len(episodes)), "blocked": int(blocked.sum()),
        "allowed": int(allowed.sum()), "allowed_relapses": relapse_allowed,
        "allowed_relapse_rate": relapse_allowed / int(allowed.sum()) if allowed.sum() else None,
        "blocked_relapses": relapse_blocked,
        "blocked_relapse_rate": relapse_blocked / int(blocked.sum()) if blocked.sum() else None,
        "potentially_avoided_relapses": relapse_blocked,
        "blocked_positive_recoveries": int((blocked & pos).sum()),
        "blocked_positive_rate": float((blocked & pos).sum() / blocked.sum()) if blocked.sum() else None,
        "allowed_positive_rate": float((allowed & pos).sum() / allowed.sum()) if allowed.sum() else None,
        "mean_fwd6_allowed": float(episodes.loc[allowed, "fwd_spy_6m"].mean()) if allowed.sum() else None,
        "mean_fwd6_blocked": float(episodes.loc[blocked, "fwd_spy_6m"].mean()) if blocked.sum() else None,
    }


def main():
    print("=" * 92)
    print("RC-4 FALSE-REENTRY TEST — NORMALIZED INSIDER/NEWS FILTERS")
    print("=" * 92)
    episodes, breadth, stress, n_available = build_episodes()
    if episodes.empty:
        raise RuntimeError("No complete recovery episodes were constructed")
    episodes["calendar_regime"] = pd.cut(pd.to_datetime(episodes.recovery_month).dt.year,
                                          bins=[0, 2009, 2020, 2023, 2100], labels=["pre-2010", "2010-2020", "2021-2023", "2024+"])
    OUT_EP.parent.mkdir(parents=True, exist_ok=True); episodes.to_csv(OUT_EP, index=False)
    filters = [None, "insider_candidate", "news_candidate", "any", "both"]
    results = [evaluate(episodes, x) for x in filters]
    print(f"[1] complete recovery episodes: {len(episodes)}")
    print(f"    stress months: {int(stress.sum())}; equity panel history: {breadth.first_valid_index().date()} -> {breadth.last_valid_index().date()}")
    print(f"    expanding panel: valid breadth requires >=5 names; latest completed month has {int(n_available.iloc[-1])} names")
    print("\n[2] fixed filter comparison")
    for r in results:
        print(f"  {r['filter']:>18}: blocked={r['blocked']:2d} allowed={r['allowed']:2d} "
              f"allowed_relapse={r['allowed_relapse_rate'] if r['allowed_relapse_rate'] is not None else np.nan:.1%} "
              f"blocked_relapse={r['blocked_relapse_rate'] if r['blocked_relapse_rate'] is not None else np.nan:.1%} "
              f"blocked_positive={r['blocked_positive_rate'] if r['blocked_positive_rate'] is not None else np.nan:.1%}")
    print("\n[3] episodes")
    print(episodes[["stress_onset", "recovery_month", "breadth_at_recovery", "insider_candidate", "news_candidate", "relapse_6m", "fwd_spy_6m", "calendar_regime"]].to_string(index=False))
    print("\n[4] special regimes")
    for y in [2008, 2020, 2022]:
        x = episodes[pd.to_datetime(episodes.recovery_month).dt.year.eq(y)]
        stress_y = stress[stress.index.year == y]
        coverage_y = n_available[n_available.index.year == y]
        status = "OBSERVED" if len(x) else ("NO_COMPLETE_RECOVERY" if stress_y.any() else "NO_VALID_STRESS")
        print(f"  {y}: {status} stress={int(stress_y.sum())} max_valid_names={int(coverage_y.max()) if len(coverage_y) else 0} "
              f"n={len(x)} relapse={int(x.relapse_6m.sum()) if len(x) else 0} "
              f"insider={int(x.insider_candidate.sum()) if len(x) else 0} news={int(x.news_candidate.sum()) if len(x) else 0}")
    print("\n[5] conclusion")
    print("  This is a fixed-episode diagnostic, not causal validation.")
    print("  A channel cannot be promoted unless it blocks few profitable recoveries,")
    print("  has adequate coverage, and reduces relapse in independent episodes.")
    summary = {"script": "81_megatrend_false_reentry_test.py", "status": "research_only",
               "definitions": {"stress": "available equity breadth <50%, SPY trailing 3m return <0, >=5 valid names",
                               "recovery": "first later month after onset with breadth >=60%; non-overlapping episodes", "partial_current_month_excluded": True,
                               "horizon": "six following month-end observations"},
               "n_episodes": len(episodes), "results": results,
               "regimes": {str(y): episodes[pd.to_datetime(episodes.recovery_month).dt.year.eq(y)].to_dict("records") for y in [2008, 2020, 2022]}}
    OUT_SUM.write_text(json.dumps(summary, indent=1, default=str))
    print(f"\n[6] artifacts: {OUT_EP.relative_to(ROOT.parent)}")
    print(f"             {OUT_SUM.relative_to(ROOT.parent)}")

if __name__ == "__main__": main()

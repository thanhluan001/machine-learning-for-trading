#!/usr/bin/env python3
"""84_megatrend_partial_exposure_short_gate.py — RC-4 short-list gate.

Tests only the four configurations fixed after Cycle 2:
 A price-only, cap 50%, step 10%
 B price-only, cap 50%, step 15%
 C price+point-in-time-capex, cap 50%, step 10%
 D price+point-in-time-capex, cap 70%, step 10%

Costs: 0 / 50 / 100 bps one-way turnover.
Reports fixed calendar blocks, 2020/2022, and a monthly moving-block bootstrap
for full-sample total return. This is a stability check, not a winner search.
No recession overlay and no operational rule change.
"""
from __future__ import annotations
import importlib.util
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
OUT_CSV = HERE / "archive" / "experiments" / "rc4_partial_exposure_short_gate.csv"
OUT_JSON = HERE / "archive" / "experiments" / "rc4_partial_exposure_short_gate_summary.json"
CYCLE2 = HERE / "83_megatrend_partial_exposure_cycle2_robustness.py"
spec = importlib.util.spec_from_file_location("cycle2", CYCLE2)
cycle2 = importlib.util.module_from_spec(spec); sys.modules["cycle2"] = cycle2
assert spec.loader is not None; spec.loader.exec_module(cycle2)

CONFIGS = [
    {"id": "A", "rotation": "price", "floor": .10, "cap": .50, "step": .10},
    {"id": "B", "rotation": "price", "floor": .10, "cap": .50, "step": .15},
    {"id": "C", "rotation": "price_plus_capex", "floor": .10, "cap": .50, "step": .10},
    {"id": "D", "rotation": "price_plus_capex", "floor": .10, "cap": .70, "step": .10},
]
COSTS = [0, 50, 100]
BLOCKS = {"2014_2019": ("2014-02-28", "2019-12-31"),
          "2020_2022": ("2020-01-31", "2022-12-31"),
          "2023_2026": ("2023-01-31", "2026-07-31")}


def moving_block_bootstrap(x: pd.Series, n=2000, block=12, seed=8401):
    x = x.dropna().to_numpy(dtype=float); rng = np.random.default_rng(seed)
    if len(x) < block: return {"n": 0}
    totals = []
    starts = np.arange(len(x) - block + 1)
    for _ in range(n):
        sample=[]
        while len(sample) < len(x): sample.extend(x[rng.choice(starts):][:block])
        sample=np.asarray(sample[:len(x)])
        totals.append(float(np.prod(1+sample)-1))
    q=np.quantile(totals,[.025,.50,.975])
    return {"n": n, "block_months": block, "p025": float(q[0]), "median": float(q[1]), "p975": float(q[2]),
            "prob_positive": float(np.mean(np.asarray(totals)>0))}


def main():
    print("="*96); print("RC-4 SHORT-LIST GATE — PARTIAL THEME EXPOSURE"); print("="*96)
    R, spy = cycle2.cycle1.monthly_theme_returns(); months=R.index
    shares=cycle2.cycle1.point_capex_share(months)
    recession, _, _ = cycle2.cycle1.macro_recession(months)
    rows=[]; summaries=[]
    for cfg in CONFIGS:
        for cost in COSTS:
            W, net=cycle2.run_variant(R, spy, shares, recession, cfg["rotation"], cfg["floor"], cfg["cap"], cfg["step"], cost, False)
            overall=cycle2.stats(net, spy)
            record={**cfg,"cost_bps":cost,"overall_total":overall["total"],"overall_ann":overall["annualized"],"overall_dd":overall["max_dd"],
                    "mean_turnover":float(W.theme_turnover.mean()),"bootstrap":moving_block_bootstrap(net)}
            for block,(start,end) in BLOCKS.items():
                s=cycle2.stats(net,spy,start,end)
                record[f"{block}_total"]=s["total"]; record[f"{block}_dd"]=s["max_dd"]
                record[f"{block}_2022"]=s.get("2022",np.nan); record[f"{block}_2020"]=s.get("2020",np.nan)
            rows.append(record)
            print(f"{cfg['id']} {cfg['rotation']:>16} cap={cfg['cap']:.0%} step={cfg['step']:.0%} cost={cost:3d}bp "
                  f"total={overall['total']:+.1%} ann={overall['annualized']:+.1%} dd={overall['max_dd']:+.1%} "
                  f"2022={record['2020_2022_2022']:+.1%} boot+={record['bootstrap']['prob_positive']:.1%}")
    result=pd.DataFrame(rows); OUT_CSV.parent.mkdir(parents=True,exist_ok=True); result.to_csv(OUT_CSV,index=False)
    # Fixed descriptive gate checks; do not select a best cell.
    c50=result[result.cost_bps==50]
    checks={}
    for cid in [c["id"] for c in CONFIGS]:
        x=c50[c50.id==cid]
        checks[cid]={"positive_all_blocks":bool((x[[f"{b}_total" for b in BLOCKS]].to_numpy()>0).all()),
                     "maxdd_all_blocks_above_minus50":bool((x[[f"{b}_dd" for b in BLOCKS]].to_numpy()>-.50).all()),
                     "bootstrap_positive":float(x.bootstrap.map(lambda z:z["prob_positive"]).iloc[0])}
    summary={"script":"84_megatrend_partial_exposure_short_gate.py","status":"research_only",
             "contract":{"configs":CONFIGS,"costs_bps":COSTS,"blocks":BLOCKS,"bootstrap_block_months":12,"bootstrap_iterations":2000},
             "checks_at_50bps":checks,"records":rows,
             "conclusion":"short-list gate completed; no configuration promoted; manual watcher unchanged"}
    OUT_JSON.write_text(json.dumps(summary,indent=1,default=str))
    print("\n[CHECKS @ 50bps]")
    print(pd.DataFrame(checks).T.to_string())
    print("\n[STATUS] descriptive stability gate only; no operational allocation rule promoted.")
    print(f"[ARTIFACTS] {OUT_CSV.relative_to(ROOT.parent)}")
    print(f"            {OUT_JSON.relative_to(ROOT.parent)}")

if __name__ == "__main__": main()

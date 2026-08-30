#!/usr/bin/env python3
"""85_megatrend_partial_exposure_final_stress_gate.py — RC-4 benchmark/stress gate.

No new parameter search. Tests the four fixed shortlist configurations from
script 84 against SPY and static 60/40, with paired excess-return block
bootstrap, rolling 36-month stability, capex delay/missingness, and
leave-one-theme-out stress.

Research only. No operational or allocation change.
"""
from __future__ import annotations
import importlib.util
import itertools
import json
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
OUT_CSV=HERE/"archive"/"experiments"/"rc4_partial_exposure_final_stress_gate.csv"
OUT_JSON=HERE/"archive"/"experiments"/"rc4_partial_exposure_final_stress_gate_summary.json"
SRC=HERE/"83_megatrend_partial_exposure_cycle2_robustness.py"
spec=importlib.util.spec_from_file_location("cycle2",SRC); m=importlib.util.module_from_spec(spec);sys.modules["cycle2"]=m;assert spec.loader;spec.loader.exec_module(m)
CONFIGS=[{"id":"A","rotation":"price","floor":.10,"cap":.50,"step":.10},{"id":"B","rotation":"price","floor":.10,"cap":.50,"step":.15},{"id":"C","rotation":"price_plus_capex","floor":.10,"cap":.50,"step":.10},{"id":"D","rotation":"price_plus_capex","floor":.10,"cap":.70,"step":.10}]
COSTS=[50,100]; BLOCKS={"2014_2019":("2014-02-28","2019-12-31"),"2020_2022":("2020-01-31","2022-12-31"),"2023_2026":("2023-01-31","2026-07-31")}


def load_benchmarks(index):
    with pd.HDFStore(m.cycle1.DB_MT,"r") as s:
        def ret(sym): return s[f"/mt/{sym}"].set_index("date").adjClose.resample("ME").last().pct_change().reindex(index)
        spy=ret("SPY"); agg=ret("AGG")
    return spy,agg,.6*spy+.4*agg


def run_with_capex(R,shares,spy,recession,cfg,cost,capex_mode="normal",drop=None):
    # Reuse the fixed transition engine. Delayed/missing capex is applied before
    # target construction; price-only is unaffected.
    sh=shares.copy()
    if capex_mode.startswith("delay"):
        sh=sh.shift(int(capex_mode.replace("delay","")))
    elif capex_mode=="missing":
        sh.iloc[:]=np.nan
    if drop:
        keep=[t for t in R.columns if t!=drop]; RR=R[keep].copy(); SS=sh[keep].copy()
        # Two-theme stress uses equal feasible target construction through the
        # same bounded transition logic, preserving the fixed cap/step contract.
        prev=np.ones(len(keep))/len(keep); rows=[]; pnav=(1+RR.fillna(0)).cumprod(); tr=pnav/pnav.shift(12)-1
        for d in RR.index:
            if cfg["rotation"]=="price":
                order=list(tr.loc[d].dropna().sort_values(ascending=False).index); raw=np.ones(len(keep))/len(keep)
                if len(order)>=1:
                    raw=np.array([.6 if t==order[0] else .4/(len(keep)-1) for t in keep]) if len(keep)>1 else np.array([1.])
            else:
                ptarget=np.array([.6 if t in list(tr.loc[d].dropna().sort_values(ascending=False).index)[:1] else .4 for t in keep]);ptarget=ptarget/ptarget.sum()
                c=SS.loc[d].fillna(0).values.astype(float);ctarget=c/c.sum() if c.sum()>0 else np.ones(len(keep))/len(keep)
                raw=.5*ptarget+.5*ctarget
            # clip feasible bounds using the shared projection; cap/floor are
            # tightened if two assets cannot support the three-theme cap.
            floor=min(cfg["floor"],1/len(keep)); cap=max(cfg["cap"],1/len(keep))
            target=m.project_simplex(raw,floor,cap); w=m.bounded_step(prev,target,cfg["step"],floor,cap)
            turnover=.5*np.abs(w-prev).sum(); rows.append((d,w,turnover));prev=w
        W=pd.DataFrame([x[1] for x in rows],index=RR.index,columns=keep); turn=pd.Series([x[2] for x in rows],index=RR.index)
        gross=(W.shift(1)*RR).sum(axis=1); net=gross-turn.shift(1).fillna(0)*cost/10000
        return net.dropna(),W,turn
    return m.run_variant(R,spy,sh,recession,cfg["rotation"],cfg["floor"],cfg["cap"],cfg["step"],cost,False)[1],None,None


def block_bootstrap(excess,dd_diff,n=2000,block=12,seed=8501):
    x=excess.dropna().to_numpy(); y=dd_diff.dropna().to_numpy(); rng=np.random.default_rng(seed); starts=np.arange(len(x)-block+1); totals=[]; dds=[]
    if len(x)<block:return {"n":0}
    for _ in range(n):
        ix=[]
        while len(ix)<len(x): ix.extend(starts[rng.integers(0,len(starts)):][:block])
        z=x[np.array(ix[:len(x)])]; totals.append(np.prod(1+z)-1)
        path=np.cumprod(1+z); dds.append(np.min(path/np.maximum.accumulate(path)-1))
    q=np.quantile(totals,[.025,.5,.975]); dq=np.quantile(dds,[.025,.5,.975])
    return {"n":n,"block":block,"excess_p025":float(q[0]),"excess_median":float(q[1]),"excess_p975":float(q[2]),"prob_excess_positive":float(np.mean(np.array(totals)>0)),"boot_active_maxdd_p025":float(dq[0]),"boot_active_maxdd_median":float(dq[1]),"boot_active_maxdd_p975":float(dq[2])}


def report(net,bench):
    x=net.dropna(); b=bench.reindex(x.index).fillna(0); ex=x-b; nav=(1+x).cumprod(); bnav=(1+b).cumprod(); dd=nav/nav.cummax()-1; bdd=bnav/bnav.cummax()-1
    roll=(1+x).rolling(36).apply(np.prod,raw=True)-1; br=(1+b).rolling(36).apply(np.prod,raw=True)-1; rex=roll-br
    return {"total":float(nav.iloc[-1]-1),"benchmark_total":float(bnav.iloc[-1]-1),"excess_total":float(np.prod(1+ex)-1),"max_dd":float(dd.min()),"benchmark_dd":float(bdd.min()),"dd_diff":float(dd.min()-bdd.min()),"positive_excess_months":float((ex>0).mean()),"rolling36_positive":float(rex.dropna().gt(0).mean()) if len(rex.dropna()) else np.nan,"rolling36_min_excess":float(rex.min()) if len(rex.dropna()) else np.nan,"bootstrap":block_bootstrap(ex,dd-bdd)}


def main():
    print("="*100);print("RC-4 FINAL STRESS GATE — BENCHMARK / WALK-FORWARD / LEAVE-ONE-THEME-OUT");print("="*100)
    R,spy=m.cycle1.monthly_theme_returns();months=R.index;shares=m.cycle1.point_capex_share(months);rec,_,_=m.cycle1.macro_recession(months);spy,agg,static=load_benchmarks(months)
    records=[]; configs_summary=[]
    for cfg,cost in itertools.product(CONFIGS,COSTS):
        for mode in ["normal","delay3","delay6","delay12","missing"]:
            net,_,_=run_with_capex(R,shares,spy,rec,cfg,cost,mode)
            for bn,bench in [("SPY",spy),("static6040",static)]:
                s=report(net,bench); row={**cfg,"cost_bps":cost,"capex_mode":mode,"benchmark":bn,**{k:v for k,v in s.items() if k!="bootstrap"},"bootstrap":s["bootstrap"]}
                records.append(row)
                if mode=="normal" and bn=="SPY": configs_summary.append((cfg["id"],cost,s))
    # Leave-one-theme-out is deliberately limited to the strongest apparent D
    # candidate and 50bp: stress, not another optimization dimension.
    loo=[]; cfg=CONFIGS[-1]
    for drop in R.columns:
        net,_,_=run_with_capex(R,shares,spy,rec,cfg,50,"normal",drop=drop)
        s=report(net,spy);loo.append({"drop":drop,"benchmark":"SPY","cost_bps":50,**{k:v for k,v in s.items() if k!="bootstrap"},"bootstrap":s["bootstrap"]})
    result=pd.DataFrame(records);OUT_CSV.parent.mkdir(parents=True,exist_ok=True);result.to_csv(OUT_CSV,index=False)
    summary={"script":"85_megatrend_partial_exposure_final_stress_gate.py","status":"research_only","contract":{"configs":CONFIGS,"costs_bps":COSTS,"capex_modes":["normal","delay3","delay6","delay12","missing"],"benchmarks":["SPY","static6040"],"rolling_window_months":36,"bootstrap_block_months":12},"records":records,"leave_one_theme_out":loo,"conclusion":"benchmark/stress gate completed; no operational promotion"}
    OUT_JSON.write_text(json.dumps(summary,indent=1,default=str))
    print("[1] SPY benchmark, normal capex, 50bp")
    for cid,cost,s in configs_summary:
        if cost==50: print(f"  {cid}: active={s['total']:+.1%} SPY={s['benchmark_total']:+.1%} excess={s['excess_total']:+.1%} DD={s['max_dd']:+.1%} SPYDD={s['benchmark_dd']:+.1%} roll36+={s['rolling36_positive']:.1%} boot+={s['bootstrap']['prob_excess_positive']:.1%}")
    print("\n[2] capex stress, 50bp, SPY excess")
    z=result[(result.cost_bps==50)&(result.benchmark=="SPY")];print(z.groupby(["id","capex_mode"])["excess_total"].first().unstack().round(3).to_string())
    print("\n[3] leave-one-theme-out D, 50bp")
    print(pd.DataFrame(loo)[["drop","total","benchmark_total","excess_total","max_dd","rolling36_positive"]].to_string(index=False))
    print("\n[4] status: no configuration promoted; operational watcher unchanged.")
    print(f"[ARTIFACTS] {OUT_CSV.relative_to(ROOT.parent)}\n            {OUT_JSON.relative_to(ROOT.parent)}")

if __name__=="__main__":main()

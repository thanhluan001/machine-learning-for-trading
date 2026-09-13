#!/usr/bin/env python3
"""Final untouched holdout for frozen V6 gate-decomposition policy.

Development: rows through 2025-06-30.
Sweep/refit: rows through 2025-12-31.
Final test: 2026 H1 (rows after 2025-12-31 through 2026-06-30).

The V6 gate HPs and min-probability threshold are read from policy.json and
must not be changed after this holdout is inspected. V4 is evaluated with its
frozen production HP and theta on the same final test window.
"""
from __future__ import annotations
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
os.chdir(Path(__file__).resolve().parents[2])
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
HERE=Path(__file__).resolve().parent
OUT=HERE/'archive'/'experiments'/'gate_decomposition_v6'
POLICY=json.load(open(OUT/'policy.json',encoding='utf-8'))

def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
bt=load('bt_holdout',HERE/'51_hp_theta_sweep_23feat.py')
DB=bt.DB; MATRIX='/features/train_matrix_v4_timing_correct'; FEATURES=bt.DEPLOY_FEATURES
GATES=['pass_g1','pass_g2','pass_g3']
V4_HP={'gamma':3,'min_child_weight':100,'max_depth':2,'n_estimators':300,'learning_rate':0.05,'reg_lambda':1.0,'subsample':0.7,'colsample_bytree':0.7,'random_state':42}

def fit(X,y,Xe,ye,hp):
 import xgboost as xgb
 return xgb.XGBClassifier(objective='binary:logistic',eval_metric=['logloss','auc'],n_estimators=hp['n_estimators'],learning_rate=hp['learning_rate'],max_depth=hp['max_depth'],min_child_weight=hp['min_child_weight'],gamma=hp['gamma'],reg_lambda=hp['reg_lambda'],subsample=hp['subsample'],colsample_bytree=hp['colsample_bytree'],random_state=hp['random_state'],n_jobs=-1).fit(X,y,eval_set=[(Xe,ye)],verbose=False)

def summary(ex,all_test=None,raw=None):
 r=np.asarray(ex.pregap_return if len(ex) else [],float)
 out={'executed':int(len(r)),'wins':int((r>0).sum()),'losses':int((r<=0).sum()),'win_rate_pct':float((r>0).mean()*100) if len(r) else 0.,'avg_trade_pct':float(r.mean()*100) if len(r) else 0.,'median_trade_pct':float(np.median(r)*100) if len(r) else 0.,'avg_win_pct':float(r[r>0].mean()*100) if (r>0).any() else 0.,'avg_loss_pct':float(r[r<=0].mean()*100) if (r<=0).any() else 0.,'raw_picks':int(len(raw)) if raw is not None else 0,'raw_precision_pct':float(raw.pead_pass.mean()*100) if raw is not None and len(raw) else 0.}
 if len(ex):
  z=ex.copy(); iso=pd.to_datetime(z.entry_date).dt.isocalendar();z['week']=iso.year.astype(str)+'-W'+iso.week.astype(str).str.zfill(2);nav=1.
  for _,w in z.groupby('week',sort=True):nav*=1+float((w.pregap_return/bt.N_SLOTS).sum())
  out['nav_pct']=float((nav-1)*100);out['weeks']=int(z.week.nunique());out['weekly_returns']=z.groupby('week',sort=True).pregap_return.sum().div(bt.N_SLOTS).to_dict()
 else: out.update({'nav_pct':0.,'weeks':0,'weekly_returns':{}})
 return out

def select(raw):
 if raw.empty:return raw
 raw=raw.copy();raw['entry_date']=pd.to_datetime(raw.pregap_entry_date);raw['exit_date']=pd.to_datetime(raw.pregap_exit_date)
 return bt.select_weekly(raw,bt.N_SLOTS)

def boot_trade(rng,r,n=10000):
 r=np.asarray(r,float);idx=rng.integers(0,len(r),size=(n,len(r)));m=r[idx].mean(axis=1);w=(r[idx]>0).mean(axis=1)
 return {'avg_ci95_pct':[float(np.percentile(m,2.5)*100),float(np.percentile(m,97.5)*100)],'win_ci95_pct':[float(np.percentile(w,2.5)*100),float(np.percentile(w,97.5)*100)]}

def boot_week(rng,w,n=10000,block=4):
 a=np.asarray(list(w.values()),float);L=len(a);blocks=[a[i:min(i+block,L)] for i in range(L)];vals=[]
 for _ in range(n):
  q=[]
  while len(q)<L:q.extend(blocks[int(rng.integers(0,len(blocks)))])
  vals.append(np.prod(1+np.asarray(q[:L]))-1)
 return {'ci95_nav_pct':[float(np.percentile(vals,2.5)*100),float(np.percentile(vals,97.5)*100)],'block_weeks':block}

def default(v):
 if isinstance(v,(pd.Timestamp,np.datetime64)):return pd.Timestamp(v).isoformat()
 if isinstance(v,np.integer):return int(v)
 if isinstance(v,np.floating):return float(v)
 raise TypeError(type(v).__name__)

def main():
 print('='*100);print('V6 FINAL HOLDOUT: frozen policy, 2026 H1 untouched test');print('='*100)
 print('Policy:',POLICY['ensemble']); print('Gate HPs:',{g:POLICY['gate_models'][g] for g in GATES})
 df=pd.read_hdf(DB,MATRIX);rd=pd.to_datetime(df.report_date)
 train=df[rd<=pd.Timestamp('2025-06-30')].copy();sweep=df[(rd>pd.Timestamp('2025-06-30'))&(rd<=pd.Timestamp('2025-12-31'))].copy();test=df[(rd>pd.Timestamp('2025-12-31'))&(rd<=pd.Timestamp('2026-06-30'))].copy()
 print(f'rows train={len(train)} sweep={len(sweep)} final_test={len(test)}')
 all_train=pd.concat([train,sweep],ignore_index=True);v6=test.copy();
 for g in GATES:
  hp=POLICY['gate_models'][g];m=fit(all_train[FEATURES],all_train[g].astype(int),test[FEATURES],test[g].astype(int),hp);v6['p_'+g]=m.predict_proba(test[FEATURES])[:,1]
 v6['score']=v6[['p_pass_g1','p_pass_g2','p_pass_g3']].min(axis=1); mask=(v6.score>=POLICY['ensemble']['threshold'])&(~v6.sector.isin(POLICY['ensemble']['sector_exclusion']))&v6.pregap_return.notna();raw6=v6[mask].copy();raw6['p']=raw6.score;ex6=select(raw6)
 # V4 final holdout
 v4=fit(all_train[FEATURES],all_train.pead_pass.astype(int),test[FEATURES],test.pead_pass.astype(int),V4_HP);v4t=test.copy();v4t['p']=v4.predict_proba(test[FEATURES])[:,1];mask4=(v4t.p>=.20)&(~v4t.sector.isin(bt.EXCLUDE_SECTORS))&v4t.pregap_return.notna();raw4=v4t[mask4].copy();ex4=select(raw4)
 s6=summary(ex6,raw=raw6);s4=summary(ex4,raw=raw4);rng=np.random.default_rng(20260809)
 result={'model_version':'phase_g_v6_gate_decomposition','policy_file':str(OUT/'policy.json'),'holdout':{'train_end':'2025-06-30','sweep_end':'2025-12-31','test_start':'2026-01-01','test_end':'2026-06-30','rows':{'train':len(train),'sweep':len(sweep),'test':len(test)}},'v6':s6,'v4':s4,'bootstrap':{'v6_trade':boot_trade(rng,ex6.pregap_return.to_numpy(float)) if len(ex6) else {},'v4_trade':boot_trade(rng,ex4.pregap_return.to_numpy(float)) if len(ex4) else {},'v6_week_block4':boot_week(rng,s6['weekly_returns']) if s6['weekly_returns'] else {},'v4_week_block4':boot_week(rng,s4['weekly_returns']) if s4['weekly_returns'] else {}},'v6_raw_picks':raw6.to_dict(orient='records'),'v6_executed_trades':ex6.to_dict(orient='records'),'v4_raw_picks':raw4.to_dict(orient='records'),'v4_executed_trades':ex4.to_dict(orient='records')}
 with open(OUT/'final_holdout.json','w',encoding='utf-8') as f:json.dump(result,f,indent=2,default=default)
 print('\nV6:',json.dumps(s6,indent=2,default=default));print('\nV4:',json.dumps(s4,indent=2,default=default));print('\nBootstrap:',json.dumps(result['bootstrap'],indent=2,default=default));print('Saved',OUT/'final_holdout.json')
if __name__=='__main__':main()

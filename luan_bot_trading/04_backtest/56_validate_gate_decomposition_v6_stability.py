#!/usr/bin/env python3
"""V6 stability validation: fixed ensemble policies and dependence-aware tests.

Gate HPs are the HPs selected within each outer fold's preceding sweep window.
The policies below are pre-registered for this validation and are evaluated on
untouched outer test windows. No production artifacts or HDF5 writes.
"""
from __future__ import annotations
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
os.chdir(Path(__file__).resolve().parents[2])
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
HERE=Path(__file__).resolve().parent

def load(n,p):
 s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
bt=load('bt_stab', HERE/'51_hp_theta_sweep_23feat.py')
DB=bt.DB; MATRIX='/features/train_matrix_v4_timing_correct'; FEATURES=bt.DEPLOY_FEATURES
GATES=['pass_g1','pass_g2','pass_g3']; N_BOOT=10000; SEED=20260808
OUT=HERE/'archive'/'experiments'/'gate_decomposition_v6'
POLICIES={
 'adaptive_nested': None,
 'product_005': ('product',.05),
 'minimum_025': ('minimum',.25),
 'minimum_030': ('minimum',.30),
 'hard_025': ('hard',.25),
}

def split(df,fi):
 te,sve,tse=bt.DEFAULT_FOLDS[fi-1]; rd=pd.to_datetime(df.report_date)
 return df[rd<=pd.Timestamp(te)].copy(),df[(rd>pd.Timestamp(te))&(rd<=pd.Timestamp(sve))].copy(),df[(rd>pd.Timestamp(sve))&(rd<=pd.Timestamp(tse))].copy()

def fit(X,y,Xe,ye,hp):
 import xgboost as xgb
 return xgb.XGBClassifier(objective='binary:logistic',eval_metric=['logloss','auc'],n_estimators=hp['n_estimators'],learning_rate=.05,max_depth=hp['max_depth'],min_child_weight=hp['min_child_weight'],gamma=hp['gamma'],reg_lambda=1.,subsample=.7,colsample_bytree=.7,random_state=42,n_jobs=-1).fit(X,y,eval_set=[(Xe,ye)],verbose=False)

def trade_summary(x):
 r=np.asarray(x,dtype=float);r=r[np.isfinite(r)]
 if not len(r):return {'n':0,'win_rate_pct':0.,'avg_trade_pct':0.,'median_trade_pct':0.}
 return {'n':int(len(r)),'win_rate_pct':float((r>0).mean()*100),'avg_trade_pct':float(r.mean()*100),'median_trade_pct':float(np.median(r)*100),'avg_win_pct':float(r[r>0].mean()*100) if (r>0).any() else 0.,'avg_loss_pct':float(r[r<=0].mean()*100) if (r<=0).any() else 0.}

def reconstruct_v4(df):
 import xgboost as xgb
 rd=pd.to_datetime(df.report_date);parts=[]
 for fi,(te,sve,tse) in enumerate(bt.DEFAULT_FOLDS,1):
  tr=df[rd<=pd.Timestamp(te)];sv=df[(rd>pd.Timestamp(te))&(rd<=pd.Timestamp(sve))];ts=df[(rd>pd.Timestamp(sve))&(rd<=pd.Timestamp(tse))].copy();train=pd.concat([tr,sv],ignore_index=True)
  m=xgb.XGBClassifier(objective='binary:logistic',eval_metric=['logloss','auc'],learning_rate=.05,reg_lambda=1.,subsample=.7,colsample_bytree=.7,random_state=42,n_jobs=-1,gamma=3,min_child_weight=100,max_depth=2,n_estimators=300)
  m.fit(train[FEATURES],train.pead_pass.astype(int),eval_set=[(ts[FEATURES],ts.pead_pass.astype(int))],verbose=False);ts['p']=m.predict_proba(ts[FEATURES])[:,1]
  q=ts[(ts.p>=.2)&ts.pregap_return.notna()&(~ts.sector.isin(bt.EXCLUDE_SECTORS))].copy()
  if q.empty:continue
  q['entry_date']=pd.to_datetime(q.pregap_entry_date);q['exit_date']=pd.to_datetime(q.pregap_exit_date);q['fold']=fi;z=bt.select_weekly(q,bt.N_SLOTS)
  if not z.empty:parts.append(z)
 return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

def choose_take(x,rule,t):
 if rule=='product': take=x[['p1','p2','p3']].prod(axis=1)>=t
 elif rule=='minimum': take=x[['p1','p2','p3']].min(axis=1)>=t
 elif rule=='hard': take=(x[['p1','p2','p3']]>=t).all(axis=1)
 else: raise ValueError(rule)
 return take & (~x.sector.isin(bt.EXCLUDE_SECTORS)) & x.pregap_return.notna()

def selected_trades(test,rule,t,fi):
 raw=test[choose_take(test,rule,t)].copy()
 if raw.empty:return raw
 raw['p']=raw[['p1','p2','p3']].prod(axis=1) if rule=='product' else raw[['p1','p2','p3']].min(axis=1)
 raw['entry_date']=pd.to_datetime(raw.pregap_entry_date); raw['exit_date']=pd.to_datetime(raw.pregap_exit_date)
 raw['fold']=fi
 return bt.select_weekly(raw,bt.N_SLOTS)

def policy_stats(all_test,policy):
 pieces=[]
 for fi,x in all_test.items():
  rule,t=policy[fi] if isinstance(policy,dict) else policy
  q=selected_trades(x,rule,t,fi)
  if not q.empty:pieces.append(q)
 ex=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()
 s=trade_summary(ex.pregap_return if not ex.empty else [])
 s.update({'executed':int(len(ex)),'precision':float(ex.pead_pass.mean()*100) if len(ex) else 0.,'raw_picks':0})
 if not ex.empty:
  z=ex.copy(); iso=z.entry_date.dt.isocalendar();z['week']=iso.year.astype(str)+'-W'+iso.week.astype(str).str.zfill(2)
  nav=1.
  for _,w in z.groupby('week',sort=True):nav*=1+float((w.pregap_return/bt.N_SLOTS).sum())
  s['nav_pct']=float((nav-1)*100);s['weekly_returns']=z.groupby('week',sort=True).pregap_return.sum().div(bt.N_SLOTS).to_dict()
  fnav=[]
  for fi in range(1,5):
   q=z[z.fold==fi]; nf=1.
   if len(q):
    for _,w in q.groupby('week',sort=True):nf*=1+float((w.pregap_return/bt.N_SLOTS).sum())
   fnav.append((nf-1)*100)
  s['fold_navs_pct']=fnav;s['min_fold_nav_pct']=min(fnav)
 else:s.update({'nav_pct':0.,'weekly_returns':{},'fold_navs_pct':[0.,0.,0.,0.],'min_fold_nav_pct':0.})
 return s,ex

def week_returns(ex):
 if ex.empty:return {}
 z=ex.copy();z['entry_date']=pd.to_datetime(z.entry_date);iso=z.entry_date.dt.isocalendar();z['week']=iso.year.astype(str)+'-W'+iso.week.astype(str).str.zfill(2)
 return z.groupby('week',sort=True).pregap_return.sum().div(bt.N_SLOTS).to_dict()

def iid_week_ci(rng, vals, n=N_BOOT):
 a=np.asarray(list(vals.values()),float); idx=rng.integers(0,len(a),size=(n,len(a))); nav=np.prod(1+a[idx],axis=1)-1
 return {'weeks':int(len(a)),'estimate_nav_pct':float((np.prod(1+a)-1)*100),'ci95_nav_pct':[float(np.percentile(nav,2.5)*100),float(np.percentile(nav,97.5)*100)]}

def block_week_ci(rng, vals, block=4,n=N_BOOT):
 keys=sorted(vals);a=np.array([vals[k] for k in keys],float); L=len(a); blocks=[a[i:min(i+block,L)] for i in range(L)]
 samples=[]
 for _ in range(n):
  chunks=[]
  while len(chunks)<L:chunks.extend(blocks[int(rng.integers(0,len(blocks)))])
  samples.append(np.prod(1+np.array(chunks[:L]))-1)
 return {'block_weeks':block,'estimate_nav_pct':float((np.prod(1+a)-1)*100),'ci95_nav_pct':[float(np.percentile(samples,2.5)*100),float(np.percentile(samples,97.5)*100)]}

def trade_bootstrap(rng, ex, n=N_BOOT):
 r=np.asarray(ex.pregap_return if not ex.empty else [],float);r=r[np.isfinite(r)]
 if not len(r):return {'n':0}
 idx=rng.integers(0,len(r),size=(n,len(r)));means=r[idx].mean(axis=1);wins=(r[idx]>0).mean(axis=1)
 return {'n':int(len(r)),'avg_trade_pct':float(r.mean()*100),'avg_trade_ci95_pct':[float(np.percentile(means,2.5)*100),float(np.percentile(means,97.5)*100)],'win_rate_pct':float((r>0).mean()*100),'win_rate_ci95_pct':[float(np.percentile(wins,2.5)*100),float(np.percentile(wins,97.5)*100)]}

def paired_week_bootstrap(rng,a,b,block=1,n=N_BOOT):
 keys=sorted(set(a)|set(b));d=np.array([a.get(k,0.)-b.get(k,0.) for k in keys],float);L=len(d)
 if block==1: idx=rng.integers(0,L,size=(n,L)); x=d[idx].mean(axis=1)
 else:
  blocks=[d[i:min(i+block,L)] for i in range(L)];x=[]
  for _ in range(n):
   q=[]
   while len(q)<L:q.extend(blocks[int(rng.integers(0,len(blocks)))])
   x.append(np.mean(q[:L]))
  x=np.array(x)
 return {'weeks':L,'block_weeks':block,'estimate_delta_weekly_return_pp':float(d.mean()*100),'ci95_delta_pp':[float(np.percentile(x,2.5)*100),float(np.percentile(x,97.5)*100)],'prob_v6_gt':float((x>0).mean())}

def json_default(v):
 if isinstance(v,(pd.Timestamp,np.datetime64)):return pd.Timestamp(v).isoformat()
 if isinstance(v,np.integer):return int(v)
 if isinstance(v,np.floating):return float(v)
 raise TypeError(type(v).__name__)

def main():
 print('='*100);print('V6 STABILITY: fixed-policy and dependence-aware validation');print('='*100)
 saved=json.load(open(OUT/'nested_results.json',encoding='utf-8')); fold_saved=saved['outer_folds']
 df=pd.read_hdf(DB,MATRIX).reset_index(drop=False).rename(columns={'index':'_row_id'})
 all_test={}
 for fi in range(1,5):
  tr,sv,te=split(df,fi);x=te.copy();hpmap=fold_saved[fi-1]['gate_hp']
  train=pd.concat([tr,sv],ignore_index=True)
  for gate,pc in zip(GATES,['p1','p2','p3']):
   hp=hpmap[gate];m=fit(train[FEATURES],train[gate].astype(int),te[FEATURES],te[gate].astype(int),hp);x[pc]=m.predict_proba(te[FEATURES])[:,1]
  all_test[fi]=x
 adaptive={fi:(fold_saved[fi-1]['choice']['rule'],fold_saved[fi-1]['choice']['threshold']) for fi in range(1,5)}
 stats={}; trades={}
 for name,policy in POLICIES.items():
  p=adaptive if name=='adaptive_nested' else policy
  stats[name],trades[name]=policy_stats(all_test,p)
  print(name,stats[name])
 # Reconstructed v4 exactly.
 v4=reconstruct_v4(pd.read_hdf(DB,MATRIX)); v4w=week_returns(v4)
 v6=trades['adaptive_nested'];v6w=week_returns(v6)
 rng=np.random.default_rng(SEED)
 dependence={'v6_week_iid':iid_week_ci(rng,v6w),'v4_week_iid':iid_week_ci(rng,v4w),'v6_week_block4':block_week_ci(rng,v6w,4),'v4_week_block4':block_week_ci(rng,v4w,4),'paired_week_iid':paired_week_bootstrap(rng,v6w,v4w,1),'paired_week_block4':paired_week_bootstrap(rng,v6w,v4w,4)}
 fixed30=trades['minimum_030'];fixed30w=week_returns(fixed30)
 fixed30_validation={'trade_bootstrap':trade_bootstrap(rng,fixed30),'week_iid':iid_week_ci(rng,fixed30w),'week_block4':block_week_ci(rng,fixed30w,4),'paired_week_iid_vs_v4':paired_week_bootstrap(rng,fixed30w,v4w,1),'paired_week_block4_vs_v4':paired_week_bootstrap(rng,fixed30w,v4w,4)}
 overlap=pd.DataFrame()
 if not v6.empty and not v4.empty:
  a=v6.assign(_key=v6.permaTicker.astype(str)+'|'+pd.to_datetime(v6.entry_date).dt.strftime('%Y-%m-%d'))
  b=v4.assign(_key=v4.permaTicker.astype(str)+'|'+pd.to_datetime(v4.entry_date).dt.strftime('%Y-%m-%d'))
  overlap=a[[' _key']].copy() if False else a[['permaTicker','entry_date','pregap_return','fold','_key']].merge(b[['permaTicker','entry_date','pregap_return','fold','_key']],on='_key',suffixes=('_v6','_v4'))
 overlap_stats={'common_events':int(len(overlap))}
 if len(overlap):
  d=overlap.pregap_return_v6-overlap.pregap_return_v4;overlap_stats.update({'v6_avg_common_pct':float(overlap.pregap_return_v6.mean()*100),'v4_avg_common_pct':float(overlap.pregap_return_v4.mean()*100),'delta_pct_points':float(d.mean()*100),'v6_wins_common':int((overlap.pregap_return_v6>0).sum()),'v4_wins_common':int((overlap.pregap_return_v4>0).sum())})
 result={'model_version':'phase_g_v6_gate_decomposition','baseline':'phase_g_v4_timing_correct','policies':{k:{kk:vv for kk,vv in v.items() if kk!='weekly_returns'} for k,v in stats.items()},'dependence_bootstrap':dependence,'fixed_minimum_030_validation':fixed30_validation,'overlap':overlap_stats,'fold_policy_choices':adaptive,'pre_registered_fixed_policies':{k:v for k,v in POLICIES.items() if v is not None}}
 with open(OUT/'stability_validation.json','w',encoding='utf-8') as f:json.dump(result,f,indent=2,default=json_default)
 print('\nDEPENDENCE');print(json.dumps(dependence,indent=2,default=json_default));print('\nFIXED MINIMUM 0.30 VALIDATION');print(json.dumps(fixed30_validation,indent=2,default=json_default));print('\nOVERLAP',overlap_stats);print('Saved',OUT/'stability_validation.json')

if __name__=='__main__':main()

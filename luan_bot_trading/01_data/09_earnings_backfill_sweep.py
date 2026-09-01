"""09_earnings_backfill_sweep.py — full back-fill of passed null-actual rows.

One-time/periodic maintenance companion to 05b_alpaca_live/01 step 4.7
(which is lazy: scored set + held book only). This sweep covers the
WHOLE /earnings/fmp table — run it after each earnings season, or before
any training-matrix rebuild, so history is complete for research use.

2026-08-31 first run: 834 tickers, 602 quarters filled, 0 fetch errors.
Remaining nulls are FMP coverage gaps (mostly pre-2015) or FMP-side lag
on recent prints (the lazy 4.7 catches those as they matter).
"""
import os, time, urllib.request, urllib.parse, json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
load_dotenv('luan_bot_trading/.env')
KEY=os.getenv('FMP_API_KEY'); BASE='https://financialmodelingprep.com/stable/earnings'
DB='luan_bot_trading/01_data/db.h5'
with pd.HDFStore(DB,'r') as s:
    ev=s['/earnings/fmp']
    pt=s['/metadata/sp400_permatickers']
ev['report_date']=pd.to_datetime(ev['report_date'])
m=dict(zip(pt.permaTicker.astype(str), pt.canonical_ticker))
mask=(ev.report_date < pd.Timestamp.now()) & (ev.eps_actual.isna() | ev.eps_estimated.isna())
sub=ev[mask]
tickers=sorted({(str(r.permaTicker), m.get(str(r.permaTicker), r.canonical_ticker)) for r in sub.itertuples()})
print(f"null-actual passed rows: {len(sub)} across {len(tickers)} tickers")
filled=missed=0
t0=time.time()
for i,(ptt,ct) in enumerate(tickers):
    try:
        u=f"{BASE}?symbol={urllib.parse.quote(ct)}&includeReportTimes=true&apikey={KEY}"
        with urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent':'Mozilla/5.0'}), timeout=30) as r:
            rows=json.loads(r.read())
    except Exception:
        missed+=1; time.sleep(0.15); continue
    upd={pd.Timestamp(x['date']).normalize(): x for x in rows if x.get('date')}
    idxs=ev[(ev.permaTicker==ptt)&mask].index
    n=0
    for ix in idxs:
        hit=upd.get(pd.Timestamp(ev.loc[ix,'report_date']).normalize())
        if not hit: continue
        a,e=hit.get('epsActual'), hit.get('epsEstimated')
        if a is None or e is None: continue
        ev.loc[ix,'eps_actual']=float(a); ev.loc[ix,'eps_estimated']=float(e)
        for col,key in [('eps_difference','epsDifference'),('eps_surprise_pct','epsSurprisePct'),('revenue_actual','revenueActual'),('revenue_estimated','revenueEstimated')]:
            v=hit.get(key)
            if v is not None: ev.loc[ix,col]=v
        tm=(hit.get('time') or '').lower()
        if tm in ('bmo','amc'): ev.loc[ix,'before_after_market']=tm
        n+=1
    filled+=n
    if (i+1)%50==0: print(f"  [{i+1}/{len(tickers)}] filled so far {filled} ({time.time()-t0:.0f}s)")
    time.sleep(0.12)
still=((ev.report_date < pd.Timestamp.now()) & (ev.eps_actual.isna() | ev.eps_estimated.isna())).sum()
print(f"\nfilled {filled} quarters | fetch-errors {missed} | null-actual passed rows remaining: {still}")
if filled:
    with pd.HDFStore(DB,'a') as w:
        w.remove('/earnings/fmp')
        w.put('/earnings/fmp', ev, format='table', data_columns=['permaTicker','report_date'])
    print("table rewritten")

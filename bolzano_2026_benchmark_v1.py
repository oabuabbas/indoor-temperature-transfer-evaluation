"""Prospective-in-time 2026 benchmark with frozen configurations and nominal clock."""
import hashlib,json,platform,sys
from unittest.mock import patch
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from audit_nest import OUT
from ideal_development_v1 import frame_from_series,H
from thermbuild_gating import FEATURES
from bolzano_external_v1 import aggregate,evaluate,source_model
import bolzano_external_v1 as prior

ROOT=OUT.parent;START=pd.Timestamp('2026-01-01');END=pd.Timestamp('2026-05-01')
METHODS=['persistence','daily_seasonal','source_daily_ridge','local_ridge_30d','rolling_selector','fallback','local_hist_gradient_boosting']

def make_boosting():
    return HistGradientBoostingRegressor(loss='squared_error',max_iter=100,learning_rate=.1,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=20261029)

def add_boosting(d,s,choices):
    frame=frame_from_series(s.sort_index().asfreq('1h'));d=d.copy();d['local_hist_gradient_boosting']=np.nan
    for row in choices.itertuples(index=False):
        start=pd.Timestamp(row.month);hist=frame[(frame.index>=start-pd.Timedelta(days=30))&(frame.index+pd.Timedelta(hours=H)<start)]
        if len(hist)!=row.history_n:raise ValueError('History mismatch')
        if len(hist)<24:continue
        model=make_boosting();model.fit(hist[FEATURES].to_numpy(),(hist.target-hist.seasonal_yesterday).to_numpy())
        mask=d.month.eq(row.month)&d[['temp_now','temp_lag1','seasonal_yesterday']].notna().all(axis=1)
        x=d.loc[mask,['temp_now','temp_lag1','seasonal_yesterday']].to_numpy()
        if len(x):d.loc[mask,'local_hist_gradient_boosting']=d.loc[mask,'seasonal_yesterday'].to_numpy()+model.predict(x)
    return d

def score(d):
    target=d.target.to_numpy();obs=np.isfinite(target);common=obs.copy();own={}
    for name in METHODS:
        x=d[name].to_numpy();ready=np.isfinite(x);common &= ready;scored=ready&obs;err=x[scored]-target[scored]
        own[name]={'ready':int(ready.sum()),'scored':int(scored.sum()),'mae_c':float(np.abs(err).mean()) if len(err) else None}
    metrics=[]
    for name in METHODS:
        err=d.loc[common,name].to_numpy()-target[common]
        metrics.append({'model':name,'n':int(common.sum()),'mae_c':float(np.abs(err).mean()),'rmse_c':float(np.sqrt(np.square(err).mean()))})
    x=d.fallback.to_numpy();y=d.persistence.to_numpy();ok=obs&np.isfinite(x)&np.isfinite(y)
    paired={'n':int(ok.sum()),'fallback_mae_c':float(np.abs(x[ok]-target[ok]).mean()),'persistence_mae_c':float(np.abs(y[ok]-target[ok]).mean())}
    monthly=[]
    for month in d.month.unique():
        cm=common & d.month.eq(month).to_numpy();n=cm.sum()
        for name in METHODS:
            err=d.loc[cm,name].to_numpy()-target[cm]
            monthly.append({'month':month,'model':name,'n':int(n),'mae_c':float(np.abs(err).mean()) if n else None,'rmse_c':float(np.sqrt(np.square(err).mean())) if n else None})
    return {'opportunities':len(d),'observed_targets':int(obs.sum()),'common_n':int(common.sum()),'own_support':own,'common_metrics':metrics,'fallback_vs_persistence':paired,'branches':{str(k):int(v) for k,v in d.branch.value_counts().items()}},pd.DataFrame(monthly)

def main():
    protocol=ROOT/'protocol_bolzano_2026_benchmark_v1.md';out=OUT/'bolzano_2026_benchmark_v1_summary.json'
    if out.exists():raise ValueError('Completed 2026 benchmark protected')
    audit=json.loads((OUT/'bolzano_quality_v1.json').read_text(encoding='utf8'));parts=[];hashes={}
    for f in audit['files']:
        if f['year'] not in ['2025','2026']:continue
        path=ROOT/'data'/'bolzano'/f['file'];data=path.read_bytes();h=hashlib.sha256(data).hexdigest()
        if hashlib.md5(data).hexdigest()!=f['provider_md5'] or h!=f['sha256_local']:raise ValueError('Input checksum mismatch')
        hashes[f['file']]=h;parts.append(pd.read_csv(path,usecols=['timestamp','Temperature']))
    raw=pd.concat(parts,ignore_index=True);model=source_model();results={}
    for scenario in ['nominal_mean','hypothetical_rome']:
        hourly,meta=aggregate(raw,scenario)
        if scenario=='nominal_mean':
            old=pd.read_csv(OUT/'bolzano_external_v1_nominal_mean_hourly.csv.gz',parse_dates=['hour_start']).set_index('hour_start').temperature_c
            idx=pd.date_range('2025-02-01','2025-12-01',freq='h',inclusive='left')
            if not np.allclose(hourly.reindex(idx),old.reindex(idx),equal_nan=True,atol=1e-8):raise ValueError('2025 preprocessing differs from earlier run')
        with patch.object(prior,'START',START),patch.object(prior,'END',END):d,choices=evaluate(hourly,model)
        d=add_boosting(d,hourly,choices);s,monthly=score(d);results[scenario]={'preprocessing':meta,**s,'choices':choices.to_dict(orient='records')}
        d.to_csv(OUT/f'bolzano_2026_benchmark_v1_{scenario}_predictions.csv.gz',index=False,compression='gzip',float_format='%.10f')
        monthly.to_csv(OUT/f'bolzano_2026_benchmark_v1_{scenario}_monthly.csv',index=False,float_format='%.10f')
        choices.to_csv(OUT/f'bolzano_2026_benchmark_v1_{scenario}_choices.csv',index=False)
        print(scenario,json.dumps(s),flush=True)
    out.write_text(json.dumps({'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'input_sha256':hashes,'source_homes':55,'source_rows':125779,'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'platform':platform.platform()},'results':results,'scope':'Four later months of one room; clock unverified; locally fixed after previous results, no independent site inference.'},indent=2),encoding='utf8')
if __name__=='__main__':main()

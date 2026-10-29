"""Locally frozen single-room external evaluation; nominal time is not verified UTC."""
import hashlib,json
import numpy as np
import pandas as pd
from pathlib import Path
from audit_nest import OUT
from ideal_development_prepare import select
from ideal_development_v1 import load_sensor, frame_from_series, H, SOURCE_END
from adaptation_stress import fit,predict
from thermbuild_gating import FEATURES,CANDIDATES,predictions
from bremen_fallback_v1 import route

ROOT=OUT.parent
START=pd.Timestamp('2024-01-01'); END=pd.Timestamp('2026-01-01')
SCENARIOS=['nominal_mean','nominal_drop_conflict','hypothetical_rome']

def aggregate(raw,scenario):
    d=raw[['timestamp','Temperature']].copy(); d['t']=pd.to_datetime(d.timestamp,errors='coerce'); d['v']=pd.to_numeric(d.Temperature,errors='coerce')
    good=d.t.notna() & d.v.between(10,45);d=d[good].copy();removed_invalid=int((~good).sum())
    clock_removed=0
    if scenario=='hypothetical_rome':
        local=d.t.dt.tz_localize('Europe/Rome',ambiguous='NaT',nonexistent='NaT')
        clock_removed=int(local.isna().sum());d=d[local.notna()].copy();d['t']=local[local.notna()].dt.tz_convert('UTC').dt.tz_localize(None)
    elif scenario not in ['nominal_mean','nominal_drop_conflict']:raise ValueError('Unknown scenario')
    conflicts=d.groupby('t').v.transform('nunique')>1
    conflict_rows=int(conflicts.sum()); removed_conflict=0
    if scenario=='nominal_drop_conflict':removed_conflict=conflict_rows;d=d[~conflicts].copy()
    exact=d.groupby('t',sort=True).v.mean();five=exact.groupby(exact.index.floor('5min')).mean()
    hourly_group=five.groupby(five.index.floor('h'));hourly=hourly_group.mean().where(hourly_group.count()>=8)
    return hourly,{'input_rows':len(raw),'invalid_rows':removed_invalid,'clock_removed_rows':clock_removed,'conflict_rows_before_policy':conflict_rows,'conflict_removed_rows':removed_conflict,'distinct_five_min_bins':len(five),'valid_hours':int(hourly.notna().sum())}

def source_model():
    selected=select('development').reset_index(drop=True)
    if not selected.equals(pd.read_csv(OUT/'ideal_development_sensors_v1.csv')):raise ValueError('Source selection differs')
    parts=[]
    for x in selected[selected.dev_role.eq('source')].itertuples(index=False):
        d=load_sensor(int(x.sensor));d=d[d.index+pd.Timedelta(hours=H)<SOURCE_END]
        if len(d)>=24:parts.append(d)
    train=pd.concat(parts)
    if len(parts)!=55 or len(train)!=125779:raise ValueError('Source counts differ')
    return fit(train[FEATURES].to_numpy(),(train.target-train.seasonal_yesterday).to_numpy())

def evaluate(s,model):
    s=s.sort_index().asfreq('1h');frame=frame_from_series(s)
    out=[];choices=[]
    for start in pd.date_range(START,END,freq='MS',inclusive='left'):
        end=start+pd.offsets.MonthBegin(1)
        hist=frame[(frame.index>=start-pd.Timedelta(days=30)) & (frame.index+pd.Timedelta(hours=H)<start)]
        choice=None
        if len(hist)>=24:
            errors=predictions(hist,model).sub(hist.target,axis=0).abs().mean()
            choice=min(CANDIDATES,key=lambda k:errors[k])
        choices.append({'month':start.isoformat(),'history_n':len(hist),'choice':choice,'last_history_target_start':str(hist.index.max()+pd.Timedelta(hours=H)) if len(hist) else None})
        idx=pd.date_range(start,end-pd.Timedelta(hours=H),freq='h',inclusive='left')
        now=s.reindex(idx).to_numpy();lag=s.reindex(idx-pd.Timedelta(hours=1)).to_numpy();daily=s.reindex(idx-pd.Timedelta(hours=18)).to_numpy()
        target=s.reindex(idx+pd.Timedelta(hours=H)).to_numpy();features=np.column_stack([now,lag,daily]);good=np.isfinite(features).all(axis=1)
        src=np.full(len(idx),np.nan);selector=np.full(len(idx),np.nan);local=np.full(len(idx),np.nan)
        if good.any():src[good]=daily[good]+predict(model,features[good])
        if choice is not None:
            candidate={'persistence':now,'daily_seasonal':daily,'source_daily_ridge':src}[choice];selector[good]=candidate[good]
            lm=fit(hist[FEATURES].to_numpy(),(hist.target-hist.seasonal_yesterday).to_numpy())
            if good.any():local[good]=daily[good]+predict(lm,features[good])
        fb,branch=route(selector,now,daily)
        out.append(pd.DataFrame({'issue_hour_start':idx,'month':start.isoformat(),'choice':choice,'temp_now':now,'temp_lag1':lag,'seasonal_yesterday':daily,'target':target,'persistence':now,'daily_seasonal':daily,'source_daily_ridge':src,'local_ridge_30d':local,'rolling_selector':selector,'fallback':fb,'branch':branch}))
    return pd.concat(out,ignore_index=True),pd.DataFrame(choices)

def summarize(d,c):
    target=d.target.to_numpy();observed=np.isfinite(target);methods={}
    for name in ['persistence','daily_seasonal','source_daily_ridge','local_ridge_30d','rolling_selector','fallback']:
        p=d[name].to_numpy();ready=np.isfinite(p);scored=ready&observed
        methods[name]={'ready':int(ready.sum()),'scored':int(scored.sum()),'coverage':float(ready.mean()),'mae_c':float(np.abs(p[scored]-target[scored]).mean()) if scored.any() else None}
    def pair(a,b):
        x=d[a].to_numpy();y=d[b].to_numpy();ok=np.isfinite(x)&np.isfinite(y)&observed
        return {'n':int(ok.sum()),a+'_mae_c':float(np.abs(x[ok]-target[ok]).mean()) if ok.any() else None,b+'_mae_c':float(np.abs(y[ok]-target[ok]).mean()) if ok.any() else None}
    extra=np.isfinite(d.fallback)&~np.isfinite(d.persistence)&observed
    monthly=[]
    for month,g in d.groupby('month'):
        ok=g.target.notna()&g.fallback.notna()&g.persistence.notna()
        x=g.loc[ok];n=len(x)
        monthly.append({'month':month,'n':n,'fallback_mae_c':float((x.fallback-x.target).abs().mean()) if n else None,'persistence_mae_c':float((x.persistence-x.target).abs().mean()) if n else None})
    return {'opportunities':len(d),'observed_targets':int(observed.sum()),'methods':methods,'branches':{str(k):int(v) for k,v in d.branch.value_counts().items()},'choices':{str(k):int(v) for k,v in c.choice.fillna('insufficient_history').value_counts().items()},'fallback_vs_persistence':pair('fallback','persistence'),'selector_vs_source':pair('rolling_selector','source_daily_ridge'),'extra_fallback_scored':int(extra.sum()),'extra_fallback_mae_c':float((d.loc[extra,'fallback']-d.loc[extra,'target']).abs().mean()) if extra.any() else None},pd.DataFrame(monthly)

def main():
    proto=ROOT/'protocol_bolzano_external_v1.md';summary=OUT/'bolzano_external_v1_summary.json'
    if summary.exists():raise ValueError('Prior completed run protected')
    audit=json.loads((OUT/'bolzano_quality_v1.json').read_text(encoding='utf8'));parts=[]
    for f in audit['files']:
        path=ROOT/'data'/'bolzano'/f['file'];raw=path.read_bytes()
        if hashlib.md5(raw).hexdigest()!=f['provider_md5'] or hashlib.sha256(raw).hexdigest()!=f['sha256_local']:raise ValueError('File checksum mismatch')
        if f['year']!='2026':parts.append(pd.read_csv(path,usecols=['timestamp','Temperature']))
    raw=pd.concat(parts,ignore_index=True);model=source_model();results={}
    for name in SCENARIOS:
        h,meta=aggregate(raw,name);d,c=evaluate(h,model);metrics,m=summarize(d,c)
        pd.DataFrame({'hour_start':h.index,'temperature_c':h.values}).to_csv(OUT/f'bolzano_external_v1_{name}_hourly.csv.gz',index=False,compression='gzip',float_format='%.10f')
        d.to_csv(OUT/f'bolzano_external_v1_{name}_predictions.csv.gz',index=False,compression='gzip',float_format='%.10f')
        c.to_csv(OUT/f'bolzano_external_v1_{name}_choices.csv',index=False)
        m.to_csv(OUT/f'bolzano_external_v1_{name}_monthly.csv',index=False,float_format='%.10f')
        results[name]={'preprocessing':meta,**metrics};print(name,json.dumps(results[name]),flush=True)
    summary.write_text(json.dumps({'protocol_sha256':hashlib.sha256(proto.read_bytes()).hexdigest(),'source_homes':55,'source_rows':125779,'results':results,'scope':'One room; original clock timezone and DST unknown; monthly selections use past labels; no causal, population or energy inference.'},indent=2),encoding='utf8')
if __name__=='__main__':main()

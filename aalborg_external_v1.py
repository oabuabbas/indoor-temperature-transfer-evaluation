"""Fixed external benchmark on Aalborg six rooms, one building; no future labels in routing."""
import hashlib,json,platform
from unittest.mock import patch
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
from audit_nest import OUT
from ideal_development_v1 import frame_from_series,H
from thermbuild_gating import FEATURES
from ideal_nonlinear_tuning_v1 import GRID,model_for,sha
from bolzano_external_v1 import source_model
import bolzano_external_v1 as base
import bolzano_2026_benchmark_v1 as boost
ROOT=OUT.parent;RAW=ROOT/'data'/'aalborg'/'dataset__2023_02_27__2023_12_31.csv'
START=pd.Timestamp('2023-04-01',tz='UTC');END=pd.Timestamp('2024-01-01',tz='UTC')
ROOMS=[f'Room{x}' for x in 'ABCDEF'];METHODS=['persistence','daily_seasonal','source_daily_ridge','local_ridge_30d','rolling_selector','fallback','hgb_bolzano_fixed','hgb_ideal_selected']

def prepare(raw):
    t=pd.to_datetime(raw.timestamp,utc=True,errors='coerce');valid=t.notna();times=t[valid]
    if times.duplicated().any() or (times.diff().dt.total_seconds().dropna()<0).any():raise ValueError('Duplicate or reversed source timestamps')
    clean=raw.loc[valid].copy();clean['t']=times;output={};quality=[]
    for room in ROOMS:
        v=pd.to_numeric(clean[f'{room}:Sensor__room_temperature'],errors='coerce');good=v.between(10,45)
        selected=pd.DataFrame({'t':clean.t[good], 'v':v[good]}).set_index('t').v
        five=selected.groupby(selected.index.floor('5min')).mean();group=five.groupby(five.index.floor('h'));hour=group.mean().where(group.count()>=8)
        output[room]=hour.sort_index().asfreq('1h')
        quality.append({'room':room,'rows_with_valid_time':len(clean),'numeric_values':int(v.notna().sum()),'outside_range':int((v.notna()&~good).sum()),'valid_5min_bins':len(five),'valid_hours':int(hour.notna().sum())})
    return output,{'source_rows':len(raw),'missing_or_invalid_timestamps':int((~valid).sum()),'duplicate_valid_timestamps':int(times.duplicated().sum()),'first_utc':str(times.min()),'last_utc':str(times.max()),'rooms':quality}

def add_ideal(d,s,choice):
    frame=frame_from_series(s);out=d.copy();out['hgb_ideal_selected']=np.nan
    for month in out.month.unique():
        start=pd.Timestamp(month);hist=frame[(frame.index>=start-pd.Timedelta(days=30))&(frame.index+pd.Timedelta(hours=H)<start)]
        if len(hist)<24:continue
        model=model_for(choice).fit(hist[FEATURES].to_numpy(),(hist.target-hist.seasonal_yesterday).to_numpy())
        take=out.month.eq(month)&out[['temp_now','temp_lag1','seasonal_yesterday']].notna().all(axis=1);x=out.loc[take,FEATURES].to_numpy()
        if len(x):out.loc[take,'hgb_ideal_selected']=out.loc[take,'seasonal_yesterday'].to_numpy()+model.predict(x)
    return out

def metrics(d):
    observed=d.target.notna();common=observed&d[METHODS].notna().all(axis=1);roomrows=[];monthrows=[];own=[];pair=[]
    for room,g in d.groupby('room'):
        q=g[common.loc[g.index]]
        if len(q)==0:raise ValueError('No common support for room')
        for method in METHODS:
            e=q[method]-q.target;roomrows.append({'room':room,'model':method,'n':len(q),'abs_sum':float(e.abs().sum()),'square_sum':float((e**2).sum()),'mae_c':float(e.abs().mean()),'rmse_c':float(np.sqrt((e**2).mean()))})
            ready=g[method].notna();valid=ready&g.target.notna();own.append({'room':room,'model':method,'opportunities':len(g),'ready':int(ready.sum()),'scored':int(valid.sum()),'abs_sum':float((g.loc[valid,method]-g.loc[valid,'target']).abs().sum())})
        for month,x in q.groupby('month'):
            for method in METHODS:
                e=x[method]-x.target;monthrows.append({'room':room,'month':month,'model':method,'n':len(x),'abs_sum':float(e.abs().sum()),'mae_c':float(e.abs().mean())})
        p=g[g.target.notna()&g.fallback.notna()&g.persistence.notna()]
        pair.append({'room':room,'n':len(p),'fallback_abs_sum':float((p.fallback-p.target).abs().sum()),'persistence_abs_sum':float((p.persistence-p.target).abs().sum())})
    return pd.DataFrame(roomrows),pd.DataFrame(monthrows),pd.DataFrame(own),pd.DataFrame(pair)

def main():
    protocol=ROOT/'protocol_aalborg_external_v1.md';out=OUT/'aalborg_external_v1_summary.json'
    if out.exists():raise ValueError('Existing evaluation protected')
    data=RAW.read_bytes()
    if hashlib.md5(data).hexdigest()!='51b6528edbc2ccd0d78282a2e75668e1' or hashlib.sha256(data).hexdigest()!='1c9b6197294e89da864cfac69e0d25165ed95da27a17c318518d0a0e010c5464':raise ValueError('Source integrity mismatch')
    selection_path=OUT/'ideal_nonlinear_tuning_v1_choice.json';selection=json.loads(selection_path.read_text(encoding='utf8'))
    if selection['grid']!=GRID or selection['chosen_candidate']!='shallow' or sha(ROOT/'protocol_ideal_nonlinear_tuning_v1.md')!=selection['protocol_sha256'] or sha(OUT/'ideal_nonlinear_tuning_v1_dev_homes.csv')!=selection['home_output_sha256']:raise ValueError('Selected HGB differs')
    cols=['timestamp']+[f'{room}:Sensor__room_temperature' for room in ROOMS]
    raw=pd.read_csv(RAW,sep=';',encoding='latin1',usecols=cols);series,quality=prepare(raw);model=source_model();points=[];choices=[]
    with threadpool_limits(limits=1):
      for room,s in series.items():
        with patch.object(base,'START',START),patch.object(base,'END',END):d,c=base.evaluate(s,model)
        d=boost.add_boosting(d,s,c).rename(columns={'local_hist_gradient_boosting':'hgb_bolzano_fixed'})
        d=add_ideal(d,s,selection['chosen_candidate']);d.insert(0,'room',room);c.insert(0,'room',room)
        points.append(d);choices.append(c);print('Completed',room,flush=True)
    d=pd.concat(points,ignore_index=True);choices=pd.concat(choices,ignore_index=True);rooms,months,own,pair=metrics(d)
    aggregated=[]
    for method,g in rooms.groupby('model'):
        aggregated.append({'model':method,'rooms':len(g),'n':int(g.n.sum()),'equal_room_mae_c':float(g.mae_c.mean()),'equal_room_rmse_c':float(g.rmse_c.mean())})
    contrasts=[]
    for method in ['source_daily_ridge','local_ridge_30d','hgb_ideal_selected','hgb_bolzano_fixed','persistence']:
        v=rooms[rooms.model.eq('rolling_selector')].set_index('room').mae_c-rooms[rooms.model.eq(method)].set_index('room').mae_c
        contrasts.append({'comparator':method,'selector_better_rooms':int((v < -1e-8).sum()),'selector_worse_rooms':int((v>1e-8).sum()),'ties':int((v.abs()<=1e-8).sum())})
    d.to_csv(OUT/'aalborg_external_v1_predictions.csv.gz',index=False,compression='gzip',float_format='%.10f')
    for frame,label in [(rooms,'rooms'),(months,'monthly'),(own,'own_support'),(pair,'fallback_pair'),(choices,'choices')]:frame.to_csv(OUT/f'aalborg_external_v1_{label}.csv',index=False,float_format='%.10f')
    result={'protocol_sha256':sha(protocol),'source_sha256':hashlib.sha256(data).hexdigest(),'selected_candidate_sha256':sha(selection_path),'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__},'source_homes':55,'source_rows':125779,'source_quality':quality,'opportunities':len(d),'observed_targets':int(d.target.notna().sum()),'common_issue_times':int((d.target.notna()&d[METHODS].notna().all(axis=1)).sum()),'equal_room_metrics':aggregated,'selector_contrasts':contrasts,'fallback_pair':{'n':int(pair.n.sum()),'fallback_equal_room_mae_c':float((pair.fallback_abs_sum/pair.n).mean()),'persistence_equal_room_mae_c':float((pair.persistence_abs_sum/pair.n).mean())},'branches':{str(k):int(v) for k,v in d.branch.value_counts().items()},'scope':'One new office building, six correlated rooms; not six independent sites; prior methods chosen after other outcomes, local histories required.'}
    out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='source_quality'},indent=2),flush=True)
if __name__=='__main__':main()

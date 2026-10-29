"""Frozen policy evaluation on unseen Bremen office-room dataset."""
import hashlib,json
import numpy as np
import pandas as pd
from audit_nest import OUT
from adaptation_stress import fit,predict
from thermbuild_gating import FEATURES,CANDIDATES,predictions
from ideal_development_prepare import select
from ideal_development_v1 import load_sensor,frame_from_series,H,SOURCE_END
from audit_bremen import HOURLY,FILES,verify

START=pd.Timestamp('2023-03-01',tz='UTC')
END=pd.Timestamp('2024-01-01',tz='UTC')

def route(selector,now,daily):
    """Choose using only current/past prediction availability, never target."""
    out=np.full(len(now),np.nan);branch=np.full(len(now),'abstain',dtype='<U12')
    a=np.isfinite(selector);b=~a & np.isfinite(now);c=~a & ~b & np.isfinite(daily)
    out[a]=selector[a];branch[a]='selector'
    out[b]=now[b];branch[b]='persistence'
    out[c]=daily[c];branch[c]='daily'
    return out,branch

def main():
    protocol=OUT.parent/'protocol_bremen_fallback_v1.md';summary_path=OUT/'bremen_fallback_v1_summary.json'
    if not protocol.exists() or not (OUT/'bremen_quality.csv').exists():raise ValueError('Protocol or quality audit missing')
    if summary_path.exists():raise ValueError('External test results already exist; preserve original run')
    manifest=json.loads((OUT/'bremen_manifest.json').read_text(encoding='utf-8'))
    if manifest['commit']!='d83ff1e66d41ac8f77a44d5d94fb877b00b842eb':raise ValueError('Repository commit mismatch')
    for entry in manifest['files']:
        if hashlib.sha256((HOURLY/entry['hourly_file']).read_bytes()).hexdigest()!=entry['hourly_sha256_local']:
            raise ValueError('Hourly audit output changed')
    for name,(size,sha) in FILES.items():verify(OUT.parent/'data'/'bremen'/name,size,sha)
    selected=select('development').reset_index(drop=True)
    if not pd.read_csv(OUT/'ideal_development_sensors_v1.csv').equals(selected):raise ValueError('Source cohort differs')
    parts=[]
    for x in selected[selected.dev_role.eq('source')].itertuples(index=False):
        s=load_sensor(int(x.sensor));d=s[s.index+pd.Timedelta(hours=H)<SOURCE_END]
        if len(d)>=24:parts.append(d)
    if len(parts)!=55:raise ValueError('Source house count mismatch')
    source=pd.concat(parts)
    if len(source)!=125779:raise ValueError('Source row count mismatch')
    model=fit(source[FEATURES].to_numpy(),(source.target-source.seasonal_yesterday).to_numpy())
    month_rows=[];branch_rows=[];pair_rows=[];choice_rows=[]
    for filename in FILES:
        sensor=filename[:-4];table=pd.read_csv(HOURLY/f'{sensor}.csv.gz',parse_dates=['hour_utc'])
        if table.empty or table.hour_utc.duplicated().any():raise ValueError('Invalid hourly data')
        s=table.set_index('hour_utc').temp_c.sort_index().asfreq('1h');all_examples=frame_from_series(s)
        for start in pd.date_range(START,END,freq='MS',inclusive='left'):
            end=min(start+pd.offsets.MonthBegin(1),END)
            target_index=all_examples.index+pd.Timedelta(hours=H)
            history=all_examples[(all_examples.index>=start-pd.Timedelta(days=30))&(target_index<start)]
            choice=None
            if len(history)>=24:
                score=predictions(history,model).sub(history.target,axis=0).abs().mean()
                choice=min(CANDIDATES,key=lambda k:score[k])
            choice_rows.append({'sensor':sensor,'month':start.isoformat(),'history_n':len(history),'choice':choice})
            idx=pd.date_range(start,end-pd.Timedelta(hours=H),freq='h',inclusive='left')
            now=s.reindex(idx).to_numpy();lag1=s.reindex(idx-pd.Timedelta(hours=1)).to_numpy()
            daily=s.reindex(idx-pd.Timedelta(hours=18)).to_numpy()
            target=s.reindex(idx+pd.Timedelta(hours=H)).to_numpy()
            features=np.column_stack([now,lag1,daily]);good=np.isfinite(features).all(axis=1)
            source_pred=np.full(len(idx),np.nan)
            if good.any():source_pred[good]=daily[good]+predict(model,features[good])
            selector=np.full(len(idx),np.nan);local=np.full(len(idx),np.nan)
            if choice is not None:
                candidate={'persistence':now,'daily_seasonal':daily,'source_daily_ridge':source_pred}[choice]
                selector[good]=candidate[good]
                loc=fit(history[FEATURES].to_numpy(),(history.target-history.seasonal_yesterday).to_numpy())
                if good.any():local[good]=daily[good]+predict(loc,features[good])
            fallback,branch=route(selector,now,daily)
            methods={'persistence':now,'daily_seasonal':daily,'source_daily_ridge':source_pred,
                     'local_ridge_30d':local,'rolling_selector':selector,'fallback':fallback}
            observed=np.isfinite(target)
            for label,pred in methods.items():
                ready=np.isfinite(pred);scored=ready & observed
                err=np.abs(pred[scored]-target[scored])
                month_rows.append({'sensor':sensor,'month':start.isoformat(),'model':label,
                    'opportunities':len(idx),'ready':int(ready.sum()),'scored':int(scored.sum()),'abs_error_sum':float(err.sum())})
            for label in ['selector','persistence','daily','abstain']:
                take=branch==label;scored=take & observed
                branch_rows.append({'sensor':sensor,'month':start.isoformat(),'branch':label,'opportunities':int(take.sum()),
                    'scored':int(scored.sum()),'abs_error_sum':float(np.abs(fallback[scored]-target[scored]).sum())})
            common=np.isfinite(fallback) & np.isfinite(now) & observed
            pair_rows.append({'sensor':sensor,'month':start.isoformat(),'n':int(common.sum()),
                'fallback_abs_sum':float(np.abs(fallback[common]-target[common]).sum()),
                'persistence_abs_sum':float(np.abs(now[common]-target[common]).sum())})
    m=pd.DataFrame(month_rows); branches=pd.DataFrame(branch_rows);pair=pd.DataFrame(pair_rows);choice=pd.DataFrame(choice_rows)
    sensor=m.groupby(['sensor','model'],as_index=False).agg(opportunities=('opportunities','sum'),ready=('ready','sum'),scored=('scored','sum'),abs_error_sum=('abs_error_sum','sum'))
    sensor['coverage']=sensor.ready/sensor.opportunities
    sensor['mae_c']=np.divide(sensor.abs_error_sum,sensor.scored,out=np.full(len(sensor),np.nan),where=sensor.scored.to_numpy()>0)
    paired=pair.groupby('sensor',as_index=False).agg(n=('n','sum'),fallback_abs_sum=('fallback_abs_sum','sum'),persistence_abs_sum=('persistence_abs_sum','sum'))
    paired=paired[paired.n>0].copy();paired['fallback_mae_c']=paired.fallback_abs_sum/paired.n;paired['persistence_mae_c']=paired.persistence_abs_sum/paired.n
    stats={}
    for label,g in sensor.groupby('model'):
        stats[label]={'sensors_with_scores':int(g.mae_c.notna().sum()),'opportunities':int(g.opportunities.sum()),
                      'ready':int(g.ready.sum()),'scored':int(g.scored.sum()),
                      'opportunity_weighted_coverage':float(g.ready.sum()/g.opportunities.sum()),
                      'equal_sensor_mean_own_support_mae_c':float(g.mae_c.mean()) if g.mae_c.notna().any() else None}
    choice_counts=choice.choice.fillna('insufficient_history').value_counts().to_dict()
    summary={'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'source_homes':len(parts),'source_rows':len(source),
             'rooms':len(FILES),'room_months':len(choice),'choice_counts':{k:int(v) for k,v in choice_counts.items()},
             'methods_own_support':stats,'branches':{str(k):int(v) for k,v in branches.groupby('branch').opportunities.sum().items()},
             'paired_fallback_vs_persistence':{'rooms_with_common_observations':len(paired),'common_issue_times':int(paired.n.sum()),
                'equal_room_fallback_mae_c':float(paired.fallback_mae_c.mean()),'equal_room_persistence_mae_c':float(paired.persistence_mae_c.mean()),
                'fallback_better_rooms':int((paired.fallback_mae_c<paired.persistence_mae_c-1e-8).sum()),
                'fallback_worse_rooms':int((paired.fallback_mae_c>paired.persistence_mae_c+1e-8).sum()),
                'ties':int(((paired.fallback_mae_c-paired.persistence_mae_c).abs()<=1e-8).sum())},
             'scope':'Four correlated office rooms in one dataset; no population/site inference, energy or control outcome.'}
    m.to_csv(OUT/'bremen_fallback_v1_months.csv',index=False,float_format='%.8f')
    sensor.to_csv(OUT/'bremen_fallback_v1_sensors.csv',index=False,float_format='%.8f')
    branches.to_csv(OUT/'bremen_fallback_v1_branches.csv',index=False,float_format='%.8f')
    paired.to_csv(OUT/'bremen_fallback_v1_common.csv',index=False,float_format='%.8f')
    choice.to_csv(OUT/'bremen_fallback_v1_choices.csv',index=False)
    summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

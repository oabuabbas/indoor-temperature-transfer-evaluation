"""Compare filters on common issue times, development homes only."""
import numpy as np
import pandas as pd
from audit_nest import OUT
from adaptation_stress import fit,predict
from thermbuild_gating import FEATURES,CANDIDATES,predictions
from ideal_development_prepare import HOURLY,select
from ideal_development_v1 import frame_from_series,windows,H,SOURCE_END,TEST_START,TEST_END
from ideal_prepare_strict import STRICT

def load(directory,sensor):
    df=pd.read_csv(directory/f'{sensor}.csv.gz',parse_dates=['hour_utc'])
    if df.empty:return pd.DataFrame(columns=FEATURES+['target'],index=pd.DatetimeIndex([],tz='UTC'))
    if df.hour_utc.duplicated().any():raise ValueError('Duplicate hour')
    return frame_from_series(df.set_index('hour_utc').temp_c)

def main():
    selected=select()
    if not pd.read_csv(OUT/'ideal_development_sensors_v1.csv').equals(selected.reset_index(drop=True)):raise ValueError('Cohort changed')
    dirs={'10_45':HOURLY,'15_40':STRICT}
    for d in dirs.values():
        if any(not (d/f'{x.sensor}.csv.gz').exists() for x in selected.itertuples(index=False)):raise ValueError('Hourly data incomplete')
    frames={name:{int(x.home):load(folder,int(x.sensor)) for x in selected.itertuples(index=False)} for name,folder in dirs.items()}
    models={};sources=[]
    for name in dirs:
        pieces=[]
        for x in selected[selected.dev_role.eq('source')].itertuples(index=False):
            d=frames[name][int(x.home)];part=d[d.index+pd.Timedelta(hours=H)<SOURCE_END]
            if len(part)>=24:pieces.append(part)
        if not pieces:raise ValueError('No source training data')
        train=pd.concat(pieces);models[name]=fit(train[FEATURES].to_numpy(),(train.target-train.seasonal_yesterday).to_numpy())
        sources.append({'filter':name,'source_homes':len(pieces),'source_rows':len(train)})
    rows=[];target_differences=[]
    for x in selected[selected.dev_role.eq('validation')].itertuples(index=False):
        for start in pd.date_range(TEST_START,TEST_END,freq='MS',inclusive='left'):
            end=min(start+pd.offsets.MonthBegin(1),TEST_END)
            pair={name:windows(frames[name][int(x.home)],start,end) for name in dirs}
            common=pair['10_45'][1].index.intersection(pair['15_40'][1].index)
            if len(common)<24 or any(len(v[0])<24 for v in pair.values()):continue
            tw=pair['10_45'][1].loc[common,'target'].to_numpy()
            ts=pair['15_40'][1].loc[common,'target'].to_numpy()
            td=np.abs(tw-ts)
            target_differences.append({'home':int(x.home),'period_start':start.isoformat(),'n':len(common),
                                       'changed_n':int((td>1e-9).sum()),'abs_sum_c':float(td.sum())})
            for name,(hist,test) in pair.items():
                test=test.loc[common]
                prior=predictions(hist,models[name]).sub(hist.target,axis=0).abs().mean()
                choice=min(CANDIDATES,key=lambda c:prior[c])
                pred=predictions(test,models[name])
                local=fit(hist[FEATURES].to_numpy(),(hist.target-hist.seasonal_yesterday).to_numpy())
                pred['local_ridge_30d']=test.seasonal_yesterday.to_numpy()+predict(local,test[FEATURES].to_numpy())
                for label in CANDIDATES+['rolling_selector','local_ridge_30d']:
                    use=choice if label=='rolling_selector' else label
                    e=np.abs(pred[use].to_numpy()-test.target.to_numpy())
                    rows.append({'filter':name,'home':int(x.home),'sensor':int(x.sensor),'period_start':start.isoformat(),
                                 'model':label,'choice':choice,'history_n':len(hist),'test_n':len(test),
                                 'error_sum':float(e.sum()),'mae_c':float(e.mean())})
    if not rows:raise ValueError('No paired eligible windows')
    d=pd.DataFrame(rows)
    d.to_csv(OUT/'ideal_prediction_range_home_months.csv',index=False,float_format='%.8f')
    pd.DataFrame(target_differences).to_csv(OUT/'ideal_prediction_range_target_differences.csv',index=False,float_format='%.8f')
    pd.DataFrame(sources).to_csv(OUT/'ideal_prediction_range_source.csv',index=False)
    per=d.groupby(['filter','home','sensor','model'],as_index=False).agg(n=('test_n','sum'),error_sum=('error_sum','sum'),months=('period_start','nunique'))
    per['mae_c']=per.error_sum/per.n
    per.to_csv(OUT/'ideal_prediction_range_homes.csv',index=False,float_format='%.8f')
    summary=per.groupby(['filter','model'],as_index=False).agg(homes=('home','nunique'),mae_c=('mae_c','mean'))
    summary.to_csv(OUT/'ideal_prediction_range_summary.csv',index=False,float_format='%.8f')
    print(pd.DataFrame(sources).to_string(index=False));print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()

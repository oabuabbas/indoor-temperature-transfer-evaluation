"""Chronological source-to-unseen-development-house test. No holdout access."""
import numpy as np
import pandas as pd
from audit_nest import OUT
from adaptation_stress import fit
from thermbuild_gating import FEATURES,CANDIDATES,predictions
from ideal_development_prepare import HOURLY,select

H=6
SOURCE_END=pd.Timestamp('2017-09-01',tz='UTC')
TEST_START=pd.Timestamp('2017-10-01',tz='UTC')
TEST_END=pd.Timestamp('2018-06-30',tz='UTC')

def frame_from_series(s):
    s=s.sort_index().asfreq('1h')
    return pd.DataFrame({'temp_now':s,'temp_lag1':s.shift(1),'seasonal_yesterday':s.shift(24-H),
                         'target':s.shift(-H)}).dropna()

def load_sensor(sensor):
    d=pd.read_csv(HOURLY/f'{sensor}.csv.gz',parse_dates=['hour_utc'])
    if d.hour_utc.duplicated().any():raise ValueError('Duplicate hourly timestamp')
    return frame_from_series(d.set_index('hour_utc').temp_c)

def windows(frame,start,end):
    target=frame.index+pd.Timedelta(hours=H)
    return (frame[(frame.index>=start-pd.Timedelta(days=30)) & (target<start)],
            frame[(frame.index>=start)&(target<end)])

def main():
    selected=select()
    homes=set(selected.home)
    if not (OUT/'ideal_development_sensors_v1.csv').exists():raise ValueError('Prepare development data first')
    recorded=pd.read_csv(OUT/'ideal_development_sensors_v1.csv')
    if not recorded.equals(selected.reset_index(drop=True)):raise ValueError('Development selection changed')
    if any(not (HOURLY/f'{r.sensor}.csv.gz').exists() for r in selected.itertuples(index=False)):
        raise ValueError('Hourly preparation not complete')
    frames={int(r.home):load_sensor(int(r.sensor)) for r in selected.itertuples(index=False)}
    source=[]
    for r in selected[selected.dev_role.eq('source')].itertuples(index=False):
        d=frames[int(r.home)]
        eligible=d[d.index+pd.Timedelta(hours=H)<SOURCE_END]
        if len(eligible)>=24:source.append(eligible)
    if not source:raise ValueError('No source data before cutoff')
    train=pd.concat(source)
    source_model=fit(train[FEATURES].to_numpy(),(train.target-train.seasonal_yesterday).to_numpy())
    rows=[]
    starts=pd.date_range(TEST_START,TEST_END,freq='MS',inclusive='left')
    for r in selected[selected.dev_role.eq('validation')].itertuples(index=False):
        frame=frames[int(r.home)]
        for start in starts:
            end=min(start+pd.offsets.MonthBegin(1),TEST_END)
            validation,test=windows(frame,start,end)
            if min(len(validation),len(test))<24:continue
            v=predictions(validation,source_model).sub(validation.target,axis=0).abs().mean()
            choice=min(CANDIDATES,key=lambda c:v[c])
            errors=predictions(test,source_model).sub(test.target,axis=0).abs()
            for method in CANDIDATES+['rolling_selector']:
                use=choice if method=='rolling_selector' else method
                rows.append({'home':int(r.home),'sensor':int(r.sensor),'period_start':start.isoformat(),
                             'period_end':end.isoformat(),'model':method,'choice':choice,'selection_n':len(validation),
                             'test_n':len(test),'error_sum':float(errors[use].sum()),'mae_c':float(errors[use].mean())})
    if not rows:raise ValueError('No validation windows, no results produced')
    d=pd.DataFrame(rows)
    d.to_csv(OUT/'ideal_development_v1_home_months.csv',index=False,float_format='%.8f')
    per=d.groupby(['home','sensor','model'],as_index=False).agg(n=('test_n','sum'),error_sum=('error_sum','sum'),months=('period_start','nunique'))
    per['mae_c']=per.error_sum/per.n
    per.to_csv(OUT/'ideal_development_v1_homes.csv',index=False,float_format='%.8f')
    summ=per.groupby('model',as_index=False).agg(homes=('home','nunique'),mae_c=('mae_c','mean'))
    summ.to_csv(OUT/'ideal_development_v1_summary.csv',index=False,float_format='%.8f')
    choices=d[d.model.eq('rolling_selector')].groupby('choice').size().rename('home_months').reset_index()
    choices.to_csv(OUT/'ideal_development_v1_choices.csv',index=False)
    print('source homes',len(source),'source samples',len(train),'selected dev homes',len(homes))
    print(summ.to_string(index=False));print(choices.to_string(index=False),flush=True)

if __name__=='__main__':main()

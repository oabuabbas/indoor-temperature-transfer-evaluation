"""Exploratory rolling, past-only model selection in PLEIAData."""
import numpy as np
import pandas as pd
from audit_nest import ITEMS, OUT
from adaptation_stress import fit
from daily_residual import nest_source
from external_pleia import read_pleia
from thermbuild_gating import CANDIDATES, FEATURES, predictions

H=6
INTERVAL=pd.Timedelta(days=30)
FIRST=pd.Timestamp('2021-01-31',tz='UTC')
END=pd.Timestamp('2021-12-18',tz='UTC')

def make_frame(group):
    s=group.set_index('hour').temp.sort_index().asfreq('1h')
    return pd.DataFrame({'temp_now':s,'temp_lag1':s.shift(1),
                         'seasonal_yesterday':s.shift(24-H),'target':s.shift(-H)}).dropna()

def partition(frame,start,end):
    target=frame.index+pd.Timedelta(hours=H)
    return (frame[(frame.index>=start-INTERVAL)&(target<start)],
            frame[(frame.index>=start)&(target<end)])

def main():
    hour,metadata=read_pleia()
    source=pd.concat([nest_source(name,item[3],H) for unit,item in ITEMS.items()
                      for name in [item[2],unit.lower()+'_2020-2021.csv']])
    model=fit(source[FEATURES].to_numpy(),(source.target-source.seasonal_yesterday).to_numpy())
    rows=[]
    for sensor,group in hour.groupby('IDdevice'):
        frame=make_frame(group)
        for block,start in enumerate(pd.date_range(FIRST,END,freq=INTERVAL,inclusive='left'),1):
            end=min(start+INTERVAL,END)
            valid,test=partition(frame,start,end)
            if min(len(valid),len(test))<24:continue
            valid_pred=predictions(valid,model)
            valid_mae=valid_pred.sub(valid.target,axis=0).abs().mean()
            selected=min(CANDIDATES,key=lambda c:valid_mae[c])
            errors=predictions(test,model).sub(test.target,axis=0).abs()
            for method in CANDIDATES+['rolling_selector']:
                choice=selected if method=='rolling_selector' else method
                rows.append({'sensor':str(sensor),'building_block':str(metadata.loc[sensor,'block']),
                             'room':str(metadata.loc[sensor,'room']),'block':block,'start':start.isoformat(),
                             'end':end.isoformat(),'model':method,'selected':selected,
                             'validation_n':len(valid),'test_n':len(test),
                             'absolute_error_sum':errors[choice].sum(),'mae_c':errors[choice].mean()})
    d=pd.DataFrame(rows)
    if d.empty: raise ValueError('No eligible PLEIA sensor-blocks')
    d.to_csv(OUT/'pleia_rolling_room_blocks.csv',index=False,float_format='%.8f')
    sensor=d.groupby(['sensor','building_block','room','model'],as_index=False).agg(
        n=('test_n','sum'),error_sum=('absolute_error_sum','sum'),blocks=('block','nunique'))
    sensor['mae_c']=sensor.error_sum/sensor.n
    sensor.to_csv(OUT/'pleia_rolling_sensors.csv',index=False,float_format='%.8f')
    sums=[]
    for label,subset in [('all_eligible',sensor),('at_least_8_blocks',sensor[sensor.blocks>=8])]:
        for method,g in subset.groupby('model'):
            sums.append({'population':label,'model':method,'sensors':g.sensor.nunique(),'mae_c':g.mae_c.mean()})
    summary=pd.DataFrame(sums)
    summary.to_csv(OUT/'pleia_rolling_summary.csv',index=False,float_format='%.8f')
    chosen=d[d.model.eq('rolling_selector')].groupby('selected').size().rename('sensor_blocks').reset_index()
    chosen.to_csv(OUT/'pleia_rolling_choices.csv',index=False)
    print(summary.to_string(index=False));print('\nChoices\n',chosen.to_string(index=False),flush=True)

if __name__=='__main__':main()

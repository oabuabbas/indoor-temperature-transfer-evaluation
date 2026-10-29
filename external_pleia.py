"""NEST to PLEIA raw sensor zero-shot exploratory forecasting; no weather inputs."""
import json,zipfile
import numpy as np
import pandas as pd
from audit_nest import DATA,OUT,ITEMS
from adaptation_stress import fit,predict,quantile
from temporal_windows import window

HORIZONS=[1,6,24]
F=['temp_now','temp_lag1','hour_sin','hour_cos']
SRC_END=pd.Timestamp('2020-11-01')
CAL_START=pd.Timestamp('2021-01-01',tz='UTC'); CAL_END=pd.Timestamp('2021-02-01',tz='UTC')
TEST_END=pd.Timestamp('2021-12-18',tz='UTC')

def nest_data(unit,name,rooms,h):
    d=pd.read_csv(DATA/name,usecols=['time']+rooms,low_memory=False)
    t=pd.to_datetime(d.pop('time'),errors='coerce'); ok=t.notna() & ~t.duplicated(keep=False)
    d=d.loc[ok].apply(pd.to_numeric,errors='coerce'); d=d.where((d>=15)&(d<=40));d.index=t[ok]
    r=d.resample('1h'); v=r.mean().where(r.count()>=45); hour=v.index.hour.to_numpy()
    rows=[]
    for room in rooms:
        s=v[room]; frame=pd.DataFrame({'temp_now':s,'temp_lag1':s.shift(1),'hour_sin':np.sin(2*np.pi*hour/24),'hour_cos':np.cos(2*np.pi*hour/24),'target':s.shift(-h)},index=v.index)
        rows.append(frame.dropna())
    return pd.concat(rows)

def read_pleia(min_slots=4, temp_range=(15,40)):
    print('Reading PLEIA raw sensor data...',flush=True)
    parts=[]; counters={'raw_rows':0,'bad_time':0,'bad_temp_or_outside_range':0}
    with zipfile.ZipFile(DATA/'PLEIAData_zenodo_7620136.zip') as z:
        with z.open('Data_Nature/raw_data/relations-sensor.csv') as f:
            m=pd.read_csv(f,sep=';',dtype={'ID':'string','block':'string','room':'string'}).dropna().drop_duplicates()
        bad=m.groupby('ID').size(); conflicts=set(bad[bad>1].index)
        m=m[~m.ID.isin(conflicts)].set_index('ID')
        with z.open('Data_Nature/raw_data/data-sensor.csv') as f:
            for c in pd.read_csv(f,sep=';',usecols=['IDdevice','Date','V2'],dtype={'IDdevice':'string'},chunksize=200000):
                counters['raw_rows']+=len(c)
                c['time']=pd.to_datetime(c.Date,format='mixed',errors='coerce',utc=True)
                c['value']=pd.to_numeric(c.V2,errors='coerce')
                counters['bad_time']+=int(c.time.isna().sum())
                counters['bad_temp_or_outside_range']+=int((~c.value.between(*temp_range)).sum())
                c=c[c.IDdevice.isin(m.index)&c.time.notna()&c.value.between(*temp_range)].copy()
                c['slot']=c.time.dt.floor('10min')
                parts.append(c.groupby(['IDdevice','slot']).value.agg(['sum','count']).reset_index())
    slot=pd.concat(parts,ignore_index=True).groupby(['IDdevice','slot'],as_index=False)[['sum','count']].sum()
    slot['value']=slot['sum']/slot['count'];slot['hour']=slot.slot.dt.floor('h')
    hour=slot.groupby(['IDdevice','hour']).value.agg(['mean','count']).reset_index()
    hour=hour[hour['count']>=min_slots].copy();hour=hour.rename(columns={'mean':'temp','count':'observed_slots'})
    counters.update({'temperature_range_c':list(temp_range),'minimum_valid_slots_per_hour':min_slots,'conflicting_sensor_mappings':len(conflicts),'mapped_sensors':len(m),'valid_10min_slots':len(slot),'eligible_observed_hours':len(hour),'target_sensors_with_hours':int(hour.IDdevice.nunique())})
    filename=('pleia_raw_quality.json' if min_slots==4 else f'pleia_raw_quality_min{min_slots}.json') if temp_range==(15,40) else f'pleia_raw_quality_range{temp_range[0]}_{temp_range[1]}_min{min_slots}.json'
    (OUT/filename).write_text(json.dumps(counters,indent=2),encoding='utf-8')
    return hour,m

def forecast_for_sensor(group,h):
    s=group.set_index('hour').temp.sort_index().asfreq('1h')
    local=s.index.tz_convert('Europe/Madrid').hour.to_numpy()
    df=pd.DataFrame({'temp_now':s,'temp_lag1':s.shift(1),'hour_sin':np.sin(2*np.pi*local/24),'hour_cos':np.cos(2*np.pi*local/24),'target':s.shift(-h)},index=s.index)
    return df.dropna()

def main():
    hour,m=read_pleia(); results=[]; sensors=[]
    for h in HORIZONS:
        sources=[]
        for unit,item in ITEMS.items():
            for name in [item[2],unit.lower()+'_2020-2021.csv']:
                d=nest_data(unit,name,item[3],h)
                sources.append(d[d.index+pd.Timedelta(hours=h)<SRC_END])
        src=pd.concat(sources); model=fit(src[F].to_numpy(),(src.target-src.temp_now).to_numpy())
        for sensor,group in hour.groupby('IDdevice'):
            d=forecast_for_sensor(group,h); tt=d.index+pd.Timedelta(hours=h)
            cal=window(d,h,CAL_START,CAL_END,CAL_START-pd.Timedelta(hours=24)); test=window(d,h,CAL_END,TEST_END,CAL_END)
            if len(test)<24: continue
            meta=m.loc[sensor]
            for label,mod in [('persistence',None),('source_zero_fit',model)]:
                pcal=np.zeros(len(cal)) if mod is None else predict(mod,cal[F].to_numpy())
                ptest=np.zeros(len(test)) if mod is None else predict(mod,test[F].to_numpy())
                q=quantile(np.abs(cal.temp_now.to_numpy()+pcal-cal.target.to_numpy()))
                err=test.temp_now.to_numpy()+ptest-test.target.to_numpy()
                results.append({'horizon_h':h,'sensor':str(sensor),'block':str(meta.block),'room':str(meta.room),'model':label,'source_n':len(src),'cal_n':len(cal),'test_n':len(test),'mae_c':float(np.abs(err).mean()),'rmse_c':float(np.sqrt(np.square(err).mean())),'coverage90':float(np.mean(np.abs(err)<=q)) if np.isfinite(q) else np.nan,'width90_c':2*q})
        print('External horizon',h,'completed',flush=True)
    pd.DataFrame(results).round(6).to_csv(OUT/'pleia_external_results.csv',index=False)

if __name__=='__main__': main()

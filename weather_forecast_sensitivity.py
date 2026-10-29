"""Exploratory NEST->PLEIA transfer under two unverified weather clock hypotheses."""
import zipfile
import numpy as np
import pandas as pd
from audit_nest import ITEMS,DATA,OUT
from adaptation_stress import fit,predict,quantile
from external_pleia import read_pleia,forecast_for_sensor,F,SRC_END,HORIZONS,CAL_START,CAL_END,TEST_END
from temporal_windows import window

FW=F+['ambient_lag1','solar_lag1']
def lag_weather(hour):
    result=hour.rename(columns={'ambient':'ambient_lag1','solar':'solar_lag1'}).copy()
    result.index=result.index+pd.Timedelta(hours=1)
    return result

def source_frames(name,rooms,h):
    cols=['time','temp_amb','irrad']+rooms
    d=pd.read_csv(DATA/name,usecols=cols,low_memory=False)
    t=pd.to_datetime(d.pop('time'),errors='coerce');ok=t.notna() & ~t.duplicated(keep=False)
    d=d.loc[ok].apply(pd.to_numeric,errors='coerce');d.index=t[ok]
    d[rooms]=d[rooms].where(d[rooms].ge(15)&d[rooms].le(40))
    d.temp_amb=d.temp_amb.where(d.temp_amb.between(-40,60));d.irrad=d.irrad.where(d.irrad.between(0,1500))
    r=d.resample('1h');v=r.mean().where(r.count()>=45);hr=v.index.hour.to_numpy();rows=[]
    for room in rooms:
        s=v[room];a=pd.DataFrame({'temp_now':s,'temp_lag1':s.shift(1),'hour_sin':np.sin(2*np.pi*hr/24),'hour_cos':np.cos(2*np.pi*hr/24),'ambient_lag1':v.temp_amb.shift(1),'solar_lag1':v.irrad.shift(1),'target':s.shift(-h)},index=v.index).dropna()
        rows.append(a[a.index+pd.Timedelta(hours=h)<SRC_END])
    return pd.concat(rows)

def weather_hourly(hypothesis):
    with zipfile.ZipFile(DATA/'PLEIAData_zenodo_7620136.zip') as z:
        with z.open('Data_Nature/raw_data/MU62_dm.txt') as f: d=pd.read_csv(f,sep=';',skipinitialspace=True)
    t=pd.to_datetime(d.fecha.astype(str).str.strip()+' '+d.hora.astype(str).str.strip(),format='%d/%m/%y %H:%M:%S',errors='coerce')
    if hypothesis=='utc':t=t.dt.tz_localize('UTC')
    else:t=t.dt.tz_localize('Europe/Madrid',ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')
    a=pd.DataFrame({'time':t,'ambient':pd.to_numeric(d.tmed,errors='coerce'),'solar':pd.to_numeric(d.radmed,errors='coerce')})
    a=a[a.time.notna()&~a.time.duplicated(keep=False)].copy();a.ambient=a.ambient.where(a.ambient.between(-40,60));a.solar=a.solar.where(a.solar.between(0,1500))
    # One representative mean per absolute 10-minute slot; no imputation.
    a['slot']=a.time.dt.floor('10min');slot=a.groupby('slot')[['ambient','solar']].mean();r=slot.resample('1h');hour=r.mean().where(r.count()>=4)
    return lag_weather(hour)

def main():
    hour,meta=read_pleia();weather={key:weather_hourly(key) for key in ['utc','madrid']};out=[]
    for h in HORIZONS:
        src=pd.concat([source_frames(name,item[3],h) for unit,item in ITEMS.items() for name in [item[2],unit.lower()+'_2020-2021.csv']])
        y=(src.target-src.temp_now).to_numpy();no=fit(src[F].to_numpy(),y);yes=fit(src[FW].to_numpy(),y)
        for sensor,g in hour.groupby('IDdevice'):
            temp=forecast_for_sensor(g,h);joined={key:temp.join(w,how='inner').dropna(subset=FW+['target']) for key,w in weather.items()}
            common=joined['utc'].index.intersection(joined['madrid'].index)
            data={key:v.loc[common] for key,v in joined.items()}
            c={key:window(v,h,CAL_START,CAL_END,CAL_START-pd.Timedelta(hours=24)) for key,v in data.items()}
            t={key:window(v,h,CAL_END,TEST_END,CAL_END) for key,v in data.items()}
            if min(len(c['utc']),len(t['utc']))<24:continue
            methods=[('persistence','utc',None,F),('source_no_weather','utc',no,F),('source_weather_utc','utc',yes,FW),('source_weather_madrid','madrid',yes,FW)]
            for label,key,model,features in methods:
                ca=c[key];te=t[key];cp=np.zeros(len(ca)) if model is None else predict(model,ca[features].to_numpy());tp=np.zeros(len(te)) if model is None else predict(model,te[features].to_numpy())
                q=quantile(np.abs(ca.temp_now.to_numpy()+cp-ca.target.to_numpy()));err=te.temp_now.to_numpy()+tp-te.target.to_numpy()
                out.append({'horizon_h':h,'sensor':str(sensor),'block':str(meta.loc[sensor].block),'room':str(meta.loc[sensor].room),'model':label,'source_train_n':len(src),'cal_n':len(ca),'test_n':len(te),'mae_c':np.abs(err).mean(),'coverage90':np.mean(np.abs(err)<=q),'width90_c':2*q})
        print('weather horizon',h,'done',flush=True)
    pd.DataFrame(out).round(6).to_csv(OUT/'weather_forecast_sensitivity.csv',index=False)

if __name__=='__main__':main()

"""Exploratory multi-horizon evaluation using only inputs observed at forecast issue time."""
import numpy as np
import pandas as pd
from audit_nest import DATA, ITEMS, OUT
from adaptation_stress import COLS, fit, predict, adapt, quantile
from multiseason import SEASONS
from temporal_windows import window

HORIZONS = [1, 6, 24]
SOURCE_END = pd.Timestamp('2020-11-01')
CAL_START, CAL_END = pd.Timestamp('2020-12-01'), pd.Timestamp('2021-01-01')

def make_frames(unit,name,rooms,horizon):
    fields=['time','temp_amb','irrad']+rooms
    d=pd.read_csv(DATA/name,usecols=fields,low_memory=False)
    t=pd.to_datetime(d.pop('time'),errors='coerce')
    ok=t.notna() & ~t.duplicated(keep=False)
    d=d.loc[ok].apply(pd.to_numeric,errors='coerce'); d.index=t[ok]
    g=d.resample('1h'); mean=g.mean().where(g.count()>=45)
    hour=mean.index.hour.to_numpy(); entries=[]
    for room in rooms:
        temp=mean[room]
        frame=pd.DataFrame({'temp_now':temp,'temp_lag1':temp.shift(1),'ambient_now':mean.temp_amb,
           'solar_now':mean.irrad,'hour_sin':np.sin(2*np.pi*hour/24),'hour_cos':np.cos(2*np.pi*hour/24),
           'target':temp.shift(-horizon)},index=mean.index)
        frame['room']=room; frame['unit']=unit
        entries.append(frame.dropna())
    return pd.concat(entries)

def metrics(cal,test,model):
    cx=cal[COLS].to_numpy(); tx=test[COLS].to_numpy()
    cp=np.zeros(len(cal)) if model is None else predict(model,cx)
    tp=np.zeros(len(test)) if model is None else predict(model,tx)
    q=quantile(np.abs(cal.temp_now.to_numpy()+cp-cal.target.to_numpy()))
    err=test.temp_now.to_numpy()+tp-test.target.to_numpy()
    return {'mae_c':np.abs(err).mean(),'rmse_c':np.sqrt(np.square(err).mean()),
        'coverage90':np.mean(np.abs(err)<=q) if np.isfinite(q) else np.nan,'width90_c':2*q}

def main():
    results=[]
    for h in HORIZONS:
        pieces=[]
        for unit,item in ITEMS.items():
            for name in [item[2],unit.lower()+'_2020-2021.csv']:
                pieces.append(make_frames(unit,name,item[3],h))
        d=pd.concat(pieces); targettime=d.index+pd.Timedelta(hours=h)
        for unit in ITEMS:
            src=d[(d.unit!=unit)&(targettime<SOURCE_END)]
            sx=src[COLS].to_numpy(); sy=(src.target-src.temp_now).to_numpy(); source=fit(sx,sy)
            for room in d[d.unit==unit].room.unique():
                rr=d[(d.unit==unit)&(d.room==room)]; rt=rr.index+pd.Timedelta(hours=h)
                cal=window(rr,h,CAL_START,CAL_END,CAL_START)
                models={'persistence':None,'source_zero_fit':source}; counts={}
                for days in [7,30]:
                    a=rr[(rt>=SOURCE_END)&(rt<SOURCE_END+pd.Timedelta(days=days))]; counts[days]=len(a)
                    if len(a)>=24:
                        x=a[COLS].to_numpy(); y=(a.target-a.temp_now).to_numpy()
                        models[f'source_adapt_{days}d']=adapt(source,x,y)
                        models[f'target_only_{days}d']=fit(x,y)
                for season,(start,end) in SEASONS.items():
                    test=window(rr,h,start,end,CAL_END)
                    if len(cal)<24 or len(test)==0: continue
                    for label,model in models.items():
                        row={'horizon_h':h,'unit':unit,'room':room,'season':season,'model':label,
                             'source_n':len(src),'target_7d_n':counts[7],'target_30d_n':counts[30],
                             'cal_n':len(cal),'test_n':len(test)}
                        row.update(metrics(cal,test,model)); results.append(row)
            print('horizon',h,unit,'done',flush=True)
    pd.DataFrame(results).round(6).to_csv(OUT/'horizon_results.csv',index=False)

if __name__=='__main__': main()

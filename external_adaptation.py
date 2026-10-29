"""Exploratory cross-site few-shot adaptation against daily seasonal persistence."""
import sys
import numpy as np
import pandas as pd
from audit_nest import ITEMS, OUT
from adaptation_stress import fit,predict,adapt,quantile
from external_pleia import read_pleia,nest_data,forecast_for_sensor,F,SRC_END,HORIZONS
from temporal_windows import window

START=pd.Timestamp('2021-01-01',tz='UTC')
CAL_START=pd.Timestamp('2021-02-01',tz='UTC'); TEST_START=pd.Timestamp('2021-03-01',tz='UTC'); TEST_END=pd.Timestamp('2021-12-18',tz='UTC')
PERIODS={'all':(TEST_START,TEST_END),'spring':(TEST_START,pd.Timestamp('2021-06-01',tz='UTC')),
 'summer':(pd.Timestamp('2021-06-01',tz='UTC'),pd.Timestamp('2021-09-01',tz='UTC')),
 'autumn':(pd.Timestamp('2021-09-01',tz='UTC'),pd.Timestamp('2021-12-01',tz='UTC')),
 'dec_partial':(pd.Timestamp('2021-12-01',tz='UTC'),TEST_END)}

def make_sensor(group,h):
    df=forecast_for_sensor(group,h)
    s=group.set_index('hour').temp.sort_index().asfreq('1h')
    df['seasonal_yesterday']=s.shift(24-h)
    return df.dropna()

def model_delta(label,model,frame):
    if label=='persistence': return np.zeros(len(frame))
    if label=='daily_seasonal': return frame.seasonal_yesterday.to_numpy()-frame.temp_now.to_numpy()
    return predict(model,frame[F].to_numpy())

def main():
    min_slots=int(sys.argv[1]) if len(sys.argv)>1 else 4
    if min_slots not in (4,6): raise ValueError('min_slots must be 4 or 6')
    hour,metadata=read_pleia(min_slots=min_slots); output=[]
    for h in HORIZONS:
        pieces=[]
        for unit,item in ITEMS.items():
            for name in [item[2],unit.lower()+'_2020-2021.csv']:
                d=nest_data(unit,name,item[3],h)
                pieces.append(d[d.index+pd.Timedelta(hours=h)<SRC_END])
        src=pd.concat(pieces); source=fit(src[F].to_numpy(),(src.target-src.temp_now).to_numpy())
        for sensor,group in hour.groupby('IDdevice'):
            df=make_sensor(group,h);tt=df.index+pd.Timedelta(hours=h)
            cal=window(df,h,CAL_START,TEST_START,START+pd.Timedelta(days=30))
            if len(cal)<24: continue
            models={'persistence':None,'daily_seasonal':None,'source_zero_fit':source}; counts={}
            for days in (7,30):
                a=df[(tt>=START)&(tt<START+pd.Timedelta(days=days))];counts[days]=len(a)
                if len(a)>=24:
                    x=a[F].to_numpy();y=(a.target-a.temp_now).to_numpy()
                    models[f'source_adapt_{days}d']=adapt(source,x,y)
                    models[f'target_only_{days}d']=fit(x,y)
            for label,model in models.items():
                q=quantile(np.abs(cal.temp_now.to_numpy()+model_delta(label,model,cal)-cal.target.to_numpy()))
                for period,(begin,end) in PERIODS.items():
                    test=window(df,h,begin,end,TEST_START)
                    if len(test)<24:continue
                    err=test.temp_now.to_numpy()+model_delta(label,model,test)-test.target.to_numpy()
                    meta=metadata.loc[sensor]
                    output.append({'horizon_h':h,'sensor':str(sensor),'block':str(meta.block),'room':str(meta.room),'period':period,'model':label,'source_n':len(src),'adapt_7d_n':counts[7],'adapt_30d_n':counts[30],'cal_n':len(cal),'test_n':len(test),'mae_c':np.abs(err).mean(),'rmse_c':np.sqrt(np.square(err).mean()),'coverage90':np.mean(np.abs(err)<=q),'width90_c':2*q})
        print('horizon',h,'complete',flush=True)
    filename='pleia_adaptation_results.csv' if min_slots==4 else 'pleia_adaptation_min6_results.csv'
    pd.DataFrame(output).round(6).to_csv(OUT/filename,index=False)

if __name__=='__main__':main()

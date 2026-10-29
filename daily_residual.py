"""Target-site validation-gated residual corrections to previous-day reference."""
import numpy as np
import pandas as pd
from audit_nest import DATA,OUT,ITEMS
from adaptation_stress import fit,predict,adapt,quantile
from external_pleia import read_pleia
from external_adaptation import make_sensor

HORIZONS=[1,6,24]
FEATURES=['temp_now','temp_lag1','seasonal_yesterday','hour_sin','hour_cos']
SRC_END=pd.Timestamp('2020-11-01')
T0=pd.Timestamp('2021-01-01',tz='UTC'); T1=pd.Timestamp('2021-01-15',tz='UTC'); T2=pd.Timestamp('2021-01-31',tz='UTC')
CAL0=pd.Timestamp('2021-02-01',tz='UTC'); CAL1=pd.Timestamp('2021-03-01',tz='UTC'); END=pd.Timestamp('2021-12-18',tz='UTC')
PERIODS={'all':(CAL1,END),'spring':(CAL1,pd.Timestamp('2021-06-01',tz='UTC')),'summer':(pd.Timestamp('2021-06-01',tz='UTC'),pd.Timestamp('2021-09-01',tz='UTC')),'autumn':(pd.Timestamp('2021-09-01',tz='UTC'),pd.Timestamp('2021-12-01',tz='UTC')),'dec_partial':(pd.Timestamp('2021-12-01',tz='UTC'),END)}

def nest_source(name,rooms,h):
    d=pd.read_csv(DATA/name,usecols=['time']+rooms,low_memory=False)
    t=pd.to_datetime(d.pop('time'),errors='coerce'); ok=t.notna() & ~t.duplicated(keep=False)
    d=d.loc[ok].apply(pd.to_numeric,errors='coerce');d=d.where(d.ge(15)&d.le(40));d.index=t[ok]
    r=d.resample('1h');v=r.mean().where(r.count()>=45);hour=v.index.hour.to_numpy();rows=[]
    for room in rooms:
        s=v[room]
        a=pd.DataFrame({'temp_now':s,'temp_lag1':s.shift(1),'seasonal_yesterday':s.shift(24-h),'hour_sin':np.sin(2*np.pi*hour/24),'hour_cos':np.cos(2*np.pi*hour/24),'target':s.shift(-h)},index=v.index).dropna()
        rows.append(a[a.index+pd.Timedelta(hours=h)<SRC_END])
    return pd.concat(rows)

def pval(label,model,w,df):
    if label=='persistence':return df.temp_now.to_numpy()
    baseline=df.seasonal_yesterday.to_numpy()
    if label=='daily_seasonal': return baseline
    if label=='blend_selected': return baseline+w*predict(model,df[FEATURES].to_numpy())
    return baseline+predict(model,df[FEATURES].to_numpy())

def select_weight(source_model,validate):
    x=validate[FEATURES].to_numpy(); base=validate.seasonal_yesterday.to_numpy();delta=predict(source_model,x); y=validate.target.to_numpy()
    choices=[0.,0.25,0.5,0.75,1.]
    return min(choices,key=lambda w:(float(np.abs(base+w*delta-y).mean()),w))

def main():
    hour,metadata=read_pleia();output=[];selected=[];change_rows=[]
    for h in HORIZONS:
        src=pd.concat([nest_source(name,item[3],h) for unit,item in ITEMS.items() for name in [item[2],unit.lower()+'_2020-2021.csv']])
        source=fit(src[FEATURES].to_numpy(),(src.target-src.seasonal_yesterday).to_numpy())
        for sensor,group in hour.groupby('IDdevice'):
            d=make_sensor(group,h);tt=d.index+pd.Timedelta(hours=h)
            fitting=d[(tt>=T0)&(tt<T1)];val=d[(d.index>=T1)&(tt<T2)];cal=d[(d.index>=T2)&(tt>=CAL0)&(tt<CAL1)]
            if min(len(fitting),len(val),len(cal))<24:continue
            x=fitting[FEATURES].to_numpy();y=(fitting.target-fitting.seasonal_yesterday).to_numpy()
            adjusted=adapt(source,x,y);target=fit(x,y);w=select_weight(adjusted,val)
            meta=metadata.loc[sensor];selected.append({'horizon_h':h,'sensor':str(sensor),'block':str(meta.block),'room':str(meta.room),'weight':w,'fit_n':len(fitting),'validation_n':len(val),'cal_n':len(cal)})
            models={'persistence':None,'daily_seasonal':None,'source_residual':source,'adapted_residual':adjusted,'target_residual':target,'blend_selected':adjusted}
            for label,model in models.items():
                q=quantile(np.abs(pval(label,model,w,cal)-cal.target.to_numpy()))
                for period,(start,end) in PERIODS.items():
                    test=d[(d.index>=CAL1)&(tt>=start)&(tt<end)]
                    if len(test)<24:continue
                    error=pval(label,model,w,test)-test.target.to_numpy()
                    output.append({'horizon_h':h,'sensor':str(sensor),'block':str(meta.block),'room':str(meta.room),'model':label,'period':period,'weight':w,'source_n':len(src),'fit_n':len(fitting),'validation_n':len(val),'cal_n':len(cal),'test_n':len(test),'mae_c':np.abs(error).mean(),'rmse_c':np.sqrt(np.square(error).mean()),'coverage90':np.mean(np.abs(error)<=q),'width90_c':2*q})
                    if period=='all':
                        # Retrospective outcome-based strata: not available at forecast issue time.
                        change=np.abs(test.target.to_numpy()-test.temp_now.to_numpy())
                        for label_bin,lo,hi in [('under_0.5',0.,.5),('0.5_to_1',.5,1.),('1_to_2',1.,2.),('at_least_2',2.,np.inf)]:
                            take=(change>=lo)&(change<hi)
                            if take.sum()<24:continue
                            ee=error[take]
                            change_rows.append({'horizon_h':h,'sensor':str(sensor),'block':str(meta.block),'room':str(meta.room),'model':label,'outcome_change_bin_c':label_bin,'test_n':int(take.sum()),'mae_c':np.abs(ee).mean(),'coverage90':np.mean(np.abs(ee)<=q)})
        print('horizon',h,'complete',flush=True)
    pd.DataFrame(output).round(6).to_csv(OUT/'daily_residual_results.csv',index=False)
    pd.DataFrame(selected).to_csv(OUT/'daily_residual_selected_weights.csv',index=False)
    pd.DataFrame(change_rows).round(6).to_csv(OUT/'daily_residual_by_observed_change.csv',index=False)

if __name__=='__main__':main()

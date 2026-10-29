"""Exploratory leakage-aware cross-unit few-shot and sensor-stress evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
from audit_nest import ITEMS, OUT, TEST_START, TEST_END
from transfer_pilot import frames

COLS = ['temp_now','temp_lag1','ambient_now','solar_now','hour_sin','hour_cos']
SOURCE_END = pd.Timestamp('2020-01-01')
ADAPT_START = pd.Timestamp('2020-01-01')
CAL_START, CAL_END = pd.Timestamp('2020-02-01'), pd.Timestamp('2020-03-01')
RIDGE, ADAPT_PENALTY = 100., 100.

def fit(x, y, center=None, penalty=RIDGE):
    # Intercept is regularized only in adapted fits to keep the solution near source.
    mu=x.mean(axis=0); sd=x.std(axis=0); sd[sd < 1e-8]=1
    z=np.column_stack([np.ones(len(x)),(x-mu)/sd])
    reg=np.diag([0.]+[penalty]*x.shape[1])
    b=np.linalg.solve(z.T@z+reg,z.T@y)
    return mu,sd,b

def predict(model, x):
    mu,sd,b=model
    return np.column_stack([np.ones(len(x)),(x-mu)/sd])@b

def adapt(source, x, y):
    mu,sd,b=source
    z=np.column_stack([np.ones(len(x)),(x-mu)/sd])
    beta=np.linalg.solve(z.T@z+ADAPT_PENALTY*np.eye(z.shape[1]),z.T@y+ADAPT_PENALTY*b)
    return mu,sd,beta

def quantile(scores):
    if len(scores)<24: return np.nan
    k=min(len(scores), int(np.ceil((len(scores)+1)*0.9)))
    return float(np.partition(scores,k-1)[k-1])

def main():
    data=pd.concat([frames(unit,item[2],item[3]) for unit,item in ITEMS.items()])
    target_time=data.index+pd.Timedelta(hours=1)
    output=[]
    for unit in ITEMS:
        src=data[(data.unit!=unit)&(target_time<SOURCE_END)]
        sx=src[COLS].to_numpy(); sy=(src.target-src.temp_now).to_numpy()
        source=fit(sx,sy)
        for room in data[data.unit==unit].room.unique():
            rr=data[(data.unit==unit)&(data.room==room)]
            rt=rr.index+pd.Timedelta(hours=1)
            cal=rr[(rt>=CAL_START)&(rt<CAL_END)]
            test=rr[(rt>=TEST_START)&(rt<TEST_END)]
            models={'persistence':None,'source_zero_fit':source}
            counts={}
            for days in [7,30]:
                a=rr[(rt>=ADAPT_START)&(rt<ADAPT_START+pd.Timedelta(days=days))]
                counts[days]=len(a)
                if len(a)>=24:
                    ax=a[COLS].to_numpy(); ay=(a.target-a.temp_now).to_numpy()
                    models[f'target_only_{days}d']=fit(ax,ay)
                    models[f'adapt_source_{days}d']=adapt(source,ax,ay)
            if len(test)==0: continue
            cx=cal[COLS].to_numpy(); tx=test[COLS].to_numpy()
            for name,model in models.items():
                cdelta=np.zeros(len(cal)) if model is None else predict(model,cx)
                calibration=np.abs(cal.temp_now.to_numpy()+cdelta-cal.target.to_numpy())
                q=quantile(calibration)
                for scenario in ['observed','missing_ambient','missing_solar']:
                    xx=tx.copy()
                    if model is not None and scenario!='observed':
                        pos=2 if scenario=='missing_ambient' else 3
                        xx[:,pos]=model[0][pos]
                    delta=np.zeros(len(test)) if model is None else predict(model,xx)
                    error=test.temp_now.to_numpy()+delta-test.target.to_numpy()
                    output.append({'unit':unit,'room':room,'model':name,'scenario':scenario,'source_train_n':len(src),'target_7d_n':counts.get(7,0),'target_30d_n':counts.get(30,0),'cal_n':len(cal),'test_n':len(test),'mae_c':np.abs(error).mean(),'rmse_c':np.sqrt(np.square(error).mean()),'nominal_90_coverage':np.mean(np.abs(error)<=q) if np.isfinite(q) else np.nan,'interval_width_c':2*q})
        print(unit,'complete',flush=True)
    pd.DataFrame(output).round(6).to_csv(OUT/'adaptation_stress.csv',index=False)

if __name__=='__main__': main()

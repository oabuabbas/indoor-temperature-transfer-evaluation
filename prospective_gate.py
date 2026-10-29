"""Exploratory risk-based selection using information available at forecast issue."""
import numpy as np
import pandas as pd
from audit_nest import ITEMS,OUT
from adaptation_stress import fit,predict,adapt,quantile
from external_pleia import read_pleia
from external_adaptation import make_sensor
from daily_residual import HORIZONS,FEATURES,nest_source,T0,T1,T2,CAL0,CAL1,PERIODS

def forecasts(df,model):
    daily=df.seasonal_yesterday.to_numpy()
    return {'persistence':df.temp_now.to_numpy(),'daily':daily,
            'adapted':daily+predict(model,df[FEATURES].to_numpy())}

def scores(df,preds):
    return {'recent_change':np.abs(df.temp_now.to_numpy()-df.temp_lag1.to_numpy()),
            'daily_gap':np.abs(df.temp_now.to_numpy()-df.seasonal_yesterday.to_numpy()),
            'correction':np.abs(preds['adapted']-preds['daily'])}

def apply_gate(policy,preds,score_values):
    if policy['type']=='constant':
        label=policy['label'];return preds[label],np.full(len(preds[label]),label=='adapted',dtype=bool)
    take=score_values[policy['score']]<=policy['threshold']
    return np.where(take,preds['adapted'],preds[policy['fallback']]),take

def choose_policy(df,model):
    preds=forecasts(df,model);sv=scores(df,preds); y=df.target.to_numpy()
    candidates=[{'type':'constant','label':x} for x in ('persistence','daily','adapted')]
    for fallback in ('persistence','daily'):
        for name,score in sv.items():
            for q in (.25,.5,.75):
                candidates.append({'type':'threshold','fallback':fallback,'score':name,'quantile':q,'threshold':float(np.quantile(score,q))})
    errors=[float(np.abs(apply_gate(c,preds,sv)[0]-y).mean()) for c in candidates]
    best=int(np.argmin(errors))
    return candidates[best],errors[best]

def main():
    hour,metadata=read_pleia();output=[];choices=[]
    for h in HORIZONS:
        src=pd.concat([nest_source(name,item[3],h) for unit,item in ITEMS.items() for name in [item[2],unit.lower()+'_2020-2021.csv']])
        source=fit(src[FEATURES].to_numpy(),(src.target-src.seasonal_yesterday).to_numpy())
        for sensor,group in hour.groupby('IDdevice'):
            df=make_sensor(group,h);tt=df.index+pd.Timedelta(hours=h)
            fitting=df[(tt>=T0)&(tt<T1)];val=df[(df.index>=T1)&(tt<T2)];cal=df[(df.index>=T2)&(tt>=CAL0)&(tt<CAL1)]
            if min(len(fitting),len(val),len(cal))<24: continue
            model=adapt(source,fitting[FEATURES].to_numpy(),(fitting.target-fitting.seasonal_yesterday).to_numpy())
            selected,validation_mae=choose_policy(val,model);meta=metadata.loc[sensor]
            choices.append({'horizon_h':h,'sensor':str(sensor),'block':str(meta.block),'room':str(meta.room),'policy':str(selected),'validation_mae':validation_mae,'validation_n':len(val)})
            cpre=forecasts(cal,model);csv=scores(cal,cpre)
            rules={'persistence':{'type':'constant','label':'persistence'},'daily':{'type':'constant','label':'daily'},'adapted':{'type':'constant','label':'adapted'},'validation_gate':selected}
            for label,rule in rules.items():
                cpred,_=apply_gate(rule,cpre,csv); q=quantile(np.abs(cpred-cal.target.to_numpy()))
                for period,(begin,end) in PERIODS.items():
                    test=df[(df.index>=CAL1)&(tt>=begin)&(tt<end)]
                    if len(test)<24:continue
                    pr=forecasts(test,model);out,take=apply_gate(rule,pr,scores(test,pr));err=out-test.target.to_numpy()
                    output.append({'horizon_h':h,'sensor':str(sensor),'block':str(meta.block),'room':str(meta.room),'policy':label,'period':period,'test_n':len(test),'fit_n':len(fitting),'validation_n':len(val),'cal_n':len(cal),'mae_c':np.abs(err).mean(),'rmse_c':np.sqrt(np.square(err).mean()),'coverage90':np.mean(np.abs(err)<=q),'width90_c':2*q,'adapted_fraction':take.mean()})
        print('horizon',h,'complete',flush=True)
    pd.DataFrame(output).round(6).to_csv(OUT/'prospective_gate_results.csv',index=False)
    pd.DataFrame(choices).to_csv(OUT/'prospective_gate_choices.csv',index=False)

if __name__=='__main__':main()

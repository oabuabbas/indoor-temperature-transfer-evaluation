"""Matched-time sensitivity to target raw measurement filter; exploratory."""
import numpy as np
import pandas as pd
from audit_nest import ITEMS,OUT
from adaptation_stress import fit,predict,quantile
from external_pleia import read_pleia,nest_data,forecast_for_sensor,F,SRC_END,HORIZONS,CAL_START,CAL_END,TEST_END
from temporal_windows import window

def main():
    strict,metadata=read_pleia(temp_range=(15,40));wide,_=read_pleia(temp_range=(10,45))
    strict_groups={k:g for k,g in strict.groupby('IDdevice')};wide_groups={k:g for k,g in wide.groupby('IDdevice')}
    sensors=sorted(set(strict_groups)&set(wide_groups));output=[];matches=[]
    for h in HORIZONS:
        sources=[]
        for unit,item in ITEMS.items():
            for name in [item[2],unit.lower()+'_2020-2021.csv']:
                d=nest_data(unit,name,item[3],h)
                sources.append(d[d.index+pd.Timedelta(hours=h)<SRC_END])
        src=pd.concat(sources); model=fit(src[F].to_numpy(),(src.target-src.temp_now).to_numpy())
        for sensor in sensors:
            frames={'strict':forecast_for_sensor(strict_groups[sensor],h),'wide':forecast_for_sensor(wide_groups[sensor],h)}
            cal={k:window(v,h,CAL_START,CAL_END,CAL_START-pd.Timedelta(hours=24)) for k,v in frames.items()}
            test={k:window(v,h,CAL_END,TEST_END,CAL_END) for k,v in frames.items()}
            common=test['strict'].index.intersection(test['wide'].index)
            if min(len(cal['strict']),len(cal['wide']),len(common))<24:continue
            difference=(test['strict'].loc[common,'target']-test['wide'].loc[common,'target']).abs()
            matches.append({'horizon_h':h,'sensor':str(sensor),'block':str(metadata.loc[sensor].block),'common_n':len(common),'strict_n':len(test['strict']),'wide_n':len(test['wide']),'mean_abs_label_diff_c':difference.mean(),'max_abs_label_diff_c':difference.max(),'fraction_changed_labels':(difference>1e-9).mean()})
            for arm in ('strict','wide'):
                for label,mod in [('persistence',None),('source_zero_fit',model)]:
                    ca=cal[arm];cp=np.zeros(len(ca)) if mod is None else predict(mod,ca[F].to_numpy())
                    q=quantile(np.abs(ca.temp_now.to_numpy()+cp-ca.target.to_numpy()))
                    for sample in ('all','common'):
                        te=test[arm] if sample=='all' else test[arm].loc[common]
                        if len(te)<24:continue
                        pp=np.zeros(len(te)) if mod is None else predict(mod,te[F].to_numpy())
                        err=te.temp_now.to_numpy()+pp-te.target.to_numpy()
                        output.append({'horizon_h':h,'sensor':str(sensor),'block':str(metadata.loc[sensor].block),'arm':arm,'sample':sample,'model':label,'cal_n':len(ca),'test_n':len(te),'mae_c':np.abs(err).mean(),'coverage90':np.mean(np.abs(err)<=q),'width90_c':2*q})
        print('horizon',h,'done',flush=True)
    pd.DataFrame(output).round(8).to_csv(OUT/'temperature_filter_sensitivity.csv',index=False)
    pd.DataFrame(matches).round(8).to_csv(OUT/'temperature_filter_matches.csv',index=False)

if __name__=='__main__':main()

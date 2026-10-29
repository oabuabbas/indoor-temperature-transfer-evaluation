"""Post-result locally fitted 30-day baseline on identical development rows."""
import pandas as pd
from audit_nest import OUT
from adaptation_stress import fit,predict
from thermbuild_gating import FEATURES
from ideal_development_v1 import load_sensor,windows

def main():
    orig=pd.read_csv(OUT/'ideal_development_v1_home_months.csv')
    reference=orig[orig.model.eq('rolling_selector')]
    if reference.duplicated(['home','sensor','period_start']).any():raise ValueError('Repeated reference window')
    rows=[]
    for (home,sensor),group in reference.groupby(['home','sensor']):
        frame=load_sensor(sensor)
        for row in group.itertuples(index=False):
            start=pd.Timestamp(row.period_start);end=pd.Timestamp(row.period_end)
            train,test=windows(frame,start,end)
            if len(train)!=row.selection_n or len(test)!=row.test_n:raise ValueError('Row support changed')
            model=fit(train[FEATURES].to_numpy(),(train.target-train.seasonal_yesterday).to_numpy())
            pred=test.seasonal_yesterday.to_numpy()+predict(model,test[FEATURES].to_numpy())
            error=abs(pred-test.target.to_numpy())
            rows.append({'home':home,'sensor':sensor,'period_start':row.period_start,'period_end':row.period_end,
                         'model':'local_ridge_30d','choice':'local_ridge_30d','selection_n':len(train),
                         'test_n':len(test),'error_sum':error.sum(),'mae_c':error.mean()})
    combined=pd.concat([orig,pd.DataFrame(rows)],ignore_index=True)
    combined.to_csv(OUT/'ideal_dev_local_home_months.csv',index=False,float_format='%.8f')
    per=combined.groupby(['home','sensor','model'],as_index=False).agg(n=('test_n','sum'),error_sum=('error_sum','sum'),months=('period_start','nunique'))
    per['mae_c']=per.error_sum/per.n
    per.to_csv(OUT/'ideal_dev_local_homes.csv',index=False,float_format='%.8f')
    summary=per.groupby('model',as_index=False).agg(homes=('home','nunique'),mae_c=('mae_c','mean'))
    summary.to_csv(OUT/'ideal_dev_local_summary.csv',index=False,float_format='%.8f')
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()

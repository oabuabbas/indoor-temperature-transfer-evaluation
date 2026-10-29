"""Post-result fixed nonlinear local comparator on original development support."""
import json,platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from audit_nest import OUT
from thermbuild_gating import FEATURES
from ideal_development_v1 import load_sensor,windows

def new_model():
    return HistGradientBoostingRegressor(learning_rate=.05,max_iter=70,max_leaf_nodes=7,
        min_samples_leaf=20,l2_regularization=10,early_stopping=False,random_state=42)

def main():
    orig=pd.read_csv(OUT/'ideal_dev_local_home_months.csv')
    reference=orig[orig.model.eq('rolling_selector')]
    if reference.duplicated(['home','sensor','period_start']).any():raise ValueError('Repeated reference window')
    rows=[]
    with threadpool_limits(limits=1):
        for i,((home,sensor),group) in enumerate(reference.groupby(['home','sensor']),1):
            frame=load_sensor(sensor)
            for row in group.itertuples(index=False):
                history,test=windows(frame,pd.Timestamp(row.period_start),pd.Timestamp(row.period_end))
                if len(history)!=row.selection_n or len(test)!=row.test_n:raise ValueError('Support changed')
                learner=new_model().fit(history[FEATURES].to_numpy(),(history.target-history.seasonal_yesterday).to_numpy())
                yhat=test.seasonal_yesterday.to_numpy()+learner.predict(test[FEATURES].to_numpy())
                err=np.abs(yhat-test.target.to_numpy())
                rows.append({'home':home,'sensor':sensor,'period_start':row.period_start,'period_end':row.period_end,
                             'model':'local_histgb_30d','choice':'local_histgb_30d','selection_n':len(history),
                             'test_n':len(test),'error_sum':err.sum(),'mae_c':err.mean()})
            if i%10==0:print('Completed',i,'of',reference.home.nunique(),'homes',flush=True)
    combined=pd.concat([orig,pd.DataFrame(rows)],ignore_index=True)
    combined.to_csv(OUT/'ideal_dev_nonlinear_home_months.csv',index=False,float_format='%.8f')
    per=combined.groupby(['home','sensor','model'],as_index=False).agg(n=('test_n','sum'),error_sum=('error_sum','sum'),months=('period_start','nunique'))
    per['mae_c']=per.error_sum/per.n
    per.to_csv(OUT/'ideal_dev_nonlinear_homes.csv',index=False,float_format='%.8f')
    summary=[]
    for threshold in (0,6,8):
        eligible=per[per.months>=threshold]
        for model,group in eligible.groupby('model'):
            summary.append({'min_months':threshold,'model':model,'homes':group.home.nunique(),'mae_c':group.mae_c.mean()})
    summary=pd.DataFrame(summary)
    summary.to_csv(OUT/'ideal_dev_nonlinear_summary.csv',index=False,float_format='%.8f')
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__}
    (OUT/'ideal_dev_nonlinear_versions.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()

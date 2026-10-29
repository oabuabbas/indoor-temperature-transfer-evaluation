"""Choose a fixed HGB candidate on development-validation homes only."""
import hashlib,json,platform
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingRegressor
from audit_nest import OUT
from ideal_development_prepare import HOURLY,select
from ideal_development_v1 import load_sensor,windows
from thermbuild_gating import FEATURES
ROOT=OUT.parent
GRID={'shallow':{'max_iter':60,'max_leaf_nodes':5,'min_samples_leaf':40,'l2_regularization':20.},'moderate':{'max_iter':100,'max_leaf_nodes':10,'min_samples_leaf':30,'l2_regularization':10.},'richer':{'max_iter':140,'max_leaf_nodes':20,'min_samples_leaf':20,'l2_regularization':1.}}

def model_for(name):
    return HistGradientBoostingRegressor(loss='squared_error',learning_rate=.05,early_stopping=False,random_state=20261029,**GRID[name])
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    protocol=ROOT/'protocol_ideal_nonlinear_tuning_v1.md';choice_path=OUT/'ideal_nonlinear_tuning_v1_choice.json';window_path=OUT/'ideal_nonlinear_tuning_v1_dev_windows.csv';home_path=OUT/'ideal_nonlinear_tuning_v1_dev_homes.csv'
    if any(x.exists() for x in [choice_path,window_path,home_path]):raise ValueError('Existing tuning outputs protected')
    dev=select('development').reset_index(drop=True);held=select('holdout').reset_index(drop=True)
    if not pd.read_csv(OUT/'ideal_development_sensors_v1.csv').equals(dev):raise ValueError('Development cohort changed')
    validation=dev[dev.dev_role.eq('validation')]
    if set(dev.home)&set(held.home):raise ValueError('Development and holdout overlap')
    original=pd.read_csv(OUT/'ideal_dev_local_home_months.csv');ref=original[original.model.eq('rolling_selector') & (pd.to_datetime(original.period_start,utc=True)<pd.Timestamp('2018-02-01',tz='UTC'))].copy()
    if ref.duplicated(['home','sensor','period_start']).any():raise ValueError('Repeated development window')
    allowed=set(zip(validation.home,validation.sensor))
    if not set(zip(ref.home,ref.sensor)).issubset(allowed):raise ValueError('Not development validation cohort')
    rows=[]; hashes={}
    with threadpool_limits(limits=1):
      for i,((home,sensor),g) in enumerate(ref.groupby(['home','sensor']),1):
        hashes[str(sensor)]=sha(HOURLY/f'{sensor}.csv.gz');frame=load_sensor(sensor)
        for row in g.itertuples(index=False):
            history,test=windows(frame,pd.Timestamp(row.period_start),pd.Timestamp(row.period_end))
            if len(history)!=row.selection_n or len(test)!=row.test_n or min(len(history),len(test))<24:raise ValueError('Development support mismatch')
            x=history[FEATURES].to_numpy();y=(history.target-history.seasonal_yesterday).to_numpy();xx=test[FEATURES].to_numpy();target=test.target.to_numpy();daily=test.seasonal_yesterday.to_numpy()
            for candidate in GRID:
                tree=model_for(candidate).fit(x,y);e=np.abs(daily+tree.predict(xx)-target)
                rows.append({'home':int(home),'sensor':int(sensor),'period_start':row.period_start,'period_end':row.period_end,'candidate':candidate,'history_n':len(history),'test_n':len(test),'error_sum':float(e.sum())})
        if i%10==0:print('Development homes',i,'of',ref.home.nunique(),flush=True)
    d=pd.DataFrame(rows);per=d.groupby(['home','sensor','candidate'],as_index=False).agg(n=('test_n','sum'),error_sum=('error_sum','sum'),months=('period_start','nunique'));per['mae_c']=per.error_sum/per.n
    piv=per.pivot(index='home',columns='candidate',values='mae_c')
    if piv.isna().any().any() or set(piv.columns)!=set(GRID):raise ValueError('Development candidate support differs')
    means={name:float(piv[name].mean()) for name in GRID};chosen=min(GRID,key=lambda name:means[name])
    d.to_csv(window_path,index=False,float_format='%.10f');per.to_csv(home_path,index=False,float_format='%.10f')
    result={'protocol_sha256':sha(protocol),'reference_sha256':sha(OUT/'ideal_dev_local_home_months.csv'),'development_selection_sha256':sha(OUT/'ideal_development_sensors_v1.csv'),'validation_hourly_sha256':hashes,'grid':GRID,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__},'development_homes':len(piv),'development_home_months':len(ref),'development_equal_home_mae_c':means,'chosen_candidate':chosen,'window_output_sha256':sha(window_path),'home_output_sha256':sha(home_path),'scope':'Development only. Earlier holdout outcomes of other methods are known; selection is not public preregistration.'}
    choice_path.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='validation_hourly_sha256'},indent=2),flush=True)
if __name__=='__main__':main()

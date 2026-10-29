"""Post-result secondary holdout comparison using candidate selected on validation homes."""
import hashlib,json,platform
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
from audit_nest import OUT
from ideal_development_prepare import select
from ideal_development_v1 import windows
from ideal_holdout_v1 import read_target
from ideal_holdout_prepare import HOURLY
from thermbuild_gating import FEATURES
from ideal_nonlinear_tuning_v1 import GRID,model_for,sha
ROOT=OUT.parent

def main():
    protocol=ROOT/'protocol_ideal_nonlinear_tuning_v1.md';selection=OUT/'ideal_nonlinear_tuning_v1_choice.json';summary=OUT/'ideal_nonlinear_holdout_secondary_v1_summary.json';winpath=OUT/'ideal_nonlinear_holdout_secondary_v1_windows.csv';homepath=OUT/'ideal_nonlinear_holdout_secondary_v1_homes.csv'
    if any(x.exists() for x in [summary,winpath,homepath]):raise ValueError('Existing holdout outputs protected')
    picked=json.loads(selection.read_text(encoding='utf8'))
    if sha(protocol)!=picked['protocol_sha256'] or GRID!=picked['grid']:raise ValueError('Protocol or grid mismatch')
    if sha(OUT/'ideal_nonlinear_tuning_v1_dev_windows.csv')!=picked['window_output_sha256'] or sha(OUT/'ideal_nonlinear_tuning_v1_dev_homes.csv')!=picked['home_output_sha256']:raise ValueError('Tuning output changed')
    if sha(OUT/'ideal_dev_local_home_months.csv')!=picked['reference_sha256'] or sha(OUT/'ideal_development_sensors_v1.csv')!=picked['development_selection_sha256']:raise ValueError('Development input changed')
    for sensor,h in picked['validation_hourly_sha256'].items():
        from ideal_development_prepare import HOURLY as DEV_HOURLY
        if sha(DEV_HOURLY/f'{sensor}.csv.gz')!=h:raise ValueError('Development hourly file changed')
    devhome=pd.read_csv(OUT/'ideal_nonlinear_tuning_v1_dev_homes.csv');devmean=devhome.groupby('candidate').mae_c.mean();choice=min(GRID,key=lambda n:devmean[n])
    if choice!=picked['chosen_candidate']:raise ValueError('Candidate choice mismatch')
    oldsum=json.loads((OUT/'ideal_holdout_v1_summary.json').read_text(encoding='utf8'))
    if sha(ROOT/'protocol_ideal_holdout_v1.md')!=oldsum['protocol_sha256']:raise ValueError('Original holdout protocol changed')
    held=select('holdout').drop(columns='dev_role').reset_index(drop=True);dev=select('development').reset_index(drop=True)
    if not held.equals(pd.read_csv(OUT/'ideal_holdout_sensors_v1.csv')) or set(dev.home)&set(held.home):raise ValueError('Cohort mismatch or overlap')
    old=pd.read_csv(OUT/'ideal_holdout_v1_home_months.csv');oldhome=pd.read_csv(OUT/'ideal_holdout_v1_homes.csv');old=old[pd.to_datetime(old.period_start,utc=True)>=pd.Timestamp('2018-02-01',tz='UTC')].copy();ref=old[old.model.eq('rolling_selector')]
    if ref.duplicated(['home','sensor','period_start']).any():raise ValueError('Duplicate holdout windows')
    if not set(zip(ref.home,ref.sensor)).issubset(set(zip(held.home,held.sensor))):raise ValueError('Unexpected holdout sensor')
    comparison=['rolling_selector','local_ridge_30d','source_daily_ridge','local_histgb_30d','persistence','daily_seasonal']
    pivot=old.pivot(index=['home','sensor','period_start'],columns='model',values='test_n')
    if set(pivot.columns)!=set(comparison) or pivot.isna().any().any() or (pivot.nunique(axis=1)!=1).any():raise ValueError('Original saved methods differ in test counts')
    rows=[]; hashes={}
    with threadpool_limits(limits=1):
      for i,((home,sensor),group) in enumerate(ref.groupby(['home','sensor']),1):
        hashes[str(sensor)]=sha(HOURLY/f'{sensor}.csv.gz');frame=read_target(int(sensor))
        for row in group.itertuples(index=False):
            history,test=windows(frame,pd.Timestamp(row.period_start),pd.Timestamp(row.period_end))
            if len(history)!=row.history_n or len(test)!=row.test_n or min(len(history),len(test))<24:raise ValueError('Holdout support mismatch')
            tree=model_for(choice).fit(history[FEATURES].to_numpy(),(history.target-history.seasonal_yesterday).to_numpy())
            pred=test.seasonal_yesterday.to_numpy()+tree.predict(test[FEATURES].to_numpy());err=np.abs(pred-test.target.to_numpy())
            rows.append({'home':int(home),'sensor':int(sensor),'period_start':row.period_start,'period_end':row.period_end,'candidate':choice,'history_n':len(history),'test_n':len(test),'error_sum':float(err.sum())})
        if i%10==0:print('Holdout homes',i,'of',ref.home.nunique(),flush=True)
    d=pd.DataFrame(rows)
    if len(d)!=len(ref):raise ValueError('Missing holdout windows')
    per=d.groupby(['home','sensor'],as_index=False).agg(n=('test_n','sum'),error_sum=('error_sum','sum'),months=('period_start','nunique'));per['mae_c']=per.error_sum/per.n
    baseper=old.groupby(['home','sensor','model'],as_index=False).agg(n=('test_n','sum'),error_sum=('error_sum','sum'),months=('period_start','nunique'));baseper['mae_c']=baseper.error_sum/baseper.n
    base=baseper.pivot(index=['home','sensor'],columns='model',values='mae_c');base_n=baseper[baseper.model.eq('rolling_selector')].set_index(['home','sensor'])
    new=per.set_index(['home','sensor']);base=base.reindex(new.index);base_n=base_n.reindex(new.index)
    if base.isna().any().any() or not np.array_equal(new.n,base_n.n) or not np.array_equal(new.months,base_n.months):raise ValueError('Unequal house support')
    joined=new.join(base);metrics={'selected_boosting':float(new.mae_c.mean())}
    contrasts={}
    for c in comparison:
        delta=(new.mae_c-base[c]);metrics[c]=float(base[c].mean())
        contrasts[c]={'mean_selected_minus_comparator_c':float(delta.mean()),'better_homes':int((delta < -1e-8).sum()),'worse_homes':int((delta>1e-8).sum()),'ties':int((delta.abs()<=1e-8).sum())}
    d.to_csv(winpath,index=False,float_format='%.10f');joined.reset_index().to_csv(homepath,index=False,float_format='%.10f')
    result={'protocol_sha256':sha(protocol),'selection_sha256':sha(selection),'original_holdout_protocol_sha256':oldsum['protocol_sha256'],'original_windows_sha256':sha(OUT/'ideal_holdout_v1_home_months.csv'),'original_homes_sha256':sha(OUT/'ideal_holdout_v1_homes.csv'),'holdout_hourly_sha256':hashes,'candidate':choice,'homes':len(new),'home_months':len(d),'scored_rows':int(new.n.sum()),'equal_home_mae_c':metrics,'selected_vs_existing':contrasts,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__},'scope':'Secondary post-result comparison of previously known holdout, same dataset and clustered region; no population/site inference.'}
    summary.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='holdout_hourly_sha256'},indent=2),flush=True)
if __name__=='__main__':main()

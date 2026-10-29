"""Locked local holdout evaluation; protocol_ideal_holdout_v1.md."""
import hashlib,json,platform
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
from audit_nest import OUT
from adaptation_stress import fit,predict
from thermbuild_gating import FEATURES,CANDIDATES,predictions
from ideal_development_prepare import select
from ideal_development_v1 import load_sensor,frame_from_series,windows,H,SOURCE_END,TEST_START,TEST_END
from ideal_dev_nonlinear import new_model
from ideal_holdout_prepare import HOURLY as HOLDOUT_HOURLY

def read_target(sensor):
    d=pd.read_csv(HOLDOUT_HOURLY/f'{sensor}.csv.gz',parse_dates=['hour_utc'])
    if d.empty: return pd.DataFrame(columns=FEATURES+['target'],index=pd.DatetimeIndex([],tz='UTC'))
    if d.hour_utc.duplicated().any():raise ValueError('Duplicate hourly target')
    return frame_from_series(d.set_index('hour_utc').temp_c)

def main():
    summary_file=OUT/'ideal_holdout_v1_summary.json'
    if summary_file.exists():raise ValueError('Holdout results already exist; do not rerun without audit')
    protocol=OUT.parent/'protocol_ideal_holdout_v1.md'
    if not protocol.exists():raise ValueError('Frozen protocol missing')
    held=select('holdout').drop(columns='dev_role').reset_index(drop=True)
    recorded=pd.read_csv(OUT/'ideal_holdout_sensors_v1.csv')
    if not recorded.equals(held):raise ValueError('Holdout selection changed')
    source_sel=select('development').reset_index(drop=True)
    if not pd.read_csv(OUT/'ideal_development_sensors_v1.csv').equals(source_sel):raise ValueError('Development selection changed')
    if set(source_sel.home)&set(held.home):raise ValueError('Home split leakage')
    if any(not (HOLDOUT_HOURLY/f'{x.sensor}.csv.gz').exists() for x in held.itertuples(index=False)):
        raise ValueError('Target hourly files incomplete')
    pieces=[]
    for x in source_sel[source_sel.dev_role.eq('source')].itertuples(index=False):
        d=load_sensor(int(x.sensor));part=d[d.index+pd.Timedelta(hours=H)<SOURCE_END]
        if len(part)>=24:pieces.append(part)
    if not pieces:raise ValueError('No source data')
    train=pd.concat(pieces)
    if len(pieces)!=55 or len(train)!=125779:raise ValueError('Source data differ from development v1')
    src=fit(train[FEATURES].to_numpy(),(train.target-train.seasonal_yesterday).to_numpy())
    starts=pd.date_range(TEST_START,TEST_END,freq='MS',inclusive='left')
    rows=[];reasons=[]
    with threadpool_limits(limits=1):
        for i,x in enumerate(held.itertuples(index=False),1):
            frame=read_target(int(x.sensor))
            for start in starts:
                end=min(start+pd.offsets.MonthBegin(1),TEST_END)
                history,test=windows(frame,start,end)
                if min(len(history),len(test))<24:
                    reasons.append({'home':int(x.home),'period_start':start.isoformat(),'history_n':len(history),'test_n':len(test)})
                    continue
                historical=predictions(history,src).sub(history.target,axis=0).abs().mean()
                choice=min(CANDIDATES,key=lambda c:historical[c])
                pred=predictions(test,src)
                xx=history[FEATURES].to_numpy(); yy=(history.target-history.seasonal_yesterday).to_numpy()
                local=fit(xx,yy)
                pred['local_ridge_30d']=test.seasonal_yesterday.to_numpy()+predict(local,test[FEATURES].to_numpy())
                tree=new_model().fit(xx,yy)
                pred['local_histgb_30d']=test.seasonal_yesterday.to_numpy()+tree.predict(test[FEATURES].to_numpy())
                for method in CANDIDATES+['local_ridge_30d','local_histgb_30d','rolling_selector']:
                    use=choice if method=='rolling_selector' else method
                    err=np.abs(pred[use].to_numpy()-test.target.to_numpy())
                    rows.append({'home':int(x.home),'sensor':int(x.sensor),'period_start':start.isoformat(),'period_end':end.isoformat(),
                                 'model':method,'choice':choice,'history_n':len(history),'test_n':len(test),
                                 'error_sum':float(err.sum()),'mae_c':float(err.mean())})
            if i%10==0 or i==len(held):print('Evaluated',i,'of',len(held),'holdout homes',flush=True)
    if not rows:raise ValueError('No eligible test periods')
    d=pd.DataFrame(rows)
    per=d.groupby(['home','sensor','model'],as_index=False).agg(n=('test_n','sum'),error_sum=('error_sum','sum'),months=('period_start','nunique'))
    per['mae_c']=per.error_sum/per.n
    piv=per.pivot(index='home',columns='model',values='mae_c')
    if piv.isna().any().any() or set(piv.columns)!=set(CANDIDATES+['local_ridge_30d','local_histgb_30d','rolling_selector']):
        raise ValueError('Unequal method support')
    delta=(piv.rolling_selector-piv.local_ridge_30d).to_numpy()
    rng=np.random.default_rng(20261028)
    draws=rng.integers(0,len(delta),size=(5000,len(delta)))
    interval=np.quantile(delta[draws].mean(axis=1),[.025,.975])
    mae=piv.mean()
    summary={'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'original_holdout_homes':74,
             'selected_homes':len(held),'contributing_homes':len(piv),'eligible_home_months':int(len(d)/6),
             'excluded_home_months':len(reasons),'source_homes':len(pieces),'source_rows':len(train),
             'mean_mae_c':{k:float(v) for k,v in mae.items()},
             'primary_selector_minus_local_c':float(delta.mean()),
             'primary_percent_relative_to_local':float(100*delta.mean()/mae.local_ridge_30d),
             'primary_median_difference_c':float(np.median(delta)),
             'selector_better_homes':int((delta < -1e-8).sum()),'ties':int((np.abs(delta)<=1e-8).sum()),
             'selector_worse_homes':int((delta>1e-8).sum()),
             'descriptive_home_resample_interval_c':interval.tolist(),'bootstrap_draws':5000,
             'bootstrap_seed':20261028,'interval_scope':'Descriptive home reweighting within one clustered region; not a population or site confidence interval.',
             'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__}}
    d.to_csv(OUT/'ideal_holdout_v1_home_months.csv',index=False,float_format='%.8f')
    per.to_csv(OUT/'ideal_holdout_v1_homes.csv',index=False,float_format='%.8f')
    pd.DataFrame(reasons,columns=['home','period_start','history_n','test_n']).to_csv(OUT/'ideal_holdout_v1_exclusions.csv',index=False)
    summary_file.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

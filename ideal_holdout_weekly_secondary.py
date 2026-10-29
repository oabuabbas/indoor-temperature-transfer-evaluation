"""Reconstruct primary predictions and resample shared calendar weeks."""
import json,platform
import numpy as np
import pandas as pd
from audit_nest import OUT
from adaptation_stress import fit,predict
from thermbuild_gating import FEATURES,CANDIDATES,predictions
from ideal_development_prepare import select
from ideal_development_v1 import load_sensor,windows,H,SOURCE_END
from ideal_holdout_v1 import read_target

SEED=20261029
DRAWS=2000

def weekly_bootstrap(weekly):
    homes=np.sort(weekly.home.unique()); weeks=np.sort(weekly.week.unique())
    index=pd.MultiIndex.from_product([homes,weeks],names=['home','week'])
    w=weekly.set_index(['home','week']).reindex(index).fillna(0)
    shape=(len(homes),len(weeks))
    n=w['n'].to_numpy().reshape(shape)
    selector=w['selector_sum'].to_numpy().reshape(shape)
    local=w['local_sum'].to_numpy().reshape(shape)
    d0=(selector.sum(axis=1)-local.sum(axis=1))/n.sum(axis=1)
    rng=np.random.default_rng(SEED);choices=rng.integers(0,len(weeks),size=(DRAWS,len(weeks)))
    ns=n[:,choices].sum(axis=2)
    err=(selector[:,choices].sum(axis=2)-local[:,choices].sum(axis=2))
    per_home=np.divide(err,ns,out=np.full(err.shape,np.nan),where=ns>0)
    with np.errstate(invalid='ignore'):delta=np.nanmean(per_home,axis=0)
    valid=np.isfinite(per_home).sum(axis=0)
    return {'weeks':len(weeks),'homes':len(homes),'draws':DRAWS,'seed':SEED,
            'observed_mean_delta_c':float(d0.mean()),'q025_c':float(np.quantile(delta,.025)),
            'q975_c':float(np.quantile(delta,.975)),'min_homes_per_draw':int(valid.min()),
            'draws_with_missing_homes':int((valid<len(homes)).sum())}

def main():
    output=OUT/'ideal_holdout_weekly_secondary_summary.json'
    if output.exists():raise ValueError('Secondary weekly results already exist')
    if not (OUT.parent/'protocol_ideal_holdout_weekly_secondary.md').exists():raise ValueError('Protocol missing')
    reference=pd.read_csv(OUT/'ideal_holdout_v1_home_months.csv')
    ref=reference[reference.model.isin(['rolling_selector','local_ridge_30d'])]
    if ref.duplicated(['home','period_start','model']).any():raise ValueError('Repeated reference window')
    refidx=ref.set_index(['home','period_start','model'])
    held=select('holdout').drop(columns='dev_role').reset_index(drop=True)
    if not pd.read_csv(OUT/'ideal_holdout_sensors_v1.csv').equals(held):raise ValueError('Holdout selection changed')
    srcsel=select('development').reset_index(drop=True)
    if not pd.read_csv(OUT/'ideal_development_sensors_v1.csv').equals(srcsel):raise ValueError('Source selection changed')
    if set(held.home)&set(srcsel.home):raise ValueError('House overlap')
    pieces=[]
    for row in srcsel[srcsel.dev_role.eq('source')].itertuples(index=False):
        frame=load_sensor(int(row.sensor));part=frame[frame.index+pd.Timedelta(hours=H)<SOURCE_END]
        if len(part)>=24:pieces.append(part)
    source=pd.concat(pieces)
    if len(pieces)!=55 or len(source)!=125779:raise ValueError('Source training changed')
    model=fit(source[FEATURES].to_numpy(),(source.target-source.seasonal_yesterday).to_numpy())
    pieces=[]; verified=set()
    for home,sensor in held[['home','sensor']].itertuples(index=False,name=None):
        windows_ref=ref[(ref.home==home)&ref.model.eq('rolling_selector')]
        if windows_ref.empty:continue
        frame=read_target(int(sensor))
        for row in windows_ref.itertuples(index=False):
            history,test=windows(frame,pd.Timestamp(row.period_start),pd.Timestamp(row.period_end))
            if len(history)!=row.history_n or len(test)!=row.test_n:raise ValueError('History/test count changed')
            choices=predictions(history,model).sub(history.target,axis=0).abs().mean()
            choice=min(CANDIDATES,key=lambda c:choices[c])
            if choice!=row.choice:raise ValueError('Recorded selector choice differs')
            local=fit(history[FEATURES].to_numpy(),(history.target-history.seasonal_yesterday).to_numpy())
            y=test.target.to_numpy()
            a=np.abs(predictions(test,model)[choice].to_numpy()-y)
            b=np.abs(test.seasonal_yesterday.to_numpy()+predict(local,test[FEATURES].to_numpy())-y)
            key=(home,row.period_start)
            for label,err in [('rolling_selector',a),('local_ridge_30d',b)]:
                saved=refidx.loc[(home,row.period_start,label)]
                if abs(float(err.sum())-float(saved.error_sum))>1e-5:raise ValueError('Original monthly error does not reproduce')
            verified.add(key)
            target_time=(test.index+pd.Timedelta(hours=H)).tz_localize(None)
            week=target_time.to_period('W-SUN').start_time.strftime('%Y-%m-%d')
            w=pd.DataFrame({'home':int(home),'week':week,'selector_sum':a,'local_sum':b,'n':1})
            pieces.append(w.groupby(['home','week'],as_index=False).sum())
    expected=set(zip(ref[ref.model.eq('rolling_selector')].home,ref[ref.model.eq('rolling_selector')].period_start))
    if verified!=expected or len(verified)!=477:raise ValueError('Original window membership changed')
    weekly=pd.concat(pieces,ignore_index=True).groupby(['home','week'],as_index=False).sum()
    result=weekly_bootstrap(weekly)
    original=json.loads((OUT/'ideal_holdout_v1_summary.json').read_text(encoding='utf-8'))
    if abs(result['observed_mean_delta_c']-original['primary_selector_minus_local_c'])>1e-7:raise ValueError('Primary observed difference not reproduced')
    result.update({'verified_home_months':len(verified),'test_issue_times':int(weekly.n.sum()),
                   'original_protocol_sha256':original['protocol_sha256'],
                   'interpretation':'Descriptive joint calendar-week reweighting within one clustered region; not a population/site confidence interval.',
                   'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}})
    weekly.to_csv(OUT/'ideal_holdout_weekly_secondary.csv',index=False,float_format='%.8f')
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

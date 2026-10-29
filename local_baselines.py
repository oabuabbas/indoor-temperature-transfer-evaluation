"""Exploratory local nonlinear benchmarks and joint calendar-week bootstrap."""
import json,platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from audit_nest import ITEMS,OUT
from adaptation_stress import fit,predict,quantile
from external_pleia import read_pleia
from external_adaptation import make_sensor
from daily_residual import FEATURES,nest_source,T0,T1,T2,CAL0,CAL1,END

H=6
def source_predict(m,d):return d.seasonal_yesterday.to_numpy()+predict(m,d[FEATURES].to_numpy())
def target_predict(m,d):return d.seasonal_yesterday.to_numpy()+m.predict(d[FEATURES].to_numpy())
def make_tree(leaves):
    return HistGradientBoostingRegressor(learning_rate=.05,max_iter=70,max_leaf_nodes=leaves,min_samples_leaf=20,l2_regularization=10,early_stopping=False,random_state=42)

def choose_models(fitting,val,refit):
    x=fitting[FEATURES].to_numpy(); y=(fitting.target-fitting.seasonal_yesterday).to_numpy()
    xv=val[FEATURES].to_numpy();yv=(val.target-val.seasonal_yesterday).to_numpy()
    xr=refit[FEATURES].to_numpy();yr=(refit.target-refit.seasonal_yesterday).to_numpy()
    grid=[]
    for alpha in [1.,100.,1000.]:
        model=fit(x,y,penalty=alpha);grid.append((float(np.abs(predict(model,xv)-yv).mean()),alpha))
    best_alpha=min(grid)[1];linear=fit(xr,yr,penalty=best_alpha)
    treegrid=[]
    for leaves in [7,15]:
        model=make_tree(leaves).fit(x,y);treegrid.append((float(np.abs(model.predict(xv)-yv).mean()),leaves))
    best_leaves=min(treegrid)[1];tree=make_tree(best_leaves).fit(xr,yr)
    return linear,tree,best_alpha,best_leaves

def bootstrap(weekly):
    rng=np.random.default_rng(20261028); output=[]
    weeks=np.sort(weekly.week.unique()); sensors=np.sort(weekly.sensor.unique());
    methods=['persistence','source_residual','local_ridge','local_histgb']
    def matrices(label):
        d=weekly[weekly.model==label].set_index(['sensor','week'])
        grid=pd.MultiIndex.from_product([sensors,weeks],names=['sensor','week'])
        v=d.reindex(grid).fillna(0)
        return v.abs_sum.to_numpy().reshape(len(sensors),len(weeks)),v.n.to_numpy().reshape(len(sensors),len(weeks))
    s0,n0=matrices('daily_seasonal');draws=rng.integers(0,len(weeks),size=(2000,len(weeks)))
    base_draw=np.divide(s0[:,draws].sum(axis=2),n0[:,draws].sum(axis=2),out=np.full((len(sensors),len(draws)),np.nan),where=n0[:,draws].sum(axis=2)>0)
    base_obs=np.divide(s0.sum(axis=1),n0.sum(axis=1),out=np.full(len(sensors),np.nan),where=n0.sum(axis=1)>0)
    for label in methods:
        s,n=matrices(label);den=n[:,draws].sum(axis=2);pred=np.divide(s[:,draws].sum(axis=2),den,out=np.full((len(sensors),len(draws)),np.nan),where=den>0)
        delta=np.nanmean(pred-base_draw,axis=0);obs=np.divide(s.sum(axis=1),n.sum(axis=1),out=np.full(len(sensors),np.nan),where=n.sum(axis=1)>0)
        output.append({'comparison':label+' minus daily_seasonal','observed_diff_c':float(np.nanmean(obs-base_obs)),'q025_c':float(np.quantile(delta,.025)),'q975_c':float(np.quantile(delta,.975)),'bootstrap_draws':len(draws),'calendar_weeks':len(weeks),'sensors':len(sensors),'min_sensors_per_draw':int(np.isfinite(pred-base_draw).sum(axis=0).min())})
    return pd.DataFrame(output)

def main():
    versions={'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scikit_learn':sklearn.__version__}
    (OUT/'local_baseline_versions.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    hour,meta=read_pleia();source_data=pd.concat([nest_source(name,item[3],H) for unit,item in ITEMS.items() for name in [item[2],unit.lower()+'_2020-2021.csv']])
    source=fit(source_data[FEATURES].to_numpy(),(source_data.target-source_data.seasonal_yesterday).to_numpy())
    results=[];choices=[];weekly=[]
    for sensor,g in hour.groupby('IDdevice'):
        d=make_sensor(g,H);tt=d.index+pd.Timedelta(hours=H)
        first=d[(tt>=T0)&(tt<T1)];val=d[(d.index>=T1)&(tt<T2)];alljan=d[(tt>=T0)&(tt<T2)]
        cal=d[(d.index>=T2)&(tt>=CAL0)&(tt<CAL1)];test=d[(d.index>=CAL1)&(tt>=CAL1)&(tt<END)]
        if min(len(first),len(val),len(cal),len(test))<24:continue
        lin,tree,alpha,leaves=choose_models(first,val,alljan)
        methods={'persistence':lambda x:x.temp_now.to_numpy(),'daily_seasonal':lambda x:x.seasonal_yesterday.to_numpy(),'source_residual':lambda x:source_predict(source,x),'local_ridge':lambda x:source_predict(lin,x),'local_histgb':lambda x:target_predict(tree,x)}
        m=meta.loc[sensor];choices.append({'sensor':str(sensor),'block':str(m.block),'room':str(m.room),'ridge_alpha':alpha,'tree_max_leaves':leaves,'fit_n':len(first),'validation_n':len(val),'refit_n':len(alljan),'cal_n':len(cal),'test_n':len(test)})
        target_time=test.index+pd.Timedelta(hours=H)
        week=target_time.tz_localize(None).to_period('W-SUN').start_time.strftime('%Y-%m-%d')
        for label,fun in methods.items():
            cp=fun(cal);tp=fun(test);q=quantile(np.abs(cp-cal.target.to_numpy()));err=tp-test.target.to_numpy()
            results.append({'sensor':str(sensor),'block':str(m.block),'room':str(m.room),'model':label,'test_n':len(test),'cal_n':len(cal),'mae_c':np.abs(err).mean(),'rmse_c':np.sqrt(np.square(err).mean()),'coverage90':np.mean(np.abs(err)<=q),'width90_c':2*q})
            w=pd.DataFrame({'week':week,'abs_err':np.abs(err)}).groupby('week').abs_err.agg(['sum','count']).reset_index()
            for row in w.itertuples(index=False):weekly.append({'sensor':str(sensor),'model':label,'week':row.week,'abs_sum':row.sum,'n':row.count})
        if len(choices)%10==0:print('sensors complete',len(choices),flush=True)
    result=pd.DataFrame(results);weekdf=pd.DataFrame(weekly)
    result.round(8).to_csv(OUT/'local_baselines_results.csv',index=False)
    pd.DataFrame(choices).to_csv(OUT/'local_baselines_choices.csv',index=False)
    weekdf.to_csv(OUT/'local_baselines_weekly.csv',index=False)
    bootstrap(weekdf).round(8).to_csv(OUT/'local_baselines_bootstrap.csv',index=False)
    print('Completed',len(choices),'sensors',flush=True)

if __name__=='__main__':main()

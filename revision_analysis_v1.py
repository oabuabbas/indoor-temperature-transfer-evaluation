from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
O=ROOT/'outputs'; D=O/'revision_v1'; D.mkdir(exist_ok=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 if (D/'summary.json').exists():raise RuntimeError('Preserve completed revision; choose a new version')
 primary=O/'ideal_holdout_v1_home_months.csv'; before=digest(primary)
 scores=pd.read_csv(primary); one=scores[scores.model.eq('rolling_selector')]
 hs=[]; rs=[]; hashes={str(primary.relative_to(ROOT)):before}
 with threadpool_limits(limits=1):
  for j,(home,windows) in enumerate(one.groupby('home')):
   sensor=int(windows.sensor.iloc[0]); path=O/'ideal_holdout_hourly_v1'/f'{sensor}.csv.gz';hashes[str(path.relative_to(ROOT))]=digest(path)
   raw=pd.read_csv(path,parse_dates=['hour_utc']);s=raw.set_index('hour_utc').temp_c.sort_index().asfreq('1h')
   frame=pd.DataFrame({f'lag{k}':s.shift(k) for k in [0,1,18,2,3,6,12,24,48]});frame['target']=s.shift(-6)
   base=frame.dropna(subset=['lag0','lag1','lag18','target'])
   for w in windows.itertuples():
    start=pd.Timestamp(w.period_start);end=pd.Timestamp(w.period_end)
    hist=base[(base.index>=start-pd.Timedelta(days=30))&(base.index+pd.Timedelta(hours=6)<start)]
    test=base[(base.index>=start)&(base.index+pd.Timedelta(hours=6)<end)]
    assert len(hist)==w.history_n and len(test)==w.test_n
    hs.append(dict(home=home,period_start=w.period_start,history_sd=hist.lag0.std(),history_change=(hist.lag0-hist.lag1).abs().mean()))
    hist=hist.dropna();test=test.dropna()
    if min(len(hist),len(test))<24:continue
    y=hist.target-hist.lag18
    models={
     'ridge_3':(make_pipeline(StandardScaler(),Ridge(alpha=100)),3),
     'ridge_9':(make_pipeline(StandardScaler(),Ridge(alpha=100)),9),
     'boosting_3':(HistGradientBoostingRegressor(learning_rate=.05,max_iter=70,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,early_stopping=False,random_state=42),3),
     'boosting_9':(HistGradientBoostingRegressor(learning_rate=.05,max_iter=70,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,early_stopping=False,random_state=42),9),
     'svr_9':(make_pipeline(StandardScaler(),SVR(C=1,epsilon=.1,gamma='scale')),9)}
    for name,(model,n) in models.items():
     model.fit(hist.iloc[:,:n],y); pred=test.lag18.to_numpy()+model.predict(test.iloc[:,:n]);err=np.abs(pred-test.target.to_numpy())
     rs.append(dict(home=home,period_start=w.period_start,model=name,n=len(test),error_sum=float(err.sum())))
    rs.append(dict(home=home,period_start=w.period_start,model='persistence',n=len(test),error_sum=float((test.lag0-test.target).abs().sum())))
   if (j+1)%10==0:print('homes',j+1,flush=True)
 h=pd.DataFrame(hs).groupby('home')[['history_sd','history_change']].mean()
 p=pd.read_csv(O/'ideal_holdout_v1_homes.csv').pivot(index='home',columns='model',values='mae_c')
 h['selector_minus_local']=p.rolling_selector-p.local_ridge_30d;h['source_minus_local']=p.source_daily_ridge-p.local_ridge_30d
 corr={outcome:{x:float(h[x].corr(h[outcome],method='spearman')) for x in ['history_sd','history_change']} for outcome in ['selector_minus_local','source_minus_local']}
 paired=scores.pivot(index=['home','period_start'],columns='model',values='mae_c');paired['delta']=paired.rolling_selector-paired.local_ridge_30d
 monthly=paired.groupby('period_start').delta.agg(['mean','median','count'])
 r=pd.DataFrame(rs); per=r.groupby(['home','model'])[['n','error_sum']].sum();per['mae']=per.error_sum/per.n
 agg=per.mae.groupby('model').mean().to_dict()
 h.to_csv(D/'home_characteristics_private.csv');monthly.to_csv(D/'monthly.csv');r.to_csv(D/'benchmark_windows_private.csv',index=False)
 plt.rcParams.update({'font.size':10})
 fig,axs=plt.subplots(1,2,figsize=(7,3.2))
 for ax,x,label in zip(axs,['history_sd','history_change'],['Prior-history temperature SD (°C)','Prior-history mean |hourly change| (°C)']):
  ax.scatter(h[x],h.selector_minus_local,s=18,alpha=.7);ax.axhline(0,color='gray',lw=.8);ax.set_xlabel(label);ax.set_ylabel('Selector minus local MAE (°C)')
 fig.tight_layout();fig.savefig(D/'figure3_heterogeneity.png',dpi=600);plt.close(fig)
 fig,ax=plt.subplots(figsize=(6.5,3)); ax.plot(range(len(monthly)),monthly['mean'],marker='o');ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(len(monthly)),[pd.Timestamp(x).strftime('%b %y') for x in monthly.index],rotation=30);ax.set_ylabel('Mean paired MAE difference (°C)')
 for i,row in enumerate(monthly.itertuples()):ax.annotate(f'n={row.count}',(i,row.mean),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
 fig.tight_layout();fig.savefig(D/'figure4_monthly.png',dpi=600);plt.close(fig)
 assert digest(primary)==before
 result={'status':'post_result_exploratory','homes':len(h),'correlations':corr,'benchmark_equal_home_mae':agg,'benchmark_homes':int(r.home.nunique()),'benchmark_home_months':int(len(r)/6),'benchmark_test_rows':int(r[r.model.eq('persistence')].n.sum()),'primary_unchanged':True,'input_sha256':hashes,'protocol_sha256':digest(ROOT/'protocol_revision_v1.md')}
 (D/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2),flush=True)
if __name__=='__main__':run()

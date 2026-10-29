"""Exploratory two-year NEST evaluation with feature-loss augmentation."""
from __future__ import annotations
import json
import urllib.request
import numpy as np
import pandas as pd
from audit_nest import DATA, OUT, ITEMS, download, digest
from transfer_pilot import frames
from adaptation_stress import COLS, fit, predict, adapt, quantile
from temporal_windows import window

YEAR2 = {'UMAR':43480689, 'SolAce':43480707, 'DFAB':43480695}
SOURCE_END = pd.Timestamp('2020-11-01')
ADAPT_START = SOURCE_END
CAL_START, CAL_END = pd.Timestamp('2020-12-01'), pd.Timestamp('2021-01-01')
SEASONS = {'winter':(pd.Timestamp('2021-01-01'),pd.Timestamp('2021-04-01')),
           'spring':(pd.Timestamp('2021-04-01'),pd.Timestamp('2021-07-01'))}

def ensure_files():
    rows=[]
    for unit,(article,_,_,_) in ITEMS.items():
        fid=YEAR2[unit]; name=unit.lower()+'_2020-2021.csv'; path=DATA/name
        print('Checking',name,flush=True)
        download(fid,path)
        meta=json.load(urllib.request.urlopen(f'https://api.figshare.com/v2/articles/{article}',timeout=30))
        info=next(f for f in meta['files'] if f['id']==fid)
        actual=digest(path,'md5')
        if actual!=info['computed_md5'] or path.stat().st_size!=info['size']:
            raise ValueError('Source checksum mismatch: '+unit)
        rows.append({'unit':unit,'file_id':fid,'filename':name,'bytes':path.stat().st_size,'sha256':digest(path,'sha256'),'md5':actual,'figshare_md5':info['computed_md5']})
    (OUT/'manifest_year2.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

def augment_features(x, mu, sd, scenario):
    z=(x-mu)/sd
    flags=np.zeros((len(x),2))
    if scenario=='missing_ambient': z[:,2]=0; flags[:,0]=1
    if scenario=='missing_solar': z[:,3]=0; flags[:,1]=1
    return np.column_stack([np.ones(len(x)),z,flags])

def augmented_fit(x,y):
    mu=x.mean(axis=0); sd=x.std(axis=0); sd[sd<1e-8]=1
    zs=[augment_features(x,mu,sd,s) for s in ['observed','missing_ambient','missing_solar']]
    reg=np.diag([0.]+[100.]*(zs[0].shape[1]-1))
    gram=reg.copy(); rhs=np.zeros(zs[0].shape[1])
    for z in zs:
        gram += z.T@z; rhs += z.T@y
    return mu,sd,np.linalg.solve(gram,rhs)

def augmented_predict(model,x,scenario):
    mu,sd,b=model
    return augment_features(x,mu,sd,scenario)@b

def score(cal,test,model,scenario):
    cx=cal[COLS].to_numpy(); tx=test[COLS].to_numpy().copy()
    kind,m=model
    if kind=='persistence':
        pc=np.zeros(len(cal)); pt=np.zeros(len(test))
    elif kind=='augmented':
        pc=augmented_predict(m,cx,'observed'); pt=augmented_predict(m,tx,scenario)
    else:
        pc=predict(m,cx)
        if scenario!='observed':
            pos=2 if scenario=='missing_ambient' else 3
            tx[:,pos]=m[0][pos]
        pt=predict(m,tx)
    q=quantile(np.abs(cal.temp_now.to_numpy()+pc-cal.target.to_numpy()))
    err=test.temp_now.to_numpy()+pt-test.target.to_numpy()
    return {'mae_c':np.abs(err).mean(), 'rmse_c':np.sqrt(np.square(err).mean()),
            'coverage90':np.mean(np.abs(err)<=q) if np.isfinite(q) else np.nan,
            'width90_c':2*q}

def main():
    ensure_files(); datasets=[]
    for unit,item in ITEMS.items():
        # Process each year separately: no accidental interpolation at file boundaries.
        for name in [item[2],unit.lower()+'_2020-2021.csv']:
            datasets.append(frames(unit,name,item[3]))
    d=pd.concat(datasets); tt=d.index+pd.Timedelta(hours=1)
    results=[]
    for unit in ITEMS:
        src=d[(d.unit!=unit)&(tt<SOURCE_END)]
        sx=src[COLS].to_numpy(); sy=(src.target-src.temp_now).to_numpy()
        source=fit(sx,sy); aug=augmented_fit(sx,sy)
        for room in d[d.unit==unit].room.unique():
            rr=d[(d.unit==unit)&(d.room==room)]; rt=rr.index+pd.Timedelta(hours=1)
            a=rr[(rt>=ADAPT_START)&(rt<ADAPT_START+pd.Timedelta(days=7))]
            cal=window(rr,1,CAL_START,CAL_END,CAL_START)
            models={'persistence':('persistence',None),'source_ridge':('ridge',source),'source_mask_augmented':('augmented',aug)}
            if len(a)>=24:
                ax=a[COLS].to_numpy(); ay=(a.target-a.temp_now).to_numpy()
                models['source_adapt_7d']=('ridge',adapt(source,ax,ay))
                models['target_only_7d']=('ridge',fit(ax,ay))
            for season,(start,end) in SEASONS.items():
                test=window(rr,1,start,end,CAL_END)
                if len(cal)<24 or len(test)==0: continue
                for label,model in models.items():
                    for sc in ['observed','missing_ambient','missing_solar']:
                        row={'unit':unit,'room':room,'season':season,'model':label,'scenario':sc,
                             'source_train_n':len(src),'target_7d_n':len(a),'cal_n':len(cal),'test_n':len(test)}
                        row.update(score(cal,test,model,sc)); results.append(row)
        print(unit,'done',flush=True)
    pd.DataFrame(results).round(6).to_csv(OUT/'multiseason_results.csv',index=False)

if __name__=='__main__': main()

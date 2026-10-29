"""Joint and state-dependent synthetic missingness; see frozen protocol."""
import json,hashlib
from unittest.mock import patch
import numpy as np
import pandas as pd
from audit_nest import OUT
from audit_bremen import HOURLY,FILES,verify
from ideal_development_prepare import select
from ideal_development_v1 import load_sensor,H,SOURCE_END
from thermbuild_gating import FEATURES
from adaptation_stress import fit
import bremen_outage_stress_v1 as core

def create_masks(name,rep,sensors):
    names=list(sensors);n=len(next(iter(sensors.values())))
    if name=='common_iid10':
        rng=np.random.default_rng(20261029+11000+10*rep);a=rng.random(n)<.10
        return {k:a.copy() for k in names}
    if name=='common_block24_14d':
        a=core.make_mask(n,'block24h_14d',(24,336),20261029+12000+10*rep)
        return {k:a.copy() for k in names}
    if name=='temperature_dependent':
        masks={}
        for room_i,k in enumerate(names):
            lag=sensors[k].shift(1).to_numpy();p=np.where(np.isfinite(lag)&(lag>=25),.5,.05)
            masks[k]=np.random.default_rng(20261029+13000+10*rep+room_i).random(n)<p
        return masks
    raise ValueError(name)

def main():
    path=OUT/'bremen_joint_missing_v1_summary.json';protocol=OUT.parent/'protocol_bremen_joint_missing_v1.md'
    if path.exists():raise ValueError('Existing results protected')
    manifest=json.loads((OUT/'bremen_manifest.json').read_text(encoding='utf8'))
    for e in manifest['files']:
        if hashlib.sha256((HOURLY/e['hourly_file']).read_bytes()).hexdigest()!=e['hourly_sha256_local']:raise ValueError('Hourly mismatch')
    for n,(size,blob) in FILES.items():verify(OUT.parent/'data'/'bremen'/n,size,blob)
    selected=select('development').reset_index(drop=True)
    if not selected.equals(pd.read_csv(OUT/'ideal_development_sensors_v1.csv')):raise ValueError('Cohort mismatch')
    parts=[]
    for x in selected[selected.dev_role.eq('source')].itertuples(index=False):
        s=load_sensor(int(x.sensor)); d=s[s.index+pd.Timedelta(hours=H)<SOURCE_END]
        if len(d)>=24:parts.append(d)
    train=pd.concat(parts)
    if len(parts)!=55 or len(train)!=125779:raise ValueError('Source mismatch')
    model=fit(train[FEATURES].to_numpy(),(train.target-train.seasonal_yesterday).to_numpy())
    timeline=pd.date_range(core.MASK_START,core.END,freq='h',inclusive='left')
    sensors={}
    for n in FILES:
        df=pd.read_csv(HOURLY/(n[:-4]+'.csv.gz'),parse_dates=['hour_utc'])
        if df.hour_utc.duplicated().any():raise ValueError('Duplicate timestamps')
        sensors[n[:-4]]=df.set_index('hour_utc').temp_c.reindex(timeline)
    original=pd.read_csv(OUT/'bremen_fallback_v1_sensors.csv').set_index(['sensor','model'])
    baseline={}
    for j,(name,series) in enumerate(sensors.items()):
        with patch.object(core,'make_mask',return_value=np.zeros(len(series),bool)):
            m,_,_,points,_=core.evaluate_room(series,model,'none',0,0,0,j)
        agg=pd.DataFrame(m).groupby('model')[['opportunities','ready','scored','abs_error_sum']].sum()
        for label,row in agg.iterrows():
            ref=original.loc[(name,label)]
            for col in ['opportunities','ready','scored','abs_error_sum']:
                if not np.isclose(row[col],ref[col],atol=1e-6,rtol=1e-9):raise ValueError('Baseline mismatch')
        baseline[name]=points.set_index('idx').fallback
    rows=[];synchronous=[];choices=[];branches=[]
    for scenario in ['common_iid10','common_block24_14d','temperature_dependent']:
        for rep in range(5):
            masks=create_masks(scenario,rep,sensors);points_all=[]
            for j,(name,s) in enumerate(sensors.items()):
                mask=masks[name]
                with patch.object(core,'make_mask',return_value=mask):
                    method,br,ch,points,_=core.evaluate_room(s,model,scenario,1,0,rep,j)
                keys={'scenario':scenario,'replicate':rep,'sensor':name}
                choices.extend([{**keys,**x} for x in ch]);branches.extend([{**keys,**x} for x in br])
                x=points.copy().set_index('idx'); target=x.target.to_numpy();fb=x.fallback.to_numpy();per=x.persistence.to_numpy()
                targetmask=pd.Series(mask,index=timeline).reindex(x.index+pd.Timedelta(hours=H)).fillna(True).to_numpy(dtype=bool)
                prior=s.reindex(x.index-pd.Timedelta(hours=1)).to_numpy()
                original_ok=np.isfinite(target);same_ok=original_ok&~targetmask
                common=np.isfinite(fb)&np.isfinite(per);ready=np.isfinite(fb)
                base=baseline[name].reindex(x.index).to_numpy();stable=original_ok&ready&np.isfinite(base)
                entry={**keys,'opportunities':len(x),'fallback_ready':int(ready.sum()),'persistence_ready':int(np.isfinite(per).sum()),'oracle_target':int(original_ok.sum()),'same_sensor_target':int(same_ok.sum()),'baseline_common_n':int(stable.sum()),'baseline_error_sum':float(np.abs(base[stable]-target[stable]).sum()),'stress_error_sum':float(np.abs(fb[stable]-target[stable]).sum()),'latent_warm_count':int((np.isfinite(prior)&(prior>=25)).sum()),'latent_warm_ready':int((np.isfinite(prior)&(prior>=25)&ready).sum()),'latent_cool_count':int((np.isfinite(prior)&(prior<25)).sum()),'latent_cool_ready':int((np.isfinite(prior)&(prior<25)&ready).sum()),'synthetic_mask_count':int(mask.sum())}
                for tag,obs in [('oracle',original_ok),('same_sensor',same_ok)]:
                    cp=obs&common; extra=obs&ready&~np.isfinite(per)
                    entry.update({tag+'_paired_n':int(cp.sum()),tag+'_fallback_error':float(np.abs(fb[cp]-target[cp]).sum()),tag+'_persistence_error':float(np.abs(per[cp]-target[cp]).sum()),tag+'_extra_scored':int(extra.sum()),tag+'_extra_error':float(np.abs(fb[extra]-target[extra]).sum())})
                rows.append(entry)
                points_all.append(pd.DataFrame({'idx':x.index,'sensor':name,'ready':ready,'same_target':same_ok}))
            allp=pd.concat(points_all).groupby('idx').agg(rooms=('sensor','size'),ready=('ready','sum'),same_target=('same_target','sum'))
            assert (allp.rooms==4).all()
            synchronous.append({'scenario':scenario,'replicate':rep,'common_issue_times':len(allp),'all_four_ready':int((allp.ready==4).sum()),'all_four_target_observed':int((allp.same_target==4).sum())})
            print(scenario,rep,flush=True)
    d=pd.DataFrame(rows);sync=pd.DataFrame(synchronous);c=pd.DataFrame(choices);b=pd.DataFrame(branches)
    for obj,label in [(d,'rooms'),(sync,'synchronous'),(c,'choices'),(b,'branches')]:obj.to_csv(OUT/f'bremen_joint_missing_v1_{label}.csv',index=False,float_format='%.10f')
    stats=[]
    for (scenario,rep),g in d.groupby(['scenario','replicate']):
        q=sync[(sync.scenario==scenario)&(sync.replicate==rep)].iloc[0]
        row={'scenario':scenario,'replicate':int(rep),'opportunities':int(g.opportunities.sum()),'fallback_ready':int(g.fallback_ready.sum()),'persistence_ready':int(g.persistence_ready.sum()),'oracle_target':int(g.oracle_target.sum()),'same_sensor_target':int(g.same_sensor_target.sum()),'all_four_ready':int(q.all_four_ready),'all_four_target_observed':int(q.all_four_target_observed),'common_issue_times':int(q.common_issue_times),'warm_ready_fraction':float(g.latent_warm_ready.sum()/g.latent_warm_count.sum()),'cool_ready_fraction':float(g.latent_cool_ready.sum()/g.latent_cool_count.sum())}
        for tag in ['oracle','same_sensor']:
            n=g[tag+'_paired_n'];row[tag+'_paired_n']=int(n.sum());row[tag+'_fallback_equal_room_mae']=float((g[tag+'_fallback_error']/n.replace(0,np.nan)).mean());row[tag+'_persistence_equal_room_mae']=float((g[tag+'_persistence_error']/n.replace(0,np.nan)).mean());row[tag+'_extra_scored']=int(g[tag+'_extra_scored'].sum())
        n=g.baseline_common_n;row['stress_equal_room_mae_same_baseline_support']=float((g.stress_error_sum/n).mean());row['baseline_equal_room_mae_same_support']=float((g.baseline_error_sum/n).mean())
        stats.append(row)
    pd.DataFrame(stats).to_csv(OUT/'bremen_joint_missing_v1_replicates.csv',index=False,float_format='%.10f')
    path.write_text(json.dumps({'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'baseline_all_models_reproduced':True,'scenarios':stats,'scope':'Four rooms; common outage simulated, state failure hypothetical; no real uptime or independent sites.'},indent=2),encoding='utf8')
    print('Done; baseline confirmed',flush=True)
if __name__=='__main__':main()

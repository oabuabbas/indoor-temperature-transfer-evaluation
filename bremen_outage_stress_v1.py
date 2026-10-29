"""Synthetic input outage experiment. Original target is an oracle for scoring only."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_nest import OUT
from audit_bremen import HOURLY, FILES, verify
from ideal_development_prepare import select
from ideal_development_v1 import load_sensor, frame_from_series, H, SOURCE_END
from thermbuild_gating import FEATURES, CANDIDATES, predictions
from adaptation_stress import fit, predict
from bremen_fallback_v1 import START, END, route

SCENARIOS=[('none',0,0),('iid05',1,.05),('iid20',2,.20),('block6h_7d',3,(6,168)),('block24h_14d',4,(24,336)),('block72h_30d',5,(72,720))]
MASK_START=pd.Timestamp('2023-01-25',tz='UTC')

def make_mask(n, kind, param, seed):
    if kind=='none': return np.zeros(n,dtype=bool)
    rng=np.random.default_rng(seed)
    if kind.startswith('iid'): return rng.random(n)<param
    length,period=param
    phase=int(rng.integers(period))
    return (np.arange(n)-phase)%period<length

def evaluate_room(s, model, scenario, i, param, rep, room_i, baseline=None):
    seed=20261029+1000*i+10*rep+room_i
    mask=make_mask(len(s),scenario,param,seed)
    masked=s.mask(mask)
    frame=frame_from_series(masked)
    method_rows=[];branch_rows=[];choice_rows=[];point_rows=[]
    for start in pd.date_range(START,END,freq='MS',inclusive='left'):
        end=min(start+pd.offsets.MonthBegin(1),END)
        history=frame[(frame.index>=start-pd.Timedelta(days=30)) & (frame.index+pd.Timedelta(hours=H)<start)]
        choice=None
        if len(history)>=24:
            errors=predictions(history,model).sub(history.target,axis=0).abs().mean()
            choice=min(CANDIDATES,key=lambda k:errors[k])
        choice_rows.append({'month':start.isoformat(),'history_n':len(history),'choice':choice})
        idx=pd.date_range(start,end-pd.Timedelta(hours=H),freq='h',inclusive='left')
        now=masked.reindex(idx).to_numpy();lag1=masked.reindex(idx-pd.Timedelta(hours=1)).to_numpy()
        daily=masked.reindex(idx-pd.Timedelta(hours=18)).to_numpy()
        target=s.reindex(idx+pd.Timedelta(hours=H)).to_numpy()
        features=np.column_stack([now,lag1,daily]); good=np.isfinite(features).all(axis=1)
        src=np.full(len(idx),np.nan)
        if good.any():src[good]=daily[good]+predict(model,features[good])
        selector=np.full(len(idx),np.nan);local=np.full(len(idx),np.nan)
        if choice is not None:
            candidate={'persistence':now,'daily_seasonal':daily,'source_daily_ridge':src}[choice]
            selector[good]=candidate[good]
            loc=fit(history[FEATURES].to_numpy(),(history.target-history.seasonal_yesterday).to_numpy())
            if good.any():local[good]=daily[good]+predict(loc,features[good])
        fallback,branch=route(selector,now,daily)
        methods={'persistence':now,'daily_seasonal':daily,'source_daily_ridge':src,'local_ridge_30d':local,'rolling_selector':selector,'fallback':fallback}
        observed=np.isfinite(target)
        for name,pred in methods.items():
            ready=np.isfinite(pred);scored=ready & observed
            method_rows.append({'model':name,'opportunities':len(idx),'ready':int(ready.sum()),'scored':int(scored.sum()),'abs_error_sum':float(np.abs(pred[scored]-target[scored]).sum())})
        for name in ['selector','persistence','daily','abstain']:
            take=branch==name; scored=take & observed & np.isfinite(fallback)
            branch_rows.append({'branch':name,'opportunities':int(take.sum()),'scored':int(scored.sum()),'abs_error_sum':float(np.abs(fallback[scored]-target[scored]).sum())})
        point_rows.append(pd.DataFrame({'idx':idx,'target':target,'fallback':fallback,'persistence':now,'source':src,'selector':selector}))
    points=pd.concat(point_rows,ignore_index=True)
    return method_rows,branch_rows,choice_rows,points,{'synthetic_mask_fraction_full_period':float(mask.mean()),'seed':seed}

def main():
    protocol=OUT.parent/'protocol_bremen_outage_stress_v1.md'; path=OUT/'bremen_outage_stress_v1_summary.json'
    if path.exists():raise ValueError('Existing stress results must not be overwritten')
    original=json.loads((OUT/'bremen_fallback_v1_summary.json').read_text(encoding='utf8'))
    manifest=json.loads((OUT/'bremen_manifest.json').read_text(encoding='utf8'))
    for e in manifest['files']:
        if hashlib.sha256((HOURLY/e['hourly_file']).read_bytes()).hexdigest()!=e['hourly_sha256_local']:raise ValueError('Hourly checksum changed')
    for name,(size,blob) in FILES.items():verify(OUT.parent/'data'/'bremen'/name,size,blob)
    selected=select('development').reset_index(drop=True)
    if not pd.read_csv(OUT/'ideal_development_sensors_v1.csv').equals(selected):raise ValueError('Source selection mismatch')
    parts=[]
    for row in selected[selected.dev_role.eq('source')].itertuples(index=False):
        d=load_sensor(int(row.sensor)); d=d[d.index+pd.Timedelta(hours=H)<SOURCE_END]
        if len(d)>=24:parts.append(d)
    source=pd.concat(parts)
    if len(parts)!=55 or len(source)!=125779:raise ValueError('Source counts mismatch')
    model=fit(source[FEATURES].to_numpy(),(source.target-source.seasonal_yesterday).to_numpy())
    sensors={}
    for f in FILES:
        d=pd.read_csv(HOURLY/(f[:-4]+'.csv.gz'),parse_dates=['hour_utc'])
        if d.hour_utc.duplicated().any():raise ValueError('Duplicate hours')
        sensors[f[:-4]]=d.set_index('hour_utc').temp_c.reindex(pd.date_range(MASK_START,END,freq='h',inclusive='left'))
    methods=[];branches=[];choices=[];pairs=[];metadata=[];baseline={}
    for scenario,i,param in SCENARIOS:
        for rep in range(1 if i==0 else 5):
            for room_i,(name,s) in enumerate(sensors.items()):
                ms,bs,cs,points,meta=evaluate_room(s,model,scenario,i,param,rep,room_i)
                keys={'scenario':scenario,'replicate':rep,'sensor':name}
                for x in ms:methods.append({**keys,**x})
                for x in bs:branches.append({**keys,**x})
                for x in cs:choices.append({**keys,**x})
                metadata.append({**keys,**meta})
                target=points.target.to_numpy(); fb=points.fallback.to_numpy(); per=points.persistence.to_numpy()
                common=np.isfinite(target)&np.isfinite(fb)&np.isfinite(per)
                extra=np.isfinite(target)&np.isfinite(fb)&~np.isfinite(per)
                pair={'n':int(common.sum()),'fallback_abs_sum':float(np.abs(fb[common]-target[common]).sum()),'persistence_abs_sum':float(np.abs(per[common]-target[common]).sum()),'extra_scored':int(extra.sum()),'extra_abs_sum':float(np.abs(fb[extra]-target[extra]).sum())}
                if scenario=='none': baseline[name]=points.set_index('idx').fallback
                b=baseline[name].reindex(points.idx).to_numpy(); stable=np.isfinite(target)&np.isfinite(fb)&np.isfinite(b)
                pair.update({'baseline_common_n':int(stable.sum()),'stress_abs_sum':float(np.abs(fb[stable]-target[stable]).sum()),'baseline_abs_sum':float(np.abs(b[stable]-target[stable]).sum())})
                pairs.append({**keys,**pair})
            print(scenario,rep,flush=True)
    m=pd.DataFrame(methods).groupby(['scenario','replicate','sensor','model'],as_index=False)[['opportunities','ready','scored','abs_error_sum']].sum()
    b=pd.DataFrame(branches).groupby(['scenario','replicate','sensor','branch'],as_index=False)[['opportunities','scored','abs_error_sum']].sum()
    c=pd.DataFrame(choices);p=pd.DataFrame(pairs);meta=pd.DataFrame(metadata)
    bm=m[(m.scenario=='none') & (m.model=='fallback')]
    base=original['methods_own_support']['fallback']
    if int(bm.ready.sum())!=base['ready'] or int(bm.scored.sum())!=base['scored'] or not np.isclose((bm.abs_error_sum/bm.scored).mean(),base['equal_sensor_mean_own_support_mae_c'],atol=1e-9):raise ValueError('No-outage fallback baseline mismatch')
    bp=p[p.scenario=='none']; expected=original['paired_fallback_vs_persistence']
    if int(bp.n.sum())!=expected['common_issue_times'] or not np.isclose((bp.fallback_abs_sum/bp.n).mean(),expected['equal_room_fallback_mae_c'],atol=1e-9):raise ValueError('No-outage pair baseline mismatch')
    stats=[]
    for (scenario,rep),group in m.groupby(['scenario','replicate']):
        fallback=group[group.model=='fallback'];pers=group[group.model=='persistence'];sel=group[group.model=='rolling_selector'];src=group[group.model=='source_daily_ridge']
        q=p[(p.scenario==scenario)&(p.replicate==rep)]
        row={'scenario':scenario,'replicate':int(rep),'opportunities':int(fallback.opportunities.sum()),'fallback_ready':int(fallback.ready.sum()),'persistence_ready':int(pers.ready.sum()),'selector_ready':int(sel.ready.sum()),'source_ready':int(src.ready.sum()),'fallback_scored':int(fallback.scored.sum()),'paired_n':int(q.n.sum()),'paired_fallback_equal_room_mae':float((q.fallback_abs_sum/q.n.replace(0,np.nan)).mean()),'paired_persistence_equal_room_mae':float((q.persistence_abs_sum/q.n.replace(0,np.nan)).mean()),'extra_scored':int(q.extra_scored.sum()),'extra_pooled_mae':float(q.extra_abs_sum.sum()/q.extra_scored.sum()) if q.extra_scored.sum() else None,'baseline_common_n':int(q.baseline_common_n.sum()),'stress_same_support_equal_room_mae':float((q.stress_abs_sum/q.baseline_common_n.replace(0,np.nan)).mean()),'baseline_same_support_equal_room_mae':float((q.baseline_abs_sum/q.baseline_common_n.replace(0,np.nan)).mean())}
        stats.append(row)
    for obj,name in [(m,'methods'),(b,'branches'),(c,'choices'),(p,'pairs'),(meta,'mask_metadata'),(pd.DataFrame(stats),'replicates')]:obj.to_csv(OUT/('bremen_outage_stress_v1_'+name+'.csv'),index=False,float_format='%.10f')
    summary={'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'source_homes':len(parts),'source_rows':len(source),'baseline_reproduced':True,'replicates':stats,'interpretation':'Synthetic input outages; unmasked original outcomes are a retrospective oracle. Four correlated rooms; no live operational or population inference.'}
    path.write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('Finished; baseline confirmed.',flush=True)
if __name__=='__main__':main()

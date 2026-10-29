"""Exploratory seven-model matched-support comparison across Bremen room series."""
import hashlib,json,sys
from unittest.mock import patch
import numpy as np
import pandas as pd
import sklearn
from audit_nest import OUT
from audit_bremen import HOURLY, FILES, verify
from bremen_fallback_v1 import START,END
import bolzano_external_v1 as base
from bolzano_external_v1 import source_model
from bolzano_2026_benchmark_v1 import add_boosting,METHODS
ROOT=OUT.parent

def verify_old_results(room,d,choices,old,oldchoices):
    o=old[old.sensor.eq(room)].set_index(['month','model'])
    for month,g in d.groupby('month'):
        observed=g.target.notna()
        for model in METHODS[:-1]:
            entry=o.loc[(month,model)];ready=g[model].notna();scored=ready&observed
            err=(g.loc[scored,model]-g.loc[scored,'target']).abs().sum()
            if len(g)!=entry.opportunities or ready.sum()!=entry.ready or scored.sum()!=entry.scored or not np.isclose(err,entry.abs_error_sum,rtol=1e-9,atol=1e-6):raise ValueError(f'Baseline mismatch: {room} {month} {model}')
    saved=oldchoices[oldchoices.sensor.eq(room)].sort_values('month').reset_index(drop=True)
    c=choices.sort_values('month').reset_index(drop=True)
    if not (saved.month.equals(c.month) and saved.history_n.equals(c.history_n) and saved.choice.fillna('NA').equals(c.choice.fillna('NA'))):raise ValueError(f'Baseline choices mismatch {room}')

def compute(d):
    observed=d.target.notna();common=observed & d[METHODS].notna().all(axis=1);rooms=[];months=[]
    for room,g in d.groupby('sensor'):
        s=g[common.loc[g.index]]
        if s.empty:raise ValueError('No common support')
        for model in METHODS:
            err=s[model]-s.target
            rooms.append({'sensor':room,'model':model,'n':len(s),'abs_sum':float(err.abs().sum()),'squared_sum':float((err**2).sum()),'mae_c':float(err.abs().mean()),'rmse_c':float(np.sqrt((err**2).mean()))})
        for month,v in s.groupby('month'):
            for model in METHODS:
                err=v[model]-v.target
                months.append({'sensor':room,'month':month,'model':model,'n':len(v),'abs_sum':float(err.abs().sum()),'squared_sum':float((err**2).sum()),'mae_c':float(err.abs().mean()),'rmse_c':float(np.sqrt((err**2).mean()))})
    return pd.DataFrame(rooms),pd.DataFrame(months)

def main():
    protocol=ROOT/'protocol_bremen_nonlinear_benchmark_v1.md';out=OUT/'bremen_nonlinear_benchmark_v1_summary.json'
    if out.exists():raise ValueError('Existing benchmark results protected')
    prior=json.loads((OUT/'bremen_fallback_v1_summary.json').read_text(encoding='utf8'))
    if hashlib.sha256((ROOT/'protocol_bremen_fallback_v1.md').read_bytes()).hexdigest()!=prior['protocol_sha256']:raise ValueError('Prior protocol changed')
    manifest=json.loads((OUT/'bremen_manifest.json').read_text(encoding='utf8'))
    for x in manifest['files']:
        if hashlib.sha256((HOURLY/x['hourly_file']).read_bytes()).hexdigest()!=x['hourly_sha256_local']:raise ValueError('Hourly file changed')
        if hashlib.sha256((ROOT/'data'/'bremen'/x['file']).read_bytes()).hexdigest()!=x['sha256_local']:raise ValueError('Raw file changed')
    for n,(size,sha) in FILES.items():verify(ROOT/'data'/'bremen'/n,size,sha)
    old=pd.read_csv(OUT/'bremen_fallback_v1_months.csv');oldchoices=pd.read_csv(OUT/'bremen_fallback_v1_choices.csv');model=source_model()
    cached=[]
    for filename in FILES:
        name=filename[:-4];t=pd.read_csv(HOURLY/f'{name}.csv.gz',parse_dates=['hour_utc'])
        if t.hour_utc.duplicated().any():raise ValueError('Duplicate hourly timestamps')
        s=t.set_index('hour_utc').temp_c.sort_index().asfreq('1h')
        with patch.object(base,'START',START),patch.object(base,'END',END): d,c=base.evaluate(s,model)
        verify_old_results(name,d,c,old,oldchoices)
        cached.append((name,s,d,c))
    print('All 40 room-months: existing six-model counts, error sums, and choices reconstructed.',flush=True)
    points=[];choice_rows=[]
    for name,s,d,c in cached:
        d=add_boosting(d,s,c);d.insert(0,'sensor',name);c.insert(0,'sensor',name);points.append(d);choice_rows.append(c)
        print('Evaluated',name,flush=True)
    d=pd.concat(points,ignore_index=True);choices=pd.concat(choice_rows,ignore_index=True);rooms,monthly=compute(d)
    stats=[]
    for model,g in rooms.groupby('model'):
        stats.append({'model':model,'rooms':len(g),'n':int(g.n.sum()),'equal_room_mae_c':float(g.mae_c.mean()),'equal_room_rmse_c':float(g.rmse_c.mean())})
    own=[]
    for (room,model),g in d.melt(id_vars=['sensor','target'],value_vars=METHODS,var_name='model',value_name='prediction').groupby(['sensor','model']):
        ready=g.prediction.notna();scored=ready&g.target.notna();own.append({'sensor':room,'model':model,'opportunities':len(g),'ready':int(ready.sum()),'scored':int(scored.sum()),'abs_sum':float((g.loc[scored,'prediction']-g.loc[scored,'target']).abs().sum())})
    contrasts=[]
    for comparator in ['source_daily_ridge','local_ridge_30d','local_hist_gradient_boosting']:
        v=rooms[rooms.model.eq('rolling_selector')].set_index('sensor').mae_c-rooms[rooms.model.eq(comparator)].set_index('sensor').mae_c
        pivot=monthly.pivot(index=['sensor','month'],columns='model',values='mae_c');diff=pivot.rolling_selector-pivot[comparator]
        contrasts.append({'comparator':comparator,'rooms_better':int((v < -1e-8).sum()),'rooms_worse':int((v>1e-8).sum()),'rooms_tied':int((v.abs()<=1e-8).sum()),'room_months_better':int((diff < -1e-8).sum()),'room_months_worse':int((diff>1e-8).sum()),'room_months_tied':int((diff.abs()<=1e-8).sum())})
    d.to_csv(OUT/'bremen_nonlinear_benchmark_v1_predictions.csv.gz',index=False,compression='gzip',float_format='%.10f')
    for x,label in [(rooms,'rooms'),(monthly,'monthly'),(choices,'choices'),(pd.DataFrame(own),'own_support')]:x.to_csv(OUT/f'bremen_nonlinear_benchmark_v1_{label}.csv',index=False,float_format='%.10f')
    result={'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'prior_protocol_sha256':prior['protocol_sha256'],'reconstructed_prior_room_months':len(cached)*10,'source_homes':55,'source_rows':125779,'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__},'opportunities':len(d),'observed_targets':int(d.target.notna().sum()),'common_issue_times':int((d.target.notna()&d[METHODS].notna().all(axis=1)).sum()),'metrics_equal_room':stats,'selector_contrasts':contrasts,'branches':{str(k):int(v) for k,v in d.branch.value_counts().items()},'scope':'Post-result exploratory analysis, four correlated room series, not independent sites; no energy or control inference.'}
    out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

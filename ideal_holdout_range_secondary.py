"""Post-primary, paired-issue holdout sensitivity; does not alter primary files."""
import hashlib,json
import numpy as np
import pandas as pd
from audit_nest import OUT
from adaptation_stress import fit,predict
from thermbuild_gating import FEATURES,CANDIDATES,predictions
from ideal_development_prepare import HOURLY as DEV_WIDE,select
from ideal_prepare_strict import STRICT as DEV_STRICT
from ideal_holdout_prepare import HOURLY as TARGET_WIDE
from ideal_holdout_prepare_strict import STRICT as TARGET_STRICT
from ideal_prediction_range import load
from ideal_development_v1 import windows,H,SOURCE_END,TEST_START,TEST_END

def main():
    output=OUT/'ideal_holdout_range_secondary_summary.json'
    if output.exists():raise ValueError('Secondary results already exist')
    protocol=OUT.parent/'protocol_ideal_holdout_range_secondary.md'
    if not protocol.exists() or not (OUT/'ideal_holdout_v1_summary.json').exists():raise ValueError('Protocol or primary results missing')
    src=select('development').reset_index(drop=True)
    if not pd.read_csv(OUT/'ideal_development_sensors_v1.csv').equals(src):raise ValueError('Development cohort changed')
    target=select('holdout').drop(columns='dev_role').reset_index(drop=True)
    if not pd.read_csv(OUT/'ideal_holdout_sensors_v1.csv').equals(target):raise ValueError('Holdout cohort changed')
    if set(src.home)&set(target.home):raise ValueError('Overlapping homes')
    source_dirs={'10_45':DEV_WIDE,'15_40':DEV_STRICT}
    target_dirs={'10_45':TARGET_WIDE,'15_40':TARGET_STRICT}
    models={};source_stats=[]
    for key,folder in source_dirs.items():
        pieces=[]
        for x in src[src.dev_role.eq('source')].itertuples(index=False):
            frame=load(folder,int(x.sensor));part=frame[frame.index+pd.Timedelta(hours=H)<SOURCE_END]
            if len(part)>=24:pieces.append(part)
        if not pieces:raise ValueError('No source examples')
        data=pd.concat(pieces);models[key]=fit(data[FEATURES].to_numpy(),(data.target-data.seasonal_yesterday).to_numpy())
        source_stats.append({'filter':key,'source_homes':len(pieces),'source_rows':len(data)})
    if source_stats[0]['source_homes']!=55 or source_stats[0]['source_rows']!=125779:raise ValueError('Primary source changed')
    starts=pd.date_range(TEST_START,TEST_END,freq='MS',inclusive='left'); rows=[];target_diff=[];excluded=[]
    for x in target.itertuples(index=False):
        frames={key:load(folder,int(x.sensor)) for key,folder in target_dirs.items()}
        for start in starts:
            end=min(start+pd.offsets.MonthBegin(1),TEST_END)
            pair={key:windows(frame,start,end) for key,frame in frames.items()}
            common=pair['10_45'][1].index.intersection(pair['15_40'][1].index)
            if len(common)<24 or any(len(v[0])<24 for v in pair.values()):
                excluded.append({'home':int(x.home),'period_start':start.isoformat(),'common_n':len(common),
                                 'wide_history_n':len(pair['10_45'][0]),'strict_history_n':len(pair['15_40'][0])})
                continue
            diff=np.abs(pair['10_45'][1].loc[common,'target'].to_numpy()-pair['15_40'][1].loc[common,'target'].to_numpy())
            target_diff.append({'home':int(x.home),'period_start':start.isoformat(),'n':len(common),
                                'changed_n':int((diff>1e-9).sum()),'abs_sum_c':float(diff.sum())})
            for key,(history,test) in pair.items():
                test=test.loc[common]; prior=predictions(history,models[key]).sub(history.target,axis=0).abs().mean()
                choice=min(CANDIDATES,key=lambda c:prior[c])
                pred=predictions(test,models[key])
                local=fit(history[FEATURES].to_numpy(),(history.target-history.seasonal_yesterday).to_numpy())
                pred['local_ridge_30d']=test.seasonal_yesterday.to_numpy()+predict(local,test[FEATURES].to_numpy())
                for label in CANDIDATES+['rolling_selector','local_ridge_30d']:
                    use=choice if label=='rolling_selector' else label
                    err=np.abs(pred[use].to_numpy()-test.target.to_numpy())
                    rows.append({'filter':key,'home':int(x.home),'sensor':int(x.sensor),'period_start':start.isoformat(),
                                 'model':label,'choice':choice,'history_n':len(history),'test_n':len(test),
                                 'error_sum':float(err.sum()),'mae_c':float(err.mean())})
    if not rows:raise ValueError('No eligible paired windows')
    d=pd.DataFrame(rows);per=d.groupby(['filter','home','sensor','model'],as_index=False).agg(n=('test_n','sum'),error_sum=('error_sum','sum'),months=('period_start','nunique'))
    per['mae_c']=per.error_sum/per.n
    means=per.groupby(['filter','model'],as_index=False).agg(homes=('home','nunique'),mae_c=('mae_c','mean'))
    targets=pd.DataFrame(target_diff)
    choices=d[d.model.eq('rolling_selector')].pivot(index=['home','period_start'],columns='filter',values='choice')
    detail={'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'homes':int(d.home.nunique()),
            'eligible_home_months':len(targets),'excluded_home_months':len(excluded),'test_issue_times':int(targets.n.sum()),
            'changed_target_times':int(targets.changed_n.sum()),
            'mean_absolute_target_difference_c':float(targets.abs_sum_c.sum()/targets.n.sum()),
            'selector_choice_changes':int((choices['10_45']!=choices['15_40']).sum()),'source':source_stats,
            'mean_mae_c':{key:{row.model:float(row.mae_c) for row in group.itertuples(index=False)} for key,group in means.groupby('filter')}}
    d.to_csv(OUT/'ideal_holdout_range_home_months.csv',index=False,float_format='%.8f')
    per.to_csv(OUT/'ideal_holdout_range_homes.csv',index=False,float_format='%.8f')
    targets.to_csv(OUT/'ideal_holdout_range_targets.csv',index=False,float_format='%.8f')
    pd.DataFrame(excluded).to_csv(OUT/'ideal_holdout_range_exclusions.csv',index=False)
    output.write_text(json.dumps(detail,indent=2),encoding='utf-8')
    print(json.dumps(detail,indent=2),flush=True)

if __name__=='__main__':main()

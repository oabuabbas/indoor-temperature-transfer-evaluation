"""Compare isolated rebuild outputs against recorded research run."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
REPLAY=ROOT/'replay_clean_20261029'
ORIG=ROOT/'outputs'
NEW=REPLAY/'outputs'

def same_csv(name,keys,columns=None):
    a=pd.read_csv(ORIG/name); b=pd.read_csv(NEW/name)
    if columns is not None:a=a[columns];b=b[columns]
    if set(a.columns)!=set(b.columns):raise AssertionError('Columns differ: '+name)
    b=b[a.columns]
    a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-9,atol=1e-8,obj=name)
    return len(a)

def compare_json(a,b,path='root'):
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):raise AssertionError('JSON keys differ at '+path)
        for key in a:compare_json(a[key],b[key],path+'.'+key)
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):raise AssertionError('JSON list length differs at '+path)
        for i,(x,y) in enumerate(zip(a,b)):compare_json(x,y,f'{path}[{i}]')
    elif isinstance(a,float) and isinstance(b,float):
        if not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8):raise AssertionError(f'JSON value differs at {path}: {a} vs {b}')
    elif a!=b:
        raise AssertionError(f'JSON value differs at {path}: {a} vs {b}')

def main():
    old=json.loads((ORIG/'ideal_house_split.json').read_text(encoding='utf-8'))
    new=json.loads((NEW/'ideal_house_split.json').read_text(encoding='utf-8'))
    if old!=new:raise AssertionError('House split changed')
    quality=['sensor','home','member','position_flag','status','raw_rows','bad_time','outside_time',
             'bad_value','duplicate_adjacent','time_reversals','outside_15_40','valid_samples','qualified_hours',
             'max_qualified_hours_60d','coverage_eligible','first_utc','last_utc','error']
    quality_wide=[x if x!='outside_15_40' else 'outside_10_45' for x in quality]
    comparisons={}
    for name,keys,cols in [
        ('ideal_sensor_quality.csv',['sensor'],quality),
        ('ideal_range_10_45_sensors.csv',['sensor'],quality_wide),
        ('ideal_home_quality.csv',['home'],None),
        ('ideal_development_sensors_v1.csv',['home'],None),
        ('ideal_development_v1_home_months.csv',['home','period_start','model'],None),
        ('ideal_dev_local_home_months.csv',['home','period_start','model'],None),
        ('ideal_dev_nonlinear_home_months.csv',['home','period_start','model'],None),
        ('ideal_holdout_sensors_v1.csv',['home'],None),
        ('ideal_holdout_v1_home_months.csv',['home','period_start','model'],None),
        ('ideal_holdout_v1_exclusions.csv',['home','period_start'],None),
        ('ideal_holdout_range_home_months.csv',['home','period_start','filter','model'],None),
        ('ideal_holdout_range_targets.csv',['home','period_start'],None)]:
        comparisons[name]=same_csv(name,keys,cols)
        print('Matched',name,comparisons[name],'rows',flush=True)
    for name in ['ideal_quality_summary.json','ideal_range_sensitivity_summary.json',
                 'ideal_holdout_v1_summary.json','ideal_holdout_range_secondary_summary.json']:
        a=json.loads((ORIG/name).read_text(encoding='utf-8'));b=json.loads((NEW/name).read_text(encoding='utf-8'))
        compare_json(a,b,name)
        print('Matched',name,flush=True)
    result={'status':'matched','comparison_atol':1e-8,'comparison_rtol':1e-9,'csv_rows':comparisons,
            'scope':'Isolated regeneration using hardlinked same source archives, same code and machine; not independent analyst, dataset, software environment or peer review.'}
    (NEW/'clean_replay_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

"""Read-only arithmetic and artifact verification; no refit or independent timestamp audit."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from ideal_nonlinear_tuning_v1 import GRID
p=Path(__file__).resolve().parent;o=p/'outputs';choice_path=o/'ideal_nonlinear_tuning_v1_choice.json';s=json.loads(choice_path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(p/'protocol_ideal_nonlinear_tuning_v1.md')==s['protocol_sha256'] and s['grid']==GRID
w=o/'ideal_nonlinear_tuning_v1_dev_windows.csv';h=o/'ideal_nonlinear_tuning_v1_dev_homes.csv'
assert sha(w)==s['window_output_sha256'] and sha(h)==s['home_output_sha256']
d=pd.read_csv(w);homes=pd.read_csv(h);assert len(d)==3*s['development_home_months'] and homes.home.nunique()==s['development_homes']
assert (d.groupby(['home','sensor','period_start']).candidate.nunique()==3).all()
assert (pd.to_datetime(d.period_start,utc=True)<pd.Timestamp('2018-02-01',tz='UTC')).all()
a=d.groupby(['home','sensor','candidate']).agg(n=('test_n','sum'),error_sum=('error_sum','sum'),months=('period_start','nunique')).sort_index();b=homes.set_index(['home','sensor','candidate']).sort_index()
assert np.allclose(a[['n','error_sum','months']],b[['n','error_sum','months']]) and np.allclose(b.mae_c,b.error_sum/b.n)
means=homes.groupby('candidate').mae_c.mean();assert min(GRID,key=lambda x:means[x])==s['chosen_candidate']
for k in GRID:assert np.isclose(means[k],s['development_equal_home_mae_c'][k],atol=1e-9)
t=o/'ideal_nonlinear_holdout_secondary_v1_summary.json'
if t.exists():
 z=json.loads(t.read_text());assert sha(choice_path)==z['selection_sha256'] and sha(o/'ideal_holdout_v1_home_months.csv')==z['original_windows_sha256'] and sha(o/'ideal_holdout_v1_homes.csv')==z['original_homes_sha256']
 d=pd.read_csv(o/'ideal_nonlinear_holdout_secondary_v1_windows.csv');h=pd.read_csv(o/'ideal_nonlinear_holdout_secondary_v1_homes.csv')
 assert len(d)==z['home_months'] and len(h)==z['homes'] and h.n.sum()==z['scored_rows']
 assert (pd.to_datetime(d.period_start,utc=True)>=pd.Timestamp('2018-02-01',tz='UTC')).all()
 a=d.groupby(['home','sensor']).agg(n=('test_n','sum'),error_sum=('error_sum','sum'),months=('period_start','nunique')).sort_index();b=h.set_index(['home','sensor']).sort_index()
 assert np.allclose(a[['n','error_sum','months']],b[['n','error_sum','months']]) and np.allclose(b.mae_c,b.error_sum/b.n)
 assert np.isclose(b.mae_c.mean(),z['equal_home_mae_c']['selected_boosting'])
 for label,stat in z['selected_vs_existing'].items():
  delta=b.mae_c-b[label];assert np.isclose(delta.mean(),stat['mean_selected_minus_comparator_c']);assert int((delta < -1e-8).sum())==stat['better_homes'];assert int((delta>1e-8).sum())==stat['worse_homes'];assert int((delta.abs()<=1e-8).sum())==stat['ties']
 print('PASS: development selection, locked artifacts, holdout aggregation and home-level differences; no independent model refit')
else:print('PASS: development selection and saved arithmetic; holdout has not been evaluated')

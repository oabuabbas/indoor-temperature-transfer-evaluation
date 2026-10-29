"""Audit saved 2026 prediction and matched-support summaries without model retraining."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from bremen_fallback_v1 import route
p=Path(__file__).resolve().parent;o=p/'outputs';s=json.loads((o/'bolzano_2026_benchmark_v1_summary.json').read_text())
assert hashlib.sha256((p/'protocol_bolzano_2026_benchmark_v1.md').read_bytes()).hexdigest()==s['protocol_sha256']
for name,h in s['input_sha256'].items():assert hashlib.sha256((p/'data'/'bolzano'/name).read_bytes()).hexdigest()==h
for scenario,result in s['results'].items():
 d=pd.read_csv(o/f'bolzano_2026_benchmark_v1_{scenario}_predictions.csv.gz',parse_dates=['issue_hour_start','month'])
 c=pd.read_csv(o/f'bolzano_2026_benchmark_v1_{scenario}_choices.csv',parse_dates=['month','last_history_target_start'])
 m=pd.read_csv(o/f'bolzano_2026_benchmark_v1_{scenario}_monthly.csv',parse_dates=['month'])
 assert len(c)==4 and len(d)==2856==result['opportunities'] and not d.issue_hour_start.duplicated().any()
 assert (d.issue_hour_start>=pd.Timestamp('2026-01-01')).all() and (d.issue_hour_start<pd.Timestamp('2026-05-01')).all()
 assert (d.issue_hour_start+pd.Timedelta(hours=6)<d.month+pd.offsets.MonthBegin(1)).all()
 assert (c.last_history_target_start<c.month).all()
 assert all(d.groupby('month').choice.nunique()==1)
 f,branch=route(d.rolling_selector.to_numpy(),d.temp_now.to_numpy(),d.seasonal_yesterday.to_numpy())
 assert np.allclose(f,d.fallback,equal_nan=True) and np.array_equal(branch,d.branch.to_numpy())
 target=d.target.to_numpy();common=np.isfinite(target)
 for row in result['common_metrics']:common &= np.isfinite(d[row['model']].to_numpy())
 assert common.sum()==result['common_n']
 for row in result['common_metrics']:
  err=d.loc[common,row['model']].to_numpy()-target[common]
  assert np.isclose(np.abs(err).mean(),row['mae_c'],atol=1e-9) and np.isclose(np.sqrt(np.square(err).mean()),row['rmse_c'],atol=1e-9)
  x=m[m.model.eq(row['model'])];assert x.n.sum()==result['common_n']
 for label,v in result['own_support'].items():
  ready=d[label].notna();scored=ready&d.target.notna();assert ready.sum()==v['ready'] and scored.sum()==v['scored']
  assert np.isclose((d.loc[scored,label]-d.loc[scored,'target']).abs().mean(),v['mae_c'],atol=1e-9)
 x=d[d.target.notna()&d.fallback.notna()&d.persistence.notna()];r=result['fallback_vs_persistence']
 assert len(x)==r['n'] and np.isclose((x.fallback-x.target).abs().mean(),r['fallback_mae_c'],atol=1e-9)
print('PASS: frozen protocol, raw hashes, target cutoff, history timestamps, routing, seven-model identical-support MAE/RMSE, monthly counts')
print('No source or boosting model is independently retrained; timestamp semantics and calibration are unverified.')

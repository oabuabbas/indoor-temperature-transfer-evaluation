"""Audit stored Aalborg outcomes; does not rerun fitting or audit live timing."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from bremen_fallback_v1 import route
from aalborg_external_v1 import METHODS
p=Path(__file__).resolve().parent;o=p/'outputs';s=json.loads((o/'aalborg_external_v1_summary.json').read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(p/'protocol_aalborg_external_v1.md')==s['protocol_sha256']
assert sha(p/'data'/'aalborg'/'dataset__2023_02_27__2023_12_31.csv')==s['source_sha256']
assert sha(o/'ideal_nonlinear_tuning_v1_choice.json')==s['selected_candidate_sha256']
d=pd.read_csv(o/'aalborg_external_v1_predictions.csv.gz',parse_dates=['issue_hour_start','month']);r=pd.read_csv(o/'aalborg_external_v1_rooms.csv');m=pd.read_csv(o/'aalborg_external_v1_monthly.csv');c=pd.read_csv(o/'aalborg_external_v1_choices.csv',parse_dates=['month','last_history_target_start']);own=pd.read_csv(o/'aalborg_external_v1_own_support.csv');pair=pd.read_csv(o/'aalborg_external_v1_fallback_pair.csv')
assert len(d)==s['opportunities']==39276 and len(c)==54 and d.room.nunique()==6
assert not d.duplicated(['room','issue_hour_start']).any() and (c.last_history_target_start<c.month).all()
assert (d.issue_hour_start+pd.Timedelta(hours=6)<d.month+pd.offsets.MonthBegin(1)).all()
f,b=route(d.rolling_selector.to_numpy(),d.temp_now.to_numpy(),d.seasonal_yesterday.to_numpy());assert np.allclose(f,d.fallback,equal_nan=True) and np.array_equal(b,d.branch)
for name in ['source_daily_ridge','daily_seasonal','persistence']:
 x=d[d.choice.eq(name)&d.rolling_selector.notna()];assert np.allclose(x.rolling_selector,x[name])
common=d.target.notna()&d[METHODS].notna().all(axis=1);assert common.sum()==s['common_issue_times']
for (room,method),row in r.set_index(['room','model']).iterrows():
 x=d[common&d.room.eq(room)];e=x[method]-x.target
 assert len(x)==row['n'] and np.isclose(e.abs().mean(),row.mae_c,atol=1e-9) and np.isclose(np.sqrt((e**2).mean()),row.rmse_c,atol=1e-9)
 mm=m[(m.room==room)&(m.model==method)];assert mm.n.sum()==row['n'] and np.isclose(mm.abs_sum.sum(),row.abs_sum,atol=1e-7)
for row in s['equal_room_metrics']:
 x=r[r.model.eq(row['model'])];assert len(x)==6 and x.n.sum()==row['n'] and np.isclose(x.mae_c.mean(),row['equal_room_mae_c']) and np.isclose(x.rmse_c.mean(),row['equal_room_rmse_c'])
for (room,method),row in own.set_index(['room','model']).iterrows():
 x=d[d.room.eq(room)];valid=x[method].notna()&x.target.notna();assert len(x)==row.opportunities and x[method].notna().sum()==row.ready and valid.sum()==row.scored
 assert np.isclose((x.loc[valid,method]-x.loc[valid,'target']).abs().sum(),row.abs_sum,atol=1e-7)
assert pair.n.sum()==s['fallback_pair']['n'] and np.isclose((pair.fallback_abs_sum/pair.n).mean(),s['fallback_pair']['fallback_equal_room_mae_c'])
assert d.branch.value_counts().to_dict()==s['branches']
print('PASS: pinned input, protocol, history timing, routing, choices, eight-method matched-support and pair aggregation')
print('Does not retrain models, independently verify sensor calibration, or establish live telemetry latency.')

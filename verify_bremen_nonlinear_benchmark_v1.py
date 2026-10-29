"""Read-only recomputation of saved matched-support and per-room metrics."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from bremen_fallback_v1 import route
from bolzano_2026_benchmark_v1 import METHODS
p=Path(__file__).resolve().parent;o=p/'outputs';s=json.loads((o/'bremen_nonlinear_benchmark_v1_summary.json').read_text())
assert hashlib.sha256((p/'protocol_bremen_nonlinear_benchmark_v1.md').read_bytes()).hexdigest()==s['protocol_sha256']
assert hashlib.sha256((p/'protocol_bremen_fallback_v1.md').read_bytes()).hexdigest()==s['prior_protocol_sha256']
d=pd.read_csv(o/'bremen_nonlinear_benchmark_v1_predictions.csv.gz',parse_dates=['issue_hour_start','month'])
r=pd.read_csv(o/'bremen_nonlinear_benchmark_v1_rooms.csv');m=pd.read_csv(o/'bremen_nonlinear_benchmark_v1_monthly.csv');c=pd.read_csv(o/'bremen_nonlinear_benchmark_v1_choices.csv',parse_dates=['month','last_history_target_start'])
assert len(d)==s['opportunities']==29136 and len(c)==40 and d.sensor.nunique()==4
assert not d.duplicated(['sensor','issue_hour_start']).any()
assert (c.last_history_target_start<c.month).all()
assert (d.issue_hour_start+pd.Timedelta(hours=6)<d.month+pd.offsets.MonthBegin(1)).all()
f,b=route(d.rolling_selector.to_numpy(),d.temp_now.to_numpy(),d.seasonal_yesterday.to_numpy())
assert np.allclose(f,d.fallback,equal_nan=True) and np.array_equal(b,d.branch)
common=d.target.notna() & d[METHODS].notna().all(axis=1)
assert common.sum()==s['common_issue_times']==28948
for (room,method),row in r.set_index(['sensor','model']).iterrows():
 x=d[common & d.sensor.eq(room)];err=x[method]-x.target
 assert len(x)==row['n'] and np.isclose(err.abs().mean(),row.mae_c,atol=1e-9) and np.isclose(np.sqrt((err**2).mean()),row.rmse_c,atol=1e-9)
 mm=m[(m.sensor==room)&(m.model==method)];assert mm.n.sum()==row['n'] and np.isclose(mm.abs_sum.sum(),row.abs_sum,atol=1e-7) and np.isclose(mm.squared_sum.sum(),row.squared_sum,atol=1e-7)
for row in s['metrics_equal_room']:
 x=r[r.model.eq(row['model'])];assert len(x)==4 and x.n.sum()==row['n'] and np.isclose(x.mae_c.mean(),row['equal_room_mae_c']) and np.isclose(x.rmse_c.mean(),row['equal_room_rmse_c'])
print('PASS: protocols, temporal windows, branches and identical-support error aggregation across four rooms and 40 room-months')
print('Does not independently retrain models or confirm sensor timing or calibration.')

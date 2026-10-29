"""Read-only verification of saved per-issue results and protocol integrity."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from bremen_fallback_v1 import route
p=Path(__file__).resolve().parent;o=p/'outputs';s=json.loads((o/'bolzano_external_v1_summary.json').read_text())
assert hashlib.sha256((p/'protocol_bolzano_external_v1.md').read_bytes()).hexdigest()==s['protocol_sha256']
for name,r in s['results'].items():
    d=pd.read_csv(o/f'bolzano_external_v1_{name}_predictions.csv.gz',parse_dates=['issue_hour_start'])
    c=pd.read_csv(o/f'bolzano_external_v1_{name}_choices.csv',parse_dates=['month','last_history_target_start'])
    monthly=pd.read_csv(o/f'bolzano_external_v1_{name}_monthly.csv')
    assert len(d)==r['opportunities']==17400 and len(c)==24 and not d.issue_hour_start.duplicated().any()
    assert (c.last_history_target_start.dropna()<c.loc[c.last_history_target_start.notna(),'month']).all()
    f,branch=route(d.rolling_selector.to_numpy(),d.temp_now.to_numpy(),d.seasonal_yesterday.to_numpy())
    assert np.allclose(f,d.fallback,equal_nan=True) and np.array_equal(branch,d.branch.to_numpy())
    target=d.target.to_numpy();obs=np.isfinite(target);assert obs.sum()==r['observed_targets']
    for method,z in r['methods'].items():
        x=d[method].to_numpy();good=np.isfinite(x);score=good&obs
        assert good.sum()==z['ready'] and score.sum()==z['scored']
        assert np.isclose(np.abs(x[score]-target[score]).mean(),z['mae_c'],atol=1e-9)
    for a,b,key in [('fallback','persistence','fallback_vs_persistence'),('rolling_selector','source_daily_ridge','selector_vs_source')]:
        x=d[a].to_numpy();y=d[b].to_numpy();common=np.isfinite(x)&np.isfinite(y)&obs; z=r[key]
        assert common.sum()==z['n'] and np.isclose(np.abs(x[common]-target[common]).mean(),z[a+'_mae_c'],atol=1e-9) and np.isclose(np.abs(y[common]-target[common]).mean(),z[b+'_mae_c'],atol=1e-9)
    assert monthly.n.sum()==r['fallback_vs_persistence']['n']
    assert sum(r['branches'].values())==len(d)
print('PASS: protocol, past history timestamps, route, per-issue method totals, paired errors and monthly support for 3 scenarios')
print('Does not refit source or independently verify temperature timestamps, aggregation or model semantics.')

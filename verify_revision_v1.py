from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
P=Path(__file__).resolve().parent; O=P/'outputs/revision_v1'
s=json.loads((O/'summary.json').read_text())
for file,expected in s['input_sha256'].items():
 assert hashlib.sha256((P/file).read_bytes()).hexdigest()==expected,file
r=pd.read_csv(O/'benchmark_windows_private.csv')
assert r.groupby(['home','period_start']).model.nunique().eq(6).all()
assert r.groupby(['home','period_start']).n.nunique().eq(1).all()
assert r.n.ge(24).all() and r.error_sum.ge(0).all()
p=r.groupby(['home','model'])[['n','error_sum']].sum();v=(p.error_sum/p.n).groupby('model').mean()
for k,value in v.items():assert np.isclose(value,s['benchmark_equal_home_mae'][k],rtol=1e-10)
assert r.home.nunique()==73 and len(r)==471*6
h=pd.read_csv(O/'home_characteristics_private.csv')
assert len(h)==73 and not h.isna().any().any()
report={'input_hashes_checked':len(s['input_sha256']),'common_support_and_aggregate_checks':'passed','scope':'Stored arithmetic and input identity; not independent model refitting.'}
(O/'verification.json').write_text(json.dumps(report,indent=2));print(report)

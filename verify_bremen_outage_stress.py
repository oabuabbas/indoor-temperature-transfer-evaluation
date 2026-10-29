"""Read-only internal consistency checks on stored stress outcomes."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parent; o=p/'outputs'
s=json.loads((o/'bremen_outage_stress_v1_summary.json').read_text())
assert hashlib.sha256((p/'protocol_bremen_outage_stress_v1.md').read_bytes()).hexdigest()==s['protocol_sha256']
m=pd.read_csv(o/'bremen_outage_stress_v1_methods.csv');b=pd.read_csv(o/'bremen_outage_stress_v1_branches.csv');q=pd.read_csv(o/'bremen_outage_stress_v1_pairs.csv');c=pd.read_csv(o/'bremen_outage_stress_v1_choices.csv');r=pd.read_csv(o/'bremen_outage_stress_v1_replicates.csv')
key=['scenario','replicate','sensor']
assert len(r)==26 and len(q)==104 and len(c)==1040 and not m.duplicated(key+['model']).any()
assert (m.groupby(key).model.nunique()==6).all() and (m.scored<=m.ready).all() and (m.ready<=m.opportunities).all()
f=m[m.model.eq('fallback')].set_index(key).sort_index(); bs=b.groupby(key)[['opportunities','scored','abs_error_sum']].sum().sort_index()
assert np.array_equal(f.opportunities,bs.opportunities) and np.array_equal(f.scored,bs.scored) and np.allclose(f.abs_error_sum,bs.abs_error_sum)
per=m[m.model.eq('persistence')].set_index(key).sort_index(); pair=q.set_index(key).sort_index()
assert np.array_equal(pair.n,per.scored)
assert np.array_equal(pair.extra_scored+pair.n,f.scored)
for x in r.itertuples(index=False):
    t=q[(q.scenario==x.scenario)&(q.replicate==x.replicate)]
    assert t.n.sum()==x.paired_n
    assert np.isclose((t.fallback_abs_sum/t.n).mean(),x.paired_fallback_equal_room_mae)
    assert np.isclose((t.persistence_abs_sum/t.n).mean(),x.paired_persistence_equal_room_mae)
original=pd.read_csv(o/'bremen_fallback_v1_sensors.csv').set_index(['sensor','model']).sort_index()
base=m[m.scenario.eq('none')].set_index(['sensor','model']).sort_index()
for col in ['opportunities','ready','scored','abs_error_sum']:
    assert np.allclose(base[col],original[col],atol=1e-7,rtol=1e-9)
oldchoices=pd.read_csv(o/'bremen_fallback_v1_choices.csv').sort_values(['sensor','month']).reset_index(drop=True)
newchoices=c[c.scenario.eq('none')][['sensor','month','history_n','choice']].sort_values(['sensor','month']).reset_index(drop=True)
assert oldchoices.equals(newchoices)
print('PASS: protocol hash, 26 runs, opportunity/branch totals, paired and oracle extra scoring arithmetic')
print('This does not rerun mask generation, inference, source training, or audit the independent-oracle assumption.')

import hashlib,json
from pathlib import Path
import pandas as pd
import numpy as np
p=Path(__file__).resolve().parent;o=p/'outputs';s=json.loads((o/'bremen_joint_missing_v1_summary.json').read_text())
assert hashlib.sha256((p/'protocol_bremen_joint_missing_v1.md').read_bytes()).hexdigest()==s['protocol_sha256']
d=pd.read_csv(o/'bremen_joint_missing_v1_rooms.csv');b=pd.read_csv(o/'bremen_joint_missing_v1_branches.csv');r=pd.read_csv(o/'bremen_joint_missing_v1_replicates.csv');q=pd.read_csv(o/'bremen_joint_missing_v1_synchronous.csv');c=pd.read_csv(o/'bremen_joint_missing_v1_choices.csv')
key=['scenario','replicate','sensor'];assert len(d)==60 and len(r)==15 and len(q)==15 and len(c)==600 and not d.duplicated(key).any()
assert (d.same_sensor_target<=d.oracle_target).all()
for tag in ['oracle','same_sensor']:
    assert (d[tag+'_paired_n']<=d.persistence_ready).all() and (d[tag+'_extra_scored']+d[tag+'_paired_n']<=d.fallback_ready).all()
assert (d.same_sensor_paired_n<=d.oracle_paired_n).all()
bg=b.groupby(key)[['opportunities','scored']].sum().sort_index();df=d.set_index(key).sort_index()
assert np.array_equal(bg.opportunities,df.opportunities)
assert np.array_equal(bg.scored,df.oracle_paired_n+df.oracle_extra_scored)
assert (b[b.branch.eq('abstain')].scored==0).all()
for x in r.itertuples(index=False):
    g=d[(d.scenario==x.scenario)&(d.replicate==x.replicate)];z=q[(q.scenario==x.scenario)&(q.replicate==x.replicate)].iloc[0]
    assert x.fallback_ready==g.fallback_ready.sum() and x.all_four_ready==z.all_four_ready
    for tag in ['oracle','same_sensor']:
        n=g[tag+'_paired_n']; assert int(n.sum())==getattr(x,tag+'_paired_n')
        assert np.isclose((g[tag+'_fallback_error']/n).mean(),getattr(x,tag+'_fallback_equal_room_mae'))
    assert x.all_four_ready<=x.common_issue_times and x.all_four_target_observed<=x.common_issue_times
print('PASS: protocol, 15 runs, branch totals, target-subset and synchronized-readiness accounting; no independent model rerun')

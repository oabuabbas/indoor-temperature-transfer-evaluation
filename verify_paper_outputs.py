"""Read-only integrity checks for reported IDEAL holdout outputs."""
import hashlib,json
import numpy as np
import pandas as pd
from audit_nest import OUT

def require(condition,message):
    if not condition:raise AssertionError(message)

def main():
    summary=json.loads((OUT/'ideal_holdout_v1_summary.json').read_text(encoding='utf-8'))
    secondary=json.loads((OUT/'ideal_holdout_range_secondary_summary.json').read_text(encoding='utf-8'))
    for name,expected in [('protocol_ideal_holdout_v1.md',summary['protocol_sha256']),
                          ('protocol_ideal_holdout_range_secondary.md',secondary['protocol_sha256'])]:
        require(hashlib.sha256((OUT.parent/name).read_bytes()).hexdigest()==expected,'Protocol hash mismatch: '+name)
    split=pd.DataFrame(json.loads((OUT/'ideal_house_split.json').read_text(encoding='utf-8'))['homes'])
    rows=pd.read_csv(OUT/'ideal_holdout_v1_home_months.csv')
    groups=rows.groupby(['home','period_start'])
    require(len(groups)==477 and groups.model.nunique().eq(6).all(),'Primary window/model counts changed')
    require(groups.test_n.nunique().eq(1).all(),'Primary test count mismatch across methods')
    require(set(rows.home).issubset(set(split.loc[split.split.eq('holdout'),'homeid'])),'Development home in holdout results')
    homes=pd.read_csv(OUT/'ideal_holdout_v1_homes.csv'); require(homes.home.nunique()==summary['contributing_homes'],'Home count mismatch')
    agg=rows.groupby(['home','model'],as_index=False).agg(n=('test_n','sum'),err=('error_sum','sum'))
    agg['mae']=agg.err/agg.n
    calc=agg.groupby('model').mae.mean()
    for model,value in summary['mean_mae_c'].items():
        require(abs(calc[model]-value)<1e-7,'Mean MAE mismatch: '+model)
    r=pd.read_csv(OUT/'ideal_holdout_range_home_months.csv');pairs=r.groupby(['home','period_start'])
    require(len(pairs)==secondary['eligible_home_months'],'Sensitivity window count mismatch')
    require(pairs['filter'].nunique().eq(2).all() and pairs.model.nunique().eq(5).all(),'Sensitivity model/filter mismatch')
    require(pairs.test_n.nunique().eq(1).all() and pairs.size().eq(10).all(),'Sensitivity support mismatch')
    verify=json.loads((OUT/'ideal_archive_verification.json').read_text(encoding='utf-8'))
    path=OUT.parent/'data'/'ideal'/'room_and_appliance_sensors.zip'
    st=path.stat()
    require(verify['matches'] and st.st_size==verify['archive_bytes'] and st.st_mtime_ns==verify['mtime_ns'],'Verified archive modified; rerun full checksum')
    print('Verified protocol hashes, 477 primary windows, 73 holdout houses, equal within-window counts, reported MAEs, 473 paired sensitivity windows, and archive verification metadata.')
    print('Archive content was not rehashed in this check; run verify_ideal_archive.py for a fresh MD5.')

if __name__=='__main__':main()

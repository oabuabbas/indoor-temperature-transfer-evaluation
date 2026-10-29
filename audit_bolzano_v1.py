"""Read-only quality inventory; timestamps deliberately kept timezone-naive."""
import hashlib,json
import pandas as pd
from pathlib import Path
root=Path(__file__).resolve().parent; data=root/'data'/'bolzano';out=root/'outputs'
expected={'2023':'55cad42273612c7ed6eef238b78f8bcf','2024':'0d82bc9ba02475934a19f522098f844c','2025':'10a75e3e455bab8e4f4fc83bb1e2e92d','2026':'6704bf400e4bd770a2aa019d7d2e837e'}
rows=[]
for year,md5 in expected.items():
    path=data/f'HomeCoach_5min_{year}.csv';raw=path.read_bytes()
    if hashlib.md5(raw).hexdigest()!=md5:raise ValueError('Source checksum mismatch')
    d=pd.read_csv(path);t=pd.to_datetime(d.timestamp,errors='coerce');v=pd.to_numeric(d.Temperature,errors='coerce');delta=t.diff().dt.total_seconds()/60
    duplicate=d[t.duplicated(keep=False)].copy(); conflicts=duplicate.groupby('timestamp').Temperature.nunique(dropna=False)
    rows.append({'year':year,'file':path.name,'bytes':len(raw),'provider_md5':md5,'sha256_local':hashlib.sha256(raw).hexdigest(),'rows':len(d),'first_timestamp_naive':str(t.min()),'last_timestamp_naive':str(t.max()),'bad_timestamp':int(t.isna().sum()),'missing_temperature':int(v.isna().sum()),'duplicate_timestamp_rows_beyond_first':int(t.duplicated().sum()),'duplicate_time_groups_with_conflicting_temperature':int((conflicts>1).sum()),'time_reversals':int((delta<0).sum()),'min_c':float(v.min()),'max_c':float(v.max()),'outside_10_45':int(((v<10)|(v>45)).sum()),'common_gap_minutes':float(delta.mode().iloc[0]),'largest_gap_minutes':float(delta.max())})
result={'source':'https://zenodo.org/records/20098783','doi':'10.5281/zenodo.20098783','license_as_declared':'CC BY 4.0','scope':'one residential room; timezone unverified; no prediction evaluation','files':rows}
(out/'bolzano_quality_v1.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))

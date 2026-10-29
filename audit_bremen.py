"""Verify pinned Git objects and audit raw Bremen room data without forecasts."""
import hashlib,json
import numpy as np
import pandas as pd
from audit_nest import DATA,OUT

ROOT=DATA/'bremen';HOURLY=OUT/'bremen_hourly_v1'
COMMIT='d83ff1e66d41ac8f77a44d5d94fb877b00b842eb'
FILES={'South_2.csv':(33433573,'3c3505df7ae46a467d3e64d8d0da71e6b9755856'),
       'South_5.csv':(13045179,'2083b268b3c0108c43d0c393d21c3d51c710e730'),
       'South_8.csv':(32769472,'417e21398f5785ed6ec2d3af924097b6f2ac5b46'),
       'South_9.csv':(33736546,'b2313f846cdc3c5b81acba7c5c83f8bfaaab0658')}

def verify(path,size,expected):
    if path.stat().st_size!=size:raise ValueError('File size mismatch '+path.name)
    sha1=hashlib.sha1(b'blob '+str(size).encode()+b'\0');sha256=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):sha1.update(b);sha256.update(b)
    if sha1.hexdigest()!=expected:raise ValueError('Git blob mismatch '+path.name)
    return sha256.hexdigest()

def aggregate(df):
    timestamp=pd.to_datetime(df.Time,format='ISO8601',errors='coerce',utc=True)
    value=pd.to_numeric(df.Temperature,errors='coerce')
    known=timestamp.notna()
    reversals=int((timestamp[known].diff()<pd.Timedelta(0)).sum())
    repeated=timestamp.duplicated(keep=False)&known
    valid=known & ~repeated & value.between(10,45)
    q={'rows':len(df),'bad_time':int((~known).sum()),'non_numeric':int((~np.isfinite(value)).sum()),
       'duplicate_time_rows':int(repeated.sum()),'time_reversals':reversals,
       'outside_10_45':int((np.isfinite(value)&~value.between(10,45)).sum()),
       'first_utc':str(timestamp[known].min()),'last_utc':str(timestamp[known].max())}
    d=pd.DataFrame({'minute':timestamp[valid].dt.floor('min'),'value':value[valid]})
    minute=d.groupby('minute').value.mean();slot=minute.groupby(minute.index.floor('10min')).agg(['mean','count'])
    slot=slot[slot['count']>=5];hour=slot['mean'].groupby(slot.index.floor('h')).agg(['mean','count'])
    hour=hour[hour['count']>=4].rename(columns={'mean':'temp_c','count':'valid_slots'})
    return hour,q

def main():
    HOURLY.mkdir(exist_ok=True);quality=[];manifest=[]
    for name,(size,git_sha) in FILES.items():
        path=ROOT/name;sha=verify(path,size,git_sha)
        d=pd.read_csv(path,usecols=['Time','Temperature'],low_memory=False)
        hourly,meta=aggregate(d)
        hp=HOURLY/f'{path.stem}.csv.gz'
        hourly.to_csv(hp,compression='gzip',index_label='hour_utc')
        meta.update({'sensor':path.stem,'valid_hours':len(hourly)})
        start=pd.Timestamp('2023-03-01',tz='UTC');end=pd.Timestamp('2024-01-01',tz='UTC')
        meta['hourly_eligible_in_planned_window']=int(((hourly.index>=start)&(hourly.index<end)).sum())
        quality.append(meta);manifest.append({'file':name,'bytes':size,'git_blob_sha1':git_sha,'sha256_local':sha,
            'git_blob_verified':True,'hourly_file':hp.name,'hourly_sha256_local':hashlib.sha256(hp.read_bytes()).hexdigest()})
        print(name,'audited',flush=True)
    (OUT/'bremen_manifest.json').write_text(json.dumps({'commit':COMMIT,'files':manifest},indent=2),encoding='utf-8')
    pd.DataFrame(quality).to_csv(OUT/'bremen_quality.csv',index=False)
    print(pd.DataFrame(quality).to_string(index=False),flush=True)

if __name__=='__main__':main()

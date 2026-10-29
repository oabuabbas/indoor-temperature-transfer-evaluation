"""Prepare one eligible raw hourly sensor per development home only."""
import concurrent.futures
import hashlib
import json
import zipfile
import pandas as pd
from audit_nest import DATA,OUT
from audit_ideal_temperature import ARCHIVE

HOURLY=OUT/'ideal_development_hourly_v1'
THRESHOLD=(7*(2**64))//10

def split_source(home):
    value=int.from_bytes(hashlib.sha256(f'ideal-development-source-v1:{int(home)}'.encode()).digest()[:8],'big')
    return 'source' if value<THRESHOLD else 'validation'

def select(partition='development'):
    if partition not in ('development','holdout'):raise ValueError('Unknown partition')
    split=pd.DataFrame(json.loads((OUT/'ideal_house_split.json').read_text(encoding='utf-8'))['homes']).rename(columns={'homeid':'home'})
    audit=pd.read_csv(OUT/'ideal_range_10_45_sensors.csv')
    audit=audit.merge(split,on='home',validate='many_to_one')
    audit=audit[(audit.split==partition) & audit.coverage_eligible & (audit.position_flag==0) & (audit.status=='ok') & (audit.time_reversals==0)].copy()
    with zipfile.ZipFile(DATA/'ideal'/'metadata_and_surveys.zip') as z:
        with z.open('metadata/sensor.csv') as f:sensor=pd.read_csv(f,usecols=['sensorid','roomid'])
        with z.open('metadata/room.csv') as f:rooms=pd.read_csv(f,usecols=['roomid','homeid','type'])
    audit=audit.merge(sensor,left_on='sensor',right_on='sensorid',validate='one_to_one')
    audit=audit.merge(rooms,on='roomid',validate='many_to_one')
    if (audit.home!=audit.homeid).any():raise ValueError('House metadata mismatch')
    rank={'livingroom':0,'kitchenliving':1,'bedroom':2}
    audit['rank']=audit.type.map(rank).fillna(3)
    audit=audit.sort_values(['home','rank','sensor']).drop_duplicates('home')
    audit['dev_role']=audit.home.map(split_source)
    return audit[['home','sensor','member','type','dev_role']].sort_values('home')

def prepare(entry,lower=10,upper=45,output_root=None):
    import gzip
    home,sensor,name=entry
    sums=[];last=None
    with zipfile.ZipFile(ARCHIVE) as z, z.open(name) as m, gzip.GzipFile(fileobj=m) as f:
        for part in pd.read_csv(f,header=None,names=['time','value'],chunksize=200000):
            t=pd.to_datetime(part.time,errors='coerce',utc=True)
            value=pd.to_numeric(part.value,errors='coerce')/10.0
            if t.isna().any() or (t<pd.Timestamp('2016-01-01',tz='UTC')).any() or (t>=pd.Timestamp('2019-01-01',tz='UTC')).any():
                raise ValueError('Unexpected timestamp')
            if (t.diff()<pd.Timedelta(0)).any() or (last is not None and t.iloc[0]<last):raise ValueError('Time reversed')
            dup=t.duplicated()
            if last is not None and t.iloc[0]==last:dup.iloc[0]=True
            last=t.iloc[-1]
            valid=value.between(lower,upper) & ~dup
            d=pd.DataFrame({'slot':t[valid].dt.floor('10min'),'value':value[valid]})
            if not d.empty:sums.append(d.groupby('slot').value.agg(['sum','count']))
    if not sums:raise ValueError('No valid temperature data')
    slots=pd.concat(sums).groupby(level=0).sum()
    slots=slots[slots['count']>=25].copy()
    slots['temp']=slots['sum']/slots['count']
    slots['hour']=slots.index.floor('h')
    h=slots.groupby('hour').temp.agg(['mean','count'])
    h=h[h['count']>=4].rename(columns={'mean':'temp_c','count':'valid_slots'})
    destination=HOURLY if output_root is None else output_root
    path=destination/f'{sensor}.csv.gz';tmp=destination/f'{sensor}.tmp.gz'
    h.to_csv(tmp,compression='gzip',index_label='hour_utc');tmp.replace(path)
    return {'home':home,'sensor':sensor,'valid_hours':len(h),'first':str(h.index.min()),'last':str(h.index.max())}

def main():
    check=json.loads((OUT/'ideal_archive_verification.json').read_text(encoding='utf-8'))
    st=ARCHIVE.stat()
    if not check['matches'] or st.st_size!=check['archive_bytes'] or st.st_mtime_ns!=check['mtime_ns']:raise ValueError('Unverified archive')
    sel=select();path=OUT/'ideal_development_sensors_v1.csv'
    if path.exists() and not pd.read_csv(path).equals(sel.reset_index(drop=True)):raise ValueError('Development selection changed')
    sel.to_csv(path,index=False)
    HOURLY.mkdir(exist_ok=True)
    entries=[(int(r.home),int(r.sensor),r.member) for r in sel.itertuples(index=False) if not (HOURLY/f'{r.sensor}.csv.gz').exists()]
    print('selected',len(sel),'roles',sel.dev_role.value_counts().to_dict(),'pending',len(entries),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for i,result in enumerate(pool.map(prepare,entries,chunksize=1),1):
            if i%20==0 or i==len(entries):print('Prepared',i,'of',len(entries),flush=True)
    print('Completed; no holdout raw series processed for modeling.',flush=True)

if __name__=='__main__':main()

"""Stream IDEAL room temperature files, with resumable per-sensor audit records."""
import concurrent.futures
import contextlib
import gzip
import io
import json
import re
import zipfile
import numpy as np
import pandas as pd
from audit_nest import DATA, OUT

ROOT=DATA/'ideal'
ARCHIVE=ROOT/'room_and_appliance_sensors.zip'
CACHE=OUT/'ideal_sensor_audit_cache'
T0=pd.Timestamp('2016-01-01',tz='UTC')
T1=pd.Timestamp('2019-01-01',tz='UTC')

def audit(entry,lower=15,upper=40):
    sensor,home,name,flag=entry[:4]
    outside_key=f'outside_{lower}_{upper}'
    record={'sensor':sensor,'home':home,'member':name,'position_flag':flag,'status':'ok',
            'raw_rows':0,'bad_time':0,'outside_time':0,'bad_value':0,'duplicate_adjacent':0,
            'time_reversals':0,outside_key:0,'valid_samples':0,'qualified_hours':0,
            'max_qualified_hours_60d':0,'coverage_eligible':False,'first_utc':None,'last_utc':None,'error':None}
    allslots=[];prev=None;first=None;last=None
    try:
        with contextlib.ExitStack() as stack:
            if len(entry)==5:
                member=io.BytesIO(entry[4])
            else:
                z=stack.enter_context(zipfile.ZipFile(ARCHIVE))
                member=stack.enter_context(z.open(name))
            stream=stack.enter_context(gzip.GzipFile(fileobj=member))
            for chunk in pd.read_csv(stream,header=None,names=['time','value'],chunksize=200000):
                record['raw_rows']+=len(chunk)
                t=pd.to_datetime(chunk.time,errors='coerce',utc=True)
                value=pd.to_numeric(chunk.value,errors='coerce')/10.0
                record['bad_time']+=int(t.isna().sum())
                record['bad_value']+=int((~np.isfinite(value)).sum())
                intime=t.notna() & (t>=T0) & (t<T1)
                record['outside_time']+=int((t.notna() & ~intime).sum())
                a=pd.DataFrame({'time':t[intime], 'value':value[intime]}).reset_index(drop=True)
                if a.empty:continue
                diff=a.time.diff()
                record['time_reversals']+=int((diff<pd.Timedelta(0)).sum())
                if prev is not None and a.time.iloc[0]<prev:record['time_reversals']+=1
                duplicate=a.time.duplicated()
                if prev is not None and a.time.iloc[0]==prev:duplicate.iloc[0]=True
                record['duplicate_adjacent']+=int(duplicate.sum())
                prev=a.time.iloc[-1]
                first=a.time.min() if first is None else min(first,a.time.min())
                last=a.time.max() if last is None else max(last,a.time.max())
                good=np.isfinite(a.value)
                record[outside_key]+=int((good & ~a.value.between(lower,upper)).sum())
                good=good & a.value.between(lower,upper) & ~duplicate
                filtered=a.loc[good,'time']
                record['valid_samples']+=len(filtered)
                if len(filtered):allslots.append(filtered.dt.floor('10min').value_counts())
        if first is not None:
            record['first_utc']=first.isoformat();record['last_utc']=last.isoformat()
        if allslots:
            slots=pd.concat(allslots).groupby(level=0).sum()
            eligible=slots[slots>=25].groupby(slots[slots>=25].index.floor('h')).size()
            eligible=eligible[eligible>=4]
            grid=pd.date_range(first.floor('h'),last.floor('h'),freq='h',tz='UTC')
            hourly=pd.Series(0,index=grid,dtype='int8')
            hourly.loc[eligible.index]=1
            record['qualified_hours']=int(hourly.sum())
            if len(hourly)>=1440:
                record['max_qualified_hours_60d']=int(hourly.rolling(1440,min_periods=1440).sum().max())
        record['coverage_eligible']=bool(record['time_reversals']==0 and record['max_qualified_hours_60d']>=720)
    except Exception as e:
        record['status']='error';record['error']=f'{type(e).__name__}: {e}'
    return record

def main():
    check=json.loads((OUT/'ideal_archive_verification.json').read_text(encoding='utf-8'))
    stat=ARCHIVE.stat()
    if not check['matches'] or stat.st_size!=check['archive_bytes'] or stat.st_mtime_ns!=check['mtime_ns']:
        raise ValueError('Archive not verified, or modified since verification')
    CACHE.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT/'metadata_and_surveys.zip') as z:
        def read(name,cols):
            with z.open('metadata/'+name+'.csv') as f:return pd.read_csv(f,usecols=cols,low_memory=False)
        sensor=read('sensor',['sensorid','sensorboxid','type','roomid'])
        box=read('sensorbox',['sensorboxid','sensorbox_type','temperatureInaccuracy'])
        room=read('room',['roomid','homeid'])
    meta=sensor[sensor.type.eq('temperature')].merge(box,on='sensorboxid',validate='many_to_one')
    meta=meta[meta.sensorbox_type.eq('room')].merge(room,on='roomid',validate='many_to_one').set_index('sensorid')
    entries=[]
    pattern=re.compile(r'home(\d+)_.*?_sensor(\d+)_room_temperature\.csv\.gz$')
    with zipfile.ZipFile(ARCHIVE) as z:
        for info in z.infolist():
            if not info.filename.endswith('_room_temperature.csv.gz'):continue
            match=pattern.search(info.filename)
            if not match:raise ValueError('Unrecognized room sensor file')
            home,sid=map(int,match.groups())
            if sid not in meta.index or int(meta.loc[sid,'homeid'])!=home:raise ValueError('Metadata mismatch')
            flag=meta.loc[sid,'temperatureInaccuracy']
            entries.append((sid,home,info.filename,None if pd.isna(flag) else int(flag)))
    if len(entries)!=1958 or len({x[0] for x in entries})!=len(entries):raise ValueError('Unexpected file inventory')
    pending=[]
    for entry in entries:
        path=CACHE/f'{entry[0]}.json'
        if not path.exists() or json.loads(path.read_text(encoding='utf-8')).get('status')!='ok':pending.append(entry)
    print('Total files',len(entries),'pending',len(pending),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for i,record in enumerate(pool.map(audit,pending,chunksize=1),1):
            path=CACHE/f"{record['sensor']}.json"
            tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(record,ensure_ascii=False),encoding='utf-8');tmp.replace(path)
            if i%20==0 or i==len(pending):print('Completed',i,'of',len(pending),flush=True)
    results=pd.DataFrame([json.loads((CACHE/f'{e[0]}.json').read_text(encoding='utf-8')) for e in entries])
    results.to_csv(OUT/'ideal_sensor_quality.csv',index=False)
    perhome=results.groupby('home',as_index=False).agg(sensors=('sensor','nunique'),eligible_sensors=('coverage_eligible','sum'),
                                                       errors=('status',lambda s:int((s!='ok').sum())))
    perhome['coverage_eligible']=perhome.eligible_sensors>0
    perhome.to_csv(OUT/'ideal_home_quality.csv',index=False)
    summary={'sensor_files':len(results),'homes':int(results.home.nunique()),'processing_errors':int((results.status!='ok').sum()),
             'eligible_sensors':int(results.coverage_eligible.sum()),'eligible_homes':int(perhome.coverage_eligible.sum()),
             'raw_rows':int(results.raw_rows.sum()),'duplicate_adjacent':int(results.duplicate_adjacent.sum()),
             'time_reversals':int(results.time_reversals.sum()),'outside_time':int(results.outside_time.sum()),
             'bad_time':int(results.bad_time.sum()),'outside_15_40':int(results.outside_15_40.sum())}
    (OUT/'ideal_quality_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

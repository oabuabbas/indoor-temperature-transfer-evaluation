"""Re-audit all sensors for 10–45°C sensitivity, separate from primary cache."""
import concurrent.futures
import json
import zipfile
import pandas as pd
from audit_nest import DATA,OUT
from audit_ideal_temperature import ARCHIVE,audit

CACHE=OUT/'ideal_range_10_45_cache'

def main():
    check=json.loads((OUT/'ideal_archive_verification.json').read_text(encoding='utf-8'))
    stat=ARCHIVE.stat()
    if not check['matches'] or stat.st_size!=check['archive_bytes'] or stat.st_mtime_ns!=check['mtime_ns']:
        raise ValueError('Archive not verified or modified')
    original=pd.read_csv(OUT/'ideal_sensor_quality.csv')
    if len(original)!=1958 or not original.sensor.is_unique or not original.status.eq('ok').all():
        raise ValueError('Primary quality audit incomplete or failed')
    with zipfile.ZipFile(ARCHIVE) as archive:
        names=set(archive.namelist())
    CACHE.mkdir(exist_ok=True)
    pending=[]
    for row in original.itertuples(index=False):
        if row.member not in names:raise ValueError('Original member missing')
        path=CACHE/f'{int(row.sensor)}.json'
        if not path.exists() or json.loads(path.read_text(encoding='utf-8')).get('status')!='ok':
            pending.append((int(row.sensor),int(row.home),row.member,None if pd.isna(row.position_flag) else int(row.position_flag)))
    print('Pending',len(pending),'of',len(original),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(audit,row,10,45):row for row in pending}
        for n,future in enumerate(concurrent.futures.as_completed(futures),1):
            record=future.result();path=CACHE/f"{record['sensor']}.json";tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(record),encoding='utf-8');tmp.replace(path)
            if n%20==0 or n==len(pending):print('Completed',n,'of',len(pending),flush=True)
    wide=pd.DataFrame([json.loads((CACHE/f'{int(s)}.json').read_text(encoding='utf-8')) for s in original.sensor])
    wide.to_csv(OUT/'ideal_range_10_45_sensors.csv',index=False)
    paired=original.merge(wide,on=['sensor','home'],suffixes=('_15_40','_10_45'),validate='one_to_one')
    if not paired.status_10_45.eq('ok').all():raise ValueError('Wide-range audit has errors; inspect sensor file')
    h15=set(paired.loc[paired.coverage_eligible_15_40,'home'])
    h10=set(paired.loc[paired.coverage_eligible_10_45,'home'])
    report={'sensors':len(paired),'homes':int(paired.home.nunique()),
            'readings_outside_15_40':int(paired.outside_15_40.sum()),
            'readings_outside_10_45':int(paired.outside_10_45.sum()),
            'eligible_sensors_15_40':int(paired.coverage_eligible_15_40.sum()),
            'eligible_sensors_10_45':int(paired.coverage_eligible_10_45.sum()),
            'eligible_homes_15_40':len(h15),'eligible_homes_10_45':len(h10),
            'new_eligible_homes':len(h10-h15),'lost_eligible_homes':len(h15-h10),
            'qualified_hours_15_40':int(paired.qualified_hours_15_40.sum()),
            'qualified_hours_10_45':int(paired.qualified_hours_10_45.sum())}
    (OUT/'ideal_range_sensitivity_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

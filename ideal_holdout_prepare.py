"""Prepare held-out homes only after the local holdout protocol is frozen."""
import concurrent.futures,json
import pandas as pd
from audit_nest import OUT
from audit_ideal_temperature import ARCHIVE
from ideal_development_prepare import prepare,select

HOURLY=OUT/'ideal_holdout_hourly_v1'

def main():
    if not (OUT.parent/'protocol_ideal_holdout_v1.md').exists():raise ValueError('Holdout protocol missing')
    if (OUT/'ideal_holdout_v1_summary.csv').exists():raise ValueError('Holdout outcomes already exist; do not rerun without an audit')
    check=json.loads((OUT/'ideal_archive_verification.json').read_text(encoding='utf-8'))
    stat=ARCHIVE.stat()
    if not check['matches'] or stat.st_size!=check['archive_bytes'] or stat.st_mtime_ns!=check['mtime_ns']:raise ValueError('Unverified archive')
    selection=select('holdout').drop(columns='dev_role')
    split=pd.DataFrame(json.loads((OUT/'ideal_house_split.json').read_text(encoding='utf-8'))['homes'])
    if not set(selection.home).issubset(set(split.loc[split.split.eq('holdout'),'homeid'])):raise ValueError('Non-holdout home')
    path=OUT/'ideal_holdout_sensors_v1.csv'
    if path.exists() and not pd.read_csv(path).equals(selection.reset_index(drop=True)):raise ValueError('Holdout cohort changed')
    selection.to_csv(path,index=False)
    HOURLY.mkdir(exist_ok=True)
    items=[(int(x.home),int(x.sensor),x.member) for x in selection.itertuples(index=False) if not (HOURLY/f'{x.sensor}.csv.gz').exists()]
    print('Selected',len(selection),'holdout homes; pending',len(items),'raw sensors',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(prepare,item,10,45,HOURLY) for item in items]
        for n,f in enumerate(concurrent.futures.as_completed(futures),1):
            f.result()
            if n%10==0 or n==len(items):print('Prepared',n,'of',len(items),flush=True)
    print('Hourly preparation complete; no predictions calculated.',flush=True)

if __name__=='__main__':main()

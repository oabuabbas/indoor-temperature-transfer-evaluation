"""Construct 15–40°C hourly data for the same development sensors."""
import concurrent.futures,json
import pandas as pd
from audit_nest import OUT
from audit_ideal_temperature import ARCHIVE
from ideal_development_prepare import prepare,select

STRICT=OUT/'ideal_development_hourly_15_40_v1'

def main():
    check=json.loads((OUT/'ideal_archive_verification.json').read_text(encoding='utf-8'))
    stat=ARCHIVE.stat()
    if not check['matches'] or stat.st_size!=check['archive_bytes'] or stat.st_mtime_ns!=check['mtime_ns']:raise ValueError('Unverified archive')
    selected=select()
    if not pd.read_csv(OUT/'ideal_development_sensors_v1.csv').equals(selected.reset_index(drop=True)):
        raise ValueError('Original development cohort differs')
    STRICT.mkdir(exist_ok=True)
    items=[(int(x.home),int(x.sensor),x.member) for x in selected.itertuples(index=False) if not (STRICT/f'{x.sensor}.csv.gz').exists()]
    print('selected',len(selected),'pending',len(items),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(prepare,item,15,40,STRICT) for item in items]
        for n,f in enumerate(concurrent.futures.as_completed(futures),1):
            f.result()
            if n%20==0 or n==len(items):print('Prepared',n,'of',len(items),flush=True)

if __name__=='__main__':main()

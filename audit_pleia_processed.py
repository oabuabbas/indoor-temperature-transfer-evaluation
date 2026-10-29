"""Profile provider-processed PLEIA room records without claiming independent validation."""
import json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from audit_nest import DATA,OUT

def main():
    path=DATA/'PLEIAData_zenodo_7620136.zip'
    rows=[]
    with zipfile.ZipFile(path) as z:
        for block in 'ABC':
            name=f'Data_Nature/processed_data/data-room{block}-10T.csv'
            sensors=set(); rooms=set(); n=0; missing={c:0 for c in ['Date','IDsensor','V2','tmed','radmed']}
            tmin=None;tmax=None
            with z.open(name) as f:
                for d in pd.read_csv(f,sep=';',usecols=['Date','IDsensor','room','V2','tmed','radmed'],chunksize=100000,low_memory=False):
                    n+=len(d)
                    for c in missing: missing[c]+=int(d[c].isna().sum())
                    sensors.update(d.IDsensor.dropna().astype(str).unique());rooms.update(d.room.dropna().astype(str).unique())
                    t=pd.to_datetime(d.Date,errors='coerce',utc=True)
                    if t.notna().any():
                        a=t.min(); b=t.max();tmin=a if tmin is None else min(tmin,a);tmax=b if tmax is None else max(tmax,b)
            rows.append({'block':block,'rows':n,'unique_sensor_ids':len(sensors),'unique_room_ids':len(rooms),'first_utc':str(tmin),'last_utc':str(tmax),**{f'missing_{c}_pct':round(100*v/n,4) for c,v in missing.items()}})
            print(block,n,'rows',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'pleia_processed_profile.csv',index=False)

if __name__=='__main__':main()

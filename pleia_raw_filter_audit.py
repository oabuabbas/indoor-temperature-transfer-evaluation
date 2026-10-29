"""Disaggregate raw PLEIA exclusions; no target imputation."""
import zipfile
import pandas as pd
from audit_nest import DATA,OUT

def main():
    parts=[]
    with zipfile.ZipFile(DATA/'PLEIAData_zenodo_7620136.zip') as z:
        with z.open('Data_Nature/raw_data/relations-sensor.csv') as f:
            mapping=pd.read_csv(f,sep=';',dtype={'ID':'string','block':'string','room':'string'}).dropna().drop_duplicates()
        with z.open('Data_Nature/raw_data/data-sensor.csv') as f:
            for d in pd.read_csv(f,sep=';',usecols=['IDdevice','V2'],dtype={'IDdevice':'string'},chunksize=200000):
                x=pd.to_numeric(d.V2,errors='coerce')
                frame=pd.DataFrame({'IDdevice':d.IDdevice,'rows':1,'null':x.isna().astype(int),'below15':(x<15).astype(int),'above40':(x>40).astype(int),'inrange':x.between(15,40).astype(int)})
                parts.append(frame.groupby('IDdevice',dropna=False)[['rows','null','below15','above40','inrange']].sum())
    a=pd.concat(parts).groupby(level=0).sum().reset_index()
    a=a.merge(mapping,left_on='IDdevice',right_on='ID',how='left',validate='one_to_many')
    a.to_csv(OUT/'pleia_raw_filter_by_sensor.csv',index=False)
    print('Totals:',a[['rows','null','below15','above40','inrange']].sum().to_dict())
    print('Mapped sensor IDs',a[a.ID.notna()].IDdevice.nunique(),'raw sensor IDs',a.IDdevice.nunique())

if __name__=='__main__':main()

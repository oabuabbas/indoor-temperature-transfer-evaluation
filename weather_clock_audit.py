"""Compare weather-clock hypotheses against approximate local solar noon."""
import datetime as dt
import zipfile
import numpy as np
import pandas as pd
from audit_nest import DATA,OUT

LONGITUDE=-1.13471944
def main():
    with zipfile.ZipFile(DATA/'PLEIAData_zenodo_7620136.zip') as z:
        with z.open('Data_Nature/raw_data/MU62_dm.txt') as f:
            d=pd.read_csv(f,sep=';',skipinitialspace=True)
    naive=pd.to_datetime(d.fecha.astype(str).str.strip()+' '+d.hora.astype(str).str.strip(),format='%d/%m/%y %H:%M:%S',errors='coerce')
    rad=pd.to_numeric(d.radmed,errors='coerce'); results=[]
    hypotheses={'utc':naive.dt.tz_localize('UTC'),'madrid':naive.dt.tz_localize('Europe/Madrid',ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC'),'fixed_utc_plus_1':naive.dt.tz_localize(dt.timezone(dt.timedelta(hours=1))).dt.tz_convert('UTC')}
    for hypothesis,t in hypotheses.items():
        invalid=int(t.isna().sum());w=pd.DataFrame({'utc':t,'rad':rad}).dropna();w=w[(w.rad>=0)&(w.rad<=1500)].copy();w['date']=w.utc.dt.floor('D')
        counts=w.groupby('date').rad.agg(['count','max']);eligible=counts[(counts['count']>=100)&(counts['max']>=600)].index
        w=w[w.date.isin(eligible)].sort_values('utc'); peaks=w.loc[w.groupby('date').rad.idxmax()].copy()
        doy=peaks.date.dt.dayofyear.to_numpy(); b=2*np.pi*(doy-81)/364; e=9.87*np.sin(2*b)-7.53*np.cos(b)-1.5*np.sin(b)
        utc_min=peaks.utc.dt.hour.to_numpy()*60+peaks.utc.dt.minute.to_numpy()+peaks.utc.dt.second.to_numpy()/60
        peaks['delta_minutes']=utc_min-(720-4*LONGITUDE-e)
        for season,months in [('all',list(range(1,13))),('winter',[1,2,12]),('summer',[6,7,8])]:
            p=peaks[peaks.date.dt.month.isin(months)]
            results.append({'hypothesis':hypothesis,'season':season,'excluded_invalid_or_ambiguous_timestamps':invalid,'qualifying_days':len(p),'median_signed_offset_minutes':p.delta_minutes.median(),'median_absolute_offset_minutes':p.delta_minutes.abs().median()})
    pd.DataFrame(results).round(3).to_csv(OUT/'weather_clock_hypotheses.csv',index=False)
    print(pd.DataFrame(results).round(1).to_string(index=False))

if __name__=='__main__':main()

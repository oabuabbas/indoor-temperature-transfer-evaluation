"""Exploratory source-only NEST -> ORNL transfer across short episodes."""
import json,urllib.request
import numpy as np
import pandas as pd
from audit_nest import ITEMS,DATA,OUT,download,digest
from adaptation_stress import fit,predict
from daily_residual import nest_source,FEATURES

ARTICLE=20520438
CASE_IDS={'base_cooling':37624448,'base_heating':37624451,'freefloat_cooling':37624457,
 'freefloat_heating':37624463,'pre_heating':37624454,'setback_cooling':37624460,'setback_heating':37624469}
ROOMS=['T_Room_102','T_Room_103','T_Room_104','T_Room_105','T_Room_106','T_Room_202','T_Room_203','T_Room_204','T_Room_205','T_Room_206']
H=[1,6,24]

def load_files():
    meta=json.load(urllib.request.urlopen(f'https://api.figshare.com/v2/articles/{ARTICLE}',timeout=40)); manifest=[];cases={}
    for case,fid in CASE_IDS.items():
        info=next(f for f in meta['files'] if f['id']==fid)
        path=DATA/('ornl_'+info['name']);download(fid,path)
        md5=digest(path,'md5')
        if md5!=info['computed_md5'] or path.stat().st_size!=info['size']:raise ValueError('checksum mismatch '+case)
        manifest.append({'case':case,'id':fid,'name':info['name'],'url':info['download_url'],'md5':md5,'sha256':digest(path,'sha256'),'bytes':path.stat().st_size})
        d=pd.read_csv(path,usecols=['TIMESTAMP']+ROOMS,low_memory=False)
        t=pd.to_datetime(d.pop('TIMESTAMP'),format='%m/%d/%Y %H:%M',errors='coerce');bad=int(t.isna().sum());dup=t.duplicated(keep=False)&t.notna();keep=t.notna()&~dup
        d=d.loc[keep].apply(pd.to_numeric,errors='coerce');d=d.where(d.ge(15)&d.le(40));d.index=t[keep];d=d.sort_index()
        r=d.resample('1h');cases[case]=(r.mean().where(r.count()>=45),bad,int(dup.sum()))
        print(case,'loaded',flush=True)
    (OUT/'ornl_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return cases

def predict_frame(hourly,room,h):
    s=hourly[room];hour=s.index.hour.to_numpy()
    d=pd.DataFrame({'temp_now':s,'temp_lag1':s.shift(1),'seasonal_yesterday':s.shift(24-h),'hour_sin':np.sin(2*np.pi*hour/24),'hour_cos':np.cos(2*np.pi*hour/24),'target':s.shift(-h)},index=s.index)
    return d.dropna()

def main():
    cases=load_files();rows=[];quality=[]
    for case,(hourly,bad,dup) in cases.items():
        quality.append({'case':case,'hours':len(hourly),'unit_or_bad_timestamp_rows':bad,'duplicate_time_rows':dup,'start':str(hourly.index.min()),'end':str(hourly.index.max()),'room_hours_missing_pct':round(float(hourly[ROOMS].isna().mean().mean()*100),4)})
    for h in H:
        src=pd.concat([nest_source(name,item[3],h) for unit,item in ITEMS.items() for name in [item[2],unit.lower()+'_2020-2021.csv']])
        model=fit(src[FEATURES].to_numpy(),(src.target-src.seasonal_yesterday).to_numpy())
        for case,(hourly,_,_) in cases.items():
            for room in ROOMS:
                d=predict_frame(hourly,room,h)
                if len(d)<24:continue
                predictions={'persistence':d.temp_now.to_numpy(),'daily_seasonal':d.seasonal_yesterday.to_numpy(),'source_residual':d.seasonal_yesterday.to_numpy()+predict(model,d[FEATURES].to_numpy())}
                for label,pred in predictions.items():
                    e=pred-d.target.to_numpy()
                    rows.append({'horizon_h':h,'case':case,'room':room,'model':label,'source_n':len(src),'test_n':len(d),'mae_c':np.abs(e).mean(),'rmse_c':np.sqrt(np.square(e).mean())})
        print('horizon',h,'done',flush=True)
    pd.DataFrame(rows).round(6).to_csv(OUT/'ornl_external_results.csv',index=False)
    pd.DataFrame(quality).to_csv(OUT/'ornl_quality.csv',index=False)

if __name__=='__main__':main()

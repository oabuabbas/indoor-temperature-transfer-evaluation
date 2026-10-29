"""Audit completed raw ZIP members before full download; verify each member CRC."""
import json
import concurrent.futures
import re
import struct
import zlib
import zipfile
import pandas as pd
from audit_nest import DATA, OUT
from audit_ideal_temperature import CACHE, audit

ROOT=DATA/'ideal'
FULL_SIZE=10005114253

def main():
    prefix=ROOT/'room_and_appliance_sensors.zip.part'
    tail=ROOT/'room_archive_tail.bin'
    if not prefix.exists():raise FileNotFoundError(prefix)
    size=prefix.stat().st_size
    base=FULL_SIZE-tail.stat().st_size
    with zipfile.ZipFile(ROOT/'metadata_and_surveys.zip') as z:
        def read(name,cols):
            with z.open('metadata/'+name+'.csv') as f:return pd.read_csv(f,usecols=cols,low_memory=False)
        s=read('sensor',['sensorid','sensorboxid','type','roomid'])
        b=read('sensorbox',['sensorboxid','sensorbox_type','temperatureInaccuracy'])
        r=read('room',['roomid','homeid'])
    meta=s[s.type.eq('temperature')].merge(b,on='sensorboxid',validate='many_to_one')
    meta=meta[meta.sensorbox_type.eq('room')].merge(r,on='roomid',validate='many_to_one').set_index('sensorid')
    pattern=re.compile(r'home(\d+)_.*?_sensor(\d+)_room_temperature\.csv\.gz$')
    CACHE.mkdir(exist_ok=True)
    count=0
    futures={}
    def save_done(done):
        nonlocal count
        for future in done:
            path=futures.pop(future)
            record=future.result()
            record['prefix_member_crc_verified']=True
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record),encoding='utf-8');tmp.replace(path)
            count+=1
            if count%20==0:print('Audited',count,'available files',flush=True)
    with zipfile.ZipFile(tail) as z, concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for info in z.infolist():
            if not info.filename.endswith('_room_temperature.csv.gz'):continue
            match=pattern.search(info.filename)
            if not match:raise ValueError('Unexpected filename')
            home,sid=map(int,match.groups())
            if sid not in meta.index or int(meta.loc[sid,'homeid'])!=home:raise ValueError('Metadata mismatch')
            path=CACHE/f'{sid}.json'
            if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('status')=='ok':continue
            pos=base+info.header_offset
            if pos<0 or pos+30>size:continue
            with prefix.open('rb') as source:
                source.seek(pos)
                header=struct.unpack('<4s5H3I2H',source.read(30))
                if header[0]!=b'PK\x03\x04':raise ValueError('ZIP local header missing')
                nlen,elen=header[-2:]
                if pos+30+nlen+elen+info.compress_size>size:continue
                name=source.read(nlen).decode('utf-8')
                if name!=info.filename:raise ValueError('ZIP member name mismatch')
                source.seek(elen,1)
                compressed=source.read(info.compress_size)
            if info.compress_type==8:payload=zlib.decompress(compressed,-15)
            elif info.compress_type==0:payload=compressed
            else:raise ValueError('Unsupported compression')
            if len(payload)!=info.file_size or (zlib.crc32(payload)&0xffffffff)!=info.CRC:raise ValueError('ZIP member CRC mismatch')
            flag=meta.loc[sid,'temperatureInaccuracy']
            future=pool.submit(audit,(sid,home,info.filename,None if pd.isna(flag) else int(flag),payload))
            futures[future]=path
            if len(futures)>=16:
                done,_=concurrent.futures.wait(futures,return_when=concurrent.futures.FIRST_COMPLETED)
                save_done(done)
        while futures:
            done,_=concurrent.futures.wait(futures,return_when=concurrent.futures.FIRST_COMPLETED)
            save_done(done)
    print('Finished prefix audit; new files',count,'prefix bytes',size,flush=True)

if __name__=='__main__':main()

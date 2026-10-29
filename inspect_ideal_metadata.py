"""Structural metadata inventory only; no sensor timeseries or forecast results."""
import hashlib
import json
import re
import zipfile
import pandas as pd
from audit_nest import DATA, OUT

ROOT=DATA/'ideal'
ARCHIVES=['documentation.zip','metadata_and_surveys.zip']
PROVIDER_MD5={'documentation.zip':'9d0efd702807ce81f3ef92b5634edcb0',
              'metadata_and_surveys.zip':'910db8febadab9cc4a034afdb1a271ea'}

def file_hash(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read_table(z,name,cols):
    with z.open('metadata/'+name+'.csv') as f:
        return pd.read_csv(f,usecols=cols,low_memory=False)

def main():
    manifest=[]
    for name in ARCHIVES:
        p=ROOT/name
        with zipfile.ZipFile(p) as z:
            corrupt=z.testzip()
            if corrupt is not None:raise ValueError('Corrupt ZIP member: '+corrupt)
            md5=file_hash(p,'md5')
            if md5!=PROVIDER_MD5[name]:raise ValueError('Provider MD5 mismatch: '+name)
            manifest.append({'file':name,'bytes':p.stat().st_size,'local_sha256':file_hash(p),
                             'member_count':len(z.namelist()),'provider_md5':PROVIDER_MD5[name],
                             'provider_checksum_verified':True})
    with zipfile.ZipFile(ROOT/'metadata_and_surveys.zip') as z:
        s=read_table(z,'sensor',['sensorid','sensorboxid','type','unit','status','roomid'])
        b=read_table(z,'sensorbox',['sensorboxid','roomid','sensorbox_type','temperatureInaccuracy'])
        r=read_table(z,'room',['roomid','homeid','type'])
        home=read_table(z,'home',['homeid'])
    if any(df[idcol].duplicated().any() for df,idcol in [(s,'sensorid'),(b,'sensorboxid'),(r,'roomid'),(home,'homeid')]):
        raise ValueError('Duplicate primary metadata IDs')
    temp=s[s.type.eq('temperature')].merge(b,on='sensorboxid',how='left',suffixes=('_sensor','_box'),validate='many_to_one')
    candidate=temp[temp.sensorbox_type.eq('room')].merge(r,left_on='roomid_sensor',right_on='roomid',how='left',validate='many_to_one')
    def counts(series):return {str(k):int(v) for k,v in series.fillna('MISSING').value_counts().items()}
    status={'archives':manifest,'home_metadata_records':int(len(home)),
            'all_temperature_sensors':int(len(temp)),'temperature_by_sensorbox_type':counts(temp.sensorbox_type),
            'room_temperature_candidates':int(len(candidate)),
            'homes_with_room_temperature_metadata':int(candidate.homeid.nunique()),
            'unique_rooms_with_room_temperature_metadata':int(candidate.roomid.nunique()),
            'missing_home_or_room_links':int(candidate.homeid.isna().sum()),
            'roomid_sensor_box_disagreements':int((candidate.roomid_sensor!=candidate.roomid_box).sum()),
            'candidate_status':counts(candidate.status),'candidate_units':counts(candidate.unit),
            'candidate_inaccuracy_flags':counts(candidate.temperatureInaccuracy),
            'candidate_by_room_type':counts(candidate.type_y),
            'note':'Metadata presence only; no raw sensor files inspected, no eligibility or data quality claims.'}
    tail=ROOT/'room_archive_tail.bin'
    if tail.exists():
        # zipfile can list the central directory in the range-fetched suffix, but
        # its member offsets do not point into this truncated file: never extract.
        with zipfile.ZipFile(tail) as z:
            members=z.infolist()
            files=[info for info in members if info.filename.endswith('_room_temperature.csv.gz')]
            pattern=re.compile(r'home(\d+)_.*?_sensor(\d+)_room_temperature\.csv\.gz$')
            found=[pattern.search(info.filename) for info in files]
            if any(match is None for match in found):raise ValueError('Unexpected temperature filename')
            actual=pd.DataFrame([(int(match.group(1)),int(match.group(2))) for match in found],columns=['homeid','sensorid'])
            if actual.sensorid.duplicated().any():raise ValueError('Repeated sensor filename IDs')
            expected=candidate[['homeid','sensorid']]
            joined=actual.merge(expected,on=['homeid','sensorid'],how='left',indicator=True,validate='one_to_one')
            status['raw_archive_directory']={'tail_bytes':tail.stat().st_size,'central_directory_records':len(members),
                'room_temperature_files':len(files),'homes_with_named_temperature_files':int(actual.homeid.nunique()),
                'files_not_matching_metadata_home_and_sensor':int(joined['_merge'].ne('both').sum()),
                'metadata_candidates_without_named_file':len(set(expected.sensorid)-set(actual.sensorid)),
                'temperature_members_compressed_bytes':sum(a.compress_size for a in files),
                'temperature_members_uncompressed_bytes':sum(a.file_size for a in files),
                'note':'Central-directory names and sizes only; gzip payloads and CRCs were not downloaded or checked.'}
    (OUT/'ideal_metadata_inventory.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

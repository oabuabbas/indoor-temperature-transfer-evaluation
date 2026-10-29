"""Reproducible house split based on public metadata IDs, not measurements."""
import hashlib
import json
import zipfile
import pandas as pd
from audit_nest import DATA,OUT

NAMESPACE='ideal-room-transfer-holdout-v1:'
THRESHOLD=(7*(2**64))//10

def assignment(home):
    digest=hashlib.sha256((NAMESPACE+str(int(home))).encode('utf-8')).digest()
    return 'development' if int.from_bytes(digest[:8],'big')<THRESHOLD else 'holdout'

def main():
    with zipfile.ZipFile(DATA/'ideal'/'metadata_and_surveys.zip') as z, z.open('metadata/home.csv') as f:
        home=pd.read_csv(f,usecols=['homeid'])
    if home.homeid.isna().any() or home.homeid.duplicated().any():raise ValueError('Invalid home IDs')
    records=[{'homeid':int(v),'split':assignment(v)} for v in sorted(home.homeid)]
    data={'scheme':'sha256-first-8-bytes-big-endian','namespace':NAMESPACE,
          'development_threshold':str(THRESHOLD),'home_count':len(records),'homes':records}
    path=OUT/'ideal_house_split.json'
    content=json.dumps(data,indent=2)+'\n'
    if path.exists() and path.read_text(encoding='utf-8')!=content:
        raise ValueError('Existing split differs; do not overwrite a frozen split')
    path.write_text(content,encoding='utf-8')
    print(pd.Series([v['split'] for v in records]).value_counts().to_string())

if __name__=='__main__':main()

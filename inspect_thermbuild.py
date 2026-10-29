"""Inventory official ThermBuild raw archives without model evaluation."""
import json,urllib.request,zipfile
from audit_nest import DATA,OUT,digest

FILES={'ThermBuild_measure_raw.zip':'https://fordatis.fraunhofer.de/bitstream/fordatis/486/1/ThermBuild_measure_raw.zip',
       'ThermBuild_measure_Temp_raw.zip':'https://fordatis.fraunhofer.de/bitstream/fordatis/486/2/ThermBuild_measure_Temp_raw.zip'}
def main():
    manifest=[]
    for name,url in FILES.items():
        path=DATA/name;part=path.with_suffix('.part')
        if not path.exists():
            print('Downloading',name,flush=True)
            try:
                with urllib.request.urlopen(url,timeout=120) as src,part.open('wb') as dst:
                    while block:=src.read(1024*1024):dst.write(block)
                part.replace(path)
            except Exception:
                part.unlink(missing_ok=True);raise
        item={'name':name,'url':url,'bytes':path.stat().st_size,'sha256':digest(path,'sha256'),'note':'Locally calculated SHA-256; no provider checksum verified','members':[]}
        with zipfile.ZipFile(path) as z:
            for f in z.infolist():
                rec={'name':f.filename,'bytes':f.file_size}
                if not f.is_dir() and f.filename.lower().endswith(('.txt','.csv','.json')):
                    with z.open(f) as s:rec['header']=s.readline(16000).decode('utf-8-sig',errors='replace').strip()
                item['members'].append(rec)
        manifest.append(item)
        print(name,'members',len(item['members']),flush=True)
    (OUT/'thermbuild_inventory.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    for item in manifest:
        for m in item['members']:print(m['name'],m['bytes'],m.get('header','')[:300],flush=True)
if __name__=='__main__':main()
